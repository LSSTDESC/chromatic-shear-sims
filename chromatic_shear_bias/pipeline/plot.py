import argparse
import copy
import os

import galsim
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import run_utils, roman_rubin, DC2_stars, surveys
from chromatic_shear_bias.pipeline.pipeline import Pipeline


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}


def run_pipeline(config, seed=None, detect=False):
    rng = np.random.default_rng(seed)

    pipeline = Pipeline(config)

    pipeline.load()
    pipeline.load_galaxies()
    pipeline.load_stars()
    # pipeline.save(overwrite=True)

    measure_config = pipeline.config.get("measure")
    measure_type = measure_config.get("type")

    lsst = surveys.lsst

    # psf
    psf = pipeline.get_psf()

    # star
    dc2builder = DC2_stars.DC2Builder(
        sed_dir=pipeline.config.get("sed_dir"),
    )
    star_params = pipeline.stars.sample(
        1,
        columns=dc2builder.columns,
    )
    star = dc2builder.build_star(star_params)

    # if using a chromatic measure, generate stars at the appropriate colors
    if measure_type in CHROMATIC_MEASURES:
        colors_type = measure_config.get("colors")
        match colors_type:
            case "quantiles":
                chroma_colors = pipeline.galaxies.aggregate.get("quantiles")
            case "uniform":
                colors_min = pipeline.galaxies.aggregate.get("min_color")[0]
                colors_max = pipeline.galaxies.aggregate.get("max_color")[0]
                # TODO add config for number of colors here...
                chroma_colors = np.linspace(colors_min, colors_max, 3)
            case "centered":
                median = pipeline.galaxies.aggregate.get("median_color")[0]
                chroma_colors = [median - 0.1, median, median + 0.1]
            case _:
                raise ValueError(f"Colors type {colors_type} not valid!")

        chroma_stars = []
        for color in chroma_colors:
            TOL = 0.01
            predicate = (pc.abs_checked(pc.field("gmag") - pc.field("imag") - color) < TOL)
            star_params = pipeline.stars.sample_with(
                1,
                columns=dc2builder.columns,
                predicate=predicate,
            )
            chroma_star = dc2builder.build_star(star_params)
            chroma_stars.append(chroma_star)

    # scene & image
    image_config = pipeline.config.get("image")
    scene_config = pipeline.config.get("scene")
    match (scene_type := scene_config.get("type")):
        case "single":
            n_gals = 1
            xs = [0]
            ys = [0]
        case "random":
            n_gals = scene_config["n"]
            xs = rng.uniform(
                -image_config["xsize"] // 2 + scene_config["border"],
                image_config["xsize"] // 2 - scene_config["border"],
                n_gals,
             )
            ys = rng.uniform(
                -image_config["ysize"] // 2 + scene_config["border"],
                image_config["ysize"] // 2 - scene_config["border"],
                n_gals,
             )
        case "hex":
            v1 = np.asarray([1, 0], dtype=float)
            v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)
            xs, ys = run_utils.build_lattice(
                lsst,
                image_config["xsize"],
                image_config["ysize"],
                scene_config["separation"],
                v1,
                v2,
                rng.uniform(0, 360),
                scene_config["border"],
            )  # pixels
            if len(xs) < 1:
                raise ValueError(f"Scene containts no objects!")
            n_gals = len(xs)
        case _:
            raise ValueError(f"Scene type {scene_type} not valid!")

    scene_pos = [
        galsim.PositionD(
            x=x * lsst.scale,
            y=y * lsst.scale,
        )
        for (x, y) in zip(xs, ys)
    ]
    if (dither := scene_config.get("dither", False)):
        scene_pos = [
            pos + galsim.PositionD(
                rng.uniform(-dither, dither) * lsst.scale,
                rng.uniform(-dither, dither) * lsst.scale,
            )
            for pos in scene_pos
        ]

    # galaxies
    romanrubinbuilder = roman_rubin.RomanRubinBuilder(
        diffskypop_params=pipeline.config.get("diffskypop_params"),
        ssp_templates=pipeline.config.get("ssp_templates"),
    )
    gal_params = pipeline.galaxies.sample(
        n_gals,
        columns=romanrubinbuilder.columns,
    )
    galaxies = romanrubinbuilder.build_gals(gal_params)
    print(galaxies[0].calculateMagnitude(lsst.bandpasses["g"]) - galaxies[0].calculateMagnitude(lsst.bandpasses["i"]))

    scene = [
        gal.shift(pos)
        for (gal, pos) in zip(galaxies, scene_pos)
    ]

    pair_seed = rng.integers(1, 2**64 // 2 - 1)
    bands = image_config["bands"]
    shear = image_config["shear"]

    pair = run_utils.build_pair(
        lsst,
        scene,
        star,
        shear,
        psf,
        bands,
        image_config["xsize"],
        image_config["ysize"],
        image_config["psf_size"],
        pair_seed,
    )

    if detect:
        import metadetect
        mdet_rng_p = np.random.default_rng(42)
        mdet_rng_m = np.random.default_rng(42)

        res_p = metadetect.do_metadetect(
            pipeline.metadetect_config,
            pair["plus"],
            mdet_rng_p,
        )

        res_m = metadetect.do_metadetect(
            pipeline.metadetect_config,
            pair["minus"],
            mdet_rng_m,
        )

        model = pipeline.metadetect_config["model"]
        if model == "wmom":
            tcut = 1.2
        else:
            tcut = 0.5

        # s2n_cut = 10
        # t_ratio_cut = tcut
        # mfrac_cut = 10
        s2n_cut = 0
        t_ratio_cut = 0
        mfrac_cut = 0
        ormask_cut = None

        def _mask(data):
            if "flags" in data.dtype.names:
                flag_col = "flags"
            else:
                flag_col = model + "_flags"

            _cut_msk = (
                (data[flag_col] == 0)
                & (data[model + "_s2n"] > s2n_cut)
                & (data[model + "_T_ratio"] > t_ratio_cut)
            )
            if ormask_cut:
                _cut_msk = _cut_msk & (data["ormask"] == 0)
            if mfrac_cut is not None:
                _cut_msk = _cut_msk & (data["mfrac"] <= mfrac_cut)
            return _cut_msk

        o_p = res_p["noshear"]
        q_p = _mask(o_p)
        o_m = res_m["noshear"]
        q_m = _mask(o_m)
        p_ns = o_p[q_p]
        m_ns = o_m[q_m]

    zp_g = lsst.bandpasses["g"].zeropoint
    zp_i = lsst.bandpasses["i"].zeropoint
    if detect:
        for j in range(len(p_ns)):
            mag_g = -2.5 * np.log10(p_ns["wmom_band_flux"][j][0]) + zp_g
            mag_i = -2.5 * np.log10(p_ns["wmom_band_flux"][j][2]) + zp_i
            print(mag_g - mag_i)


    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import Divider, Size

    fig = plt.figure(figsize=(8.5, 5.5))
    h = [
        Size.Fixed(1),
        Size.Fixed(2),
        Size.Fixed(0.5),
        Size.Fixed(2),
        Size.Fixed(0.5),
        Size.Fixed(2),
        Size.Fixed(0.5),
    ]
    v = [
        Size.Fixed(0.5),
        Size.Fixed(2),
        Size.Fixed(0.5),
        Size.Fixed(2),
        Size.Fixed(0.5),
    ]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)

    # for i, band in enumerate(bands):
    #     ax = fig.add_axes(
    #         divider.get_position(),
    #         axes_locator=divider.new_locator(nx=2 * i + 1, ny=3)
    #     )
    #     ax.imshow(np.arcsinh(pair["plus"][i][0].image), origin="lower")
    #     ax.set_title(f"{band}")

    #     ax = fig.add_axes(
    #         divider.get_position(),
    #         axes_locator=divider.new_locator(nx=2 * i + 1, ny=1)
    #     )
    #     ax.imshow(np.arcsinh(pair["minus"][i][0].image), origin="lower")
    i = 1
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=3)
    )
    ax.imshow(np.arcsinh(pair["plus"][i][0].image), origin="lower")
    if detect:
        for j in range(len(p_ns)):
            # axs[i, 0].annotate(round(p_ns["wmom_s2n"][j]), (p_ns["sx_col"][j], p_ns["sx_row"][j]), c="r")
            ax.text(p_ns["sx_col"][j], p_ns["sx_row"][j], round(p_ns["wmom_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
    ax.set_title(f"{bands[i]} image")

    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=3, ny=3)
    )
    ax.imshow(np.arcsinh(pair["plus"][i][0].psf.image), origin="lower")
    ax.set_title(f"{bands[i]} psf")

    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=5, ny=3)
    )
    ax.imshow(np.arcsinh(pair["plus"][i][0].noise), origin="lower")
    ax.set_title(f"{bands[i]} noise")

    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    ax.imshow(np.arcsinh(pair["minus"][i][0].image), origin="lower")
    if detect:
        for j in range(len(m_ns)):
            # axs[i, 1].annotate(round(m_ns["wmom_s2n"][j]), (m_ns["sx_col"][j], m_ns["sx_row"][j]), c="r")
            ax.text(m_ns["sx_col"][j], m_ns["sx_row"][j], round(m_ns["wmom_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
    ax.set_title(f"{bands[i]} image")

    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=3, ny=1)
    )
    ax.imshow(np.arcsinh(pair["minus"][i][0].psf.image), origin="lower")
    ax.set_title(f"{bands[i]} psf")

    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=5, ny=1)
    )
    ax.imshow(np.arcsinh(pair["minus"][i][0].noise), origin="lower")
    ax.set_title(f"{bands[i]} noise")

    plt.show()

    return


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed [int]",
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        required=False,
        default=1,
        help="Number of sims to run [int; 1]"
    )
    # parser.add_argument(
    #     "--n_jobs",
    #     type=int,
    #     required=False,
    #     default=1,
    #     help="Number of parallel jobs to run [int; 1]"
    # )
    parser.add_argument(
        "--detect",
        type=bool,
        required=False,
        default=False,
        help="Whether to make detections [bool; False]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = args.config
    seed = args.seed
    n_sims = args.n_sims
    # n_jobs = args.n_jobs
    detect = args.detect

    pipeline = Pipeline(config)
    print("pipeline:", pipeline.name)
    print("seed:", seed)

    print("measure:", pipeline.config["measure"]["type"])

    print("image:", pipeline.config.get("image"))
    print("scene:", pipeline.config.get("scene"))

    pipeline.load()
    pipeline.load_galaxies()
    pipeline.load_stars()

    print("galxies:", pipeline.galaxies.aggregate)
    print("stars:", pipeline.stars.aggregate)

    rng = np.random.default_rng(seed)

    for seed in rng.integers(1, 2**32, n_sims):
        run_pipeline(
            config, seed=seed, detect=detect
        )

    print("done!")

