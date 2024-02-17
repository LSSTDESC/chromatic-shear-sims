import argparse
import logging
import copy
import os

import galsim
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import run_utils, roman_rubin, DC2_stars, surveys
from chromatic_shear_bias.pipeline.pipeline import Pipeline
from chromatic_shear_bias.pipeline import logging_config


logger = logging.getLogger(__name__)


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}

TOL = 0.01


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

    colors = pipeline.get_colors()
    if colors:
        chroma_stars = []
        for color in colors:
            predicate = (pc.abs_checked(pc.field("gmag_obs") - pc.field("imag_obs") - color) < TOL)
            star_params = pipeline.stars.sample_with(
                1,
                columns=dc2builder.columns,
                predicate=predicate,
            )
            chroma_star = dc2builder.build_stars(star_params)[0]
            chroma_stars.append(chroma_star)
        star = chroma_stars[1]
    else:
        star_params = pipeline.stars.sample(
            1,
            columns=dc2builder.columns,
        )
        star = dc2builder.build_stars(star_params)[0]

    # scene & image
    image_config = pipeline.config.get("image")
    scene_pos = pipeline.get_scene_pos(lsst, seed=seed)
    n_gals = len(scene_pos)

    # galaxies
    romanrubinbuilder = roman_rubin.RomanRubinBuilder(
        diffskypop_params=pipeline.config.get("diffskypop_params"),
        ssp_templates=pipeline.config.get("ssp_templates"),
        survey=lsst,
    )

    gal_params = pipeline.galaxies.sample(
        n_gals,
        columns=romanrubinbuilder.columns,
    )
    # predicate = (pc.abs_checked(pc.field("LSST_obs_g") - pc.field("LSST_obs_i") - chroma_colors[1]) < TOL)
    # gal_params = pipeline.galaxies.sample_with(
    #     n_gals,
    #     columns=romanrubinbuilder.columns,
    #     predicate=predicate,
    # )
    galaxies = romanrubinbuilder.build_gals(gal_params)

    scene = [
        gal.shift(pos)
        for (gal, pos) in zip(galaxies, scene_pos)
    ]

    pair_seed = rng.integers(1, 2**64 // 2 - 1)
    bands = image_config["bands"]
    shear = image_config["shear"]

    logger.info(f"building simulation pair")
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
        logger.info(f"running detection")
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

    logger.info(f"making plot")
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
            ax.text(p_ns["sx_col"][j], p_ns["sx_row"][j], round(p_ns[model + "_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
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
            ax.text(m_ns["sx_col"][j], m_ns["sx_row"][j], round(m_ns[model + "_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
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
        action="store_true",
        help="Whether to make detections",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    return parser.parse_args()


def main():
    args = get_args()

    logger_config = logging_config.defaults
    log_level = logging_config.get_level(args.log_level)
    logging.basicConfig(level=log_level, **logging_config.defaults)

    logger.info(f"{vars(args)}")

    pa.set_cpu_count(8)
    pa.set_io_thread_count(8)

    config = args.config
    seed = args.seed
    n_sims = args.n_sims
    # n_jobs = args.n_jobs
    detect = args.detect

    pipeline = Pipeline(config)

    pipeline.load()
    pipeline.load_galaxies()
    pipeline.load_stars()

    rng = np.random.default_rng(seed)

    for seed in rng.integers(1, 2**32, n_sims):
        run_pipeline(
            config, seed=seed, detect=detect
        )


if __name__ == "__main__":
    main()
