import argparse
import os

import galsim
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import run_utils, roman_rubin, DC2_stars, surveys
from pipeline import Pipeline


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
        "--output",
        type=str,
        required=False,
        default="",
        help="Output directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    seed = args.seed
    rng = np.random.default_rng(seed)

    pipeline = Pipeline(args.config, load=False)
    print("pipeline:", pipeline.name)
    print("seed:", seed)
    print("cpu count:", pa.cpu_count())
    print("thread_count:", pa.io_thread_count())

    # pipeline.load()
    pipeline.load_galaxies()
    pipeline.save(overwrite=True)
    pipeline.load_stars()
    pipeline.save(overwrite=True)

    print("galxies:", pipeline.galaxies.aggregate)
    print("stars:", pipeline.stars.aggregate)

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

    # more stars if we need them
    # if (quantiles := pipeline.config.get("quantiles", None)) is not None:
    if pipeline.config.get("chromatic", False):
        chroma_colors = pipeline.galaxies.aggregate.get("quantiles")
        print("colors:", chroma_colors)
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
        case "random":
            n_gals = scene_config["n"]
            xs = rng.uniform(
                -image_config["xsize"] // 2,
                image_config["xsize"] // 2,
                n_gals,
             )
            ys = rng.uniform(
                -image_config["ysize"] // 2,
                image_config["ysize"] // 2,
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

    print("scene:", scene_type)

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
    print("dither:", dither)

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

    scene = [
        gal.shift(pos)
        for (gal, pos) in zip(galaxies, scene_pos)
    ]

    pair_seed = rng.integers(1, 2**64 // 2 - 1)
    bands = image_config["bands"]
    shear = image_config["shear"]
    print(f"bands: {bands}")
    print(f"shear: {shear}")
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

    # measure
    meas_seed = rng.integers(1, 2**64 // 2 - 1)

    measure_config = pipeline.config.get("measure")
    match (measure_type := measure_config.get("type")):
        case "metadetect":
            shear_bands = measure_config.get("shear_bands")
            det_bands = measure_config.get("det_bands")
            print(f"shear bands: {det_bands}")
            print(f"det bands: {shear_bands}")
            measurement = run_utils.measure_pair(
                pair,
                shear_bands,
                det_bands,
                pipeline.metadetect_config,
                meas_seed,
            )
        case "chromatic_metadetect":
            shear_bands = measure_config.get("shear_bands")
            det_bands = measure_config.get("det_bands")
            print(f"shear bands: {det_bands}")
            print(f"det bands: {shear_bands}")
            measurement = run_utils.measure_pair_color(
                pair,
                psf,
                chroma_colors,
                chroma_stars,
                image_config["psf_size"],
                lsst.scale,
                bands,
                shear_bands,
                det_bands,
                pipeline.metadetect_config,
                meas_seed,
            )
        case "drdc":
            shear_bands = measure_config.get("shear_bands")
            det_bands = measure_config.get("det_bands")
            print(f"shear bands: {det_bands}")
            print(f"det bands: {shear_bands}")
            measurement = run_utils.measure_pair_color_response(
                lsst,
                pair,
                psf,
                chroma_colors,
                chroma_stars,
                image_config["psf_size"],
                bands,
                shear_bands,
                det_bands,
                pipeline.metadetect_config,
                meas_seed,
            )
        case _:
            raise ValueError(f"Measure type {measure_type} not valid!")

    print("measure:", measure_type)

    schema = run_utils._get_schema()
    table = pa.Table.from_pylist(
        measurement,
        schema,
    )

    # output
    output_config = pipeline.output_config
    output_path = os.path.join(
        output_config.get("path"),
        pipeline.name,
    )
    output_format = output_config.get("format", "parquet")
    print("output:", output_path)
    ds.write_dataset(
        table,
        output_path,
        basename_template=f"{seed}-part-{{i}}.{output_format}",
        format=output_format,
        schema=schema,
        existing_data_behavior="overwrite_or_ignore",
    )
    print("done!")
    exit()



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
    i = 0
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=3)
    )
    ax.imshow(np.arcsinh(pair["plus"][i][0].image), origin="lower")
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

    # jobs = []
    # for seed in rng.integers(1, 2**32, 64):
    #     jobs.append(
    #         joblib.delayed(pipeline.stars.sample)(
    #             1, columns=star_columns, seed=seed
    #         )
    #     )
    # with joblib.Parallel(n_jobs=8, verbose=100) as parallel:
    #     res = parallel(jobs)

    # jobs = []
    # for seed in rng.integers(1, 2**32, 64):
    #     jobs.append(joblib.delayed(pipeline.sample_galaxies)(300, columns=gal_columns, seed=seed))
    # with joblib.Parallel(n_jobs=8, verbose=100) as parallel:
    #     res = parallel(jobs)

