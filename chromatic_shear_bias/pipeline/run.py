import argparse
import copy
import os
from itertools import chain

import galsim
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import run_utils, roman_rubin, DC2_stars, surveys
from chromatic_shear_bias.pipeline.pipeline import Pipeline

# import time
# from psutil import Process
# from threading import Thread
# 
# # FIXME
# class MemoryMonitor(Thread):
#     """Monitor the memory usage in MB in a separate thread.
# 
#     Note that this class is good enough to highlight the memory profile of
#     Parallel in this example, but is not a general purpose profiler fit for
#     all cases.
#     """
#     def __init__(self):
#         super().__init__()
#         self.stop = False
#         self.memory_buffer = []
#         self.start()
# 
#     def get_memory(self):
#         "Get memory of a process and its children."
#         p = Process()
#         memory = p.memory_info().rss
#         for c in p.children():
#             memory += c.memory_info().rss
#         return memory
# 
#     def run(self):
#         memory_start = self.get_memory()
#         while not self.stop:
#             self.memory_buffer.append(self.get_memory() - memory_start)
#             time.sleep(0.2)
# 
#     def join(self):
#         self.stop = True
#         super().join()
# #FIXME


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}

TOL = 0.01


def run_pipeline(config, seed=None):
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
            predicate = (pc.abs_checked(pc.field("gmag") - pc.field("imag") - color) < TOL)
            star_params = pipeline.stars.sample_with(
                1,
                columns=dc2builder.columns,
                predicate=predicate,
            )
            chroma_star = dc2builder.build_star(star_params)
            chroma_stars.append(chroma_star)
        star = chroma_stars[1]
    else:
        star_params = pipeline.stars.sample(
            1,
            columns=dc2builder.columns,
        )
        star = dc2builder.build_star(star_params)

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

    # measure
    meas_seed = rng.integers(1, 2**64 // 2 - 1)

    match measure_type:
        case "metadetect":
            shear_bands = measure_config.get("shear_bands")
            det_bands = measure_config.get("det_bands")
            # FIXME return batches...
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
            # FIXME return batches...
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
            # measurement = run_utils.measure_pair_color_response(
            #     lsst,
            #     pair,
            #     psf,
            #     chroma_colors,
            #     chroma_stars,
            #     image_config["psf_size"],
            #     bands,
            #     shear_bands,
            #     det_bands,
            #     pipeline.metadetect_config,
            #     meas_seed,
            # )
            batches = run_utils.run_pair_color_response(
                pipeline,
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

    return batches


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
        help="RNG seed [int; None]",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=".",
        help="Output directory [str; .]"
    )
    parser.add_argument(
        "--n_sims",
        type=int,
        required=False,
        default=1,
        help="Number of sims to run [int; 1]"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of parallel jobs to run [int; 1]"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = args.config
    seed = args.seed
    n_sims = args.n_sims
    n_jobs = args.n_jobs
    output = args.output

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

    # TODO use command line arguments, rather than config here
    output_config = pipeline.output_config
    output_path = os.path.join(
        output,
        pipeline.name,
    )
    output_format = output_config.get("format", "parquet")
    print("output:", output_path)

    schema = pipeline.get_schema()

    rng = np.random.default_rng(seed)

    jobs = []
    for _seed in rng.integers(1, 2**32, n_sims):
        jobs.append(
            joblib.delayed(run_pipeline)(
                config, seed=_seed
            )
        )
    # with joblib.Parallel(n_jobs=n_jobs, verbose=100, return_as="generator") as parallel:
    #     res_generator = parallel(jobs)
    batches = joblib.Parallel(n_jobs=n_jobs, verbose=100, return_as="generator")(jobs)
    # with joblib.Parallel(n_jobs=n_jobs, verbose=100, return_as="list") as parallel:
    #     _batches = parallel(jobs)
    # batches = iter(_batches)

    # monitor = MemoryMonitor()
    chained = chain.from_iterable(batches)

    ds.write_dataset(
        chained,
        output_path,
        basename_template=f"{seed}-part-{{i}}.{output_format}",
        format=output_format,
        schema=schema,
        existing_data_behavior="overwrite_or_ignore",
        max_rows_per_file=1024**2 * 100,
    )
    print("done!")

    # del chained
    # del batches
    # monitor.join()

    # import matplotlib.pyplot as plt
    # plt.semilogy(
    #     np.maximum.accumulate(monitor.memory_buffer),
    # )
    # plt.xlabel("Time")
    # plt.xticks([], [])
    # plt.ylabel("Memory usage")
    # plt.yticks([1e7, 1e8, 1e9], ['10MB', '100MB', '1GB'])
    # plt.show()

