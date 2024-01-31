import argparse
import logging
import os

import ngmix
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as ft
import pyarrow.parquet as pq
from pyarrow import acero
import tqdm

from chromatic_shear_bias.pipeline.pipeline import Pipeline
from chromatic_shear_bias.pipeline import logging_config


logger = logging.getLogger(__name__)


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}


def pre_aggregate(dataset, predicate, color):
    """
    Aggregate measurements at the image level to accelerate bootstrapping
    """
    import galsim
    zp_0 = galsim.Bandpass("LSST_g.dat", wave_type="nm").withZeropoint("AB").zeropoint
    zp_2 = galsim.Bandpass("LSST_i.dat", wave_type="nm").withZeropoint("AB").zeropoint

    scan_node = acero.Declaration(
        "scan",
        acero.ScanNodeOptions(
            dataset,
            # columns=projection,
            filter=predicate,
        ),
    )
    filter_node = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(
            predicate,
        ),
    )
    pre_project_node = acero.Declaration(
        "project",
        acero.ProjectNodeOptions(
            [
                pc.field("seed"),
                pc.field("shear"),
                pc.field("color_step"),
                pc.field("mdet_step"),
                pc.list_element(pc.field("pgauss_g"), 0),
                pc.list_element(pc.field("pgauss_g"), 1),
                pc.add(
                    pc.multiply(
                        pc.scalar(-2.5),
                        pc.log10(
                            pc.divide(
                                pc.list_element(pc.field("pgauss_band_flux"), 0),
                                pc.list_element(pc.field("pgauss_band_flux"), 2)
                            )
                        )
                    ),
                    pc.subtract(pc.scalar(zp_0), pc.scalar(zp_2))
                ),
                pc.multiply(
                    pc.list_element(pc.field("pgauss_g"), 0),
                    pc.subtract(
                        pc.add(
                            pc.multiply(
                                pc.scalar(-2.5),
                                pc.log10(
                                    pc.divide(
                                        pc.list_element(pc.field("pgauss_band_flux"), 0),
                                        pc.list_element(pc.field("pgauss_band_flux"), 2)
                                    )
                                )
                            ),
                            pc.subtract(pc.scalar(zp_0), pc.scalar(zp_2))
                        ),
                        pc.scalar(color),
                    ),
                ),
                pc.multiply(
                    pc.list_element(pc.field("pgauss_g"), 1),
                    pc.subtract(
                        pc.add(
                            pc.multiply(
                                pc.scalar(-2.5),
                                pc.log10(
                                    pc.divide(
                                        pc.list_element(pc.field("pgauss_band_flux"), 0),
                                        pc.list_element(pc.field("pgauss_band_flux"), 2)
                                    )
                                )
                            ),
                            pc.subtract(pc.scalar(zp_0), pc.scalar(zp_2))
                        ),
                        pc.scalar(color),
                    ),
                ),
            ],
            names=[
                "seed",
                "shear",
                "color_step",
                "mdet_step",
                "g1",
                "g2",
                "color",
                "g1c",
                "g2c",
            ],
        )
    )
    pre_aggregate_node = acero.Declaration(
        "aggregate",
        acero.AggregateNodeOptions(
            [
                ("seed", "hash_count", None, "count"),
                ("g1", "hash_mean", None, "mean_g1"),
                ("g2", "hash_mean", None, "mean_g2"),
                ("g1c", "hash_mean", None, "mean_g1c"),
                ("g2c", "hash_mean", None, "mean_g2c"),
                ("color", "hash_mean", None, "mean_color"),
            ],
            keys=["seed", "shear", "color_step", "mdet_step"],
        )
    )
    # FIXME is there a cleaner way to address null colors?
    post_filter_node = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(
            pc.is_finite(pc.field("mean_color")),
        ),
    )
    seq = [
        scan_node,
        filter_node,
        pre_project_node,
        pre_aggregate_node,
        post_filter_node,
    ]
    plan = acero.Declaration.from_sequence(seq)
    logging.debug(plan)
    res = plan.to_table(use_threads=True)

    return res


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
        "--s2n-cut", type=int, default=10,
        help="Signal/noise cut [int; 10]",
    )
    parser.add_argument(
        "--ormask-cut", type=int, default=None,
        help="Cut to make on ormask. 0 indicates make a cut, 1 indicates no cut.",
    )
    parser.add_argument(
        "--mfrac-cut", type=float, default=None,
        help="Cut to make on mfrac. Given in percentages and comma separated. Cut keeps all objects less than the given value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n_resample",
        type=int,
        required=False,
        default=1000,
        help="Number of resample iterations"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of jobs to run [int; 1]",
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

    config = args.config
    seed = args.seed
    n_jobs = args.n_jobs
    output = args.output

    pa.set_cpu_count(n_jobs)
    pa.set_io_thread_count(n_jobs)

    pipeline = Pipeline(config)
    pipeline.load()

    logger.info(f"seed: {seed}")

    rng = np.random.default_rng(seed)

    measure_config = pipeline.config.get("measure")
    measure_type = measure_config.get("type")

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

    color = chroma_colors[1]

    dataset_path = os.path.join(
        output,
        pipeline.name,
    )
    logger.info(f"aggregating data in {aggregate_path}")
    dataset = ds.dataset(dataset_path, format="arrow")

    predicate = \
        (pc.field("pgauss_flags") == 0) \
        & (pc.field("pgauss_s2n") > args.s2n_cut) \
        & (pc.field("pgauss_T_ratio") > 0.5)

    aggregates = pre_aggregate(dataset, predicate, color)

    output_config = pipeline.output_config
    aggregate_path = os.path.join(
        output,
        f"{pipeline.name}_aggregates.feather",
    )
    logger.info(f"writing aggregates to {aggregate_path}")

    ft.write_feather(
        aggregates,
        aggregate_path,
    )


if __name__ == "__main__":
    main()
