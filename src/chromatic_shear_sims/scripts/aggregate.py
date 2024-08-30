import argparse
import logging
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as ft
from pyarrow import acero

from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder
from chromatic_shear_sims.throughputs import load_throughputs

from . import log_util


def pre_aggregate(dataset, predicate, colors=None):
    """
    Aggregate measurements at the image level to accelerate bootstrapping
    """
    throughputs = load_throughputs(bands=["g", "i"])
    color = colors[1]

    scan_node = acero.Declaration(
        "scan",
        acero.ScanNodeOptions(
            dataset,
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
                pc.field("shear_step"),
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
                    pc.subtract(pc.scalar(throughputs["g"].zeropoint), pc.scalar(throughputs["i"].zeropoint))
                ),
            ],
            names=[
                "seed",
                "shear_step",
                "color_step",
                "mdet_step",
                "e1",
                "e2",
                "c",
            ],
        )
    )
    project_node = acero.Declaration(
        "project",
        acero.ProjectNodeOptions(
            [
                pc.field("seed"),
                pc.field("shear_step"),
                pc.field("color_step"),
                pc.field("mdet_step"),
                pc.field("e1"),
                pc.field("e2"),
                pc.field("c"),
                pc.multiply(pc.field("e1"), pc.field("e2")),
                pc.multiply(pc.field("e1"), pc.field("c")),
                pc.multiply(pc.field("e2"), pc.field("c")),
                pc.multiply(pc.field("e1"), pc.power(pc.field("c"), 2)),
                pc.multiply(pc.field("e2"), pc.power(pc.field("c"), 2)),
                pc.multiply(pc.field("e1"), pc.subtract(pc.field("c"), pc.scalar(color))),
                pc.multiply(pc.field("e2"), pc.subtract(pc.field("c"), pc.scalar(color))),
                pc.multiply(pc.field("e1"), pc.power(pc.subtract(pc.field("c"), pc.scalar(color)), 2)),
                pc.multiply(pc.field("e2"), pc.power(pc.subtract(pc.field("c"), pc.scalar(color)), 2)),
            ],
            names=[
                "seed",
                "shear_step",
                "color_step",
                "mdet_step",
                "e1",
                "e2",
                "c",
                "e1e2",
                "e1c",
                "e2c",
                "e1cc",
                "e2cc",
                "e1dc",
                "e2dc",
                "e1dc2",
                "e2dc2",
            ],
        )
    )
    pre_aggregate_node = acero.Declaration(
        "aggregate",
        acero.AggregateNodeOptions(
            [
                ("seed", "hash_count", None, "count"),
                ("e1", "hash_mean", None, "mean_e1"),
                ("e2", "hash_mean", None, "mean_e2"),
                ("c", "hash_mean", None, "mean_c"),
                ("e1e2", "hash_mean", None, "mean_e1e2"),
                ("e1c", "hash_mean", None, "mean_e1c"),
                ("e2c", "hash_mean", None, "mean_e2c"),
                ("e1cc", "hash_mean", None, "mean_e1cc"),
                ("e2cc", "hash_mean", None, "mean_e2cc"),
                ("e1dc", "hash_mean", None, "mean_e1dc"),
                ("e2dc", "hash_mean", None, "mean_e2dc"),
                ("e1dc2", "hash_mean", None, "mean_e1dc2"),
                ("e2dc2", "hash_mean", None, "mean_e2dc2"),
                # ("e1", "hash_variance", None, "var_e1"),
                # ("e2", "hash_variance", None, "var_e2"),
                # ("c", "hash_variance", None, "var_c"),
                # ("e1e2", "hash_variance", None, "var_e1e2"),
                # ("e1c", "hash_variance", None, "var_e1c"),
                # ("e2c", "hash_variance", None, "var_e2c"),
                # ("e1dc", "hash_variance", None, "var_e1dc"),
                # ("e2dc", "hash_variance", None, "var_e2dc"),
            ],
            keys=["seed", "shear_step", "color_step", "mdet_step"],
        )
    )

    # FIXME is there a cleaner way to address null colors?
    post_filter_node = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(
            pc.is_finite(pc.field("mean_c")),
        ),
    )
    seq = [
        scan_node,
        filter_node,
        pre_project_node,
        project_node,
        pre_aggregate_node,
        post_filter_node,
    ]
    plan = acero.Declaration.from_sequence(seq)
    print(plan)
    res = plan.to_table(use_threads=True)

    return res



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "output",
        type=str,
        help="output directory",
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
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of parallel jobs to run [int; 1]"
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
    log_level = log_util.get_level(args.log_level)
    logging.basicConfig(format=log_util.FORMAT, level=log_level)

    config_file = args.config
    simulation_builder = SimulationBuilder.from_yaml(config_file)
    measure = measurement.get_measure(
        **simulation_builder.config["measurement"].get("builder"),
    )

    n_jobs = args.n_jobs
    pa.set_cpu_count(n_jobs)
    pa.set_io_thread_count(2 * n_jobs)

    config_name = os.path.basename(config_file).split(".")[0]

    output_path = f"{args.output}/{config_name}"
    aggregate_path = f"{args.output}/{config_name}_aggregates.feather"

    print(f"aggregating data in {output_path}")
    dataset = ds.dataset(output_path, format="parquet")

    predicate = (
        (pc.field("pgauss_flags") == 0)
        & (pc.field("pgauss_s2n") > args.s2n_cut)
        & (pc.field("pgauss_T_ratio") > 0.5)
    )
    # express predicate using DNF
    # predicate = [
    #     [
    #         ("pgauss_flags", "=", 0),
    #         ("pgauss_s2n", ">", args.s2n_cut),
    #         ("pgauss_T_ratio", ">", 0.5),
    #     ],
    # ]

    psf_colors = simulation_builder.config["measurement"].get("colors")
    aggregates = pre_aggregate(dataset, predicate, colors=psf_colors)

    print(f"writing aggregates to {aggregate_path}")

    ft.write_feather(
        aggregates,
        aggregate_path,
    )

    print(f"aggregation completed")
