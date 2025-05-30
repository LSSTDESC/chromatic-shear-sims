import argparse
import concurrent.futures
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as ft
from pyarrow import acero
import yaml

from chromatic_shear_sims.throughputs import load_throughputs

from . import log_util, name_util


logger = logging.getLogger(__name__)


def check_files(dataset_path):
    logger.debug(f"checking {dataset_path}")
    _invalid_files = []
    for path in Path(dataset_path).glob("*.parquet"):
        if path.stem not in _invalid_files:
            try:
                pq.ParquetFile(path)
            except pa.ArrowInvalid:
                _invalid_files.append(path)
        else:
            _invalid_files.append(path)

    return _invalid_files


# def pre_aggregate(dataset_path, predicate):
def pre_aggregate(dataset_path, predicate, colors=None, color_indices=None):
    """
    Aggregate measurements at the image level to accelerate bootstrapping
    """
    throughputs = load_throughputs(bands=["g", "i"])
    color = colors[color_indices[1]]

    dataset = ds.dataset(dataset_path)

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
                # pc.field("shear_step"),
                # pc.field("color_step"),
                # pc.field("mdet_step"),
                pc.scalar(1),
                pc.scalar(1),
                # pc.divide(
                #     pc.scalar(1),
                #     pc.add(
                #         pc.multiply(  # average covariance
                #             pc.scalar(0.5),
                #             pc.add(
                #                 pc.list_element(pc.list_element(pc.field("pgauss_g_cov"), 0), 0),
                #                 pc.list_element(pc.list_element(pc.field("pgauss_g_cov"), 0), 0),
                #             ),
                #         ),
                #         pc.power(pc.scalar(0.2), 2),  # intrinsic scatter term
                #     ),
                # ),
                pc.list_element(pc.field("pgauss_g"), 0),
                pc.list_element(pc.field("pgauss_g"), 1),
                pc.add(
                    pc.multiply(
                        pc.scalar(-2.5),
                        pc.log10(
                            pc.divide(
                                pc.list_element(pc.field("pgauss_band_flux"), 0),
                                pc.list_element(pc.field("pgauss_band_flux"), 2)
                            ),
                        ),
                    ),
                    pc.subtract(
                        pc.scalar(throughputs["g"].zeropoint),
                        pc.scalar(throughputs["i"].zeropoint),
                    ),
                ),
            ],
            names=[
                "seed",
                # "shear_step",
                # "color_step",
                # "mdet_step",
                "count",
                "weight",
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
                # pc.field("shear_step"),
                # pc.field("color_step"),
                # pc.field("mdet_step"),
                pc.field("count"),
                pc.field("weight"),
                # e1
                pc.multiply(pc.field("e1"), pc.field("weight")),
                # e2
                pc.multiply(pc.field("e2"), pc.field("weight")),
                # c
                pc.multiply(pc.field("c"), pc.field("weight")),
                # e1c
                pc.multiply(
                    pc.multiply(pc.field("e1"), pc.field("c")),
                    pc.field("weight")
                ),
                # e2c
                pc.multiply(
                    pc.multiply(pc.field("e2"), pc.field("c")),
                    pc.field("weight")
                ),
                # e1cc
                pc.multiply(
                    pc.multiply(pc.field("e1"), pc.power(pc.field("c"), 2)),
                    pc.field("weight")
                ),
                # e2cc
                pc.multiply(
                    pc.multiply(pc.field("e2"), pc.power(pc.field("c"), 2)),
                    pc.field("weight")
                ),
                # e1dc
                pc.multiply(
                    pc.multiply(pc.field("e1"), pc.subtract(pc.field("c"), pc.scalar(color))),
                    pc.field("weight")
                ),
                # e2dc
                pc.multiply(
                    pc.multiply(pc.field("e2"), pc.subtract(pc.field("c"), pc.scalar(color))),
                    pc.field("weight")
                ),
                # e1dcdc
                pc.multiply(
                    pc.multiply(pc.field("e1"), pc.power(pc.subtract(pc.field("c"), pc.scalar(color)), 2)),
                    pc.field("weight")
                ),
                # e2dcdc
                pc.multiply(
                    pc.multiply(pc.field("e2"), pc.power(pc.subtract(pc.field("c"), pc.scalar(color)), 2)),
                    pc.field("weight")
                ),
            ],
            names=[
                "seed",
                # "shear_step",
                # "color_step",
                # "mdet_step",
                "count",
                "weight",
                "weighted_e1",
                "weighted_e2",
                "weighted_c",
                "weighted_e1c",
                "weighted_e2c",
                "weighted_e1cc",
                "weighted_e2cc",
                "weighted_e1dc",
                "weighted_e2dc",
                "weighted_e1dcdc",
                "weighted_e2dcdc",
            ],
        )
    )
    aggregate_node = acero.Declaration(
        "aggregate",
        acero.AggregateNodeOptions(
            [
                # ("seed", "hash_count", None, "count"),
                ("count", "hash_sum", None, "count_sum"),
                ("weight", "hash_sum", None, "weight_sum"),
                ("weighted_e1", "hash_sum", None, "weighted_e1_sum"),
                ("weighted_e2", "hash_sum", None, "weighted_e2_sum"),
                ("weighted_c", "hash_sum", None, "weighted_c_sum"),
                ("weighted_e1c", "hash_sum", None, "weighted_e1c_sum"),
                ("weighted_e2c", "hash_sum", None, "weighted_e2c_sum"),
                ("weighted_e1cc", "hash_sum", None, "weighted_e1cc_sum"),
                ("weighted_e2cc", "hash_sum", None, "weighted_e2cc_sum"),
                ("weighted_e1dc", "hash_sum", None, "weighted_e1dc_sum"),
                ("weighted_e2dc", "hash_sum", None, "weighted_e2dc_sum"),
                ("weighted_e1dcdc", "hash_sum", None, "weighted_e1dcdc_sum"),
                ("weighted_e2dcdc", "hash_sum", None, "weighted_e2dcdc_sum"),
            ],
            # keys=["shear_step", "color_step", "mdet_step", "seed"],
            keys=["seed"],
        )
    )
    post_project_node = acero.Declaration(
        "project",
        acero.ProjectNodeOptions(
            [
                pc.field("seed"),
                # pc.field("shear_step"),
                # pc.field("color_step"),
                # pc.field("mdet_step"),
                # pc.field("count"),
                pc.field("count_sum"),
                pc.field("weight_sum"),
                pc.divide(pc.field("weighted_e1_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e2_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_c_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e1c_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e2c_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e1cc_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e2cc_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e1dc_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e2dc_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e1dcdc_sum"), pc.field("weight_sum")),
                pc.divide(pc.field("weighted_e2dcdc_sum"), pc.field("weight_sum")),
            ],
            names=[
                "seed",
                # "shear_step",
                # "color_step",
                # "mdet_step",
                "count",
                "weight",
                "e1",
                "e2",
                "c",
                "e1c",
                "e2c",
                "e1cc",
                "e2cc",
                "e1dc",
                "e2dc",
                "e1dcdc",
                "e2dcdc",
            ],
        )
    )

    seq = [
        scan_node,
        filter_node,
        pre_project_node,
        project_node,
        aggregate_node,
        post_project_node,
    ]
    plan = acero.Declaration.from_sequence(seq)
    logger.debug(plan)
    res = plan.to_table(use_threads=True)

    return res


def post_aggregate(aggregate_path, format=None, partitioning=None):
    aggregates = ds.dataset(
        aggregate_path,
        format=format,
        partitioning=partitioning,
    )

    # this is a really unfortuante solution of exploiting pandas for it's pivot
    # method and then returing to arrow.
    pivot_index = "seed"
    pivot_columns = ["shear_step", "color_step", "mdet_step"]
    pivot_values = [field for field in aggregates.schema.names if (field != pivot_index) and (field not in pivot_columns)]
    df = aggregates.to_table().to_pandas()
    df_pivot = df.pivot(
        index=pivot_index,
        columns=pivot_columns,
        values=pivot_values,
    )
    pivot = pa.Table.from_pandas(df_pivot)
    renaming = []
    for field in pivot.schema.names:
        if field != pivot_index:
            _pivot_names = eval(field)
            # _name = _pivot_names[0]
            _name = ""
            for _pk, _pv in zip(pivot_columns, _pivot_names[1:]):
                # _name += f":{_pk}={_pv}"
                _name += f"{_pk}={_pv}:"
            _name += _pivot_names[0]
            renaming.append(_name)
        else:
            renaming.append(field)
    pivot = pivot.rename_columns(renaming)

    return pivot


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

    with open(config_file) as fp:
        config = yaml.safe_load(fp)

    output_path = name_util.get_output_path(args.output, args.config)
    aggregate_dataset = name_util.get_aggregate_dataset(args.output, args.config)
    aggregate_path = name_util.get_aggregate_path(args.output, args.config)

    predicate = (
        (pc.field("pgauss_flags") == 0)
        & (pc.field("pgauss_s2n") > args.s2n_cut)
        & (pc.field("pgauss_T_ratio") > 0.5)
    )

    psf_colors = config["measurement"].get("colors")
    psf_color_indices = config["measurement"].get("color_indices")

    shear_steps = ["plus", "minus"]
    color_steps = [f"c{i}" for i, psf_color in enumerate(psf_colors)]
    mdet_steps = ["noshear", "1p", "1m", "2p", "2m"]

    print(f"checking files in {output_path}")

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for shear_step in shear_steps:
            for color_step in color_steps:
                for mdet_step in mdet_steps:
                    dataset_path = os.path.join(
                        output_path,
                        shear_step,
                        color_step,
                        mdet_step,
                    )
                    # print(f"checking {dataset_path}")
                    # for path in Path(dataset_path).glob("*.parquet"):
                    #     if path.stem not in invalid_files:
                    #         try:
                    #             pq.ParquetFile(path)
                    #         except pa.ArrowInvalid:
                    #             invalid_files.append(path)
                    #     else:
                    #         invalid_files.append(path)

                    # _invalid_files = task(dataset_path)
                    # for _invalid_file in _invalid_files:
                    #     invalid_files.append(_invalid_file)

                    _future = executor.submit(check_files, dataset_path)
                    futures.append(_future)

    concurrent.futures.wait(futures)

    invalid_files = []
    for future in futures:
        result = future.result()
        for _invalid_file in result:
            invalid_files.append(_invalid_file)

    print(f"found {len(invalid_files)} invalid files")

    for invalid_file in invalid_files:
        print(f"removing {invalid_file}")
        os.remove(invalid_file)

    print("done checking files")

    print(f"aggregating data in {output_path}")

    for shear_step in shear_steps:
        for color_step in color_steps:
            for mdet_step in mdet_steps:
                dataset_path = os.path.join(
                    output_path,
                    shear_step,
                    color_step,
                    mdet_step,
                )
                print(f"aggregating data in {dataset_path}")
                # _aggregates = pre_aggregate(dataset_path, predicate)
                _aggregates = pre_aggregate(dataset_path, predicate, colors=psf_colors, color_indices=psf_color_indices)

                _aggregate_dir = os.path.join(
                    aggregate_dataset,
                    shear_step,
                    color_step,
                    mdet_step,
                )
                _aggregate_path = os.path.join(
                    _aggregate_dir,
                    f"aggregates.arrow",
                )
                os.makedirs(_aggregate_dir, exist_ok=True)
                print(f"writing aggregates to {_aggregate_path}")
                ft.write_feather(
                    _aggregates,
                    _aggregate_path,
                )

    print(f"pivoting aggregates")
    aggregates = post_aggregate(
        aggregate_dataset,
        format="feather",
        partitioning=["shear_step", "color_step", "mdet_step"],
    )

    ft.write_feather(
        aggregates,
        aggregate_path,
    )
    print(f"aggregation completed")
