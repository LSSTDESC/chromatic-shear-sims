import argparse
import logging

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero

from chromatic_shear_sims import utils
from chromatic_shear_sims.simulation import SimulationBuilder

from . import log_util, name_util, plot_util


def do_aggregate(dataset, aggregates, projection=None, predicate=None):
    """
    Plan and execute aggregations for a dataset
    """
    scan_node = acero.Declaration(
        "scan",
        acero.ScanNodeOptions(
            dataset,
            columns=projection,
            filter=predicate,
        ),
    )
    if predicate is not None:
        filter_node = acero.Declaration(
            "filter",
            acero.FilterNodeOptions(
                predicate,
            ),
        )
    if projection is not None:
        project_node = acero.Declaration(
            "project",
            acero.ProjectNodeOptions(
                [v for k, v in projection.items()],
                names=[k for k, v in projection.items()],
            )
        )
    aggregate_node = acero.Declaration(
        "aggregate",
        acero.AggregateNodeOptions(
            [
                (
                    agg.get("input"),
                    agg.get("function"),
                    agg.get("options"),
                    agg.get("output"),
                )
                for agg in aggregates
            ],
        )
    )
    if (predicate is not None) & (projection is not None):
        seq = [
            scan_node,
            filter_node,
            project_node,
            aggregate_node,
        ]
    elif (predicate is not None):
        seq = [
            scan_node,
            filter_node,
            aggregate_node,
        ]
    elif (projection is not None):
        seq = [
            scan_node,
            project_node,
            aggregate_node,
        ]
    else:
        seq = [
            scan_node,
            aggregate_node,
        ]

    plan = acero.Declaration.from_sequence(seq)
    print(plan)

    res = plan.to_table(use_threads=True)

    aggregate_dict = res.to_pydict()
    aggregates_out = {}
    for k, v in aggregate_dict.items():
        if len(v) == 1:
            aggregates_out[k] = v[0]
        else:
            aggregates_out[k] = v

    return aggregates_out


def do_plot(dataset, aggregates, nbins=100, projection=None, predicate=None):
    bin_edges = np.linspace(
        aggregates["color_min"],
        aggregates["color_max"],
        nbins + 1,
    )
    hist_unfiltered = np.zeros(nbins)
    hist = np.zeros(nbins)

    for batch in dataset.to_batches(columns=projection):
        _hist, _ = np.histogram(batch["color"], bins=bin_edges)
        hist_unfiltered += _hist

    for batch in dataset.to_batches(filter=predicate, columns=projection):
        _hist, _ = np.histogram(batch["color"], bins=bin_edges)
        hist += _hist

    fig, axs = plot_util.subplots(1, 1)

    axs.stairs(hist_unfiltered / sum(hist_unfiltered), bin_edges, ls="--", label="full sample")
    axs.stairs(hist / sum(hist), bin_edges, label="maglim sample")
    axs.axvline(aggregates["color_mean"], c="b", ls="--", label="mean")
    for i, cq in enumerate(aggregates["color_quantiles"]):
        ql = "quantiles" if (i == 1) else None
        axs.axvline(cq, c="b", ls=":", label=ql)

    axs.set_xlabel("color")
    axs.legend(loc="upper right")

    return fig, axs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="configuration file [yaml]",
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
    config_name = name_util.get_config_name(config_file)
    simulation_builder = SimulationBuilder.from_yaml(config_file)

    dataset_path = simulation_builder.galaxy_loader.path
    dataset_format = simulation_builder.galaxy_loader.format
    predicate = simulation_builder.galaxy_loader.predicate

    projection = {
        # "LSST_obs_u": ds.field("LSST_obs_u"),
        # "LSST_obs_g": ds.field("LSST_obs_g"),
        # "LSST_obs_r": ds.field("LSST_obs_r"),
        # "LSST_obs_i": ds.field("LSST_obs_i"),
        # "LSST_obs_z": ds.field("LSST_obs_z"),
        # "LSST_obs_y": ds.field("LSST_obs_y"),
        "color": ds.field("LSST_obs_g") - ds.field("LSST_obs_i"),
    }

    dataset = ds.dataset(dataset_path, format=dataset_format)

    aggregates = [
        {
            "function": "count",
            "input": "color",
            "options": None,
            "output": "color_count",
        },
        {
            "function": "min",
            "input": "color",
            "options": None,
            "output": "color_min",
        },
        {
            "function": "max",
            "input": "color",
            "options": None,
            "output": "color_max",
        },
        {
            "function": "mean",
            "input": "color",
            "options": None,
            "output": "color_mean",
        },
        {
            "function": "variance",
            "input": "color",
            "options": None,
            "output": "color_variance",
        },
        {
            "function": "approximate_median",
            "input": "color",
            "options": None,
            "output": "color_median",
        },
        {
            "function": "tdigest",
            "input": "color",
            "options": pc.TDigestOptions(q=[0.25, 0.50, 0.75]),
            "output": "color_quantiles",
        },
    ]

    aggregate_res = do_aggregate(
        dataset,
        aggregates,
        projection=projection,
        predicate=predicate,
    )

    for k, v in aggregate_res.items():
        print(f"{k}: {v}")

    fig, axs = do_plot(dataset, aggregate_res, nbins=100, projection=projection, predicate=predicate)

    figname = f"{config_name}-colors.pdf"
    fig.savefig(figname)

