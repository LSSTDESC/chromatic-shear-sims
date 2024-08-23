import copy
import logging
import os
import pickle
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero
# import scipy.stats
import yaml


logger = logging.getLogger(__name__)


def parse_expression(predicate):
    """Parse a predicate tree intro a pyarrow compute expression
    """
    # Parse through the tree
    if type(predicate) is dict:
        for k, v in predicate.items():
            f = getattr(pc, k)
            # if the value is a list, then recursively parse each element and
            # pass the result
            if type(v) is list:
                return f(*[parse_expression(_v) for _v in v])
            # else, the value can be directly used as the function argument
            else:
                return f(v)
    else:
        return predicate


def parse_projection(projection):
    """Parse a projection from a dict of expressions or a list into a dict
    """
    projection_dict = {}
    if projection is not None:
        for proj in projection:
            # if a projection is itself a dict, parse the expression as a pyarrow
            # compute expression
            if type(proj) == dict:
                for k, v in proj.items():
                    projection_dict[k] = parse_expression(v)
            # else interpret as a simple field from the schema
            else:
                projection_dict[proj] = pc.field(proj)

    return projection_dict


def parse_options(options):
    """Parse an options tree intro a pyarrow options object
    """
    # Parse through the tree
    if type(options) is dict:
        for k, v in options.items():
            f = getattr(pc, k)
            return f(**v)
    else:
        return options


class Loader:
    def __init__(self, config):
        self.config = copy.copy(config)
        self.path = self.config.get("path")
        self.format = self.config.get("format")
        self.seed = self.config.get("seed", None)

        self.predicate_dict = self.config.get("predicate", None)
        self.predicate = parse_expression(self.predicate_dict)

        self.projection_dict = self.config.get("projection", None)
        self.projection = parse_projection(self.projection_dict)

        self.aggregate_dict = self.config.get("aggregate", None)

        self.num_rows = None

        logger.info(f"initializing loader for {self.path}")

    def do_aggregate(self, dataset, projection, predicate, aggregate):
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
                        parse_options(agg.get("options", None)),
                        agg.get("output"),
                    )
                    for agg in aggregate
                ],
            )
        )
        if predicate is not None:
            seq = [
                scan_node,
                filter_node,
                project_node,
                aggregate_node,
            ]
        else:
            seq = [
                scan_node,
                project_node,
                aggregate_node,
            ]
        plan = acero.Declaration.from_sequence(seq)
        logger.debug(plan)

        res = plan.to_table(use_threads=True)

        return res

    def get_scanner(self, columns=None):
        """
        Load a dataset defined in a config
        """
        dataset = ds.dataset(self.path, format=self.format)
        scanner = dataset.scanner(
            columns=columns,
            filter=self.predicate,
        )

        return scanner

    def process(self):
        """
        Process a dataset defined in a config
        """
        dataset = ds.dataset(self.path, format=self.format)
        num_rows = dataset.count_rows(filter=self.predicate)
        self.num_rows = num_rows

        logger.info(f"processing aggregates for {self.path}")
        if self.aggregate_dict is not None:
            aggregate = self.do_aggregate(
                dataset,
                self.projection,
                self.predicate,
                self.aggregate_dict,
            )
            aggregate_dict = aggregate.to_pydict()
            aggregates = {}
            for k, v in aggregate_dict.items():
                if len(v) == 1:
                    aggregates[k] = v[0]
                else:
                    aggregates[k] = v
        else:
            aggregates = None

        self.aggregates = aggregates
        return

    def get_rng(self, seed=None):
        rng_seed = seed if seed else self.seed

        logger.debug(f"spawning rng with seed {rng_seed}")
        rng = np.random.default_rng(rng_seed)

        return rng

    def select(self, n, seed=None):
        rng = self.get_rng(seed)
        indices = rng.choice(
            self.num_rows,
            size=n,
            replace=True,
            shuffle=True,
        )

        return indices

    def sample(self, n, columns=None, seed=None):
        indices = self.select(n, seed=seed)
        scanner = self.get_scanner(columns)

        _start_time = time.time()
        objs = scanner.take(indices).to_pydict()
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        logger.info(f"sampled {n} records from {self.path} in {_elapsed_time} seconds")

        return objs

    def digitize_color(self, n_bins=100):
        min_color = self.aggregates.get("min_color")
        max_color = self.aggregates.get("max_color")

        logger.info(f"making color bins ({min_color}, {max_color}, {n_bins})")
        color_bin_edges = np.linspace(
            min_color,
            max_color,
            n_bins + 1,
        )
        color_hist = np.zeros(n_bins)

        logger.info(f"accumulating color histogram...")
        scanner = self.get_scanner(columns=self.projection)
        for i, batch in enumerate(scanner.to_batches()):
            logger.debug(f"accumulating color histogram batch {i}")
            _hist, _ = np.histogram(
                batch["color"],
                bins=color_bin_edges,
            )
            color_hist += _hist

        return color_hist, color_bin_edges
        # hist_dist = scipy.stats.rv_histogram(
        #     (color_hist, color_bin_edges),
        #     density=False,
        # )
        # return hist_dist

