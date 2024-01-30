import copy
import os
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero
import yaml


from chromatic_shear_bias.pipeline import logging_config


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
    def __init__(self, config, log_name=__name__, log_level=2):
        self.logger = logging_config.get_logger(log_name=log_name, log_level=log_level)
        self.logger.info(f"Initializing loader...")

        self.config = copy.copy(config)

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
        self.logger.debug(plan)

        res = plan.to_table(use_threads=True)

        return res

    def get_scanner(self, columns=None):
        """
        Load a dataset defined in a config
        """
        _path = self.config.get("path")
        _format = self.config.get("format")
        _predicate = self.config.get("predicate", None)
        predicate = parse_expression(_predicate)

        dataset = ds.dataset(_path, format=_format)
        scanner = dataset.scanner(columns=columns, filter=predicate)

        return scanner

    def process(self):
        """
        Process a dataset defined in a config
        """
        _path = self.config.get("path")
        _format = self.config.get("format")
        _filter = self.config.get("filter", None)
        _predicate = self.config.get("predicate", None)
        _projection = self.config.get("projection", None)
        _aggregate = self.config.get("aggregate", None)

        predicate = parse_expression(_predicate)
        projection = parse_projection(_projection)

        dataset = ds.dataset(_path, format=_format)

        aggregate = self.do_aggregate(
            dataset,
            projection,
            predicate,
            _aggregate,
        )
        aggregate_dict = aggregate.to_pydict()

        self.aggregate = aggregate_dict
        return

    def select(self, n, seed=None):
        nobj = self.aggregate.get("count")[0]
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            nobj,
            size=n,
            replace=True,
            shuffle=True,
        )

        return indices

    def sample(self, n, columns=None, seed=None):
        indices = self.select(n, seed=seed)
        scanner = self.get_scanner(columns)

        obj = scanner.take(indices).to_pydict()

        return obj

    def sample_with(self, n, columns=None, predicate=None, seed=None):
        _path = self.config.get("path")
        _format = self.config.get("format")
        _predicate = self.config.get("predicate", None)
        base_predicate = parse_expression(_predicate)
        if predicate is not None:
            new_predicate = base_predicate & predicate
        else:
            new_predicate = base_predicate

        dataset = ds.dataset(_path, format=_format)
        scanner = dataset.scanner(columns=columns, filter=new_predicate)
        nobj = scanner.count_rows()
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            nobj,
            size=n,
            replace=True,
            shuffle=True,
        )

        obj = scanner.take(indices).to_pydict()

        return obj


if __name__ == "__main__":

    from pipeline import Pipeline
    pipeline = Pipeline("config.yaml")
    print("pipeline:", pipeline.name)
    print("cpu count:", pa.cpu_count())
    print("thread_count:", pa.io_thread_count())

    galaxy_loader = Loader(pipeline.galaxy_config)
    star_loader = Loader(pipeline.star_config)

    galaxy_loader.process()

    star_loader.process()

    print("galxies:", galaxy_loader.aggregate)
    print("stars:", star_loader.aggregate)

    rng = np.random.default_rng()

    star_columns = ["sedFilename", "imag"]
    star_params = star_loader.sample(
        3,
        columns=star_columns
    )

    from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import ALL_DIFFSKY_PNAMES
    morph_columns = [
       "redshift",
       "spheroidEllipticity1",
       "spheroidEllipticity2",
       "spheroidHalfLightRadiusArcsec",
       "diskEllipticity1",
       "diskEllipticity2",
       "diskHalfLightRadiusArcsec",
    ]
    gal_columns = list(set(morph_columns + ALL_DIFFSKY_PNAMES))
    gal_params = galaxy_loader.sample(
        300,
        columns=gal_columns,
    )

