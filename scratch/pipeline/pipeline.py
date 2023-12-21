import copy
import os
import pickle

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero
import yaml




def parse_expression(predicate):
    """Parse a predicate tree intro a pyarrow compute expression
    """
    # Parse through the tree
    if type(predicate) is dict:
        for k, v in predicate.items():
            f = getattr(pc, k)
            if type(v) is list:
                return f(*[parse_expression(_v) for _v in v])
            else:
                return f(v)
    else:
        return predicate


def parse_projection(projection):
    projection_dict = {}
    for proj in projection:
        if type(proj) == dict:
            for k, v in proj.items():
                projection_dict[k] = parse_expression(v)
        else:
            projection_dict[proj] = pc.field(proj)

    return projection_dict


# def match_expression(names, expressions):
#     """Match names to regular expressions
#     """
#     return [
#         name for name in names
#         for expression in expressions
#         if re.match(expression, name)
#     ]


class Pipeline:
    def __init__(self, fname):
        self.fname = fname
        self.name = os.path.splitext(os.path.basename(fname))[0]
        self.config = self.get_config(self.fname)
        self.stash = f"{self.name}.pickle"
        # self.config = copy.copy(config)
        self.galaxy_config = self.config.get("galaxies", None)
        self.star_config = self.config.get("stars", None)
        self.galsim_config = self.config.get("galsim", None)
        self.metadetect_config = self.config.get("metadetect", None)
        self.output_config = self.config.get("output", None)

    def get_config(self, fname):
        with open(fname, "r") as fobj:
            config_dict = yaml.safe_load(fobj)

        return config_dict

    def dump(self):
        print(f"saving pipeline to {self.stash}...")
        with open(self.stash, "wb") as fobj:
            pickle.dump(self, fobj, pickle.HIGHEST_PROTOCOL)

        return

    def load(stash):
        print(f"loading pipeline from {stash}...")
        with open(stash, "rb") as fobj:
            obj = pickle.load(fobj)

        return obj

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
                        agg.get("options", None),
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
        print(plan)

        res = plan.to_table(use_threads=True)

        return res

    # def load_dataset(self, dataset_config):
    #     """
    #     Load a dataset defined in a config
    #     """
    #     _path = dataset_config.get("path")
    #     _format = dataset_config.get("format")
    #     _filter = dataset_config.get("filter", None)
    #     _predicate = dataset_config.get("predicate", None)
    #     _projection = dataset_config.get("projection", None)
    #     _columns = dataset_config.get("columns", None)
    #     _aggregate = dataset_config.get("aggregate", None)

    #     predicate = parse_expression(_predicate)
    #     projection = {
    #         k: parse_expression(v)
    #         for _proj in _projection
    #         for k, v in _proj.items()
    #     }

    #     dataset = ds.dataset(_path, format=_format)

    #     return dataset

    def process_dataset(self, dataset_config):
        """
        Process a dataset defined in a config
        """
        _path = dataset_config.get("path")
        _format = dataset_config.get("format")
        _filter = dataset_config.get("filter", None)
        _predicate = dataset_config.get("predicate", None)
        _projection = dataset_config.get("projection", None)
        _aggregate = dataset_config.get("aggregate", None)

        predicate = parse_expression(_predicate)
        projection = parse_projection(_projection)

        dataset = ds.dataset(_path, format=_format)
        schema_str = dataset.schema.to_string()

        aggregate = self.do_aggregate(
            dataset,
            projection,
            predicate,
            _aggregate,
        )
        aggregate_dict = aggregate.to_pydict()

        res = {
          "config": dataset_config,
          "schema": schema_str,
          "aggregate": aggregate_dict,
        }

        return res

    def process_galaxies(self):
        """
        If the galaxies have not been processed, process the dataset
        """
        if hasattr(self, "galaxies"):
            print("galaxies already processed; skipping...")
        else:
            res = self.process_dataset(self.galaxy_config)
            self.galaxies = res

        return

    def process_stars(self):
        """
        If the stars have not been processed, process the dataset
        """
        if hasattr(self, "stars"):
            print("stars already processed; skipping...")
        else:
            res = self.process_dataset(self.star_config)
            self.stars = res

        return

    def repartition_output(self, partitioning):
        """
        Repartition and serialize output
        """
        raw_output_path = self.raw_output_path
        output_path = self.output_path

        raw_output_dataset = ds.dataset(
            raw_output_path,
            format="arrow",
        )

        ds.write_dataset(
            raw_output_dataset,
            output_path,
            format="parquet",
            partitioning=parititioning,
        )

        return


if __name__ == "__main__":
    # with open("test.pickle", "rb") as fobj:
    #     test = pickle.load(fobj)

    # config = get_config("config.yaml")
    pipeline = Pipeline("config.yaml")
    print("pipeline:", pipeline.name)
    print("cpu count:", pa.cpu_count())
    print("thread_count:", pa.io_thread_count())

    # if not os.path.exists("obj.pickle"):
    #     config = get_config("config.yaml")
    #     pipeline = Pipeline(config)

    # else:
    #     with open("obj.pickle", "rb") as fobj:
    #         pipeline = pickle.load(fobj)

    pipeline.process_galaxies()

    pipeline.dump()

    pipeline.process_stars()

    pipeline.dump()

    print("galxies:", pipeline.galaxies["aggregate"])
    print("stars:", pipeline.stars["aggregate"])

    # stash = vars(pipeline)
    # with open("stash.pickle", "wb") as fobj:
    #     pickle.dump(stash, fobj, pickle.HIGHEST_PROTOCOL)
