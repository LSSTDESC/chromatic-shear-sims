import ast
import copy
import functools
import logging
import re
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds


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


def parse_filters(filters):
    if filters is None:
        return None
    else:
        def parse_token(token):
            try:
                val = ast.literal_eval(token)
            except:
                val = token
            return val

        # see pq.core._DNF_filter_doc
        _operators = ["==", "=", "!=", "<=", ">=", "<", ">", "in", "not in"]
        _pattern = "(" + "|".join(_operators) + ")"
        pattern = re.compile(_pattern)
        disjunction = []
        for _conjunction in filters:
            conjunction = []
            for filters_string in _conjunction:
                filters_tuple = tuple(
                    map(
                        parse_token,
                        map(
                            str.strip,
                            pattern.split(
                                filters_string,
                            ),
                        ),
                    ),
                )
                conjunction.append(filters_tuple)
            disjunction.append(conjunction)

        return pq.filters_to_expression(disjunction)


class Loader:
    def __init__(self, config):
        self.config = copy.copy(config)

        self.seed = self.config.get("seed", None)

        self.predicate_dict = self.config.get("predicate", None)
        self.predicate = parse_expression(self.predicate_dict)
        # self.predicate_filters = self.config.get("predicate", None)
        # self.predicate = parse_filters(self.predicate_filters)

        self.path = self.config.get("path")
        self.format = self.config.get("format")
        self.dataset = ds.dataset(self.path, self.format)
        self.files = self.dataset.files

        logger.info(f"scanning dataset: {self.path}")
        logger.info(f"applying filter: {self.predicate}")

        num_rows = {}
        for _file in self.files:
            _dataset = ds.dataset(_file, format=self.format)
            num_rows[_file] = _dataset.count_rows(filter=self.predicate)
            del _dataset
            logger.info(f"{_file}: {num_rows[_file]} rows")

        self.num_rows = num_rows
        # self.num_rows = self.count_rows()


    def get_scanner(self, file, columns=None):
        """
        Load a dataset defined in a config
        """
        dataset = ds.dataset(file, format=self.format)
        scanner = dataset.scanner(
            columns=columns,
            filter=self.predicate,
        )

        return scanner

    # def count_rows(self):
    #     """
    #     Process a dataset defined in a config
    #     """
    #     # dataset = ds.dataset(self.path, format=self.format)
    #     # num_rows = dataset.count_rows(filter=self.predicate)
    #     scanner = self.get_scanner()
    #     num_rows = scanner.count_rows()
    #     return num_rows

    def get_rng(self, seed=None):
        rng_seed = seed if seed else self.seed

        logger.debug(f"spawning rng with seed {rng_seed}")
        rng = np.random.default_rng(rng_seed)

        return rng

    def select(self, file, n, seed=None):
        rng = self.get_rng(seed)
        indices = rng.choice(
            self.num_rows[file],
            size=n,
            replace=True,
            shuffle=True,
        )

        return indices

    def sample(self, n, columns=None, seed=None):
        rng = self.get_rng(seed)
        file = rng.choice(self.files)
        indices = self.select(file, n, seed=seed)
        scanner = self.get_scanner(file, columns)

        _start_time = time.time()
        objs = scanner.take(indices).to_pydict()
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        logger.info(f"sampled {n} records from {file} in {_elapsed_time} seconds")

        return objs

