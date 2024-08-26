import copy
import logging
import time

import numpy as np
import pyarrow as pa
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


class Loader:
    def __init__(self, config):
        self.config = copy.copy(config)
        self.path = self.config.get("path")

        logger.info(f"initializing loader for {self.path}")

        self.format = self.config.get("format")
        self.seed = self.config.get("seed", None)

        self.predicate_dict = self.config.get("predicate", None)
        self.predicate = parse_expression(self.predicate_dict)

        self.num_rows = self.count_rows()

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

    def count_rows(self):
        """
        Process a dataset defined in a config
        """
        # dataset = ds.dataset(self.path, format=self.format)
        # num_rows = dataset.count_rows(filter=self.predicate)
        scanner = self.get_scanner()
        num_rows = scanner.count_rows()
        return num_rows

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

