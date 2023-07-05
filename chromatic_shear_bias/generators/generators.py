"""
"""

import argparse
import copy
import functools
import itertools
import os
from pathlib import Path
import re

import galsim
import joblib
import metadetect
import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import run_utils


def generate_batches(dataset_files, dir=None, format=None, columns=None, predicate=None):
    dataset = ds.dataset(dataset_files)
    names = dataset.schema.names

    columns = run_utils.match_expression(names, columns)

    scanner = dataset.scanner(
        columns=columns,
        filter=predicate,
    )

    num_rows = scanner.count_rows()
    if num_rows < 1:
        raise ValueError(f"Scanner of {dataset_files} with {predicate} found 0 rows")

    # We repeat the scanner to regenerate and reuse the batches
    for scanner_rep in itertools.repeat(scanner):
        for batch in scanner_rep.to_batches():
            if len(batch) > 0:
                yield batch
            else:
                continue


def generate_rows(batch, n_sample=1, seed=None):
    size = len(batch)
    rng = np.random.default_rng(seed=seed)

    for i in range(n_sample):
        index = rng.integers(low=0, high=size)

        row = batch.take([index]).to_pydict()
        yield row
