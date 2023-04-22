"""
"""

import argparse
import copy
import functools
import itertools
# import operator
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

from chromatic_shear_bias import sed_tools, run_utils


@functools.cache
def read_sed_file(file_name, wave_type, flux_type):
    return galsim.sed.SED(file_name, wave_type, flux_type)


def build_star(star_params, sed_dir):
    _standard_dict = {
        "lte*": "starSED/phoSimMLT",
        "bergeron*": "starSED/wDs",
        "k[mp]*": "starSED/kurucz",
    }
    wave_type = "Nm"
    flux_type = "flambda"
    sed_filename = star_params.get("sedFilename")[0].strip()
    if not sed_filename.endswith(".gz"):
        # Some files are missing ".gz" in their suffix; if this is the case,
        # append to the current suffix
        sed_filename += ".gz"
    path_name = Path(sed_filename)
    for k, v in _standard_dict.items():
        matched = False
        if path_name.match(k):
            sed_path = Path(sed_dir) / v / path_name
            matched = True
            break  # we should only have one match
    if not matched:
        raise ValueError(
            f"Filename {sed_filename} does not match any known patterns in {sed_dir}"
        )
    if not sed_path.exists():
        raise ValueError(f"Filename {sed_filename} not found in {sed_dir}")

    sed_file = sed_path.as_posix()
    sed = read_sed_file(sed_file, wave_type, flux_type)
    sed = sed.withFluxDensity(1, wavelength=600)

    # print(f"\tBuilding star took {end - start} s")

    return sed


def DC2_generator(predicate=None):
    dataset = "/oak/stanford/orgs/kipac/users/smau/dc2_stellar_healpixel.parquet"
    columns = [
        "^sedFilename$",
    ]
    sed_dir = "/oak/stanford/orgs/kipac/users/smau/"
    batch_generator = generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generate_rows(batch, n_sample=batch.num_rows)
        for row in row_generator:
            built = build_star(row, sed_dir)
            yield built
