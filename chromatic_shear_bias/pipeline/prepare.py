import argparse
import os

import galsim
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import run_utils, roman_rubin, DC2_stars, surveys
from chromatic_shear_bias.pipeline.pipeline import Pipeline


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configuration file [yaml]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = args.config

    pipeline = Pipeline(config, log_level=3)

    pipeline.load_galaxies()
    pipeline.load_stars()

    pipeline.save(overwrite=True)

    print("galxies:", pipeline.galaxies.aggregate)
    print("stars:", pipeline.stars.aggregate)

