import argparse
import logging
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
from chromatic_shear_bias.pipeline import logging_config


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
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

    logger_config = logging_config.defaults
    log_level = logging_config.get_level(args.log_level)
    logging.basicConfig(level=log_level, **logging_config.defaults)

    logger.info(f"{vars(args)}")

    config = args.config

    pipeline = Pipeline(config)

    pipeline.load_galaxies()
    pipeline.load_stars()

    pipeline.save(overwrite=True)

    logger.info(f"galxies: {pipeline.galaxies.aggregate}")
    logger.info(f"stars: {pipeline.stars.aggregate}")


if __name__ == "__main__":
    main()
