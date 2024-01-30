import copy
import os
import pickle

import galsim
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero
import yaml

from chromatic_shear_bias import run_utils
from chromatic_shear_bias.pipeline import logging_config
from chromatic_shear_bias.pipeline.loader import Loader


class Pipeline:
    def __init__(self, fname, log_level=2):
        self.logger = logging_config.get_logger(log_name=__name__, log_level=log_level)
        self.logger.info(f"Initializing pipeline for {fname}")

        self.fname = fname
        self.name = os.path.splitext(os.path.basename(fname))[0]
        self.config = self.get_config(self.fname)
        self.stash = f"{self.name}.pickle"
        self.galaxy_config = self.config.get("galaxies", None)
        self.star_config = self.config.get("stars", None)
        self.galsim_config = self.config.get("galsim", None)
        self.metadetect_config = self.config.get("metadetect", None)
        self.output_config = self.config.get("output", None)

    def get_config(self, fname):
        with open(fname, "r") as fobj:
            config_dict = yaml.safe_load(fobj)

        return config_dict

    def save(self, overwrite=False):
        exists = os.path.exists(self.stash)
        if (overwrite == False) and exists:
            raise ValueError(f"{self.stash} exists and overwrite=False")
        else:
            self.logger.info(f"saving pipeline to {self.stash}...")
            with open(self.stash, "wb") as fobj:
                # pickle.dump(self, fobj, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.__dict__, fobj, pickle.HIGHEST_PROTOCOL)

        return

    def load(self):
        exists = os.path.exists(self.stash)
        if exists:
            self.logger.info(f"loading pipeline from {self.stash}...")
            with open(self.stash, "rb") as fobj:
                stash = pickle.load(fobj)

            if self.config == stash.get("config"):
                # Persist current logger
                logger = getattr(self, "logger", None)
                # self.__dict__ = stash
                for k, v in stash.items():
                    setattr(self, k, v)
                setattr(self, "logger", logger)
            else:
                raise ValueError(f"config in {self.fname} differs from {self.stash}!")
        else:
            self.logger.info(f"{self.stash} does not exist; continuing...")

        return

    def load_galaxies(self):
        if hasattr(self, "galaxies"):
            self.logger.info("galaxies already processed; skipping...")
        else:
            self.logger.info("loading galaxies...")
            if self.galaxy_config is not None:
                loader = Loader(
                    self.galaxy_config,
                    log_name="galaxy_loader",
                    log_level=self.logger.level,
                )
                loader.process()
            else:
                loader = None

            self.galaxies = loader

        return

    def load_stars(self):
        if hasattr(self, "stars"):
            self.logger.info("stars already processed; skipping...")
        else:
            self.logger.info("loading stars...")
            if self.star_config is not None:
                loader = Loader(
                    self.star_config,
                    log_name="star_loader",
                    log_level=self.logger.level,
                )
                loader.process()
            else:
                loader = None

            self.stars = loader

        return

    def get_psf(self):
        galsim_config = copy.deepcopy(self.galsim_config)
        psf, _ = galsim.config.BuildGSObject(galsim_config, "psf", logger=self.logger)

        return psf

    def get_schema(self):
        measure_config = self.config.get("measure")
        measure_type = measure_config.get("type")
        measure_model = measure_config.get("model")
        match measure_type:
            case "metadetect":
                schema = run_utils._mdet_schema
            case "chromatic_metadetect":
                schema = run_utils._mdet_schema
            case "drdc":
                schema = run_utils._chromatic_schema
            case _:
                raise ValueError(f"Measure type {measure_type} has no registered schema!")

        return schema


if __name__ == "__main__":

    pipeline = Pipeline("config.yaml", load=False)
    print("pipeline:", pipeline.name)
    print("cpu count:", pa.cpu_count())
    print("thread_count:", pa.io_thread_count())

    # pipeline.load()
    pipeline.load_galaxies()
    pipeline.save(overwrite=True)
    pipeline.load_stars()
    pipeline.save(overwrite=True)

    print("galxies:", pipeline.galaxies.aggregate)
    print("stars:", pipeline.stars.aggregate)

    import joblib

    rng = np.random.default_rng()

    psf = pipeline.get_psf()

    star_columns = ["sedFilename", "imag"]
    # star_params = pipeline.stars.sample(
    #     3,
    #     columns=star_columns
    # )

    # jobs = []
    # for seed in rng.integers(1, 2**32, 64):
    #     jobs.append(
    #         joblib.delayed(pipeline.stars.sample)(
    #             1, columns=star_columns, seed=seed
    #         )
    #     )
    # with joblib.Parallel(n_jobs=8, verbose=100) as parallel:
    #     res = parallel(jobs)

    exit()

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
    # gal_params = pipeline.galaxies.sample(
    #     300,
    #     columns=gal_columns,
    # )

    # jobs = []
    # for seed in rng.integers(1, 2**32, 64):
    #     jobs.append(joblib.delayed(pipeline.sample_galaxies)(300, columns=gal_columns, seed=seed))
    # with joblib.Parallel(n_jobs=8, verbose=100) as parallel:
    #     res = parallel(jobs)

