import copy
import logging
import os
import pickle

import galsim
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero
import yaml

from chromatic_shear_sims import run_utils, surveys
from chromatic_shear_sims.loader import Loader
from chromatic_shear_sims.DC2_stars import DC2Stars
from chromatic_shear_sims.roman_rubin import RomanRubinGalaxies, RomanRubinBlackBodyGalaxies
from chromatic_shear_sims.exponential import ExponentialBlackBodyGalaxies
from chromatic_shear_sims.blackbody import BlackBodyStars


logger = logging.getLogger(__name__)


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}


def match_color(color_expression, kwargs):
    match color_expression:
        case str():
            color = eval(color_expression, kwargs)
        case float() | int():
            color = color_expression
        case None:
            color = None
        case _:
            raise ValueError(f"Color type not valid!")
    return color


class Pipeline:
    def __init__(self, fname):
        self.fname = fname
        self.name = os.path.splitext(os.path.basename(fname))[0]
        self.config = self.get_config(self.fname)
        self.stash = f"{self.name}.pickle"
        self.survey_config = self.config.get("survey", None)
        self.galaxy_config = self.config.get("galaxies", None)
        self.star_config = self.config.get("stars", None)
        self.galsim_config = self.config.get("galsim", None)
        self.metadetect_config = self.config.get("metadetect", None)
        self.aggregates = {}

        logger.info(f"initializing pipeline for {self.name}")

        for k, v in self.config.items():
            if type(v) == dict:
                for _k, _v in v.items():
                    logger.info(f"{k}.{_k}: {_v}")
            else:
                logger.info(f"{k}: {v}")


    def get_config(self, fname):
        logger.debug(f"reading pipeline config from {fname}")
        with open(fname, "r") as fobj:
            config_dict = yaml.safe_load(fobj)

        return config_dict

    def save(self, overwrite=False):
        exists = os.path.exists(self.stash)
        if (overwrite == False) and exists:
            raise ValueError(f"{self.stash} exists and overwrite=False")
        else:
            logger.info(f"saving pipeline to {self.stash}...")
            with open(self.stash, "wb") as fobj:
                # pickle.dump(self, fobj, pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.__dict__, fobj, pickle.HIGHEST_PROTOCOL)

        return

    def load(self):
        exists = os.path.exists(self.stash)
        if exists:
            logger.info(f"loading pipeline from {self.stash}...")
            with open(self.stash, "rb") as fobj:
                stash = pickle.load(fobj)

            if self.config == stash.get("config"):
                # self.__dict__ = stash
                for k, v in stash.items():
                    setattr(self, k, v)
            else:
                raise ValueError(f"config in {self.fname} differs from {self.stash}!")
        else:
            logger.info(f"{self.stash} does not exist; continuing...")

        return

    def load_survey(self):
        if not hasattr(self, "survey"):
            logger.info("loading survey...")
            if self.survey_config is not None:
                survey = getattr(surveys, self.survey_config)
                survey.load_bandpasses(
                    self.config.get(
                        "throughput_dir",
                        os.environ.get("THROUGHPUT_DIR"),
                    )
                )
            else:
                survey = None

            self.survey = survey

        logger.info(f"survey: {self.survey}")

        return

    def load_galaxies(self):
        if not hasattr(self, "survey"):
            self.load_survey()
        if not hasattr(self, "galaxies"):
            logger.info("loading galaxies...")
            galaxy_config = self.galaxy_config

            if galaxy_config is not None:
                galaxy_type = galaxy_config.get("type")

                match galaxy_type:
                    case "ExponentialBlackBody":
                        loader = Loader(self.galaxy_config)
                        loader.process()
                        if loader.aggregates is not None:
                            self.aggregates["galaxies"] = loader.aggregates
                        galaxies = ExponentialBlackBodyGalaxies(
                            galaxy_config,
                            loader,
                            survey=self.survey,
                        )
                    case "RomanRubin":
                        ssp_templates = self.config.get(
                            "ssp_templates",
                            os.environ.get("SSP_TEMPLATES"),
                        )
                        loader = Loader(self.galaxy_config)
                        loader.process()
                        if loader.aggregates is not None:
                            self.aggregates["galaxies"] = loader.aggregates
                        galaxies = RomanRubinGalaxies(
                            galaxy_config,
                            loader,
                            ssp_templates=ssp_templates,
                            survey=self.survey,
                        )
                    case "RomanRubinBlackBody":
                        ssp_templates = self.config.get(
                            "ssp_templates",
                            os.environ.get("SSP_TEMPLATES"),
                        )
                        loader = Loader(self.galaxy_config)
                        loader.process()
                        if loader.aggregates is not None:
                            self.aggregates["galaxies"] = loader.aggregates
                        galaxies = RomanRubinBlackBodyGalaxies(
                            galaxy_config,
                            loader,
                            survey=self.survey,
                        )
                    case _:
                        raise ValueError(f"Galaxy type not valid!")

            self.galaxies = galaxies

        logger.info(f"galaxies: {self.galaxies}")

        return

    def load_stars(self):
        if not hasattr(self, "survey"):
            self.load_survey()
        if not hasattr(self, "stars"):
            logger.info("loading stars...")
            star_config = self.star_config

            if star_config is not None:
                star_type = star_config.get("type")

                match star_type:
                    case "DC2":
                        sed_dir = self.config.get(
                            "sed_dir",
                            os.environ.get("SED_DIR")
                        )
                        loader = Loader(star_config)
                        loader.process()
                        if loader.aggregates is not None:
                            self.aggregates["stars"] = loader.aggregates
                        # self.stars = loader
                        stars = DC2Stars(
                            star_config,
                            loader,
                            sed_dir=sed_dir,
                            survey=self.survey,
                        )
                    case "BlackBody":
                        stars = BlackBodyStars(star_config, self.survey)
                    case _:
                        raise ValueError(f"Star type not valid!")

            self.stars = stars

        logger.info(f"stars: {self.stars}")

        return

    def get_psf(self):
        galsim_config = copy.deepcopy(self.galsim_config)
        psf, _ = galsim.config.BuildGSObject(galsim_config, "psf")

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

    def get_colors(self):
        measure_config = self.config.get("measure")

        _color = measure_config.get("color")
        logger.info(f"Matching color: {_color}")
        # match _color:
        #     case str():
        #         color = eval(_color, self.aggregates.get("galaxies"))
        #     # case "median":
        #     #     color = self.galaxies.aggregate.get("median_color")
        #     # case "mean":
        #     #     color = self.galaxies.aggregate.get("mean_color")
        #     case float() | int():
        #         color = _color
        #     case None:
        #         color = None
        #     case _:
        #         raise ValueError(f"Color type not valid!")
        color = match_color(_color, self.aggregates.get("galaxies"))

        _colors = measure_config.get("colors")
        logger.info(f"Matching colors: {_colors}")
        match _colors:
            case list():
                colors = [match_color(_color, self.aggregates.get("galaxies")) for _color in _colors]
            case "quantiles":
                colors = self.aggregates.get("galaxies").get("quantiles")
            case "uniform":
                color_min = self.aggregates.get("galaxies").get("min_color")
                color_max = self.aggregates.get("galaxies").get("max_color")
                # TODO add config for number of colors here...
                colors = np.linspace(color_min, color_max, 3)
            case "centered":
                median = self.aggregates.get("galaxies").get("median_color")
                dc = measure_config.get("dc", 0.1)
                colors = [median - dc, median, median + dc]
            case None:
                colors = None
            case _:
                raise ValueError(f"Colors type not valid!")

        if (colors is not None) and (color is None):
            color = colors[1]

        logger.info(f"color: {color}")
        logger.info(f"colors: {colors}")

        return color, colors

    def get_scene_pos(self, seed=None):
        rng = np.random.default_rng(seed)
        image_config = self.config.get("image")
        scene_config = self.config.get("scene")
        survey = self.survey
        match (scene_type := scene_config.get("type")):
            case "single":
                n_gals = 1
                xs = [0]
                ys = [0]
            case "random":
                n_gals = scene_config["n"]
                xs = rng.uniform(
                    -image_config["xsize"] // 2 + scene_config["border"],
                    image_config["xsize"] // 2 - scene_config["border"],
                    n_gals,
                 )
                ys = rng.uniform(
                    -image_config["ysize"] // 2 + scene_config["border"],
                    image_config["ysize"] // 2 - scene_config["border"],
                    n_gals,
                 )
            case "hex":
                v1 = np.asarray([1, 0], dtype=float)
                v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)
                xs, ys = run_utils.build_lattice(
                    survey,
                    image_config["xsize"],
                    image_config["ysize"],
                    scene_config["separation"],
                    v1,
                    v2,
                    rng.uniform(0, 360),
                    scene_config["border"],
                )  # pixels
                if len(xs) < 1:
                    raise ValueError(f"Scene containts no objects!")
                n_gals = len(xs)
            case _:
                raise ValueError(f"Scene type {scene_type} not valid!")

        scene_pos = [
            galsim.PositionD(
                x=x * survey.scale,
                y=y * survey.scale,
            )
            for (x, y) in zip(xs, ys)
        ]
        if (dither := scene_config.get("dither", False)):
            scene_pos = [
                pos + galsim.PositionD(
                    rng.uniform(-dither, dither) * survey.scale,
                    rng.uniform(-dither, dither) * survey.scale,
                )
                for pos in scene_pos
            ]

        return scene_pos

    # def get_galaxies(self, builder, n_gals, seed=None):
    #     romanrubinbuilder = roman_rubin.RomanRubinBuilder(
    #         diffskypop_params=self.config.get("diffskypop_params"),
    #         ssp_templates=self.config.get("ssp_templates"),
    #     )
    #     gal_params = self.galaxies.sample(
    #         n_gals,
    #         columns=romanrubinbuilder.columns,
    #     )
    #     galaxies = romanrubinbuilder.build_gals(gal_params)

    #     return galaxies

    # def get_scene(self, survey, builder


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

