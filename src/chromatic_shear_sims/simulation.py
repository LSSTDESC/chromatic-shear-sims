import copy
import logging

import yaml

from chromatic_shear_sims import utils
from chromatic_shear_sims import observations
from chromatic_shear_sims.data import Data
from chromatic_shear_sims.loader import Loader
from chromatic_shear_sims.psf import PSF
from chromatic_shear_sims.scene import Scene
from chromatic_shear_sims.images import ImageBuilder
from chromatic_shear_sims.galaxies import GalaxyBuilder
from chromatic_shear_sims.positions import PositionBuilder
from chromatic_shear_sims.stars import StarBuilder, InterpolatedStarBuilder
from chromatic_shear_sims.throughputs import load_throughputs
from chromatic_shear_sims.darksky import load_darksky


logger = logging.getLogger(__name__)


class SimulationBuilder:
    def __init__(self, simulation_config):
        simulation_config_copy = copy.deepcopy(simulation_config)
        self.config = simulation_config_copy
        self.bands = self.config.get("bands")

        self.throughputs = load_throughputs(bands=self.bands)
        self.sky_background = load_darksky()

        if "builder" in self.config["stars"]:
            star_builder = StarBuilder(
                **self.config["stars"]["builder"]
            )
            psf_star_builder = InterpolatedStarBuilder(
                **self.config["stars"]["builder"],
                throughput_1=self.throughputs["g"],
                throughput_2=self.throughputs["i"],
            )
        else:
            star_builder = None
            psf_star_builder = None
        self.star_builder = star_builder
        self.psf_star_builder = psf_star_builder

        if "loader" in self.config["stars"]:
            star_loader = Loader(self.config["stars"]["loader"])
        else:
            star_loader = None
        self.star_loader = star_loader

        if "data" in self.config["stars"]:
            star_data = Data(
                **self.config["stars"]["data"],
                data_loader=self.star_loader,
            )
        else:
            star_data = None
        self.star_data = star_data

        if "positions" in self.config["stars"]:
            star_position_builder = PositionBuilder.from_config(self.config["stars"]["positions"])
        else:
            star_position_builder = None
        self.star_position_builder = star_position_builder

        if "builder" in self.config["galaxies"]:
            galaxy_builder = GalaxyBuilder(
                **self.config["galaxies"]["builder"],
            )
        else:
            galaxy_builder = None
        self.galaxy_builder = galaxy_builder

        if "loader" in self.config["galaxies"]:
            galaxy_loader = Loader(self.config["galaxies"]["loader"])
        else:
            galaxy_loader = None
        self.galaxy_loader = galaxy_loader

        if "data" in self.config["galaxies"]:
            galaxy_data = Data(
                **self.config["galaxies"]["data"],
                data_loader=self.galaxy_loader,
            )
        else:
            galaxy_data = None
        self.galaxy_data = galaxy_data

        if "positions" in self.config["galaxies"]:
            galaxy_position_builder = PositionBuilder.from_config(self.config["galaxies"]["positions"])
        else:
            galaxy_position_builder = None
        self.galaxy_position_builder = galaxy_position_builder

        self.image_builder = ImageBuilder.from_config(self.config["image"])

        self.psf_image_builder = ImageBuilder.from_config(self.config["psf"]["image"])

    # @classmethod
    # def from_config(cls, simulation_config):
    #     simulation_config_copy = copy.deepcopy(simulation_config)
    #     return cls(**simulation_config_copy)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as fp:
            config = yaml.safe_load(fp)
        return cls(config)
        # return cls.from_config(config)

    def make_scene(self, seed=None):
        scene_seed = utils.get_seed(seed=seed)
        star_data_seed, star_position_seed, star_rotation_seed, galaxy_data_seed, galaxy_position_seed, galaxy_rotation_seed = utils.get_seeds(6, seed=scene_seed)

        if self.star_position_builder is not None:
            star_positions = self.star_position_builder.get_positions(seed=star_position_seed)
            star_rotations = self.star_position_builder.get_rotations_for(star_positions, seed=star_rotation_seed)

        if (self.star_data is not None) and (self.star_loader is not None):
            stars = self.star_data.load(len(star_positions), seed=star_data_seed)
        else:
            stars = None

        if self.galaxy_position_builder is not None:
            galaxy_positions = self.galaxy_position_builder.get_positions(seed=galaxy_position_seed)
            galaxy_rotations = self.galaxy_position_builder.get_rotations_for(galaxy_positions, seed=galaxy_rotation_seed)

        if (self.galaxy_data is not None) and (self.galaxy_loader is not None):
            galaxies = self.galaxy_data.load(len(galaxy_positions), seed=galaxy_data_seed)
        else:
            galaxies = None

        if stars is not None:
            star_kwargs = self.config["stars"].get("kwargs", {})
            scene_stars = [
                self.star_builder(stars(i), **star_kwargs).rotate(
                    rotation
                ).shift(position)
                for i, (position, rotation) in enumerate(
                    zip(star_positions, star_rotations)
                )
            ]
        else:
            scene_stars = []

        if galaxies is not None:
            galaxy_kwargs = self.config["galaxies"].get("kwargs", {})
            scene_galaxies = [
                self.galaxy_builder(galaxies(i), **galaxy_kwargs).rotate(
                    rotation
                ).shift(position)
                for i, (position, rotation) in enumerate(
                    zip(galaxy_positions, galaxy_rotations)
                )
            ]
        else:
            scene_galaxies = []

        scene = Scene(galaxies=scene_galaxies, stars=scene_stars)

        return scene

    def make_psf(self, seed=None):
        psf = PSF(self.config["psf"]["model"], seed=seed)

        return psf

    def make_psf_obs(self, psf, color=None, seed=None):
        if color is not None:
            psf_color = color
        else:
            psf_color = self.config["psf"]["color"]

        psf_star = self.psf_star_builder(psf_color)

        psf_mbobs = observations.get_psf_mbobs(
            self.bands,
            self.throughputs,
            psf,
            psf_star,
            psf_image_builder=self.psf_image_builder,
            seed=seed,
        )

        return psf_mbobs

    def make_obs(self, psf, seed=None):
        scene_seed = utils.get_seed(seed=seed)

        scene = self.make_scene(seed=seed)

        mbobs = observations.get_mbobs(
            self.bands,
            self.throughputs,
            psf,
            scene,
            image_builder=self.image_builder,
            sky_background=self.sky_background,
            seed=scene_seed,
        )

        return mbobs

    def make_obs_pair(self, psf, g1=0.00, g2=0.00, seed=None):
        scene_seed = utils.get_seed(seed=seed)

        scene = self.make_scene(seed=seed)

        plus_scene = scene.with_shear(g1=g1, g2=g2)
        minus_scene = scene.with_shear(g1=-g1, g2=-g2)

        plus_mbobs = observations.get_mbobs(
            self.bands,
            self.throughputs,
            psf,
            plus_scene,
            image_builder=self.image_builder,
            sky_background=self.sky_background,
            seed=scene_seed,
        )
        minus_mbobs = observations.get_mbobs(
            self.bands,
            self.throughputs,
            psf,
            minus_scene,
            image_builder=self.image_builder,
            sky_background=self.sky_background,
            seed=scene_seed,
        )
        mbobs_dict = {
            "plus": plus_mbobs,
            "minus": minus_mbobs,
        }

        return mbobs_dict

    def run_sim(self, seed=None):
        psf_seed, obs_seed = utils.get_seeds(2, seed=seed)

        psf = self.make_psf(seed=psf_seed)

        obs = self.make_obs(psf, seed=obs_seed)
        psf_obs = self.make_psf_obs(psf)

        return obs, psf_obs

    def run_sim_pair(self, seed=None):
        psf_seed, obs_seed = utils.get_seeds(2, seed=seed)

        psf = self.make_psf(seed=psf_seed)

        obs_dict = self.make_obs_pair(psf, g1=0.02, g2=0.00, seed=obs_seed)
        psf_obs = self.make_psf_obs(psf)

        return obs_dict, psf_obs
