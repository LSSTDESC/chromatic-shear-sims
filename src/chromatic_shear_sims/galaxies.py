import logging

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class GalaxyBuilder:
    def __init__(self, entrypoint, **kwargs):
        self.model = utils.get_instance(entrypoint, **kwargs)
        self.name = self.model.name

    def __call__(self, galaxy_params, **kwargs):
        galaxy = self.model.get_galaxy(galaxy_params, **kwargs)
        return galaxy

