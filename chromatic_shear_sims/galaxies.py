import logging

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class GalaxyBuilder:
    def __init__(self, module_name, class_name, **kwargs):
        self.model = utils.get_instance(module_name, class_name, **kwargs)
        self.name = self.model.name

    def __call__(self, galaxy_params, **kwargs):
        # model_params = self.model.get_params(galaxy_params)
        # galaxy = self.model.get_galaxy(*model_params)
        galaxy = self.model.get_galaxy(*galaxy_params, **kwargs)
        return galaxy


class GalaxyData:
    def __init__(self, module_name, class_name, data):
        self.module_name = module_name
        self.class_name = class_name
        self._data = utils.get_instance(module_name, class_name, data)

    @property
    def data(self):
        return self._data

    @property
    def num_rows(self):
        return self.data.num_rows

    def __call__(self, i, **kwargs):
        return self.data.get_params(i, **kwargs)

