import logging

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class Galaxies:
    def __init__(self, module_name, class_name):
        self.model = utils.get_instance(module_name, class_name)
        self.name = self.model.name

    def __call__(self, galaxy_params, **kwargs):
        # model_params = self.model.get_params(galaxy_params)
        # galaxy = self.model.get_galaxy(*model_params)
        galaxy = self.model.get_galaxy(*galaxy_params, **kwargs)
        return galaxy


class GalaxyData:
    def __init__(self, module_name, class_name):
        self._data = None
        self._num_rows = None
        self.model = utils.get_class(module_name, class_name)

    @property
    def data(self):
        return self._data

    @property
    def num_rows(self):
        return self._num_rows

    def register_data(self, data):
        self._data = self.model(data)
        self._num_rows = self.data.num_rows
        logger.info(f"registered data with {self.num_rows} rows")

    def __call__(self, i, **kwargs):
        return self.data.get_params(i, **kwargs)

