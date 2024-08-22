import logging

from chromatic_weak_lensing.diffsky import Diffsky, RomanRubin

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class DiffskyGalaxies:
    def __init__(self, morphology="achromatic", knots=False, red_limit=None):
        self._data = None
        self.model = Diffsky(red_limit=red_limit)
        self.name = self.model.name
        self.morphology = morphology
        self.knots = knots
        self.columns = RomanRubin.columns
        logger.info(f"initialized {self.name} galaxies with morphology={self.morphology} and knots={self.knots}")

    @property
    def data(self):
        return self._data

    def register_data(self, data):
        self._data = RomanRubin(data)
        logger.info(f"registered data with {self.data.num_rows} rows")

    def __call__(self, i):
        galaxy_params = self.data.get_params(i, knots=self.knots)
        galaxy = self.model.get_galaxy(*galaxy_params, morphology=self.morphology)
        return galaxy


# class Galaxies:
#     def __init__(self, module_name, class_name, morphology="achromatic", knots=False, red_limit=None):
#         self._data = None
#         self.model = utils.get_instance(module_name, class_name)
#         self.name = self.model.name
#         self.morphology = morphology
#         self.knots = knots
# 
#     @property
#     def data(self):
#         return self._data
# 
#     def load_data(self, module_name, class_name, data):
#         self._data = utils.get_instance(
#             module_name,
#             class_name,
#             data,
#         )
# 
#     def __call__(self, i):
#         galaxy_params = self.data.get_params(i, knots=self.knots)
#         galaxy = self.model.get_galaxy(*galaxy_params, morphology=self.morphology)
#         return galaxy
# 
