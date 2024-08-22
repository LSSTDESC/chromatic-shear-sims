import logging

from chromatic_weak_lensing import StellarParams
from chromatic_weak_lensing.lsst_sim import LSST_Sim

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class LSSTSimStars:
    def __init__(self, module_name, class_name):
        self._data = None
        self.model = utils.get_instance(module_name, class_name)
        self.name = self.model.name
        logger.info(f"initialized {self.name} stars for LSST_Sim stars")

    @property
    def data(self):
        return self._data

    def register_data(self, data):
        self._data = LSST_Sim(data)
        logger.info(f"registered data with {self.data.num_rows} rows")

    def __call__(self, i):
        stellar_params = self.data.get_params(i)
        model_params = self.model.get_params(stellar_params)
        star = self.model.get_star(*model_params)
        return star

