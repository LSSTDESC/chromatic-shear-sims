import logging

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class CatalogData:
    def __init__(self, module_name, class_name, data_loader):
        logger.info(f"initializing catalog data for {module_name}.{class_name}")
        self.module_name = module_name
        self.class_name = class_name
        self.data_builder = utils.get_class(module_name, class_name)
        self.columns = self.data_builder.columns
        self.data_loader = data_loader
        self._data = None

    @property
    def data(self):
        return self._data

    @property
    def num_rows(self):
        return self.data.num_rows

    # TODO make this a random process that will iterate through data and load more if needed?
    def __call__(self, i, **kwargs):
        return self.data.get_params(i, **kwargs)

    def load(self, n, seed=None):
        data = self.data_loader.sample(n, columns=self.columns, seed=seed)
        self._data = self.data_builder(data)

