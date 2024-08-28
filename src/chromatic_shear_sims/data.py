import logging

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class Data:
    def __init__(self, entrypoint, data_loader):
        logger.info(f"initializing data for {entrypoint}")
        self.entrypoint = entrypoint
        self.data_builder = utils.get_class(entrypoint)
        self.columns = self.data_builder.columns
        self.data_loader = data_loader

    def load(self, n, seed=None):
        sample = self.data_loader.sample(n, columns=self.columns, seed=seed)

        data = self.data_builder(sample)

        return data
