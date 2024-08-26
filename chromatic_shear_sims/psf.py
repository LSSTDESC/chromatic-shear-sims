import copy
import logging

import galsim

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class PSF:
    def __init__(self, config, seed=None):
        psf_config_copy = copy.deepcopy(config)
        self.config = psf_config_copy
        self.seed = seed
        grng = galsim.BaseDeviate(seed)
        config = {
            "psf": psf_config_copy,
            "rng": grng,
        }
        psf, _ = galsim.config.BuildGSObject(config, "psf", logger=logger)
        self.model = psf

    def __eq__(self, other):
        return self.model == other.model

    def draw_image(self, star, throughput, image):
        # note that we provide a special method for drawing PSF images to
        # ensure normalizaiton and centering
        psf_model = self.model
        observed_psf = galsim.Convolve([star, psf_model])
        observed_psf.drawImage(
            image=image,
            bandpass=throughput,
            add_to_image=True,
        )
        rescaled_image = utils.rescale(image)
        recentered_image = utils.recenter(rescaled_image)
        return recentered_image
