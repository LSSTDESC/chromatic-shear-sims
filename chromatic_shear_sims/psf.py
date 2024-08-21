import copy
import logging

import galsim

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class PSF:
    def __init__(self, gsobject):
        self.gsobject = gsobject

    @classmethod
    def from_config(cls, psf_config):
        psf_config_copy = copy.deepcopy(psf_config)
        config = {"psf": psf_config_copy}
        psf, _ = galsim.config.BuildGSObject(config, "psf")
        return cls(psf)

    def __eq__(self, other):
        return self.gsobject == other.gsobject

    def draw_image(self, star, throughput, nx, ny, scale):
        psf_model = self.gsobject
        observed_psf = galsim.Convolve([star, psf_model])
        psf_image = galsim.Image(
            nx,
            ny,
            scale=scale,
        )
        observed_psf.drawImage(
            image=psf_image,
            bandpass=throughput,
            add_to_image=True,
        )
        rescaled_image = utils.rescale(psf_image)
        recentered_image = utils.recenter(rescaled_image)
        return recentered_image

