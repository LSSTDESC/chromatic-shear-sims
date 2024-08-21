import copy
import logging

import galsim

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class ImageBuilder:
    def __init__(
        self,
        xsize,
        ysize,
        scale,
        ncoadd=1,
        border=0,
    ):
        self.xsize = xsize
        self.ysize = ysize
        self.scale = scale
        self.ncoadd = ncoadd
        self.border = border

    @property
    def npixel(self):
        return self.xsize * self.ysize

    @classmethod
    def from_config(cls, image_config):
        image_config_copy = copy.deepcopy(image_config)
        return cls(**image_config_copy)

    def get_noise_sigma(self, darksky, throughput):
        noise_sigma = utils.get_noise_sigma(
            darksky,
            throughput,
            self.npixel,
            ncoadd=self.ncoadd,
        )
        return noise_sigma

    def get_image(self):
        image = galsim.Image(
            self.xsize,
            self.ysize,
            scale=self.scale,
        )
        return image
