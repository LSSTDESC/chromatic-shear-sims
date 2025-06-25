import copy
import logging

import galsim


logger = logging.getLogger(__name__)

gsparams = galsim.GSParams(
    maximum_fft_size=16384,
    # kvalue_accuracy=1e-8,
    # maxk_threshold=1e-5,
)

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
        self.model = psf.withGSParams(gsparams)

    def __eq__(self, other):
        return self.model == other.model

    def draw_image(self, star, throughput, image):
        # note that we provide a special method for drawing PSF images to
        # ensure normalizaiton and centering
        psf_model = self.model
        observed_psf = galsim.Convolve([star, psf_model]).withFlux(
            1,
            throughput,
        )
        _centroid = observed_psf.calculateCentroid(throughput)
        _offset = -1 * _centroid / image.scale
        observed_psf.drawImage(
            image=image,
            bandpass=throughput,
            offset=_offset,
            add_to_image=True,
        )
        return image
