# from dataclasses import dataclass
import logging

import numpy as np
import galsim


logger = logging.getLogger(__name__)


class Survey:
    """Struct for keeping track of survey info"""
    def __init__(
        self,
        scale,
        exptime,
        area,
        gain,
        ncoadd,
        zeropoints,
        sky,
        bands,
    ):
        self.scale = scale
        self.exptime = exptime
        self.area = area
        self.gain = gain
        self.ncoadd = ncoadd
        self.zeropoints = zeropoints
        self.sky = sky
        # self.bandpasses = bandpasses
        # self.bandpasses = self.get_bandpasses(bandpass_dir, bands)
        self.bands = bands
        self.sky_rms = self.get_sky_rms()
        self._bandpasses = None

    def get_sky_rms(self):
        # return {
        #     f: np.power(10, -0.4 * (self.sky[f] - self.zeropoints[f])) * self.exptime * self.scale**2 / self.area / 100
        #     for f in self.sky.keys()
        # }
        # return {
        #     f: np.power(10, -0.4 * (self.sky[f] - self.zeropoints[f]))
        #     for f in self.sky.keys()
        # }
        return {
            band: 1e-9
            for band in self.bands
        }

    @property
    def bandpasses(self):
        return self._bandpasses

    def load_bandpasses(self, throughput_dir):
        logger.info(f"loading bandpasses from {throughput_dir}")
        bandpasses = {
            band: galsim.Bandpass(
                f"{throughput_dir}/total_{band}.dat",
                "nm",
            ).withZeropoint("AB").thin(1e-3)
            for band in self.bands
        }
        self._bandpasses = bandpasses

# cf.
# 	https://lsst.org/scientists/keynumbers
# 	https://smtn-002.lsst.io/
# 	https://pstn-054.lsst.io/
lsst = Survey(
    scale=0.2,  # arcsec / pixel
    exptime=30,  # s
    area=np.pi * (6.423 / 2)**2 * 1e4,  # cm^2
    gain=1,  # photons/ADU
    ncoadd={
        "u": 100,
        "g": 100,
        "r": 100,
        "i": 100,
        "z": 100,
        "y": 100,
    },
    zeropoints={
        "u": 27.03,
        "g": 28.38,
        "r": 28.16,
        "i": 27.85,
        "z": 27.46,
        "y": 26.68,
    },
    sky={
        "u": 22.96,
        "g": 22.26,
        "r": 21.20,
        "i": 20.48,
        "z": 19.60,
        "y": 18.61,
    },
    bands=["u", "g", "r", "i", "z", "y"],
)
