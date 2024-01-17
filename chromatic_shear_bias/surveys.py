# from dataclasses import dataclass

import numpy as np
import galsim


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
        bandpasses,
    ):
        self.scale = scale
        self.exptime = exptime
        self.area = area
        self.gain = gain
        self.ncoadd = ncoadd
        self.zeropoints = zeropoints
        self.sky = sky
        self.bandpasses = bandpasses
        self.sky_rms = self.get_sky_rms()

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
            f: 1e-6
            for f in self.sky.keys()
        }

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
    bandpasses={
        band: galsim.Bandpass(f"LSST_{band}.dat", "nm").withZeropoint("AB")
        for band in {"u", "g", "r", "i", "z", "y"}
    },
)
