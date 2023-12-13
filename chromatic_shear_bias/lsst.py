import numpy as np

import galsim


# cf.
# 	https://lsst.org/scientists/keynumbers
# 	https://smtn-002.lsst.io/
# 	https://pstn-054.lsst.io/

SCALE = 0.2  # arcsec/pixel
EXPTIME = 30  # s
AREA = np.pi * (6.423 / 2)**2 * 1e4  # cm^2
GAIN = 1  # photons/ADU

NCOADD = {
    "u": 100,
    "g": 100,
    "r": 100,
    "i": 100,
    "z": 100,
    "y": 100,
}

ZEROPOINTS = {
    "u": 27.03,
    "g": 28.38,
    "r": 28.16,
    "i": 27.85,
    "z": 27.46,
    "y": 26.68,
}

SKY = {
    "u": 22.96,
    "g": 22.26,
    "r": 21.20,
    "i": 20.48,
    "z": 19.60,
    "y": 18.61,
}

SKY_RMS = {
    f: np.power(10, -0.4 * (SKY[f] - ZEROPOINTS[f])) * EXPTIME * SCALE
    for f in SKY.keys()
}

BANDPASSES = {
    band: galsim.Bandpass(f"LSST_{band}.dat", "nm").withZeropoint("AB")
    for band in {"u", "g", "r", "i", "z", "y"}
}
