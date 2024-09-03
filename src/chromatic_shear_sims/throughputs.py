# from importlib.resources import files
import functools
import logging
import os
import urllib.request
import warnings

import galsim


logger = logging.getLogger(__name__)


BANDS = ["u", "g", "r", "i", "z", "y"]

# https://github.com/LSSTDESC/lsstdesc-diffsky/blob/rr23_legacy/data/throughputs/lsst/README.rst
# /global/cfs/cdirs/descssim/imSim/lsst/data/throughputs/baseline/total_*.dat
THROUGHPUTS_URLS = {
    "u": "https://raw.githubusercontent.com/LSSTDESC/lsstdesc-diffsky/rr23_legacy/data/throughputs/lsst/total_u.dat",
    "g": "https://raw.githubusercontent.com/LSSTDESC/lsstdesc-diffsky/rr23_legacy/data/throughputs/lsst/total_g.dat",
    "r": "https://raw.githubusercontent.com/LSSTDESC/lsstdesc-diffsky/rr23_legacy/data/throughputs/lsst/total_r.dat",
    "i": "https://raw.githubusercontent.com/LSSTDESC/lsstdesc-diffsky/rr23_legacy/data/throughputs/lsst/total_i.dat",
    "z": "https://raw.githubusercontent.com/LSSTDESC/lsstdesc-diffsky/rr23_legacy/data/throughputs/lsst/total_z.dat",
    "y": "https://raw.githubusercontent.com/LSSTDESC/lsstdesc-diffsky/rr23_legacy/data/throughputs/lsst/total_y.dat",
}

THROUGHPUTS = {
    "u": "total_u.dat",
    "g": "total_g.dat",
    "r": "total_r.dat",
    "i": "total_i.dat",
    "z": "total_z.dat",
    "y": "total_y.dat",
}


def get_throughput_path(band=""):
    throughput = THROUGHPUTS[band]
    throughput_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        throughput,
    )
    return throughput_path


@functools.cache
def load_throughput(band=""):
    throughput_path = get_throughput_path(band)
    if not os.path.exists(throughput_path):
        raise ValueError(f"{throughput_path} not found")
    logger.info(f"loading throughput for {band}-band from {throughput_path}")
    throughput = galsim.Bandpass(throughput_path, "nm").thin().withZeropoint("AB")
    return throughput


def load_throughputs(bands=BANDS):
    throughputs = {}
    for band in bands:
        throughputs[band] = load_throughput(band)

    return throughputs
