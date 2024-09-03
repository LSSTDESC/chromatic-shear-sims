import functools
import logging
import os
import urllib.request
import warnings

import galsim


logger = logging.getLogger(__name__)


DARKSKY_URL = "https://raw.githubusercontent.com/lsst-pst/syseng_throughputs/main/siteProperties/darksky.dat"
DARKSKY_FILE = "darksky.dat"


def get_darksky_path():
    darksky_file = DARKSKY_FILE
    darksky_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        darksky_file,
    )
    return darksky_path


@functools.cache
def load_darksky():
    darksky_path = get_darksky_path()
    if not os.path.exists(darksky_path):
        raise ValueError(f"{darksky_path} not found")
    logger.info(f"loading darksky from {darksky_path}")
    darksky = galsim.SED(darksky_path, wave_type="nm", flux_type="flambda")
    return darksky
