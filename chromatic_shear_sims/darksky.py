import functools
import logging
import os
import urllib.request
import warnings

import galsim


logger = logging.getLogger(__name__)


DARKSY_URL = "https://raw.githubusercontent.com/lsst-pst/syseng_throughputs/main/siteProperties/darksky.dat"
DARKSKY_FILE = "darksky.dat"

THROUGHPUTS_DIR = "."


def get_darksky_path():
    darksky_file = DARKSKY_FILE
    throughputs_dir = os.environ.get("THROUGHPUTS_DIR")
    if throughputs_dir is None:
        throughputs_dir = THROUGHPUTS_DIR
    darksky_path = os.path.join(
        throughputs_dir,
        darksky_file,
    )
    return darksky_path


def retrieve_darksky(fname):
    logger.info(f"Retrieving {DARKSKY_URL} > {fname}")
    status = urllib.request.urlretrieve(DARKSKY_URL, fname)
    return status


@functools.cache
def load_darksky():
    darksky_path = get_darksky_path()
    if not os.path.exists(darksky_path):
        warnings.warn(f"{darksky_path} not found; downloading")
        retrieve_darksky()
    logger.info(f"loading darksky from {darksky_path}")
    darksky = galsim.SED(darksky_path, wave_type="nm", flux_type="flambda")
    return darksky
