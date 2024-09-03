import functools
import logging
import os
import urllib.request
import warnings

import galsim


logger = logging.getLogger(__name__)


DARKSKY_URL = "https://raw.githubusercontent.com/lsst-pst/syseng_throughputs/main/siteProperties/darksky.dat"
DARKSKY_FILE = "darksky.dat"

THROUGHPUT_DIR = "."


def get_darksky_path():
    darksky_file = DARKSKY_FILE
    throughput_dir = os.environ.get("THROUGHPUT_DIR")
    if throughput_dir is None:
        throughput_dir = THROUGHPUT_DIR
    if not os.path.isdir(throughput_dir):
        os.makedirs(throughput_dir)
    darksky_path = os.path.join(
        throughput_dir,
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
        retrieve_darksky(darksky_path)
    logger.info(f"loading darksky from {darksky_path}")
    darksky = galsim.SED(darksky_path, wave_type="nm", flux_type="flambda")
    return darksky
