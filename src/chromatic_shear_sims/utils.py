import importlib
import logging
import functools
import math
import sys

import galsim
import numpy as np

from chromatic_weak_lensing import MainSequence


logger = logging.getLogger(__name__)


def get_seed(seed=None):
    rng = np.random.default_rng(seed)
    maxint = sys.maxsize  # np.iinfo(np.int64).max
    return rng.integers(0, maxint)


def get_seeds(n, seed=None):
    rng = np.random.default_rng(seed)
    maxint = sys.maxsize  # np.iinfo(np.int64).max
    return rng.integers(0, maxint, n)


def get_class(entrypoint):
    # https://packaging.python.org/en/latest/specifications/entry-points/#data-model
    module_name, class_name = entrypoint.split(":")
    _module = importlib.import_module(module_name)
    _class = getattr(_module, class_name)
    return _class


def get_instance(entrypoint, *args, **kwargs):
    _class = get_class(entrypoint)
    _instance = _class(*args, **kwargs)
    return _instance


def rescale(image):
    _normalization = 1 / image.FindAdaptiveMom().moments_amp
    logger.debug(f"rescaling image with normalization: {_normalization}")

    rescaled_image = image * _normalization

    return rescaled_image


def recenter(image):
    _shift = image.FindAdaptiveMom().moments_centroid - image.true_center
    logger.debug(f"recentering image with pixel shift: ({_shift.x}, {_shift.y})")

    shifted = galsim.InterpolatedImage(
        image,
        offset=_shift,
    )

    shifted_image = shifted.drawImage(
        nx=image.ncol,
        ny=image.nrow,
        scale=image.scale,
    )

    return shifted_image


def get_mag(flux, throughput):
    zeropoint = throughput.zeropoint
    return -2.5 * np.log10(flux) + zeropoint
