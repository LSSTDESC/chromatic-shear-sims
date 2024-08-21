import importlib
import logging
import functools
import math

import galsim
import numpy as np

from chromatic_weak_lensing import MainSequence


logger = logging.getLogger(__name__)


def get_instance(module_name, class_name, *args, **kwargs):
    _module = importlib.import_module(module_name)
    _class = getattr(_module, class_name)
    _instance = _class(*args, **kwargs)
    return _instance


def rescale(image):
    _normalization = 1 / image.FindAdaptiveMom().moments_amp
    logger.info(f"rescaling image with normalization: {_normalization}")

    rescaled_image = image * _normalization

    return rescaled_image


def recenter(image):
    _shift = image.FindAdaptiveMom().moments_centroid - image.true_center
    logger.info(f"recentering image with pixel shift: ({_shift.x}, {_shift.y})")

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


