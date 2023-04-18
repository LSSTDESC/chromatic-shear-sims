import logging
import re

import numpy as np

import galsim
from galsim import Bandpass, LookupTable, GSObject, GSParams
from galsim.catalog import Catalog
from galsim.config.bandpass import BuildBandpass
from galsim.config.input import RegisterInputType, RegisterInputConnectedType, GetInputObj, InputLoader
from galsim.config.sed import SEDBuilder, RegisterSEDType
from galsim.config.util import LoggerWrapper, GetRNG
from galsim.config.value import GetAllParams, SetDefaultIndex, RegisterValueType
from galsim.errors import GalSimConfigError
from galsim.sed import SED
from galsim.config.gsobject import RegisterObjectType

from . import sed_tools


# FIXME what is the correct comology to use here?
cosmology = {
    "Om0": 0.2648,
    "Ob0": 0.0448,
    "H0": 71.0,
    "sigma8": 0.8,
    "n_s": 0.963,
}


def _BuildcosmoDC2Galaxy(config, base, ignore, gsparams, logger):
    """Build a cosmoDC2 galaxy using the cosmos_catalog input item.
    """
    dataset = GetInputObj("arrow_dataset", config, base, "ArrowDataset")

    SetDefaultIndex(config, dataset.getNObjects())

    req = {"index": int}
    opt = {}
    ignore = ["num"]
    ignore += [
        'dilate', 'dilation', 'ellip', 'rotate', 'rotation', 'scale_flux',
        'magnify', 'magnification', 'shear', 'lens', 'shift', 'sed',
        'gsparams', 'skip',
        'current', 'index_key', 'repeat'
    ]
    ignore += [ 'resolution', 'signal_to_noise', 'redshift', 're_from_res' ]
    # opt = { "index" : int,
    #         "gal_type" : str,
    #         "noise_pad_size" : float,
    #         "deep" : bool,
    #         "sersic_prec": float,
    #         "chromatic": bool,
    #         "area": float,
    #         "exptime": float,
    # }

    kwargs, safe = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)
    if gsparams: kwargs['gsparams'] = GSParams(**gsparams)

    cls_name = "cosmoDC2Galaxy"
    # rng = GetRNG(config, base, logger, cls_name)

    # FIXME: do we need this?
    # if 'index' not in kwargs:
    #     kwargs['index'], n_rng_calls = sample_cat.selectRandomIndex(1, rng=rng, _n_rng_calls=True)
    #     safe = False

    #     # Make sure this process gives consistent results regardless of the number of processes
    #     # being used.
    #     if not isinstance(sample_cat, GalaxySample) and rng is not None:
    #         # Then sample_cat is really a proxy, which means the rng was pickled, so we need to
    #         # discard the same number of random calls from the one in the config dict.
    #         rng.discard(int(n_rng_calls))

    # kwargs['rng'] = rng

    # NB. Even though index is officially optional, it will always be present, either because it was
    #     set by a call to selectRandomIndex, explicitly by the user, or due to the call to
    #     SetDefaultIndex.
    index = kwargs['index']
    if index >= dataset.nobjects:
        raise GalSimConfigError(
            "index=%s has gone past the number of entries in the %s"%(index, cls_name))

    logger.debug('obj %d: %s kwargs = %s',base.get('obj_num',0), cls_name, kwargs)

    galaxy_params = dataset.getRow(index)

    galaxy_id = galaxy_params.get("galaxy_id")[0]
    logger.info("obj %d: galaxy %d (%d in current batch of %s)", base.get("obj_num", 0), galaxy_id, index, dataset)

    redshift_hubble = galaxy_params.get("redshift_true", [0])[0]
    redshift = galaxy_params.get("redshift", [0])[0]

    gal_components = []
    for component in ["bulge", "disk"]:
        sersic = galaxy_params.get(f"sersic_{component}")[0]
        size = galaxy_params.get(f"size_{component}_true")[0]
        ellipticity_1 = galaxy_params.get(f"ellipticity_1_{component}_true")[0]
        ellipticity_2 = galaxy_params.get(f"ellipticity_2_{component}_true")[0]

        ellipticity = galsim.Shear(e1=ellipticity_1, e2=ellipticity_2)
        gal = galsim.Sersic(
            n=sersic,
            half_light_radius=size,
        ) \
        .shear(ellipticity)

        sed_bins = [_q for _q in galaxy_params.keys() if re.match(rf"sed_\d+_\d+_{component}$", _q)]

        sed_bins_array = np.asarray(
            [np.asarray(bin.split("_")[1:3], dtype="float") for bin in sed_bins]
        )

        sed_values_array = np.asarray(
            [galaxy_params[bin] for bin in sed_bins]
        ).ravel()

        # There are some components with no spectra. In this case,
        # skip that component
        if np.allclose(0, sed_values_array):
            continue

        sed_factory = sed_tools.ObservedSedFactory(
            sed_bins_array,
            cosmology,
        )

        sed = sed_factory.create(
            sed_values_array,
            redshift_hubble,
            redshift,
        )

        gal_components.append(gal * sed)

    gal = galsim.Add(gal_components)

    return gal, safe

# Register this as a valid gsobject type
RegisterObjectType('cosmoDC2Galaxy', _BuildcosmoDC2Galaxy, input_type='arrow_dataset')
