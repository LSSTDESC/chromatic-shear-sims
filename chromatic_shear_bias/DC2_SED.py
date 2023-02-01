import logging
from pathlib import Path

import numpy as np

from galsim import Bandpass, LookupTable
from galsim.catalog import Catalog
from galsim.config.bandpass import BuildBandpass
from galsim.config.input import RegisterInputType, RegisterInputConnectedType, GetInputObj, InputLoader
from galsim.config.sed import SEDBuilder, RegisterSEDType
from galsim.config.util import LoggerWrapper
from galsim.config.value import GetAllParams, SetDefaultIndex, RegisterValueType
from galsim.errors import GalSimConfigError
from galsim.sed import SED
from galsim.utilities import LRU_Cache


def _read_sed_file(file_name, wave_type, flux_type):
    return SED(file_name, wave_type, flux_type)


read_sed_file = LRU_Cache(_read_sed_file)


_standard_dict = {
    'lte*' : 'starSED/phoSimMLT',
    'bergeron*' : 'starSED/wDs',
    'k[mp]*' : 'starSED/kurucz',
}


class DC2_SEDBuilder(SEDBuilder):
    """SEDs from the DC2 stellar database. Some code adapted from
    https://github.com/LSSTDESC/skyCatalogs/blob/master/python/desc/skycatalogs/utils/sed_tools.py

        index (required)    The index into the DC2 SED catalog for which to build the SED
    """
    def buildSED(self, config, base, logger):
        """Build the SED based on the specifications in the config dict.

        Parameters:
            config:     The configuration dict for the SED type.
            base:       The base configuration dict.
            logger:     If provided, a logger for logging debug statements.

        Returns:
            the constructed SED object.
        """
        req = {"index": int}
        opt = {"sed_dir" : str}
        ignore = ["num"]

        # These are fixed for all SEDs in the DC2 database
        wave_type = "Nm"
        flux_type = "flambda"

        DC2_dataset = GetInputObj("arrow_dataset", config, base, "ArrowDataset")

        SetDefaultIndex(config, DC2_dataset.getNObjects())

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        index = kwargs["index"]
        sed_dir = Path(kwargs.get("sed_dir", "."))

        file_name = DC2_dataset.get(index, "sedFilename").strip()
        if not file_name.endswith(".gz"):
            # Some files are missing ".gz" in their suffix; if this is the case,
            # append to the current suffix
            file_name += ".gz"
        path_name = Path(file_name)
        for k, v in _standard_dict.items():
            matched = False
            if path_name.match(k):
                sed_path = sed_dir / v / path_name
                matched = True
                break  # we should only have one match
        if not matched:
            raise ValueError(f"Filename {file_name} does not match any known patterns in {sed_dir}")
        if not sed_path.exists():
            raise ValueError(f"Filename {file_name} not found in {sed_dir}")

        sed_file = sed_path.as_posix()
        logger.info("Using SED file: %s",sed_file)
        sed = read_sed_file(sed_file, wave_type, flux_type)

        logger.debug("obj %d: sed = %s", base.get("obj_num", 0), sed)

        return sed, safe


RegisterSEDType("DC2_SED", DC2_SEDBuilder())
