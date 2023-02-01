import logging
import re

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


class cosmoDC2_SED_Builder(SEDBuilder):
    """A class for defining an SED from a set of tophats

    DC2_SED expects the following parameters:

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
        opt = {}
        ignore = ["num"]

        cosmoDC2_dataset = GetInputObj("arrow_dataset", config, base, "ArrowDataset")  # cosmoDC2 is stored as parquet files

        SetDefaultIndex(config, cosmoDC2_dataset.getNObjects())

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        index = kwargs["index"]

        # get SED
        # DC2 SEDs are in units of 4.4659e13 W/Hz defined over bins in units of Angstrom
        # cf. https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/SCHEMA.md#extragalactic-catalogs
        _to_W_per_Hz = 4.4659e13
        _to_cgs = 1e7 * 1e-4  # (erg / Joule) * (m**2 / cm**2)

        # From https://github.com/LSSTDESC/imSim/blob/main/imsim/instcat.py
        # Using area-weighted effective aperture over FOV
        # from https://confluence.lsstcorp.org/display/LKB/LSST+Key+Numbers
        _rubin_area = 0.25 * np.pi * 649**2  # cm^2

        logger.info("obj %d: sed %d in current batch of %s", base.get("obj_num", 0), index, cosmoDC2_dataset)
        sed_array = cosmoDC2_dataset.getRow(index)
        sed_bins = [_q for _q in sed_array.keys() if re.match(r"sed_\d+_\d+_no_host_extinction$", _q)]
        redshift = sed_array["redshift"].pop()

        # DC2 SEDs are tophats; the file has columns corresponding to the value of the tophat
        # for bins in wavelength. We want to parse the bin edges from the column names and
        # repeat the tophat values for each bin edge.
        sed_bins_array = np.asarray(
            [np.cumsum(np.asarray(bin.split("_")[1:3], dtype="float")) for bin in sed_bins]
        )
        sed_values_array = np.asarray(
            [np.repeat(sed_array[bin], 2) for bin in sed_bins]
        ) * _to_W_per_Hz * _to_cgs / _rubin_area  # convert from 4.4659e13 W/Hz to erg/Hz/cm^2/s (i.e., nu)

        # We sort the SED tophats according to the bins, narrow by a small
        # factor to avoid overlapping bin edges; ravel to a 1-D array.
        _idx = np.argsort(sed_bins_array, axis=0)
        sed_bins_array = np.take_along_axis(sed_bins_array * np.asarray([1, 1 - 1e-9]), _idx, axis=0).ravel()
        sed_values_array = np.take_along_axis(sed_values_array, _idx, axis=0).ravel()

        # Set the SED response to 0 at the bluest wavelengths
        sed_bins_array = np.concatenate([np.array([1e-9, sed_bins_array[0] * (1 - 1e-9)], dtype="float"), sed_bins_array])
        sed_values_array = np.concatenate([np.array([0, 0], dtype="float"), sed_values_array])

        tophat_lookuptable = LookupTable(
            x=sed_bins_array,
            f=sed_values_array,
            interpolant="linear",
        )  # Linearly interpolate between the tophats
        sed = SED(
            tophat_lookuptable,
            wave_type="Ang",
            flux_type="fnu",
            redshift=redshift
        )
        # sed = sed.withFluxDensity(1., 500)  # TODO use appropriate units rather than rescaled flux density
        # sed = sed.atRedshift(redshift)

        # TODO the following isn't a great solution and is verified to be incorrect at the ~1% level
        bands = ["u", "g", "r", "i", "z", "y"]
        bps = [Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm") for band in bands]
        bps_zp = [bp.withZeropoint("AB") for bp in bps]
        sed = sed.withMagnitude(sed_array["mag_true_r_lsst"].pop(), bps_zp[2])  # use the provided magnitudes to set overall normalization

        return sed, safe


RegisterSEDType("cosmoDC2_SED", cosmoDC2_SED_Builder())
