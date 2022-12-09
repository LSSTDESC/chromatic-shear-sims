import logging
import re

import fitsio

from galsim.catalog import Catalog
from galsim.config.input import RegisterInputType, RegisterInputConnectedType, GetInputObj, InputLoader
from galsim.config.sed import SEDBuilder, RegisterSEDType
from galsim.config.util import LoggerWrapper
from galsim.config.value import GetAllParams, SetDefaultIndex, RegisterValueType
from galsim.errors import GalSimConfigError
from galsim.sed import SED


class DC2_SEDCatalog(Catalog):
    """A class storing the data from a DC2 SED input catalog
    We primarily inherit from Catalog but redefine the readFits method
    and define a new getRow method.
    """
    def readFits(self):
        # Here we explicitly do _not_ read the entire catalog into memory; this is because
        # the DC2 SED catalogs are large and we only need one row at a time. However, we do
        # set certain properties of the table
        with fitsio.FITS(self.file_name) as fits:
            self._data = None  # Data is only ingested when the getRow method is called
            self.names = fits[1].get_colnames()
            self.nobjects = fits[1].get_nrows()
            self._ncols = fits[1].get_nrows()
            self.isfits = True

    def getRow(self, index):
        # read row with fits
        with fitsio.FITS(self.file_name) as fits:
            data = fits[1].read(rows=[index])
        return data


# def _GenerateFromDC2_SEDCatalog(config, base, value_type):
#     """Return a value read from an input catalog
#     """
#     input_cat = GetInputObj('dc2_sed_catalog', config, base, 'DC2_SED_Catalog')
# 
#     # Setup the indexing sequence if it hasn't been specified.
#     # The normal thing with a Catalog is to just use each object in order,
#     # so we don't require the user to specify that by hand.  We can do it for them.
#     SetDefaultIndex(config, input_cat.getNObjects())
# 
#     # Coding note: the and/or bit is equivalent to a C ternary operator:
#     #     input_cat.isFits() ? str : int
#     # which of course doesn't exist in python.  This does the same thing (so long as the
#     # middle item evaluates to true).
#     req = { 'index' : int }
#     opt = { 'num' : int }
#     kwargs, safe = GetAllParams(config, base, req=req, opt=opt)
#     index = kwargs['index']
# 
#     # TODO this returns a vector -- it is unclear if GalSim handles that at the moment
#     return input_cat.getRow(index)


class DC2_SEDBuilder(SEDBuilder):
    """A class for defining an SED from a set of tophats

    DC2_SED expects the following parameters:

        sed_array (required)    The SED array
        wave_type(required)     The units (nm or Ang) of the wavelengths expected by the string function
        flux_type (required)    Which kind of flux values are in the string function
                                Allowed values: flambda, fnu, fphotons, 1
        redshift (optional)     Whether or not to redshift the SED (DC2 uses rest-frame SEDs)
                                Allowed values: True, False
                                Default: False
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
        import numpy as np
        from galsim.config.bandpass import BuildBandpass
        from galsim import LookupTable
        logger = LoggerWrapper(logger)

        req = {'wave_type': str, 'flux_type': str}
        opt = {'norm_flux_density': float, 'norm_wavelength': float,
               'norm_flux': float, 'redshift': bool,
               'index': int}
        ignore = ['norm_bandpass']

        input_cat = GetInputObj('dc2_sed_catalog', config, base, 'DC2_SED_Catalog')
        # Setup the indexing sequence if it hasn't been specified.
        # The normal thing with a Catalog is to just use each object in order,
        # so we don't require the user to specify that by hand.  We can do it for them.
        SetDefaultIndex(config, input_cat.getNObjects())

        kwargs, safe = GetAllParams(config, base, req=req, opt=opt, ignore=ignore)

        # TODO: I suspect this should happen in the _Generate function...
        index = kwargs['index']
        sed_array = input_cat.getRow(index)
        # sed_bins = [
        #     _q for _q in sed_array.dtype.names
        #     if "sed" in _q
        # ]
        # the following regex match is a bit better given that DC2 has many other SED-related
        # quantities that we don't want here
        sed_bins = [_q for _q in sed_array.dtype.names if re.match(r'sed_\d+_\d+$', _q)]

        # DC2 SEDs are tophats; the file has columns corresponding to the value of the tophat
        # for bins in wavelength. We want to parse the bin edges from the column names and
        # repeat the tophat values for each bin edge.
        # sed_bins_array = np.asarray(
        #     [(lambda x: [float(x[0]), float(x[0]) + float(x[1])])(bin.split('_')[1:]) for bin in sed_bins]
        # ).ravel()
        sed_bins_array = np.asarray(
            [np.cumsum(np.asarray(bin.split('_')[1:], dtype='float')) for bin in sed_bins]
        )
        sed_values_array = np.asarray(
            [np.repeat(sed_array[bin], 2) for bin in sed_bins]
        )
        # We sort the SED tophats according to the bins
        _idx = np.argsort(sed_bins_array, axis=0)
        # sed_bins_array = np.take_along_axis(sed_bins_array, _idx, axis=0).ravel()
        sed_bins_array = np.take_along_axis(sed_bins_array * np.asarray([1, 1 - 1e-9]), _idx, axis=0).ravel()  # narrow the bins by a small factor to avoid overlap
        sed_values_array = np.take_along_axis(sed_values_array, _idx, axis=0).ravel()
        # TODO fix the SED to 0 for the 0 bin
        sed_bins_array = np.concatenate([np.array([0, sed_bins_array[0] * (1 - 1e-9)], dtype='float'), sed_bins_array])
        sed_values_array = np.concatenate([np.array([0, 0], dtype='float'), sed_values_array])

        norm_flux_density = kwargs.pop('norm_flux_density', None)
        norm_wavelength = kwargs.pop('norm_wavelength', None)
        norm_flux = kwargs.pop('norm_flux', None)
        redshift = kwargs.pop('redshift', False)
        wave_type = kwargs.pop('wave_type')
        flux_type = kwargs.pop('flux_type')

        tophat_lookuptable = LookupTable(
            x=sed_bins_array,
            f=sed_values_array,
            interpolant='linear',
        )  # Linearly interpolate between the tophats
        sed = SED(tophat_lookuptable, wave_type, flux_type)
        if norm_flux_density:
            sed = sed.withFluxDensity(norm_flux_density, wavelength=norm_wavelength)
        elif norm_flux:
            bandpass, safe1 = BuildBandpass(config, 'norm_bandpass', base, logger)
            sed = sed.withFlux(norm_flux, bandpass=bandpass)
            safe = safe and safe1
        if redshift:
            # Cast the redshift to a float for compatibility with hashing
            sed = sed.atRedshift(float(sed_array["redshift"]))
        logger.info(f"Using SED {index} from dc2_sed_catalog")

        return sed, safe


RegisterInputType("dc2_sed_catalog", InputLoader(DC2_SEDCatalog, has_nobj=True))
RegisterSEDType("dc2_sed", DC2_SEDBuilder())
# RegisterValueType("DC2_SED_Catalog", _GenerateFromDC2_SEDCatalog, [ None ], input_type="dc2_sed_catalog")
# RegisterInputConnectedType("dc2_sed_catalog", "dc2_sed")
