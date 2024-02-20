"""
"""

import os
import re

import astropy.constants
import galsim
import numpy as np

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


# From
# https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/catalog_configs/cosmoDC2_v1.1.4_parquet.yaml
COSMOLOGY = {
    "H0": 71.0,
    "Om0": 0.2648,
    "Ob0": 0.0448,
    "sigma8": 0.8,
    "n_s": 0.963,
}


# From
# https://github.com/LSSTDESC/skyCatalogs/blob/master/python/desc/skycatalogs/utils/sed_tools.py
class ObservedSedFactory:
    _clight = astropy.constants.c.to('m/s').value
    # Conversion factor below of cosmoDC2 tophat Lnu values to W/Hz comes from
    # https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/SCHEMA.md
    _to_W_per_Hz = 4.4659e13

    def __init__(self, th_definition, cosmology, delta_wl=0.001):
        # Get wavelength and frequency bin boundaries.
        bins = th_definition
        wl0 = [_[0] for _ in bins]

        # Find index of original bin which includes 500 nm == 5000 ang
        ix = -1
        for w in wl0:
            if w > 5000:
                break
            ix += 1

        self._ix_500nm = ix

        wl0.append(bins[-1][0] + bins[-1][1])

        wl0 = 0.1*np.array(wl0)
        self.wl = np.array(wl0)
        self.nu = self._clight/(self.wl*1e-9)  # frequency in Hz

        # Also save version of wl where vertical rise is replaced by
        # steep slope
        wl_deltas = []
        for i in range(len(bins)):
            wl_deltas.extend((self.wl[i], self.wl[i+1] - delta_wl))

        # Prepend more bins which will have 0 value
        n_bins = int(wl0[0]) - 1
        pre_wl = [float(i) for i in range(n_bins)]

        wl_deltas = np.insert(wl_deltas, 0, pre_wl)

        # Also make a matching array of 0 values
        self.pre_val = [0.0 for i in range(n_bins)]

        self._wl_deltas = wl_deltas
        self._wl_deltas_u_nm = wl_deltas*u.nm

        # Create a FlatLambdaCDM cosmology from a dictionary of input
        # parameters.  This code is based on/borrowed from
        # https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/cosmodc2.py#L128
        cosmo_astropy_allowed = FlatLambdaCDM.__init__.__code__.co_varnames[1:]
        ##cosmo_astropy = {k: v for k, v in config['Cosmology'].items()
        ##                 if k in cosmo_astropy_allowed}
        cosmo_astropy = {k: v for k, v in cosmology.items()
                         if k in cosmo_astropy_allowed}
        self.cosmology = FlatLambdaCDM(**cosmo_astropy)

        self.sims_sed_library_dir = os.getenv('SIMS_SED_LIBRARY_DIR')

    # Useful for getting magnorm from f_nu values
    @property
    def ix_500nm(self):
        return self._ix_500nm

    @property
    def wl_deltas(self):
        return self._wl_deltas

    @property
    def wl_deltas_u_nm(self):
        return self._wl_deltas_u_nm

    def dl(self, z):
        """
        Return the luminosity distance in units of meters.
        """
        # Conversion factor from Mpc to meters (obtained from pyccl).
        MPC_TO_METER = 3.085677581491367e+22
        return self.cosmology.luminosity_distance(z).value*MPC_TO_METER

    def create(self, Lnu, redshift_hubble, redshift, resolution=None):
        '''
        Given tophat values from cosmoDC2 produce redshifted sed.
        Does not apply extinction.
        '''
        # Compute Llambda in units of W/nm
        Llambda = (Lnu*self._to_W_per_Hz*(self.nu[:-1] - self.nu[1:])
                   /(self.wl[1:] - self.wl[:-1]))

        # Fill the arrays for the galsim.LookupTable.   Prepend
        # zero-valued bins down to mix extinction wl to handle redshifts z > 2.
        my_Llambda = []
        my_Llambda += self.pre_val
        for i in range(len(Llambda)):
            # Dealt with wl already in __init__
            my_Llambda.extend((Llambda[i], Llambda[i]))

        # Convert to (unredshifted) flux given redshift_hubble.
        flambda = np.array(my_Llambda)/(4.0*np.pi*self.dl(redshift_hubble)**2)

        # Convert to cgs units
        flambda *= (1e7/1e4)  # (erg/joule)*(m**2/cm**2)

        # Create the lookup table.
        lut = galsim.LookupTable(self.wl_deltas, flambda, interpolant='nearest')

        if resolution:
            wl_min = min(self.wl_deltas)
            wl_max = max(self.wl_deltas)
            wl_res = np.linspace(wl_min, wl_max, int((wl_max - wl_min)/resolution))
            flambda_res = [lut(wl) for wl in wl_res]
            lut = galsim.LookupTable(wl_res, flambda_res, interpolant='linear')

        # Create the SED object and apply redshift.
        sed = galsim.SED(lut, wave_type='nm', flux_type='flambda')\
                    .atRedshift(redshift)

        return sed

    def create_pointsource(self, rel_path, redshift=0):
        '''
        Return a galsim SED from information in a file
        '''
        fpath = os.path.join(self.sims_sed_library_dir, rel_path)

        sed = galsim.SED(fpath, wave_type='nm', flux_type='flambda')
        if redshift > 0:
            sed = sed.atRedshift(redshift)
        return sed

    def magnorm(self, tophat_values, z_H):
        one_Jy = 1e-26  # W/Hz/m**2
        Lnu = tophat_values[self.ix_500nm]*self._to_W_per_Hz  # convert to W/Hz
        Fnu = Lnu/4/np.pi/self.dl(z_H)**2
        with np.errstate(divide='ignore', invalid='ignore'):
            return -2.5*np.log10(Fnu/one_Jy) + 8.90


def build_cosmoDC2_ellipse(gal_params):

    size = gal_params.get(f"size_true")[0]
    ellipticity_1 = gal_params.get(f"ellipticity_1_true")[0]
    ellipticity_2 = gal_params.get(f"ellipticity_2_true")[0]

    ellipticity = galsim.Shear(e1=ellipticity_1, e2=ellipticity_2)
    gal = galsim.Sersic(
        n=1,
        half_light_radius=size,
    ).shear(ellipticity)

    return gal


def build_simple_cosmoDC2_gal(gal_params):
    cosmology = COSMOLOGY
    redshift_hubble = gal_params.get("redshift_true", 0)
    redshift = gal_params.get("redshift", 0)

    gal = galsim.Sersic(
        n=1,
        half_light_radius=0.5,
    )

    sed_bins = [
        _q for _q in gal_params.keys() if re.match(rf"sed_\d+_\d+$", _q)
    ]

    sed_bins_array = np.asarray(
        [np.asarray(bin.split("_")[1:3], dtype="float") for bin in sed_bins]
    )

    sed_values_array = np.asarray([gal_params[bin] for bin in sed_bins]).ravel()

    sed_factory = ObservedSedFactory(
        sed_bins_array,
        cosmology,
    )

    sed = sed_factory.create(
        sed_values_array,
        redshift_hubble,
        redshift,
    )

    gal = gal * sed

    return gal


def build_cosmoDC2_gal(gal_params):
    cosmology = COSMOLOGY
    redshift_hubble = gal_params.get("redshift_true", [0])[0]
    redshift = gal_params.get("redshift", [0])[0]

    gal_components = []
    for component in ["bulge", "disk"]:
        sersic = gal_params.get(f"sersic_{component}")[0]
        size = gal_params.get(f"size_{component}_true")[0]
        ellipticity_1 = gal_params.get(f"ellipticity_1_{component}_true")[0]
        ellipticity_2 = gal_params.get(f"ellipticity_2_{component}_true")[0]

        ellipticity = galsim.Shear(e1=ellipticity_1, e2=ellipticity_2)
        gal = galsim.Sersic(
            n=sersic,
            half_light_radius=size,
        ).shear(ellipticity)

        sed_bins = [
            _q for _q in gal_params.keys() if re.match(rf"sed_\d+_\d+_{component}$", _q)
        ]

        sed_bins_array = np.asarray(
            [np.asarray(bin.split("_")[1:3], dtype="float") for bin in sed_bins]
        )

        sed_values_array = np.asarray([gal_params[bin] for bin in sed_bins]).ravel()

        # There are some components with no spectra. In this case,
        # skip that component
        if np.allclose(0, sed_values_array):
            continue

        sed_factory = ObservedSedFactory(
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

    return gal

