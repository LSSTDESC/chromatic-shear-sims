"""
Adapted from Claire-Alice (@cahebert)
"""

import functools
import logging
import os
from pathlib import Path
import time

import numpy as np
import pystellibs
import galsim


logger = logging.getLogger(__name__)


PSEC_TO_CM = 3.085677581e16 * 100 ##648000 / np.pi * 1.495978707e11 * 100

ZSUN = 0.0134

def observed_from_intrinsic(sed, mu0):
    """Normalize 'intrinsic' sed from erg/s/A to erg/s/A/cm^2."""
    # need to divide by spherical area given by distance to sun:
    dl = 10**(1 + mu0/5) * PSEC_TO_CM
    return sed / (4 * np.pi * dl**2)


def apply_dust(flux, lam, av):
    dust_extinction = extinction.odonnell94(lam * 10, av, r_v=3.1, unit="aa")
    return extinction.apply(dust_extinction, flux)


def get_spectrum(logte, logg, logl, metallicity, mu0, av=None, apply_dust=False):
    logger.info(f"Generating Phoenix spectrum")
    _start_time = time.time()
    # speclib = pystellibs.BTSettl(medres=False)
    speclib = pystellibs.Phoenix()

    lam = speclib._wavelength / 10 # convert to nm from Angstrom

    intrinsic_spec = speclib.generate_stellar_spectrum(
        logte,
        logg,
        logl,
        metallicity,
    )
    observed_spec = observed_from_intrinsic(intrinsic_spec, mu0) * 10 # convert to 1/nm from 1/A
    if apply_dust:
        observed_spec = apply_dust(observed_spec, lam, "av")
    spec_lut = galsim.LookupTable(lam, observed_spec)

    spec = galsim.SED(spec_lut, "nm", "flambda")
    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"generated spectrum in {_elapsed_time} seconds")

    return spec


def model_spectrum(mass):
    _temperature = 5780 * np.power(mass, 2.1 / 4)
    _luminosity = np.power(mass, 3.5)

    logte = np.log10(_temperature)
    logl = np.log10(_luminosity)
    logg = np.log10(1/4.13E10 * mass / _luminosity) + 4 * np.log10(_temperature)  # FSPS

    metallicity = ZSUN
    mu0 = 1

    return get_spectrum(logte, logg, logl, metallicity, mu0)


class PhoenixSpectra:
    def __init__(self, survey, m_min=0.2, m_max=2, nm=1_000):
        self.survey = survey
        self._lut = None
        self.make_lut(m_min=m_min, m_max=m_max, nm=nm)

    def make_lut(self, m_min=0.2, m_max=2, nm=1_000):
        logger.info(f"making color-mass lookup table for phoenix SED")
        logger.debug(f"interpolating for masses in lin({m_min}, {m_max}, {nm})")
        bandpass_g = self.survey.bandpasses["g"]
        bandpass_i = self.survey.bandpasses["i"]

        masses = []
        colors = []

        for mass in np.linspace(m_min, m_max, nm):
            sed = model_spectrum(mass)
            color = sed.calculateMagnitude(bandpass_g) - sed.calculateMagnitude(bandpass_i)
            logger.debug(f"(mass, color) = ({mass}, {color})")

            masses.append(mass)
            colors.append(color)

        lut = galsim.LookupTable(colors, masses, x_log=False, f_log=False)

        self._lut = lut

    @property
    def lut(self):
        return self._lut

    def __call__(self, color):
        mass = self.lut(color)
        logger.debug(f"generating SED with mass {mass}")
        sed = model_spectrum(mass)
        return sed


class PhoenixStars:
    def __init__(self, config, survey=None):
        self.config = config
        self.survey = survey
        self.spectra = PhoenixSpectra(self.survey)


    def __call__(self, color):
        sed = self.spectra(color)
        return galsim.DeltaFunction() * sed

