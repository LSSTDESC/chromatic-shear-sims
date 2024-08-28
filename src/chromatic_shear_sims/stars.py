import logging
import time

import galsim
import numpy as np

from chromatic_weak_lensing import MainSequence

from chromatic_shear_sims import utils
from chromatic_shear_sims.throughputs import load_throughputs


logger = logging.getLogger(__name__)


class StarBuilder:
    def __init__(self, entrypoint, **kwargs):
        self.model = utils.get_instance(entrypoint, **kwargs)
        self.name = self.model.name

    def __call__(self, stellar_params, **kwargs):
        model_params = self.model.get_params(stellar_params)
        star = self.model.get_star(*model_params, **kwargs)
        return star


class InterpolatedStarBuilder:
    def __init__(self, entrypoint, band_1="g", band_2="i", **kwargs):
        self.model = utils.get_instance(entrypoint, **kwargs)
        self.name = self.model.name
        self.band_1 = band_1
        self.band_2 = band_2
        self.throughputs = load_throughputs(bands=[self.band_1, self.band_2])
        self.lut = self.get_lut(
            self.throughputs[self.band_1],
            self.throughputs[self.band_2],
        )
        self.x_min = self.lut.x_min
        self.x_max = self.lut.x_max

    def get_lut(self, throughput_1, throughput_2):
        """
        Get a lookup table so as to find the mass of a star
        whose spectra produces a given color.
        """
        _start_time = time.time()
        logM_min = np.min(MainSequence.logM)
        logM_max = np.max(MainSequence.logM)
        n = 1000
        masses = np.logspace(
            logM_min,
            logM_max,
            n,
        )

        colors = []
        for mass in masses:
            sparams = MainSequence.get_params(mass)
            params = self.model.get_params(sparams)
            spec = self.model.get_spectrum(*params)
            color = spec.calculateMagnitude(throughput_1) - spec.calculateMagnitude(throughput_2)
            colors.append(color)

        lut = galsim.LookupTable(colors, masses, x_log=False, f_log=True, interpolant="linear")
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        logger.info(f"made inverse lookup table in {_elapsed_time} seconds")

        return lut

    def get_spectrum(self, color, **kwargs):
        mass = self.lut(color)
        sparams = MainSequence.get_params(mass)
        params = self.model.get_params(sparams)
        spec = self.model.get_spectrum(*params, **kwargs)
        return spec

    def get_star(self, color, **kwargs):
        mass = self.lut(color)
        sparams = MainSequence.get_params(mass)
        params = self.model.get_params(sparams)
        star = self.model.get_star(*params, **kwargs)
        return star

    def __call__(self, color, **kwargs):
        star = self.get_star(color, **kwargs)
        return star

