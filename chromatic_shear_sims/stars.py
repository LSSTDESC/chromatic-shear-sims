import logging
import time

import galsim
import numpy as np

from chromatic_weak_lensing import MainSequence

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


class StarBuilder:
    def __init__(self, module_name, class_name, **kwargs):
        self.model = utils.get_instance(module_name, class_name, **kwargs)
        self.name = self.model.name

    def __call__(self, stellar_params, **kwargs):
        model_params = self.model.get_params(stellar_params)
        star = self.model.get_star(*model_params, **kwargs)
        return star


class InterpolatedStars:
    def __init__(self, module_name, class_name, throughput_1, throughput_2):
        self.model = utils.get_instance(module_name, class_name)
        self.name = self.model.name
        self.lut = self.get_lut(throughput_1, throughput_2)
        self.x_min = self.lut.x_min
        self.x_max = self.lut.x_max

    def get_lut(self, throughput_1, throughput_2, m_min=0.1, m_max=10.0, n=1_000):
        """
        Get a lookup table so as to find the mass of a star
        whose spectra produces a given color.
        """
        _start_time = time.time()
        masses = np.geomspace(m_min, m_max, n)

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

    def get_spec(self, color):
        mass = self.lut(color)
        sparams = MainSequence.get_params(mass)
        params = self.model.get_params(sparams)
        spec = self.model.get_spectrum(*params)
        return spec

    def get_star(self, color):
        mass = self.lut(color)
        sparams = MainSequence.get_params(mass)
        params = self.model.get_params(sparams)
        star = self.model.get_star(*params)
        return star

    def __call__(self, color):
        star = self.get_star(color)
        return star


class StarData:
    def __init__(self, module_name, class_name, data):
        self.module_name = module_name
        self.class_name = class_name
        self._data = utils.get_instance(module_name, class_name, data)

    @property
    def data(self):
        return self._data

    @property
    def num_rows(self):
        return self.data.num_rows

    def __call__(self, i, **kwargs):
        return self.data.get_params(i, **kwargs)

