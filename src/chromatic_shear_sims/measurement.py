import copy
import logging
import os

import yaml
import metadetect
import numpy as np
import pyarrow as pa

from chromatic_shear_sims import utils
from chromatic_shear_sims import observations


logger = logging.getLogger(__name__)


def get_measure(module_name, class_name, **kwargs):
    return utils.get_instance(module_name, class_name, **kwargs)


class Measure:
    def __init__(self):
        self.name = None

    def get_schema(self):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError


class Metadetect(Measure):
    schema = pa.schema([
        ("pgauss_flags", pa.int64()),
        ("pgauss_psf_flags", pa.int64()),
        ("pgauss_psf_g", pa.list_(pa.float64())),
        ("pgauss_psf_T", pa.float64()),
        ("pgauss_obj_flags", pa.int64()),
        ("pgauss_s2n", pa.float64()),
        ("pgauss_g", pa.list_(pa.float64())),
        ("pgauss_g_cov", pa.list_(pa.list_(pa.float64()))),
        ("pgauss_T", pa.float64()),
        ("pgauss_T_flags", pa.int64()),
        ("pgauss_T_err", pa.float64()),
        ("pgauss_T_ratio", pa.float64()),
        ("pgauss_band_flux_flags", pa.list_(pa.int64())),
        ("pgauss_band_flux", pa.list_(pa.float64())),
        ("pgauss_band_flux_err", pa.list_(pa.float64())),
        ("shear_bands", pa.string()),
        ("sx_row", pa.float64()),
        ("sx_col", pa.float64()),
        ("sx_row_noshear", pa.float64()),
        ("sx_col_noshear", pa.float64()),
        ("ormask", pa.int64()),
        ("mfrac", pa.float64()),
        ("bmask", pa.int64()),
        ("mfrac_img", pa.float64()),
        ("ormask_noshear", pa.int64()),
        ("mfrac_noshear", pa.float64()),
        ("bmask_noshear", pa.int64()),
        ("det_bands", pa.string()),
        ("psfrec_flags", pa.int64()),
        ("psfrec_g", pa.list_(pa.float64())),
        ("psfrec_T", pa.float64()),
        ("mdet_step", pa.string()),
        ("shear", pa.string()),
        ("seed", pa.int64()),
    ])

    def __init__(self, config):
        self.name = "metadetect"
        self.config = config

    def get_schema(self):
        return self.schema

    def run(self, obs, psf_obs, *, seed=None, **kwargs):
        obs = observations.with_psf_obs(obs, psf_obs)
        rng = np.random.default_rng(seed)
        measurement = metadetect.do_metadetect(
            self.config,
            obs,
            rng,
            **kwargs,
        )

        return measurement
