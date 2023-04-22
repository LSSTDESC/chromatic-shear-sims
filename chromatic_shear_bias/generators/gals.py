"""
"""

import argparse
import functools
import itertools
# import operator
import os
from pathlib import Path
import re

import galsim
import joblib
import metadetect
import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from chromatic_shear_bias import sed_tools, run_utils


def build_simple_cosmoDC2_gal(gal_params):
    # TODO validate this is correct for cosmoDC2 and include reference
    cosmology = {
        "Om0": 0.2648,
        "Ob0": 0.0448,
        "H0": 71.0,
        "sigma8": 0.8,
        "n_s": 0.963,
    }

    redshift_hubble = gal_params.get("redshift_true", [0])[0]
    redshift = gal_params.get("redshift", [0])[0]

    sersic = gal_params.get(f"sersic")[0]
    size = gal_params.get(f"size_true")[0]

    gal = galsim.Sersic(
        n=sersic,
        half_light_radius=size,
    )

    sed_bins = [
        _q for _q in gal_params.keys() if re.match(rf"sed_\d+_\d+$", _q)
    ]

    sed_bins_array = np.asarray(
        [np.asarray(bin.split("_")[1:3], dtype="float") for bin in sed_bins]
    )

    sed_values_array = np.asarray([gal_params[bin] for bin in sed_bins]).ravel()

    sed_factory = sed_tools.ObservedSedFactory(
        sed_bins_array,
        cosmology,
    )

    sed = sed_factory.create(
        sed_values_array,
        redshift_hubble,
        redshift,
    )

    gal = (gal * sed).withFluxDensity(1e9, 500)

    return gal


def build_cosmoDC2_gal(gal_params):
    # TODO validate this is correct for cosmoDC2 and include reference
    cosmology = {
        "Om0": 0.2648,
        "Ob0": 0.0448,
        "H0": 71.0,
        "sigma8": 0.8,
        "n_s": 0.963,
    }

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

    gal = galsim.Add(gal_components).withFluxDensity(1e9, 500)
    # print(f"\tBuilding gal took {end - start} s")

    return gal


def simple_cosmoDC2_generator(predicate=None):
    dataset = "/oak/stanford/orgs/kipac/users/smau/cosmoDC2_v1.1.4_parquet"
    columns = [
        "^galaxy_id$",
        "^sersic$",
        "^size_true$",
        "^redshift_true$",
        "^redshift$",
        "^mag_true_\w_lsst$",
        "^sed_\d+_\d+$",
    ]
    batch_generator = generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generate_rows(batch, n_sample=batches.num_rows)
        for row in row_generator:
            built = build_simple_cosmoDC2_gal(row)
            yield built


def cosmoDC2_generator(predicate=None):
    dataset = "/oak/stanford/orgs/kipac/users/smau/cosmoDC2_v1.1.4_parquet"
    columns = [
        "^galaxy_id$",
        "^sersic_bulge$",
        "^sersic_disk$",
        "^size_bulge_true$",
        "^size_disk_true$",
        "^ellipticity_\d_bulge_true$",
        "^ellipticity_\d_disk_true$",
        "^redshift_true$",
        "^redshift$",
        "^mag_true_\w_lsst$",
        "^sed_\d+_\d+_bulge$",
        "^sed_\d+_\d+_disk$",
    ]
    batch_generator = generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generate_rows(batch, n_sample=batches.num_rows)
        for row in row_generator:
            built = build_cosmoDC2_gal(row)
            yield built
