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
from chromatic_shear_bias.generators import generators


# From
# https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/catalog_configs/cosmoDC2_v1.1.4_parquet.yaml
COSMOLOGY = {
    "H0": 71.0,
    "Om0": 0.2648,
    "Ob0": 0.0448,
    "sigma8": 0.8,
    "n_s": 0.963,
}


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
    redshift_hubble = gal_params.get("redshift_true", [0])[0]
    redshift = gal_params.get("redshift", [0])[0]

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

    sed_factory = sed_tools.ObservedSedFactory(
        sed_bins_array,
        cosmology,
    )

    sed = sed_factory.create(
        sed_values_array,
        redshift_hubble,
        redshift,
    )

    # gal = (gal * sed).withFluxDensity(1e9, 500)
    gal = gal * sed

    # # # # FIXME remove later
    # mag_g_true = gal_params.get("mag_true_g_lsst")[0]
    # mag_i_true = gal_params.get("mag_true_i_lsst")[0]
    # color_true = mag_g_true - mag_i_true
    # print(f"mag_g: {mag_g_true} [true]")
    # print(f"mag_i: {mag_i_true} [true]")
    # print(f"color: {color_true} [true]\v")

    # bp_g = galsim.Bandpass(f"LSST_g.dat", wave_type="nm").withZeropoint("AB")
    # bp_i = galsim.Bandpass(f"LSST_i.dat", wave_type="nm").withZeropoint("AB")
    # # # bp_g = galsim.Bandpass(f"LSST_g.dat", wave_type="nm", zeropoint=28.38)
    # # # bp_i = galsim.Bandpass(f"LSST_i.dat", wave_type="nm", zeropoint=27.85)


    # mag_g = gal.calculateMagnitude(bp_g)
    # mag_i = gal.calculateMagnitude(bp_i)
    # color = mag_g - mag_i
    # print(f"mag_g: {mag_g} [sim]")
    # print(f"mag_i: {mag_i} [sim]")
    # print(f"color: {color} [sim]\v")

    # bp_r = galsim.Bandpass(f"LSST_r.dat", wave_type="nm").withZeropoint("AB")
    # gal = gal.withMagnitude(19, bp_r)

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

    return gal


def simple_gal_generator():
    gal = galsim.Exponential(half_light_radius=0.5)
    sed = galsim.SED("1", wave_type="ang", flux_type="fphotons")
    while True:
        yield gal * sed

def elliptical_cosmoDC2_generator(predicate=None, seed=None):
    dataset = "/oak/stanford/orgs/kipac/users/smau/cosmoDC2_v1.1.4_parquet"
    columns = [
        "^galaxy_id$",
        "^size_true$",
        "^ellipticity_\d_true$",
    ]
    batch_generator = generators.generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generators.generate_rows(batch, n_sample=batch.num_rows, seed=seed)
        for row in row_generator:
            built = build_cosmoDC2_ellipse(row)
            yield built


def simple_cosmoDC2_generator(predicate=None, seed=None):
    dataset = "/oak/stanford/orgs/kipac/users/smau/cosmoDC2_v1.1.4_parquet"
    columns = [
        "^galaxy_id$",
        "^redshift_true$",
        "^redshift$",
        "^mag_true_\w_lsst$",
        "^mag_\w_lsst$",
        "^sed_\d+_\d+$",
    ]
    batch_generator = generators.generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generators.generate_rows(batch, n_sample=batch.num_rows, seed=seed)
        for row in row_generator:
            built = build_simple_cosmoDC2_gal(row)
            yield built


def cosmoDC2_generator(predicate=None, seed=None):
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
    batch_generator = generators.generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generators.generate_rows(batch, n_sample=batch.num_rows, seed=seed)
        for row in row_generator:
            built = build_cosmoDC2_gal(row)
            yield built
