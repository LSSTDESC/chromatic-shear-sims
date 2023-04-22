"""
"""

import argparse
import copy
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


def compose(*fs):
    """Functional composition of generator functions
    From https://stackoverflow.com/a/38755760
    """
    return lambda x: functools.reduce(lambda f, g: g(f), fs, x)


def match_expression(names, expressions):
    """Match names to regular expressions"""
    return [
        name
        for name in names
        for expression in expressions
        if re.match(expression, name)
    ]


def generate_batches(dataset, dir=None, format=None, columns=None, predicate=None):
    print(f"generating batches for {dataset}")
    dataset = ds.dataset(dataset)
    names = dataset.schema.names

    columns = match_expression(names, columns)

    scanner = dataset.scanner(
        columns=columns,
        filter=predicate,
    )
    batches = scanner.to_batches()

    for batch in batches:
        if len(batch) > 0:
            yield batch
        else:
            # TODO check if we can do a recursive call here to regenerate
            # our batches. Otherwise, what happens when we execute this branch?
            continue


def generate_rows(batch, n_sample=1):
    size = len(batch)
    rng = np.random.default_rng()

    for i in range(n_sample):
        index = rng.integers(low=0, high=size)

        row = batch.take([index]).to_pydict()
        yield row


# def build_cosmoDC2_gal_circle(gal_params):
#     # TODO validate this is correct for cosmoDC2 and include reference
#     cosmology = {
#         "Om0": 0.2648,
#         "Ob0": 0.0448,
#         "H0": 71.0,
#         "sigma8": 0.8,
#         "n_s": 0.963,
#     }
# 
#     redshift_hubble = gal_params.get("redshift_true", [0])[0]
#     redshift = gal_params.get("redshift", [0])[0]
# 
#     sersic = gal_params.get(f"sersic")[0]
#     size = gal_params.get(f"size_true")[0]
# 
#     gal = galsim.Sersic(
#         n=sersic,
#         half_light_radius=size,
#     )
# 
#     sed_bins = [
#         _q for _q in gal_params.keys() if re.match(rf"sed_\d+_\d+$", _q)
#     ]
# 
#     sed_bins_array = np.asarray(
#         [np.asarray(bin.split("_")[1:3], dtype="float") for bin in sed_bins]
#     )
# 
#     sed_values_array = np.asarray([gal_params[bin] for bin in sed_bins]).ravel()
# 
#     sed_factory = sed_tools.ObservedSedFactory(
#         sed_bins_array,
#         cosmology,
#     )
# 
#     sed = sed_factory.create(
#         sed_values_array,
#         redshift_hubble,
#         redshift,
#     )
# 
#     gal = (gal * sed).withFluxDensity(1e9, 500)
# 
#     return gal


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


@functools.cache
def read_sed_file(file_name, wave_type, flux_type):
    return galsim.sed.SED(file_name, wave_type, flux_type)


def build_star(star_params, sed_dir):
    _standard_dict = {
        "lte*": "starSED/phoSimMLT",
        "bergeron*": "starSED/wDs",
        "k[mp]*": "starSED/kurucz",
    }
    wave_type = "Nm"
    flux_type = "flambda"
    sed_filename = star_params.get("sedFilename")[0].strip()
    if not sed_filename.endswith(".gz"):
        # Some files are missing ".gz" in their suffix; if this is the case,
        # append to the current suffix
        sed_filename += ".gz"
    path_name = Path(sed_filename)
    for k, v in _standard_dict.items():
        matched = False
        if path_name.match(k):
            sed_path = Path(sed_dir) / v / path_name
            matched = True
            break  # we should only have one match
    if not matched:
        raise ValueError(
            f"Filename {sed_filename} does not match any known patterns in {sed_dir}"
        )
    if not sed_path.exists():
        raise ValueError(f"Filename {sed_filename} not found in {sed_dir}")

    sed_file = sed_path.as_posix()
    sed = read_sed_file(sed_file, wave_type, flux_type)
    sed = sed.withFluxDensity(1, wavelength=600)

    # print(f"\tBuilding star took {end - start} s")

    return sed


# def cosmoDC2_circle_generator(predicate=None):
#     dataset = "/oak/stanford/orgs/kipac/users/smau/cosmoDC2_v1.1.4_parquet"
#     columns = [
#         "^galaxy_id$",
#         "^sersic$",
#         "^size_true$",
#         "^redshift_true$",
#         "^redshift$",
#         "^mag_true_\w_lsst$",
#         "^sed_\d+_\d+$",
#     ]
#     batch_generator = generate_batches(dataset, columns=columns, predicate=predicate)
#     for batch in batch_generator:
#         row_generator = generate_rows(batch, n_sample=batch.num_rows)
#         for row in row_generator:
#             built = build_cosmoDC2_gal_circle(row)
#             yield built


# # FIXME test this
# def generate_rings(gals, num):
#     for gal in gals:
#         for i in range(num)
#             if i == 0:
#                 yield gal
#             else:
#                 if i % num == 0:
#                     continue
#                 else:
#                     yield gal.rotate(i)


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
        row_generator = generate_rows(batch, n_sample=batch.num_rows)
        for row in row_generator:
            built = build_cosmoDC2_gal(row)
            yield built


def DC2_generator(predicate=None):
    dataset = "/oak/stanford/orgs/kipac/users/smau/dc2_stellar_healpixel.parquet"
    columns = [
        "^sedFilename$",
    ]
    sed_dir = "/oak/stanford/orgs/kipac/users/smau/"
    batch_generator = generate_batches(dataset, columns=columns, predicate=predicate)
    for batch in batch_generator:
        row_generator = generate_rows(batch, n_sample=batch.num_rows)
        for row in row_generator:
            built = build_star(row, sed_dir)
            yield built


def build_lattice(full_xsize, full_ysize, sep, scale, v1, v2, rot=None, border=0):
    """
    Build a lattice from primitive translation vectors.
    Method adapted from https://stackoverflow.com/a/6145068 and
    https://github.com/alexkaz2/hexalattice/blob/master/hexalattice/hexalattice.py
    """
    # ensure that the lattice vectors are normalized
    v1 /= np.sqrt(v1.dot(v1))
    v2 /= np.sqrt(v2.dot(v2))

    # first, create a square lattice that covers the full image
    # scale: arcsec / pixel
    # sep: arcsec
    xs = np.arange(-full_xsize // 2, full_xsize // 2 + 1) * sep / scale
    ys = np.arange(-full_ysize // 2, full_ysize // 2 + 1) * sep / scale
    x_square, y_square = np.meshgrid(xs, ys)

    # apply the lattice vectors to the lattice
    x_lattice = v1[0] * x_square + v2[0] * y_square
    y_lattice = v1[1] * x_square + v2[1] * y_square

    # construct the rotation matrix and rotate the lattice points
    rotation = np.asarray(
        [
            [np.cos(np.radians(rot)), -np.sin(np.radians(rot))],
            [np.sin(np.radians(rot)), np.cos(np.radians(rot))],
        ]
    )
    xy_lattice_rot = (
        np.stack(
            [x_lattice.reshape(-1), y_lattice.reshape(-1)],
            axis=-1,
        )
        @ rotation.T
    )
    x_lattice_rot, y_lattice_rot = np.split(xy_lattice_rot, 2, axis=1)

    # remove points outside of the full image
    bounds_x = (-(full_xsize - 1) // 2, (full_xsize - 1) // 2)
    bounds_y = (-(full_ysize - 1) // 2, (full_ysize - 1) // 2)

    # remove points according to the border
    mask = (
        (x_lattice_rot > bounds_x[0] + border)
        & (x_lattice_rot < bounds_x[1] - border)
        & (y_lattice_rot > bounds_y[0] + border)
        & (y_lattice_rot < bounds_y[1] - border)
    )

    return x_lattice_rot[mask], y_lattice_rot[mask]


def build_scene(seed, gals, xsize, ysize, pixel_scale):
    rng = np.random.default_rng(seed)

    v1 = np.asarray([1, 0], dtype=float)
    v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)
    x_lattice, y_lattice = build_lattice(
        xsize,
        ysize,
        10,
        pixel_scale,
        v1,
        v2,
        rng.uniform(0, 360),
        100,
    )  # pixels

    # persist objects as a list for reiteration
    objects = [
        next(gals)
        .rotate(rng.uniform(0, 360) * galsim.degrees)
        .shift(x * pixel_scale, y * pixel_scale)
        .shift(
            rng.uniform(-0.5, 0.5) * pixel_scale,
            rng.uniform(-0.5, 0.5) * pixel_scale,
        )
        for (x, y) in zip(x_lattice, y_lattice)
    ]

    return objects


def generate_scenes(nsims, gals, xsize, ysize, pixel_scale, seed):
    rng = np.random.default_rng(seed)

    for i in range(nsims):
        v1 = np.asarray([1, 0], dtype=float)
        v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)
        x_lattice, y_lattice = build_lattice(
            xsize,
            ysize,
            10,
            pixel_scale,
            v1,
            v2,
            rng.uniform(0, 360),
            100,
        )  # pixels

        # persist objects as a list for reiteration
        objects = [
            next(gals)
            .rotate(rng.uniform(0, 360) * galsim.degrees)
            .shift(x * pixel_scale, y * pixel_scale)
            .shift(
                rng.uniform(-0.5, 0.5) * pixel_scale,
                rng.uniform(-0.5, 0.5) * pixel_scale,
            )
            for (x, y) in zip(x_lattice, y_lattice)
        ]

        yield objects


def build_image(band, observed, observed_psf, xsize, ysize, psf_size, pixel_scale, scene_seed, image_seed):
    scene_rng = np.random.default_rng(scene_seed)
    scene_grng = galsim.BaseDeviate(scene_seed)
    image_rng = np.random.default_rng(image_seed)
    image_grng = galsim.BaseDeviate(image_seed)

    image = galsim.Image(
        xsize,
        ysize,
        scale=pixel_scale,
    )
    bandpass = galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm")
    for obs in observed:
        obs.drawImage(image=image, bandpass=bandpass, add_to_image=True)

    noise = galsim.GaussianNoise(scene_grng, sigma=0.02)
    # image.addNoise(noise)
    snr = 1e6
    image.addNoiseSNR(noise, snr)

    psf_image = galsim.Image(
        psf_size,
        psf_size,
        scale=pixel_scale,
    )
    observed_psf.drawImage(image=psf_image, bandpass=bandpass, add_to_image=True)

    noise_image = galsim.Image(
        xsize,
        ysize,
        scale=pixel_scale,
    )
    counterfactual_noise = galsim.GaussianNoise(image_grng, sigma=0.02)
    noise_image.addNoise(counterfactual_noise)

    return image, psf_image, noise_image


def generate_pairs(scenes, shear, stars, psf, bands, xsize, ysize, psf_size, pixel_scale, seed):

    ormask = np.full((xsize, ysize), int(0))
    bmask = np.full((xsize, ysize), int(0))

    for scene in scenes:
        rng = np.random.default_rng(seed)

        _shear_seed = rng.integers(1, 2**64 // 2 - 1)

        star = galsim.DeltaFunction() * next(stars)
        observed_psf = galsim.Convolve([star, psf])

        # TODO split across plus/minus version here?
        pair = {}
        for shear_type in ["plus", "minus"]:
            shear_rng = np.random.default_rng(_shear_seed)
            if shear_type == "plus":
                g = shear
            elif shear_type == "minus":
                g = -shear
            sheared_objects = [obj.shear(g1=g, g2=0.00) for obj in scene]

            observed = [
                galsim.Convolve([sheared_obj, psf]) for sheared_obj in sheared_objects
            ]

            _image_seed = rng.integers(1, 2**64 // 2 - 1)
            image_rng = np.random.default_rng(_image_seed)

            mbobs = ngmix.MultiBandObsList()
            for band in bands:
                shear_seed = shear_rng.integers(1, 2**64 // 2 - 1)
                image_seed = image_rng.integers(1, 2**64 // 2 - 1)
                image, psf_image, noise_image = build_image(band, observed, observed_psf, xsize, ysize, psf_size, pixel_scale, shear_seed, image_seed)
                psf_obs = ngmix.Observation(psf_image.array)
                obs = ngmix.Observation(
                    image.array,
                    psf=psf_obs,
                    noise=noise_image.array,
                    ormask=ormask,
                    bmask=bmask,
                )
                obslist = ngmix.ObsList()
                obslist.append(obs)
                mbobs.append(obslist)

            pair[shear_type] = mbobs

        seed += 1

        yield pair


def build_pair(scene, star, shear, predicate, psf, bands, xsize, ysize, psf_size, pixel_scale, seed):

    ormask = np.full((xsize, ysize), int(0))
    bmask = np.full((xsize, ysize), int(0))

    rng = np.random.default_rng(seed)

    _shear_seed = rng.integers(1, 2**64 // 2 - 1)

    star = galsim.DeltaFunction() * star
    observed_psf = galsim.Convolve([star, psf])

    # TODO split across plus/minus version here?
    pair = {}
    for shear_type in ["plus", "minus"]:
        shear_rng = np.random.default_rng(_shear_seed)
        if shear_type == "plus":
            g = shear
        elif shear_type == "minus":
            g = -shear
        sheared_objects = [obj.shear(g1=g, g2=0.00) for obj in scene]

        observed = [
            galsim.Convolve([sheared_obj, psf]) for sheared_obj in sheared_objects
        ]

        _image_seed = rng.integers(1, 2**64 // 2 - 1)
        image_rng = np.random.default_rng(_image_seed)

        mbobs = ngmix.MultiBandObsList()
        for band in bands:
            shear_seed = shear_rng.integers(1, 2**64 // 2 - 1)
            image_seed = image_rng.integers(1, 2**64 // 2 - 1)
            image, psf_image, noise_image = build_image(band, observed, observed_psf, xsize, ysize, psf_size, pixel_scale, shear_seed, image_seed)
            psf_obs = ngmix.Observation(psf_image.array)
            obs = ngmix.Observation(
                image.array,
                psf=psf_obs,
                noise=noise_image.array,
                ormask=ormask,
                bmask=bmask,
            )
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        pair[shear_type] = mbobs

    seed += 1

    return pair

# def build_pair_color(scene, stars, shear, predicate, psf, bands, xsize, ysize, psf_size, pixel_scale, seed):
# 
#     ormask = np.full((xsize, ysize), int(0))
#     bmask = np.full((xsize, ysize), int(0))
# 
#     rng = np.random.default_rng(seed)
# 
#     _shear_seed = rng.integers(1, 2**64 // 2 - 1)
# 
#     chromatic_pairs = []
#     for star in stars:
#         star = galsim.DeltaFunction() * star
#         observed_psf = galsim.Convolve([star, psf])
# 
#         # TODO split across plus/minus version here?
#         pair = {}
#         for shear_type in ["plus", "minus"]:
#             shear_rng = np.random.default_rng(_shear_seed)
#             if shear_type == "plus":
#                 g = shear
#             elif shear_type == "minus":
#                 g = -shear
#             sheared_objects = [obj.shear(g1=g, g2=0.00) for obj in scene]
# 
#             observed = [
#                 galsim.Convolve([sheared_obj, psf]) for sheared_obj in sheared_objects
#             ]
# 
#             _image_seed = rng.integers(1, 2**64 // 2 - 1)
#             image_rng = np.random.default_rng(_image_seed)
# 
#             mbobs = ngmix.MultiBandObsList()
#             for band in bands:
#                 shear_seed = shear_rng.integers(1, 2**64 // 2 - 1)
#                 image_seed = image_rng.integers(1, 2**64 // 2 - 1)
#                 image, psf_image, noise_image = build_image(band, observed, observed_psf, xsize, ysize, psf_size, pixel_scale, shear_seed, image_seed)
#                 psf_obs = ngmix.Observation(psf_image.array)
#                 obs = ngmix.Observation(
#                     image.array,
#                     psf=psf_obs,
#                     noise=noise_image.array,
#                     ormask=ormask,
#                     bmask=bmask,
#                 )
#                 obslist = ngmix.ObsList()
#                 obslist.append(obs)
#                 mbobs.append(obslist)
# 
#             pair[shear_type] = mbobs
#         chromatic_pairs.append(pair)
# 
#     seed += 1
# 
# return chromatic_pairs


def measure_pair(pair, config, seed):
    mdet_rng_p = np.random.default_rng(seed)
    mdet_rng_m = np.random.default_rng(seed)
    mbobs_p = pair["plus"]
    mbobs_m = pair["minus"]

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng_p,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng_m,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    measurement = run_utils.measure_pairs(config, res_p, res_m)

    return measurement


def measure_pair_color(pair, psf, stars, psf_size, pixel_scale, bands, config, seed):
    # given a pair of mbobs with a psf drawn at the median g-i color,
    # create color_dep_mbobs at each of the provided stars
    # and run color_dep metadetect
    mdet_rng_p = np.random.default_rng(seed)
    mdet_rng_m = np.random.default_rng(seed)
    mbobs_p = pair["plus"]
    mbobs_m = pair["minus"]

    def color_key_func(fluxes):
        if np.any(~np.isfinite(fluxes)):
            return None

        if fluxes[0] > fluxes[1]:
            return 1
        else:
            return 0

    color_dep_mbobs_p = {}
    color_dep_mbobs_m = {}
    for c, star in enumerate(stars):
        _mbobs_p = copy.deepcopy(mbobs_p)
        _mbobs_m = copy.deepcopy(mbobs_m)
        star = galsim.DeltaFunction() * star
        observed_psf = galsim.Convolve([star, psf])
        # get size, etc. from obs
        psf_image = galsim.Image(
            psf_size,
            psf_size,
            scale=pixel_scale,
        )

        for i, (_obslist_p, _obslist_m) in enumerate(zip(_mbobs_p, _mbobs_m)):
            band = bands[i]
            bandpass = galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm")
            observed_psf.drawImage(image=psf_image, bandpass=bandpass)
            psf_obs = ngmix.Observation(psf_image.array)
            for _obs_p, _obs_m in zip(_obslist_p, _obslist_m):
                _obs_p.set_psf(psf_obs)
                _obs_m.set_psf(psf_obs)

        color_dep_mbobs_p[c] = _mbobs_p
        color_dep_mbobs_m[c] = _mbobs_m

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng_p,
        color_key_func=color_key_func,
        color_dep_mbobs=color_dep_mbobs_p,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng_m,
        color_key_func=color_key_func,
        color_dep_mbobs=color_dep_mbobs_m,
    )

    measurement = run_utils.measure_pairs(config, res_p, res_m)

    return measurement


def generate_measurements(pairs, config, seed):
    mdet_rng_p = np.random.default_rng(seed)
    mdet_rng_m = np.random.default_rng(seed)

    for pair in pairs:
        mbobs_p = pair["plus"]
        mbobs_m = pair["minus"]

        res_p = metadetect.do_metadetect(
            config["metadetect"],
            mbobs_p,
            mdet_rng_p,
            color_key_func=None,
            color_dep_mbobs=None,
        )

        res_m = metadetect.do_metadetect(
            config["metadetect"],
            mbobs_m,
            mdet_rng_m,
            color_key_func=None,
            color_dep_mbobs=None,
        )

        measurement = run_utils.measure_pairs(config, res_p, res_m)

        yield measurement


def generate_plots(pairs, bands):
    for pair in pairs:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(len(bands), 4)
        axs[0, 0].set_title("image +")
        axs[0, 1].set_title("image -")
        axs[0, 2].set_title("psf")
        axs[0, 3].set_title("noise")
        for i, band in enumerate(bands):
            axs[i, 0].set_ylabel(f"{band}")
            axs[i, 0].imshow(pair["plus"][i][0].image, origin="lower")
            axs[i, 1].imshow(pair["minus"][i][0].image, origin="lower")
            axs[i, 2].imshow(pair["plus"][i][0].psf.image, origin="lower")
            axs[i, 3].imshow(pair["plus"][i][0].noise, origin="lower")
        for ax in axs.ravel():
            ax.set_xticks([])
            ax.set_yticks([])

        yield fig

def build_plot(pair, bands):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(bands), 4)
    if len(bands) > 1:
        axs[0, 0].set_title("image +")
        axs[0, 1].set_title("image -")
        axs[0, 2].set_title("psf")
        axs[0, 3].set_title("noise")
        for i, band in enumerate(bands):
            axs[i, 0].set_ylabel(f"{band}")
            axs[i, 0].imshow(pair["plus"][i][0].image, origin="lower")
            axs[i, 1].imshow(pair["minus"][i][0].image, origin="lower")
            axs[i, 2].imshow(pair["plus"][i][0].psf.image, origin="lower")
            axs[i, 3].imshow(pair["plus"][i][0].noise, origin="lower")
    else:
        axs[0].set_title("image +")
        axs[1].set_title("image -")
        axs[2].set_title("psf")
        axs[3].set_title("noise")
        axs[0].set_ylabel(f"{bands}")
        axs[0].imshow(pair["plus"][0][0].image, origin="lower")
        axs[1].imshow(pair["minus"][0][0].image, origin="lower")
        axs[2].imshow(pair["plus"][0][0].psf.image, origin="lower")
        axs[3].imshow(pair["plus"][0][0].noise, origin="lower")
    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    return fig
