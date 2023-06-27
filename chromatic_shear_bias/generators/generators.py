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
    Adapted from https://stackoverflow.com/a/38755760
    """
    # Note that the structure of reduce requires that fs be a list of
    # _curried_ functions
    return lambda x: functools.reduce(lambda f, g: g(f), fs, x)


def match_expression(names, expressions):
    """Match names to regular expressions"""
    return [
        name
        for name in names
        for expression in expressions
        if re.match(expression, name)
    ]


def generate_batches(dataset_files, dir=None, format=None, columns=None, predicate=None):
    dataset = ds.dataset(dataset_files)
    names = dataset.schema.names

    columns = match_expression(names, columns)

    scanner = dataset.scanner(
        columns=columns,
        filter=predicate,
    )

    # for batch in scanner.to_batches():
    #     if len(batch) > 0:
    #         yield batch
    #     else:
    #         continue

    num_rows = scanner.count_rows()
    if num_rows < 1:
        raise ValueError(f"Scanner of {dataset_files} with {predicate} found 0 rows")

    # We repeat the scanner to regenerate and reuse the batches
    for scanner_rep in itertools.repeat(scanner):
        for batch in scanner_rep.to_batches():
            if len(batch) > 0:
                yield batch
            else:
                continue


def generate_rows(batch, n_sample=1, seed=None):
    size = len(batch)
    rng = np.random.default_rng(seed=seed)

    for i in range(n_sample):
        index = rng.integers(low=0, high=size)

        row = batch.take([index]).to_pydict()
        yield row


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
        .shift(
            rng.uniform(-0.5, 0.5) * pixel_scale,
            rng.uniform(-0.5, 0.5) * pixel_scale,
        )
        .shift(x * pixel_scale, y * pixel_scale)
        for (x, y) in zip(x_lattice, y_lattice)
    ]

    return objects


def generate_scenes(n_sims, gals, xsize, ysize, pixel_scale, seed, mag=None):
    rng = np.random.default_rng(seed)
    for i in range(n_sims):
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
        if len(x_lattice) < 1:
            raise ValueError(f"Scene containts no objects!")

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

        if mag is not None:
            bp_r = galsim.Bandpass(f"LSST_r.dat", wave_type="nm").withZeropoint("AB")
            objects = [
                obj.withMagnitude(mag, bp_r)
                for obj in objects
            ]

        yield objects


def build_image(band, observed, observed_psf, noise_sigma, xsize, ysize, psf_size, pixel_scale, scene_seed, image_seed, n_coadd=1):
    scene_rng = np.random.default_rng(scene_seed)
    scene_grng = galsim.BaseDeviate(scene_seed)
    image_rng = np.random.default_rng(image_seed)
    image_grng = galsim.BaseDeviate(image_seed)

    image = galsim.Image(
        xsize,
        ysize,
        scale=pixel_scale,
    )
    bandpass = galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm").withZeropoint("AB") if band else None
    for obs in observed:
        obs.drawImage(
            image=image,
            add_to_image=True,
            bandpass=bandpass,
        )
        # obs.drawImage(
        #     image=image,
        #     exptime=30 * n_coadd,
        #     area=319 / 9.6 * 1e4,
        #     add_to_image=True,
        #     bandpass=bandpass,
        # )
        # stamp = obs.drawImage(
        #     exptime=30 * n_coadd,
        #     area=319 / 9.6 * 1e4,
        #     bandpass=bandpass,
        # )  # Need to sort out centering...
        # b = stamp.bounds & image.bounds
        # if b.isDefined():
        #     image[b] += stamp[b]

    noise = galsim.GaussianNoise(scene_grng, sigma=noise_sigma)
    image.addNoise(noise)

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
    # counterfactual_noise = galsim.GaussianNoise(image_grng, sigma=noise_sigma)
    # noise_image.addNoise(counterfactual_noise)
    noise_image.addNoise(noise)

    ormask = np.full((xsize, ysize), int(0))
    bmask = np.full((xsize, ysize), int(0))
    weight = np.full((xsize, ysize), 1 / noise_sigma ** 2)

    return image, psf_image, noise_image, ormask, bmask, weight


def generate_pairs(scenes, star, shear, psf, bands, xsize, ysize, psf_size, pixel_scale, rng):

    for scene in scenes:
        _shear_seed = rng.integers(1, 2**64 // 2 - 1)

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
                image, psf_image, noise_image, ormask, bmask, weight = build_image(
                    band,
                    observed,
                    observed_psf,
                    xsize,
                    ysize,
                    psf_size,
                    pixel_scale,
                    shear_seed,
                    image_seed
                )
                wcs = galsim.AffineTransform(
                    pixel_scale,
                    0.0,
                    0.0,
                    pixel_scale,
                    origin=image.center,
                )
                im_jac = ngmix.jacobian.Jacobian(
                    x=(image.array.shape[1] - 1) / 2,
                    y=(image.array.shape[0] - 1) / 2,
                    dudx=wcs.dudx,
                    dudy=wcs.dudy,
                    dvdx=wcs.dvdx,
                    dvdy=wcs.dvdy,
                )
                psf_jac = ngmix.jacobian.Jacobian(
                    x=(psf_image.array.shape[1] - 1) / 2,
                    y=(psf_image.array.shape[0] - 1) / 2,
                    dudx=wcs.dudx,
                    dudy=wcs.dudy,
                    dvdx=wcs.dvdx,
                    dvdy=wcs.dvdy,
                )
                psf_obs = ngmix.Observation(
                    psf_image.array,
                    jacobian=psf_jac
                )
                obs = ngmix.Observation(
                    image.array,
                    psf=psf_obs,
                    noise=noise_image.array,
                    weight=weight,
                    ormask=ormask,
                    bmask=bmask,
                    jacobian=im_jac,
                )
                obslist = ngmix.ObsList()
                obslist.append(obs)
                mbobs.append(obslist)

            pair[shear_type] = mbobs

        yield pair


def build_pair(scene, star, shear, psf, bands, noises, xsize, ysize, psf_size, pixel_scale, seed, n_coadd=1):

    rng = np.random.default_rng(seed)
    _scene_seed = rng.integers(1, 2**64 // 2 - 1)
    _image_seed = rng.integers(1, 2**64 // 2 - 1)

    observed_psf = galsim.Convolve([star, psf])

    pair = {}
    for shear_type in ["plus", "minus"]:
        if shear_type == "plus":
            g = shear
        elif shear_type == "minus":
            g = -shear
        sheared_objects = [obj.shear(g1=g, g2=0.00) for obj in scene]

        observed = [
            galsim.Convolve([sheared_obj, psf]) for sheared_obj in sheared_objects
        ]

        scene_rng = np.random.default_rng(_scene_seed)
        image_rng = np.random.default_rng(_image_seed)

        mbobs = ngmix.MultiBandObsList()
        for band, noise_sigma in zip(bands, noises):
            scene_seed = scene_rng.integers(1, 2**64 // 2 - 1)
            image_seed = image_rng.integers(1, 2**64 // 2 - 1)
            image, psf_image, noise_image, ormask, bmask, weight = build_image(
                band,
                observed,
                observed_psf,
                noise_sigma,
                xsize,
                ysize,
                psf_size,
                pixel_scale,
                scene_seed,
                image_seed,
                n_coadd=n_coadd,
            )
            wcs = galsim.AffineTransform(
                pixel_scale,
                0.0,
                0.0,
                pixel_scale,
                origin=image.center,
            )
            im_jac = ngmix.jacobian.Jacobian(
                x=(image.array.shape[1] - 1) / 2,
                y=(image.array.shape[0] - 1) / 2,
                dudx=wcs.dudx,
                dudy=wcs.dudy,
                dvdx=wcs.dvdx,
                dvdy=wcs.dvdy,
            )
            psf_jac = ngmix.jacobian.Jacobian(
                x=(psf_image.array.shape[1] - 1) / 2,
                y=(psf_image.array.shape[0] - 1) / 2,
                dudx=wcs.dudx,
                dudy=wcs.dudy,
                dvdx=wcs.dvdx,
                dvdy=wcs.dvdy,
            )
            psf_obs = ngmix.Observation(
                psf_image.array,
                jacobian=psf_jac
            )
            obs = ngmix.Observation(
                image.array,
                psf=psf_obs,
                noise=noise_image.array,
                weight=weight,
                ormask=ormask,
                bmask=bmask,
                jacobian=im_jac,
            )
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)

        pair[shear_type] = mbobs

    return pair

# def build_pair_color(scene, stars, shear, psf, bands, xsize, ysize, psf_size, pixel_scale, seed):
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


def measure_pair(pair, shear_bands, det_bands, config, seed):
    rng = np.random.default_rng(seed)
    mdet_seed = rng.integers(1, 2**64 // 2 - 1)
    mdet_rng_p = np.random.default_rng(mdet_seed)
    mdet_rng_m = np.random.default_rng(mdet_seed)
    mbobs_p = pair["plus"]
    mbobs_m = pair["minus"]

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng_m,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    measurement = run_utils.measure_pairs(config, res_p, res_m)

    return measurement


def measure_pair_color(pair, psf, colors, stars, psf_size, pixel_scale, bands, shear_bands, det_bands, config, rng):
    # given a pair of mbobs with a psf drawn at the median g-i color,
    # create color_dep_mbobs at each of the provided stars
    # and run color_dep metadetect
    mdet_seed = rng.integers(1, 2**64 // 2 - 1)
    mdet_rng_p = np.random.default_rng(mdet_seed)
    mdet_rng_m = np.random.default_rng(mdet_seed)
    mbobs_p = pair["plus"]
    mbobs_m = pair["minus"]

    bps = {
        band.lower(): galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm").withZeropoint("AB") for band in bands
    }

    def color_key_func(fluxes):
        # g - i < 0 --> blue (higher flux in bluer band)
        # g - i > 0 --> red (lower flux in bluer band)
        if np.any(~np.isfinite(fluxes)):
            return None

        if fluxes[0] < 0 or fluxes[2] < 0:
            if fluxes[0] < fluxes[2]:
                # red end
                return len(stars) - 1
            else:
                # blue end
                return 0
        else:
            # mag0 = 28.38 - np.log10(fluxes[0])/0.4
            # mag1 = 27.85 - np.log10(fluxes[1])/0.4
            # color = mag0 - mag1
            AB_mags = sed_tools.AB_mag(bps)
            mag0 = AB_mags(fluxes[0], bands[0])
            mag1 = AB_mags(fluxes[2], bands[2])
            color = mag0 - mag1
            # color += np.random.default_rng().uniform(-0.5, 0.5)  # perturb color to induce noisiness?
            color_key = int(np.argmin(np.abs(color - colors)))
            # print(f"mag_g: {mag0} [mdet]")
            # print(f"mag_i: {mag1} [mdet]")
            # print(f"color: {color} [mdet]")
            # print(f"index: {color_key}\v")

            return color_key

    color_dep_mbobs_p = {}
    color_dep_mbobs_m = {}
    for c, star in enumerate(stars):
        _mbobs_p = copy.deepcopy(mbobs_p)
        _mbobs_m = copy.deepcopy(mbobs_m)
        observed_psf = galsim.Convolve([star, psf])
        # get size, etc. from obs
        psf_image = galsim.Image(
            psf_size,
            psf_size,
            scale=pixel_scale,
        )

        for i, (_obslist_p, _obslist_m) in enumerate(zip(_mbobs_p, _mbobs_m)):
            band = bands[i]
            observed_psf.drawImage(image=psf_image, bandpass=bps[band])
            for _obs_p, _obs_m in zip(_obslist_p, _obslist_m):
                _obs_p.psf.set_image(psf_image.array)
                _obs_m.psf.set_image(psf_image.array)

        color_dep_mbobs_p[c] = _mbobs_p
        color_dep_mbobs_m[c] = _mbobs_m

    # for c in range(len(stars)):
    #     for b in range(len(color_dep_mbobs_p[c])):
    #         assert np.all(np.equal(color_dep_mbobs_p[c][b][0].psf.image, color_dep_mbobs_m[c][b][0].psf.image))

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=color_key_func,
        color_dep_mbobs=color_dep_mbobs_p,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng_m,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=color_key_func,
        color_dep_mbobs=color_dep_mbobs_m,
    )

    measurement = run_utils.measure_pairs(config, res_p, res_m)

    return measurement


def generate_measurements(pairs, config, rng):
    mdet_seed = rng.integers(1, 2**64 // 2 - 1)
    mdet_rng_p = np.random.default_rng(mdet_seed)
    mdet_rng_m = np.random.default_rng(mdet_seed)

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

def build_plot(pair, bands, detect, config):
    if detect:
        import metadetect
        mdet_rng_p = np.random.default_rng(42)
        mdet_rng_m = np.random.default_rng(42)

        res_p = metadetect.do_metadetect(
            config["metadetect"],
            pair["plus"],
            mdet_rng_p,
        )

        res_m = metadetect.do_metadetect(
            config["metadetect"],
            pair["minus"],
            mdet_rng_m,
        )

        model = config["metadetect"]["model"]
        if model == "wmom":
            tcut = 1.2
        else:
            tcut = 0.5

        s2n_cut = 10
        t_ratio_cut = tcut
        mfrac_cut = 10
        ormask_cut = None

        def _mask(data):
            if "flags" in data.dtype.names:
                flag_col = "flags"
            else:
                flag_col = model + "_flags"

            _cut_msk = (
                (data[flag_col] == 0)
                & (data[model + "_s2n"] > s2n_cut)
                & (data[model + "_T_ratio"] > t_ratio_cut)
            )
            if ormask_cut:
                _cut_msk = _cut_msk & (data["ormask"] == 0)
            if mfrac_cut is not None:
                _cut_msk = _cut_msk & (data["mfrac"] <= mfrac_cut)
            return _cut_msk

        o_p = res_p["noshear"]
        q_p = _mask(o_p)
        o_m = res_m["noshear"]
        q_m = _mask(o_m)
        p_ns = o_p[q_p]
        m_ns = o_m[q_m]

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(bands), 5)
    if len(bands) > 1:
        axs[0, 0].set_title("image +")
        axs[0, 1].set_title("image -")
        axs[0, 2].set_title("psf")
        axs[0, 3].set_title("noise +")
        axs[0, 4].set_title("noise -")
        for i, band in enumerate(bands):
            axs[i, 0].set_ylabel(f"{band}")
            axs[i, 0].imshow(pair["plus"][i][0].image, origin="lower")
            axs[i, 1].imshow(pair["minus"][i][0].image, origin="lower")
            axs[i, 2].imshow(pair["plus"][i][0].psf.image, origin="lower")
            axs[i, 3].imshow(pair["plus"][i][0].noise, origin="lower")
            axs[i, 4].imshow(pair["minus"][i][0].noise, origin="lower")
            if detect:
                # axs[i, 0].scatter(p_ns["sx_col"], p_ns["sx_row"], c="r", marker="x")
                # axs[i, 1].scatter(m_ns["sx_col"], m_ns["sx_row"], c="r", marker="x")
                for j in range(len(p_ns)):
                    # axs[i, 0].annotate(round(p_ns["wmom_s2n"][j]), (p_ns["sx_col"][j], p_ns["sx_row"][j]), c="r")
                    axs[i, 0].text(p_ns["sx_col"][j], p_ns["sx_row"][j], round(p_ns["wmom_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
                for j in range(len(m_ns)):
                    # axs[i, 1].annotate(round(m_ns["wmom_s2n"][j]), (m_ns["sx_col"][j], m_ns["sx_row"][j]), c="r")
                    axs[i, 1].text(m_ns["sx_col"][j], m_ns["sx_row"][j], round(m_ns["wmom_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
    else:
        axs[0].set_title("image +")
        axs[1].set_title("image -")
        axs[2].set_title("psf")
        axs[3].set_title("noise +")
        axs[4].set_title("noise -")
        axs[0].set_ylabel(f"{bands}")
        axs[0].imshow(pair["plus"][0][0].image, origin="lower")
        axs[1].imshow(pair["minus"][0][0].image, origin="lower")
        axs[2].imshow(pair["plus"][0][0].psf.image, origin="lower")
        axs[3].imshow(pair["plus"][0][0].noise, origin="lower")
        axs[4].imshow(pair["minus"][0][0].noise, origin="lower")
        if detect:
            axs[0].scatter(p_ns["sx_col"], p_ns["sx_row"], c="r", marker="x")
            axs[1].scatter(m_ns["sx_col"], m_ns["sx_row"], c="r", marker="x")
            for j in range(len(p_ns)):
                # axs[0].annotate(round(p_ns["wmom_s2n"][j]), (p_ns["sx_col"][j], p_ns["sx_row"][j]), c="r")
                axs[0].text(p_ns["sx_col"][j], p_ns["sx_row"][j], round(p_ns["wmom_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")
            for j in range(len(m_ns)):
                # axs[1].annotate(round(m_ns["wmom_s2n"][j]), (m_ns["sx_col"][j], m_ns["sx_row"][j]), c="r")
                axs[1].text(m_ns["sx_col"][j], m_ns["sx_row"][j], round(m_ns["wmom_s2n"][j]), c="r", horizontalalignment="left", verticalalignment="bottom")

    for ax in axs.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    return fig
