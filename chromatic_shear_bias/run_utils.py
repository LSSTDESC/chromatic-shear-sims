"""
Utilities for running simulations
Some functions from https://github.com/beckermr/pizza-cutter-sims
"""

import copy
import re

import numpy as np
import pyarrow as pa
import yaml

import galsim
import ngmix
import metadetect

from chromatic_shear_bias import lsst


ORMASK_CUTS = [True, False]
S2N_CUTS = [7, 8, 9, 10, 15, 20]
MFRAC_CUTS = [0, 1, 2, 5, 8, 10, 20, 50, 80, 100]


class AB_mag:
    """
    Convert flux to AB magnitude for a set of bandpasses.
    From skycatalogs sed_tools
    """
    def __init__(self, bps):
        ab_sed = galsim.SED(lambda nu : 10**(8.90/2.5 - 23), wave_type='nm',
                            flux_type='fnu')
        self.ab_fluxes = {band: ab_sed.calculateFlux(bp) for
                          band, bp in bps.items()}
    def __call__(self, flux, band):
        return -2.5*np.log10(flux/self.ab_fluxes[band])


def get_sky_rms(exposure_time, zeropoint, sky_brightness, pixel_scale):
    # convert sky brightness in mag/arcsec^2 to flux
    # then add in pixel-scale
    sky_level = exposure_time * 10 ** (-0.4 * (sky_brightness - zeropoint))
    sky_level_pixel = sky_level * pixel_scale**2
    return sky_level_pixel**(1/2)


def _get_dtype():
    fkeys = ["g1p", "g1m", "g1", "g2p", "g2m", "g2"]
    ikeys = ["s2n_cut", "ormask_cut", "mfrac_cut"]
    dtype = []
    for parity in ["p", "m"]:
        for key in fkeys:
            dtype.append((f"{parity}.{key}", "f8"))
    for key in ikeys:
        dtype.append((key, "i4"))

    dtype.append(("weight", "f8"))

    return dtype

def _get_type():
    return pa.struct([
        ("g1p", pa.float64()),
        ("g1m", pa.float64()),
        ("g1", pa.float64()),
        ("g2p", pa.float64()),
        ("g2m", pa.float64()),
        ("g2", pa.float64()),
    ])


def _get_schema():
    # schema = pa.schema([
    #     ("plus", pa.struct([
    #         ("g1p", pa.float64()),
    #         ("g1m", pa.float64()),
    #         ("g1", pa.float64()),
    #         ("g2p", pa.float64()),
    #         ("g2m", pa.float64()),
    #         ("g2", pa.float64()),
    #      ])),
    #     ("minus", pa.struct([
    #         ("g1p", pa.float64()),
    #         ("g1m", pa.float64()),
    #         ("g1", pa.float64()),
    #         ("g2p", pa.float64()),
    #         ("g2m", pa.float64()),
    #         ("g2", pa.float64()),
    #     ])),
    #     ("s2n_cut", pa.int32()),
    #     ("ormask_cut", pa.int32()),
    #     ("mfrac_cut", pa.int32()),
    #     ("weight", pa.float64()),
    # ])
    schema = pa.schema([
        ("p.g1p", pa.float64()),
        ("p.g1m", pa.float64()),
        ("p.g1", pa.float64()),
        ("p.g2p", pa.float64()),
        ("p.g2m", pa.float64()),
        ("p.g2", pa.float64()),
        ("m.g1p", pa.float64()),
        ("m.g1m", pa.float64()),
        ("m.g1", pa.float64()),
        ("m.g2p", pa.float64()),
        ("m.g2m", pa.float64()),
        ("m.g2", pa.float64()),
        ("s2n_cut", pa.int32()),
        ("ormask_cut", pa.int32()),
        ("mfrac_cut", pa.int32()),
        ("weight", pa.float64()),
    ])

    return schema


def match_expression(names, expressions):
    """Match names to regular expressions"""
    return [
        name
        for name in names
        for expression in expressions
        if re.match(expression, name)
    ]


def build_lattice(
    full_xsize,
    full_ysize,
    sep,
    scale,
    v1,
    v2,
    rot=None,
    border=0,
):
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


def build_scene(
    gals,
    xsize,
    ysize,
    pixel_scale,
    seed,
    mag=None,
):
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

    return objects


def build_image(
    band,
    observed,
    observed_psf,
    xsize,
    ysize,
    psf_size,
    scene_seed,
    image_seed,
):
    scene_rng = np.random.default_rng(scene_seed)
    scene_grng = galsim.BaseDeviate(scene_seed)
    image_rng = np.random.default_rng(image_seed)
    image_grng = galsim.BaseDeviate(image_seed)

    image = galsim.Image(
        xsize,
        ysize,
        scale=lsst.SCALE,
    )
    for obs in observed:
        # obs.drawImage(
        #     image=image,
        #     add_to_image=True,
        #     bandpass=bandpass,
        # )
        obs.drawImage(
            image=image,
            exptime=lsst.EXPTIME * lsst.NCOADD[band],
            area=lsst.AREA,
            gain=lsst.GAIN,
            add_to_image=True,
            bandpass=lsst.BANDPASSES[band],
        )
        # stamp = obs.drawImage(
        #     exptime=30 * n_coadd,
        #     area=319 / 9.6 * 1e4,
        #     bandpass=bandpass,
        # )  # Need to sort out centering...
        # b = stamp.bounds & image.bounds
        # if b.isDefined():
        #     image[b] += stamp[b]

    # TODO: what is the correct expression here?
    noise_sigma = np.sqrt(lsst.SKY_RMS[band] * lsst.NCOADD[band]) / 10
    noise = galsim.GaussianNoise(scene_grng, sigma=noise_sigma)
    image.addNoise(noise)

    psf_image = galsim.Image(
        psf_size,
        psf_size,
        scale=lsst.SCALE,
    )
    observed_psf.drawImage(
        image=psf_image,
        bandpass=lsst.BANDPASSES[band],
        add_to_image=True,
    )

    noise_image = galsim.Image(
        xsize,
        ysize,
        scale=lsst.SCALE,
    )
    # counterfactual_noise = galsim.GaussianNoise(image_grng, sigma=noise_sigma)
    # noise_image.addNoise(counterfactual_noise)
    noise_image.addNoise(noise)

    ormask = np.full((xsize, ysize), int(0))
    bmask = np.full((xsize, ysize), int(0))
    weight = np.full((xsize, ysize), 1 / noise_sigma ** 2)

    return image, psf_image, noise_image, ormask, bmask, weight


def build_pair(
    scene,
    star,
    shear,
    psf,
    bands,
    xsize,
    ysize,
    psf_size,
    seed,
):

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
        for band in bands:
            scene_seed = scene_rng.integers(1, 2**64 // 2 - 1)
            image_seed = image_rng.integers(1, 2**64 // 2 - 1)
            image, psf_image, noise_image, ormask, bmask, weight = build_image(
                band,
                observed,
                observed_psf,
                xsize,
                ysize,
                psf_size,
                scene_seed,
                image_seed,
            )
            wcs = galsim.AffineTransform(
                lsst.SCALE,
                0.0,
                0.0,
                lsst.SCALE,
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


def measure_pair(
    pair,
    shear_bands,
    det_bands,
    config,
    seed,
):
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

    measurement = measure_pairs(config, res_p, res_m)

    return measurement


def measure_pair_color(
    pair,
    psf,
    colors,
    stars,
    psf_size,
    pixel_scale,
    bands,
    shear_bands,
    det_bands,
    config,
    seed,
):
    rng = np.random.default_rng(seed)
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
            AB_mags = AB_mag(bps)
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

    measurement = measure_pairs(config, res_p, res_m)

    return measurement


def build_and_measure_pair(
    scene,
    star,
    shear,
    xsize,
    ysize,
    psf_size,
    pixel_scale,
    bands,
    noises,
    psf,
    n_coadd,
    shear_bands,
    det_bands,
    config,
    pair_seed,
    meas_seed,
):
    pair = build_pair(
        scene,
        star,
        shear,
        psf,
        bands,
        xsize,
        ysize,
        psf_size,
        pair_seed,
    )

    meas = measure_pair(
        pair,
        shear_bands,
        det_bands,
        config,
        meas_seed,
    )

    return meas


def build_and_measure_pair_color(,
    scene,
    star,
    shear,
    xsize,
    ysize,
    psf_size,
    pixel_scale,
    bands,
    noises,
    psf,
    colors,
    stars,
    n_coadd,
    shear_bands,
    det_bands,
    config,
    pair_seed,
    meas_seed,
):
    pair = build_pair(
        scene,
        star,
        shear,
        psf,
        bands,
        xsize,
        ysize,
        psf_size,
        pair_seed,
    )

    meas = measure_pair_color(
        pair,
        psf,
        colors,
        stars,
        psf_size,
        pixel_scale,
        bands,
        shear_bands,
        det_bands,
        config,
        meas_seed,
    )

    return meas


def build_plot(
    pair,
    bands,
    detect,
    config,
):
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



def estimate_biases(
    data,
    calibration_shear,
    cosmic_shear,
    weights=None,
    method="bootstrap",
    n_resample=1000,
):
    """
    Estimate both additive and multiplicative biases with noise bias
    cancellation and resampled standard deviations.
    """
    g1p = np.array(data["p.g1"])
    R11p = (np.array(data["p.g1p"]) - np.array(data["p.g1m"])) / (2 * calibration_shear)

    g1m = np.array(data["m.g1"])
    R11m = (np.array(data["m.g1p"]) - np.array(data["m.g1m"])) / (2 * calibration_shear)

    g2p = np.array(data["p.g2"])
    R22p = (np.array(data["p.g2p"]) - np.array(data["p.g2m"])) / (2 * calibration_shear)

    g2m = np.array(data["m.g2"])
    R22m = (np.array(data["m.g2p"]) - np.array(data["m.g2m"])) / (2 * calibration_shear)

    if weights is not None:
        w = np.asarray(weights).astype(np.float64)
    else:
        w = np.ones(len(g1p)).astype(np.float64)
    w /= np.sum(w)

    msk = (
        np.isfinite(g1p) &
        np.isfinite(R11p) &
        np.isfinite(g1m) &
        np.isfinite(R11m) &
        np.isfinite(g2p) &
        np.isfinite(R22p) &
        np.isfinite(g2m) &
        np.isfinite(R22m)
    )
    g1p = g1p[msk]
    R11p = R11p[msk]
    g1m = g1m[msk]
    R11m = R11m[msk]
    g2p = g2p[msk]
    R22p = R22p[msk]
    g2m = g2m[msk]
    R22m = R22m[msk]
    w = w[msk]

    # Use g1 axis for multiplicative bias
    x1 = (R11p + R11m) / 2  # N.B. we assume that these are the same and average
    y1 = (g1p - g1m) / 2 / np.abs(cosmic_shear)

    # Use g2 axis for additive bias
    x2 = (R22p + R22m) / 2
    y2 = (g2p + g2m) / 2

    if method == "jackknife":
        return _jackknife(x1, y1, x2, y2, w, n_resample=n_resample)
    else:
        return _bootstrap(x1, y1, x2, y2, w, n_resample=n_resample)


def measure_shear_metadetect(
    res, *, s2n_cut, t_ratio_cut, ormask_cut, mfrac_cut, model
):
    """Measure the shear parameters for metadetect.

    NOTE: Returns None if nothing can be measured.

    Parameters
    ----------
    res : dict
        The metadetect results.
    s2n_cut : float
        The cut on `wmom_s2n`. Typically 10.
    t_ratio_cut : float
        The cut on `t_ratio_cut`. Typically 1.2.
    ormask_cut : bool
        If True, cut on the `ormask` flags.
    mfrac_cut : float or None
        If not None, cut objects with a masked fraction higher than this
        value.
    model : str
        The model kind (e.g. wmom).

    Returns
    -------
    g1p : float
        The mean 1-component shape for the plus metadetect measurement.
    g1m : float
        The mean 1-component shape for the minus metadetect measurement.
    g1 : float
        The mean 1-component shape for the zero-shear metadetect measurement.
    g2p : float
        The mean 2-component shape for the plus metadetect measurement.
    g2m : float
        The mean 2-component shape for the minus metadetect measurement.
    g2 : float
        The mean 2-component shape for the zero-shear metadetect measurement.
    """

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

    op = res["1p"]
    q = _mask(op)
    if not np.any(q):
        return None
    g1p = op[model + "_g"][q, 0]

    om = res["1m"]
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om[model + "_g"][q, 0]

    o = res["noshear"]
    q = _mask(o)
    if not np.any(q):
        return None
    g1 = o[model + "_g"][q, 0]
    g2 = o[model + "_g"][q, 1]

    op = res["2p"]
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op[model + "_g"][q, 1]

    om = res["2m"]
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om[model + "_g"][q, 1]

    return (
        np.mean(g1p),
        np.mean(g1m),
        np.mean(g1),
        np.mean(g2p),
        np.mean(g2m),
        np.mean(g2),
    )


def measure_pairs(config, res_p, res_m):
    model = config["metadetect"]["model"]
    if model == "wmom":
        tcut = 1.2
    else:
        tcut = 0.5

    if len(res_p) > 0:
        # wgt = len(res_p)
        wgt = sum(len(v) for v in res_p.values()) + sum(len(v) for v in res_m.values())

        data = []
        # data = {
        #     "plus": [],
        #     "minus": [],
        #     "s2n_cut": [],
        #     "ormask_cut": [],
        #     "mfrac_cut": [],
        #     "weight": [],
        # }
        for ormask_cut in ORMASK_CUTS:
            for s2n_cut in S2N_CUTS:
                pgm = measure_shear_metadetect(
                    res_p,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                )
                mgm = measure_shear_metadetect(
                    res_m,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                )
                if pgm is None or mgm is None:
                    continue

                # data.append(
                #     tuple(list(pgm) + list(mgm) + [s2n_cut, 0 if ormask_cut else 1, -1, wgt])
                # )
                # data.append({
                #     "pgm": list(pgm),
                #     "mgm": list(mgm),
                #     "s2n_cut": s2n_cut,
                #     "ormask_cut": 0 if ormask_cut else 1,
                #     "mfrac_cut":-1,
                #     "weight": wgt,
                # })
                data.append({
                    "p.g1p": pgm[0],
                    "p.g1m": pgm[1],
                    "p.g1": pgm[2],
                    "p.g2p": pgm[3],
                    "p.g2m": pgm[4],
                    "p.g2": pgm[5],
                    "m.g1p": mgm[0],
                    "m.g1m": mgm[1],
                    "m.g1": mgm[2],
                    "m.g2p": mgm[3],
                    "m.g2m": mgm[4],
                    "m.g2": mgm[5],
                    "s2n_cut": s2n_cut,
                    "ormask_cut": 0 if ormask_cut else 1,
                    "mfrac_cut": -1,
                    "weight": wgt,
                })
                # data["plus"].append(pgm)
                # data["minus"].append(mgm)
                # data["s2n_cut"].append(s2n_cut)
                # data["ormask_cut"].append(0 if ormask_cut else 1)
                # data["mfrac_cut"].append(1)
                # data["weight"].append(wgt)

        for mfrac_cut in MFRAC_CUTS:
            for s2n_cut in S2N_CUTS:
                pgm = measure_shear_metadetect(
                    res_p,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut / 100,
                    model=model,
                )
                mgm = measure_shear_metadetect(
                    res_m,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut / 100,
                    model=model,
                )
                if pgm is None or mgm is None:
                    continue

                # data.append(
                #     tuple(list(pgm) + list(mgm) + [s2n_cut, -1, mfrac_cut, wgt])
                # )
                # data.append({
                #     "pgm": list(pgm),
                #     "mgm": list(mgm),
                #     "s2n_cut": s2n_cut,
                #     "ormask_cut": -1,
                #     "mfrac_cut": mfrac_cut,
                #     "weight": wgt,
                # })
                data.append({
                    "p.g1p": pgm[0],
                    "p.g1m": pgm[1],
                    "p.g1": pgm[2],
                    "p.g2p": pgm[3],
                    "p.g2m": pgm[4],
                    "p.g2": pgm[5],
                    "m.g1p": mgm[0],
                    "m.g1m": mgm[1],
                    "m.g1": mgm[2],
                    "m.g2p": mgm[3],
                    "m.g2m": mgm[4],
                    "m.g2": mgm[5],
                    "s2n_cut": s2n_cut,
                    "ormask_cut": -1,
                    "mfrac_cut": mfrac_cut,
                    "weight": wgt,
                })
                # data["plus"].append(pgm)
                # data["minus"].append(mgm)
                # data["s2n_cut"].append(s2n_cut)
                # data["ormask_cut"].append(-1)
                # data["mfrac_cut"].append(mfrac_cut)
                # data["weight"].append(wgt)

        return data
    else:
        return None

