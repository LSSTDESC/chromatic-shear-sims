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


ORMASK_CUTS = [True, False]
S2N_CUTS = [7, 8, 9, 10, 15, 20]
MFRAC_CUTS = [0, 1, 2, 5, 8, 10, 20, 50, 80, 100]

gsparams = galsim.GSParams(
    maximum_fft_size=16384,
    kvalue_accuracy=1e-8,
    maxk_threshold=1e-5,
)


_mdet_schema = pa.schema([
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


_chromatic_schema = pa.schema([
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
    ("color_step", pa.string()),
    ("seed", pa.int64()),
])


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


def get_AB_zeropoint(bandpass):
    # from galsim.Bandpass.withZeropoint
    AB_source = 3631e-23 # 3631 Jy in units of erg/s/Hz/cm^2
    AB_sed = galsim.SED(lambda wave: AB_source, wave_type='nm', flux_type='fnu')
    AB_flux = AB_sed.calculateFlux(bandpass)
    AB_zeropoint = 2.5 * np.log10(AB_flux)

    return AB_zeropoint


def f2c(f_1, f_2, zp_1, zp_2):
    return -2.5 * np.log10(np.divide(f_1, f_2)) + zp_1 - zp_2


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


def _get_schema(drdc=False):
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
    match drdc:
        case False:
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
        case True:
            schema = pa.schema([
                ("p.c0.g1p", pa.float64()),
                ("p.c0.g1m", pa.float64()),
                ("p.c0.g1", pa.float64()),
                ("p.c0.g2p", pa.float64()),
                ("p.c0.g2m", pa.float64()),
                ("p.c0.g2", pa.float64()),
                ("p.c0.cg1p", pa.float64()),
                ("p.c0.cg1m", pa.float64()),
                ("p.c0.cg1", pa.float64()),
                ("p.c0.cg2p", pa.float64()),
                ("p.c0.cg2m", pa.float64()),
                ("p.c0.cg2", pa.float64()),
                ("m.c0.g1p", pa.float64()),
                ("m.c0.g1m", pa.float64()),
                ("m.c0.g1", pa.float64()),
                ("m.c0.g2p", pa.float64()),
                ("m.c0.g2m", pa.float64()),
                ("m.c0.g2", pa.float64()),
                ("m.c0.cg1p", pa.float64()),
                ("m.c0.cg1m", pa.float64()),
                ("m.c0.cg1", pa.float64()),
                ("m.c0.cg2p", pa.float64()),
                ("m.c0.cg2m", pa.float64()),
                ("m.c0.cg2", pa.float64()),
                ("p.c1.g1p", pa.float64()),
                ("p.c1.g1m", pa.float64()),
                ("p.c1.g1", pa.float64()),
                ("p.c1.g2p", pa.float64()),
                ("p.c1.g2m", pa.float64()),
                ("p.c1.g2", pa.float64()),
                ("p.c1.cg1p", pa.float64()),
                ("p.c1.cg1m", pa.float64()),
                ("p.c1.cg1", pa.float64()),
                ("p.c1.cg2p", pa.float64()),
                ("p.c1.cg2m", pa.float64()),
                ("p.c1.cg2", pa.float64()),
                ("m.c1.g1p", pa.float64()),
                ("m.c1.g1m", pa.float64()),
                ("m.c1.g1", pa.float64()),
                ("m.c1.g2p", pa.float64()),
                ("m.c1.g2m", pa.float64()),
                ("m.c1.g2", pa.float64()),
                ("m.c1.cg1p", pa.float64()),
                ("m.c1.cg1m", pa.float64()),
                ("m.c1.cg1", pa.float64()),
                ("m.c1.cg2p", pa.float64()),
                ("m.c1.cg2m", pa.float64()),
                ("m.c1.cg2", pa.float64()),
                ("p.c2.g1p", pa.float64()),
                ("p.c2.g1m", pa.float64()),
                ("p.c2.g1", pa.float64()),
                ("p.c2.g2p", pa.float64()),
                ("p.c2.g2m", pa.float64()),
                ("p.c2.g2", pa.float64()),
                ("p.c2.cg1p", pa.float64()),
                ("p.c2.cg1m", pa.float64()),
                ("p.c2.cg1", pa.float64()),
                ("p.c2.cg2p", pa.float64()),
                ("p.c2.cg2m", pa.float64()),
                ("p.c2.cg2", pa.float64()),
                ("m.c2.g1p", pa.float64()),
                ("m.c2.g1m", pa.float64()),
                ("m.c2.g1", pa.float64()),
                ("m.c2.g2p", pa.float64()),
                ("m.c2.g2m", pa.float64()),
                ("m.c2.g2", pa.float64()),
                ("m.c2.cg1p", pa.float64()),
                ("m.c2.cg1m", pa.float64()),
                ("m.c2.cg1", pa.float64()),
                ("m.c2.cg2p", pa.float64()),
                ("m.c2.cg2m", pa.float64()),
                ("m.c2.cg2", pa.float64()),
                ("s2n_cut", pa.int32()),
                ("ormask_cut", pa.int32()),
                ("mfrac_cut", pa.int32()),
                ("weight", pa.float64()),
            ])
        case _:
            schema = None

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
    survey,
    full_xsize,
    full_ysize,
    sep,
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
    xs = np.arange(-full_xsize // 2, full_xsize // 2 + 1) * sep / survey.scale
    ys = np.arange(-full_ysize // 2, full_ysize // 2 + 1) * sep / survey.scale
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
    survey,
    gals,
    xsize,
    ysize,
    seed,
    mag=None,
):
    rng = np.random.default_rng(seed)

    v1 = np.asarray([1, 0], dtype=float)
    v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)
    x_lattice, y_lattice = build_lattice(
        survey,
        xsize,
        ysize,
        10,
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
        .shift(x * survey.scale, y * survey.scale)
        .shift(
            rng.uniform(-0.5, 0.5) * survey.scale,
            rng.uniform(-0.5, 0.5) * survey.scale,
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
    survey,
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
        scale=survey.scale,
    )
    for obs in observed:
        obs = obs.withGSParams(gsparams)
        # obs.drawImage(
        #     image=image,
        #     add_to_image=True,
        #     bandpass=bandpass,
        # )
        obs.drawImage(
            image=image,
            # exptime=survey.exptime * survey.ncoadd[band],
            # area=survey.area,
            # gain=survey.gain,
            add_to_image=True,
            bandpass=survey.bandpasses[band],
        )
        # stamp = obs.drawImage(
        #     scale=survey.scale,
        #     exptime=survey.exptime * survey.ncoadd[band],
        #     area=survey.area,
        #     gain=survey.gain,
        #     bandpass=survey.bandpasses[band],
        # )  # FIXME Need to sort out centering...
        # b = stamp.bounds & image.bounds
        # if b.isDefined():
        #     image[b] += stamp[b]

    # TODO: what is the correct expression here?
    # noise_sigma = survey.sky_rms[band] * np.sqrt(survey.ncoadd[band])
    # noise = galsim.GaussianNoise(scene_grng, sigma=noise_sigma)
    noise_sigma = survey.sky_rms[band]
    noise = galsim.GaussianNoise(scene_grng, sigma=noise_sigma)
    image.addNoise(noise)

    psf_image = galsim.Image(
        psf_size,
        psf_size,
        scale=survey.scale,
    )
    observed_psf.drawImage(
        image=psf_image,
        bandpass=survey.bandpasses[band],
        add_to_image=True,
    )
    # renormalize the psf image to unity
    psf_norm = psf_image.array.sum()
    psf_image.array[:] /= psf_norm

    noise_image = galsim.Image(
        xsize,
        ysize,
        scale=survey.scale,
    )
    # counterfactual_noise = galsim.GaussianNoise(image_grng, sigma=noise_sigma)
    # noise_image.addNoise(counterfactual_noise)
    noise_image.addNoise(noise)

    ormask = np.full((xsize, ysize), int(0))
    bmask = np.full((xsize, ysize), int(0))
    weight = np.full((xsize, ysize), 1 / noise_sigma ** 2)

    return image, psf_image, noise_image, ormask, bmask, weight


def build_pair(
    survey,
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
                survey,
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
                survey.scale,
                0.0,
                0.0,
                survey.scale,
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
        config,
        mbobs_p,
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_m = metadetect.do_metadetect(
        config,
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
        band.lower(): galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm").withZeropoint("AB")
        for band in bands
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
            # renormalize the psf image to unity
            psf_norm = psf_image.array.sum()
            psf_image.array[:] /= psf_norm
            for _obs_p, _obs_m in zip(_obslist_p, _obslist_m):
                _obs_p.psf.set_image(psf_image.array)
                _obs_m.psf.set_image(psf_image.array)

        color_dep_mbobs_p[c] = _mbobs_p
        color_dep_mbobs_m[c] = _mbobs_m

    # for c in range(len(stars)):
    #     for b in range(len(color_dep_mbobs_p[c])):
    #         assert np.all(np.equal(color_dep_mbobs_p[c][b][0].psf.image, color_dep_mbobs_m[c][b][0].psf.image))

    res_p = metadetect.do_metadetect(
        config,
        mbobs_p,
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=color_key_func,
        color_dep_mbobs=color_dep_mbobs_p,
    )

    res_m = metadetect.do_metadetect(
        config,
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
    survey,
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
        survey,
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


def build_and_measure_pair_color(
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
        survey,
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


def measure_more_pairs(
    res_p_c0,
    res_p_c1,
    res_p_c2,
    res_m_c0,
    res_m_c1,
    res_m_c2,
    *,
    config,
    color=None,
):
    model = config["model"]
    if model == "wmom":
        tcut = 1.2
    else:
        tcut = 0.5

    if len(res_p_c0) > 0:
        # wgt = len(res_p)
        wgt = sum(len(v) for v in res_p_c0.values()) \
            + sum(len(v) for v in res_p_c1.values()) \
            + sum(len(v) for v in res_p_c2.values()) \
            + sum(len(v) for v in res_m_c0.values()) \
            + sum(len(v) for v in res_m_c1.values()) \
            + sum(len(v) for v in res_m_c2.values())

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
                # [shear type]gm_c[color index]
                # no idea what the m is here... "metadetect"?
                pgm_c0 = measure_shear_metadetect(
                    res_p_c0,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                pgm_c1 = measure_shear_metadetect(
                    res_p_c1,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                pgm_c2 = measure_shear_metadetect(
                    res_p_c2,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                mgm_c0 = measure_shear_metadetect(
                    res_m_c0,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                mgm_c1 = measure_shear_metadetect(
                    res_m_c1,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                mgm_c2 = measure_shear_metadetect(
                    res_m_c2,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=ormask_cut,
                    mfrac_cut=None,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                if pgm_c0 is None or pgm_c1 is None or pgm_c2 is None or mgm_c0 is None or mgm_c1 is None or mgm_c2 is None:
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

                # we need e_gp_cp * (c - c_0), etc. for each combination of
                # shear and color and so forth. note that this also means we
                # need to compute colors here
                data.append({
                    "p.c0.g1p": pgm_c0[0],
                    "p.c0.g1m": pgm_c0[1],
                    "p.c0.g1": pgm_c0[2],
                    "p.c0.g2p": pgm_c0[3],
                    "p.c0.g2m": pgm_c0[4],
                    "p.c0.g2": pgm_c0[5],
                    "p.c0.cg1p": pgm_c0[6],
                    "p.c0.cg1m": pgm_c0[7],
                    "p.c0.cg1": pgm_c0[8],
                    "p.c0.cg2p": pgm_c0[9],
                    "p.c0.cg2m": pgm_c0[10],
                    "p.c0.cg2": pgm_c0[11],
                    "m.c0.g1p": mgm_c0[0],
                    "m.c0.g1m": mgm_c0[1],
                    "m.c0.g1": mgm_c0[2],
                    "m.c0.g2p": mgm_c0[3],
                    "m.c0.g2m": mgm_c0[4],
                    "m.c0.g2": mgm_c0[5],
                    "m.c0.cg1p": mgm_c0[6],
                    "m.c0.cg1m": mgm_c0[7],
                    "m.c0.cg1": mgm_c0[8],
                    "m.c0.cg2p": mgm_c0[9],
                    "m.c0.cg2m": mgm_c0[10],
                    "m.c0.cg2": mgm_c0[11],
                    "p.c1.g1p": pgm_c1[0],
                    "p.c1.g1m": pgm_c1[1],
                    "p.c1.g1": pgm_c1[2],
                    "p.c1.g2p": pgm_c1[3],
                    "p.c1.g2m": pgm_c1[4],
                    "p.c1.g2": pgm_c1[5],
                    "p.c1.cg1p": pgm_c1[6],
                    "p.c1.cg1m": pgm_c1[7],
                    "p.c1.cg1": pgm_c1[8],
                    "p.c1.cg2p": pgm_c1[9],
                    "p.c1.cg2m": pgm_c1[10],
                    "p.c1.cg2": pgm_c1[11],
                    "m.c1.g1p": mgm_c1[0],
                    "m.c1.g1m": mgm_c1[1],
                    "m.c1.g1": mgm_c1[2],
                    "m.c1.g2p": mgm_c1[3],
                    "m.c1.g2m": mgm_c1[4],
                    "m.c1.g2": mgm_c1[5],
                    "m.c1.cg1p": mgm_c1[6],
                    "m.c1.cg1m": mgm_c1[7],
                    "m.c1.cg1": mgm_c1[8],
                    "m.c1.cg2p": mgm_c1[9],
                    "m.c1.cg2m": mgm_c1[10],
                    "m.c1.cg2": mgm_c1[11],
                    "p.c2.g1p": pgm_c2[0],
                    "p.c2.g1m": pgm_c2[1],
                    "p.c2.g1": pgm_c2[2],
                    "p.c2.g2p": pgm_c2[3],
                    "p.c2.g2m": pgm_c2[4],
                    "p.c2.g2": pgm_c2[5],
                    "p.c2.cg1p": pgm_c2[6],
                    "p.c2.cg1m": pgm_c2[7],
                    "p.c2.cg1": pgm_c2[8],
                    "p.c2.cg2p": pgm_c2[9],
                    "p.c2.cg2m": pgm_c2[10],
                    "p.c2.cg2": pgm_c2[11],
                    "m.c2.g1p": mgm_c2[0],
                    "m.c2.g1m": mgm_c2[1],
                    "m.c2.g1": mgm_c2[2],
                    "m.c2.g2p": mgm_c2[3],
                    "m.c2.g2m": mgm_c2[4],
                    "m.c2.g2": mgm_c2[5],
                    "m.c2.cg1p": mgm_c2[6],
                    "m.c2.cg1m": mgm_c2[7],
                    "m.c2.cg1": mgm_c2[8],
                    "m.c2.cg2p": mgm_c2[9],
                    "m.c2.cg2m": mgm_c2[10],
                    "m.c2.cg2": mgm_c2[11],
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
                pgm_c0 = measure_shear_metadetect(
                    res_p_c0,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut / 100,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                pgm_c1 = measure_shear_metadetect(
                    res_p_c1,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut/100,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                pgm_c2 = measure_shear_metadetect(
                    res_p_c2,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut/100,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                mgm_c0 = measure_shear_metadetect(
                    res_m_c0,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut/100,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                mgm_c1 = measure_shear_metadetect(
                    res_m_c1,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut/100,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                mgm_c2 = measure_shear_metadetect(
                    res_m_c2,
                    s2n_cut=s2n_cut,
                    t_ratio_cut=tcut,
                    ormask_cut=False,
                    mfrac_cut=mfrac_cut/100,
                    model=model,
                    color=color,
                    bands=[2, 0],
                )
                if pgm_c0 is None or pgm_c1 is None or pgm_c2 is None or mgm_c0 is None or mgm_c1 is None or mgm_c2 is None:
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
                    "p.c0.g1p": pgm_c0[0],
                    "p.c0.g1m": pgm_c0[1],
                    "p.c0.g1": pgm_c0[2],
                    "p.c0.g2p": pgm_c0[3],
                    "p.c0.g2m": pgm_c0[4],
                    "p.c0.g2": pgm_c0[5],
                    "p.c0.cg1p": pgm_c0[6],
                    "p.c0.cg1m": pgm_c0[7],
                    "p.c0.cg1": pgm_c0[8],
                    "p.c0.cg2p": pgm_c0[9],
                    "p.c0.cg2m": pgm_c0[10],
                    "p.c0.cg2": pgm_c0[11],
                    "m.c0.g1p": mgm_c0[0],
                    "m.c0.g1m": mgm_c0[1],
                    "m.c0.g1": mgm_c0[2],
                    "m.c0.g2p": mgm_c0[3],
                    "m.c0.g2m": mgm_c0[4],
                    "m.c0.g2": mgm_c0[5],
                    "m.c0.cg1p": mgm_c0[6],
                    "m.c0.cg1m": mgm_c0[7],
                    "m.c0.cg1": mgm_c0[8],
                    "m.c0.cg2p": mgm_c0[9],
                    "m.c0.cg2m": mgm_c0[10],
                    "m.c0.cg2": mgm_c0[11],
                    "p.c1.g1p": pgm_c1[0],
                    "p.c1.g1m": pgm_c1[1],
                    "p.c1.g1": pgm_c1[2],
                    "p.c1.g2p": pgm_c1[3],
                    "p.c1.g2m": pgm_c1[4],
                    "p.c1.g2": pgm_c1[5],
                    "p.c1.cg1p": pgm_c1[6],
                    "p.c1.cg1m": pgm_c1[7],
                    "p.c1.cg1": pgm_c1[8],
                    "p.c1.cg2p": pgm_c1[9],
                    "p.c1.cg2m": pgm_c1[10],
                    "p.c1.cg2": pgm_c1[11],
                    "m.c1.g1p": mgm_c1[0],
                    "m.c1.g1m": mgm_c1[1],
                    "m.c1.g1": mgm_c1[2],
                    "m.c1.g2p": mgm_c1[3],
                    "m.c1.g2m": mgm_c1[4],
                    "m.c1.g2": mgm_c1[5],
                    "m.c1.cg1p": mgm_c1[6],
                    "m.c1.cg1m": mgm_c1[7],
                    "m.c1.cg1": mgm_c1[8],
                    "m.c1.cg2p": mgm_c1[9],
                    "m.c1.cg2m": mgm_c1[10],
                    "m.c1.cg2": mgm_c1[11],
                    "p.c2.g1p": pgm_c2[0],
                    "p.c2.g1m": pgm_c2[1],
                    "p.c2.g1": pgm_c2[2],
                    "p.c2.g2p": pgm_c2[3],
                    "p.c2.g2m": pgm_c2[4],
                    "p.c2.g2": pgm_c2[5],
                    "p.c2.cg1p": pgm_c2[6],
                    "p.c2.cg1m": pgm_c2[7],
                    "p.c2.cg1": pgm_c2[8],
                    "p.c2.cg2p": pgm_c2[9],
                    "p.c2.cg2m": pgm_c2[10],
                    "p.c2.cg2": pgm_c2[11],
                    "m.c2.g1p": mgm_c2[0],
                    "m.c2.g1m": mgm_c2[1],
                    "m.c2.g1": mgm_c2[2],
                    "m.c2.g2p": mgm_c2[3],
                    "m.c2.g2m": mgm_c2[4],
                    "m.c2.g2": mgm_c2[5],
                    "m.c2.cg1p": mgm_c2[6],
                    "m.c2.cg1m": mgm_c2[7],
                    "m.c2.cg1": mgm_c2[8],
                    "m.c2.cg2p": mgm_c2[9],
                    "m.c2.cg2m": mgm_c2[10],
                    "m.c2.cg2": mgm_c2[11],
                    "s2n_cut": s2n_cut,
                    "ormask_cut": 0 if ormask_cut else 1,
                    "mfrac_cut": -1,
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


def measure_pair_color_response(
    survey,
    pair,
    psf,
    colors,
    stars,
    psf_size,
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
        band.lower(): galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm").withZeropoint("AB")
        for band in bands
    }

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
            scale=survey.scale,
        )

        for i, (_obslist_p, _obslist_m) in enumerate(zip(_mbobs_p, _mbobs_m)):
            band = bands[i]
            observed_psf.drawImage(image=psf_image, bandpass=bps[band])
            # renormalize the psf image to unity
            psf_norm = psf_image.array.sum()
            psf_image.array[:] /= psf_norm
            for _obs_p, _obs_m in zip(_obslist_p, _obslist_m):
                _obs_p.psf.set_image(psf_image.array)
                _obs_m.psf.set_image(psf_image.array)

        color_dep_mbobs_p[c] = _mbobs_p
        color_dep_mbobs_m[c] = _mbobs_m

    # for c in range(len(stars)):
    #     for b in range(len(color_dep_mbobs_p[c])):
    #         assert np.all(np.equal(color_dep_mbobs_p[c][b][0].psf.image, color_dep_mbobs_m[c][b][0].psf.image))

    res_p_c0 = metadetect.do_metadetect(
        config,
        color_dep_mbobs_p[0],
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_p_c1 = metadetect.do_metadetect(
        config,
        color_dep_mbobs_p[1],
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_p_c2 = metadetect.do_metadetect(
        config,
        color_dep_mbobs_p[2],
        mdet_rng_p,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_m_c0 = metadetect.do_metadetect(
        config,
        color_dep_mbobs_m[0],
        mdet_rng_m,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_m_c1 = metadetect.do_metadetect(
        config,
        color_dep_mbobs_m[1],
        mdet_rng_m,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    res_m_c2 = metadetect.do_metadetect(
        config,
        color_dep_mbobs_m[2],
        mdet_rng_m,
        shear_band_combs=shear_bands,
        det_band_combs=det_bands,
        color_key_func=None,
        color_dep_mbobs=None,
    )

    # calibrate chromatic response about the central color
    color = colors[1]

    measurement = measure_more_pairs(
        res_p_c0,
        res_p_c1,
        res_p_c2,
        res_m_c0,
        res_m_c1,
        res_m_c2,
        config=config,
        color=color,
    )

    return measurement


def run_pair_color_response(
    pipeline,
    survey,
    pair,
    psf,
    colors,
    stars,
    psf_size,
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
    mdet_rng = np.random.default_rng(mdet_seed)

    mbobs_p = pair["plus"]
    mbobs_m = pair["minus"]

    bps = {
        band.lower(): galsim.Bandpass(f"LSST_{band.lower()}.dat", wave_type="nm").withZeropoint("AB")
        for band in bands
    }

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
            scale=survey.scale,
        )

        for i, (_obslist_p, _obslist_m) in enumerate(zip(_mbobs_p, _mbobs_m)):
            band = bands[i]
            observed_psf.drawImage(image=psf_image, bandpass=bps[band])
            # renormalize the psf image to unity
            psf_norm = psf_image.array.sum()
            psf_image.array[:] /= psf_norm
            for _obs_p, _obs_m in zip(_obslist_p, _obslist_m):
                _obs_p.psf.set_image(psf_image.array)
                _obs_m.psf.set_image(psf_image.array)

        color_dep_mbobs_p[c] = _mbobs_p
        color_dep_mbobs_m[c] = _mbobs_m

    schema = pipeline.get_schema()
    batches = []
    for (shear, color_dep_mbobs) in [("plus", color_dep_mbobs_p), ("minus", color_dep_mbobs_m)]:
        for i, color in enumerate(colors):
            color_step = f"c{i}"
            res = metadetect.do_metadetect(
                config,
                color_dep_mbobs[i],
                mdet_rng,
                shear_band_combs=shear_bands,
                det_band_combs=det_bands,
                color_key_func=None,
                color_dep_mbobs=None,
            )

            # # FIXME
            # # pa.table({"wmom_band_flux": pa.array(list(res["wmom_band_flux"]), pa.list_(pa.float64()))})
            # _schema = []
            # for dt in res["noshear"].dtype.descr:
            #     dt_name = dt[0]
            #     dt_type = pa.from_numpy_dtype(dt[1])
            #     dt_is_list = len(dt) > 2

            #     if not dt_is_list:
            #         _schema.append((dt_name, dt_type))
            #     else:
            #         dt_list_size = dt[2][0]
            #         # _schema.append((dt_name, pa.list_(dt_type, dt_list_size)))
            #         _schema.append((dt_name, pa.list_(dt_type)))

            # _schema.append(("mdet_step", pa.string()))
            # _schema.append(("shear", pa.string()))
            # _schema.append(("color", pa.float64()))
            # _schema.append(("seed", pa.int64()))
            # schema = pa.schema(_schema)

            for mdet_step in res.keys():
                mdet_cat = res[mdet_step]
                data_dict = {name: mdet_cat[name].tolist() for name in mdet_cat.dtype.names}
                data_dict["mdet_step"] = [mdet_step for _ in range(len(mdet_cat))]
                data_dict["shear"] = [shear for _ in range(len(mdet_cat))]
                data_dict["color_step"] = [color_step for _ in range(len(mdet_cat))]
                data_dict["seed"] = [seed for _ in range(len(mdet_cat))]

                # # data_list = [mdet_cat[name].tolist() for name in schema.names]
                # data_list = [mdet_cat[name].tolist() for name in mdet_cat.dtype.names]
                # data_list.append([mdet_step for _ in range(len(mdet_cat))])
                # data_list.append([shear for _ in range(len(mdet_cat))])
                # data_list.append([color for _ in range(len(mdet_cat))])
                # data_list.append([seed for _ in range(len(mdet_cat))])

                # table = pa.table(data_dict, schema=_schema)
                # tables.append(table)
                batch = pa.RecordBatch.from_pydict(data_dict, schema=schema)
                batches.append(batch)

    return batches


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
    res, *, s2n_cut, t_ratio_cut, ormask_cut, mfrac_cut, model,
    color=None, bands=None
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
    if color is not None and bands is not None:
        assert len(bands) == 2
        assert bands == sorted(bands, reverse=True)
        bps = [
            galsim.Bandpass(f"LSST_{band}.dat", wave_type="nm").withZeropoint("AB")
            for band in ["u", "g", "r", "i", "z", "y"]
        ]
        bp_0 = bps[bands[0]]
        bp_1 = bps[bands[1]]
        # FIXME: better solution would be to pass in the survey object; then,
        #        get zeropoint with survey.bandpasses[...].zeropoint
        zp_0 = get_AB_zeropoint(bp_0)
        zp_1 = get_AB_zeropoint(bp_1)

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
    if color is not None and bands is not None:
        f_0 = op["wmom_band_flux"][q, bands[0]]
        f_1 = op["wmom_band_flux"][q, bands[1]]
        c_1p = f2c(f_0, f_1, zp_0, zp_1)

    om = res["1m"]
    q = _mask(om)
    if not np.any(q):
        return None
    g1m = om[model + "_g"][q, 0]
    if color is not None and bands is not None:
        f_0 = om["wmom_band_flux"][q, bands[0]]
        f_1 = om["wmom_band_flux"][q, bands[1]]
        c_1m = f2c(f_0, f_1, zp_0, zp_1)

    o = res["noshear"]
    q = _mask(o)
    if not np.any(q):
        return None
    g1_ns = o[model + "_g"][q, 0]
    g2_ns = o[model + "_g"][q, 1]
    if color is not None and bands is not None:
        f_0 = o["wmom_band_flux"][q, bands[0]]
        f_1 = o["wmom_band_flux"][q, bands[1]]
        c_ns = f2c(f_0, f_1, zp_0, zp_1)

    op = res["2p"]
    q = _mask(op)
    if not np.any(q):
        return None
    g2p = op[model + "_g"][q, 1]
    if color is not None and bands is not None:
        f_0 = op["wmom_band_flux"][q, bands[0]]
        f_1 = op["wmom_band_flux"][q, bands[1]]
        c_2p = f2c(f_0, f_1, zp_0, zp_1)

    om = res["2m"]
    q = _mask(om)
    if not np.any(q):
        return None
    g2m = om[model + "_g"][q, 1]
    if color is not None and bands is not None:
        f_0 = om["wmom_band_flux"][q, bands[0]]
        f_1 = om["wmom_band_flux"][q, bands[1]]
        c_2m = f2c(f_0, f_1, zp_0, zp_1)

    if color is not None and bands is not None:
        return (
            np.mean(g1p),
            np.mean(g1m),
            np.mean(g1_ns),
            np.mean(g2p),
            np.mean(g2m),
            np.mean(g2_ns),
            np.mean(g1p * (c_1p - color)),
            np.mean(g1m * (c_1m - color)),
            np.mean(g1_ns * (c_ns - color)),
            np.mean(g2p * (c_2p - color)),
            np.mean(g2m * (c_2m - color)),
            np.mean(g2_ns * (c_ns - color)),
        )
    else:
        return (
            np.mean(g1p),
            np.mean(g1m),
            np.mean(g1_ns),
            np.mean(g2p),
            np.mean(g2m),
            np.mean(g2_ns),
        )


def measure_pairs(config, res_p, res_m):
    model = config["model"]
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

