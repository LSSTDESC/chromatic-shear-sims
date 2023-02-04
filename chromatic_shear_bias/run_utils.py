"""
Utilities for running simulations
Some functions from https://github.com/beckermr/pizza-cutter-sims
"""

import copy

import numpy as np
import pyarrow as pa
import yaml

import galsim
import galsim_extra
import ngmix
import metadetect


ORMASK_CUTS = [True, False]
S2N_CUTS = [7, 8, 9, 10, 15, 20]
MFRAC_CUTS = [0, 1, 2, 5, 8, 10, 20, 50, 80, 100]


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


def generate_arguments(config, galsim_config, seed, n, logger):
    """
    Generate arguments for the measurement builder.
    """
    i = 0
    rng = np.random.default_rng(seed)
    while i < n:
        arg_dict = {
            "config": copy.deepcopy(config),
            "galsim_config": copy.deepcopy(galsim_config),
            "seed": rng.integers(1, 2**64 // 2),
            "index": i,
            "logger": logger,
        }
        yield arg_dict
        i += 1


def draw_psf(config, size, logger=None):
    """
    Draw the PSF by convolving against a chromatic star.
    """
    psf, _ = galsim.config.BuildGSObject(config, "psf", logger=logger)
    if "star" in config.keys():
        star, _ = galsim.config.BuildGSObject(config, "star", logger=logger)
    else:
        star = galsim.DeltaFunction()

    stellar_psf = galsim.Convolve([star, psf])

    # this should be set following the building of an image with a bandpass
    if "bandpass" in config.keys():
        return stellar_psf.drawImage(nx=size, ny=size, scale=config["image"]["pixel_scale"], bandpass=config["bandpass"])
    else:
        return stellar_psf.drawImage(nx=size, ny=size, scale=config["image"]["pixel_scale"])


def draw_weight(config, logger=None):
    """
    Draw the weight.
    """
    image_shape = (config["image_ysize"], config["image_xsize"])
    weight = galsim.Image(np.zeros(image_shape))
    galsim.config.noise.AddNoiseVariance(config, weight, logger=logger)
    weight.invertSelf()
    return weight


def draw_noise(config, logger=None):
    """
    Draw the noise realization.
    """
    image_shape = (config["image_ysize"], config["image_xsize"])
    noise = galsim.Image(np.zeros(image_shape))
    galsim.config.noise.AddNoise(config, noise, logger=logger)
    return noise


def draw_bmask(config, logger=None):
    """
    Draw the bmask.
    """
    # build the bmask array
    image_shape = (config["image_ysize"], config["image_xsize"])
    return np.full(image_shape, int(0))


def draw_ormask(config, logger=None):
    """
    Draw the ormask.
    """
    # build the ormask array
    image_shape = (config["image_ysize"], config["image_xsize"])
    return np.full(image_shape, int(0))


def observation_builder(config, galsim_config, seed, logger=None):
    """
    Build an ngmix MultiBandObsList from a GalSim config dictionary.
    """
    # Setup the RNGs with a fixed default
    galsim.config.SetInConfig(
        galsim_config,
        "image.random_seed",
        [
            {'type': 'Sequence', 'first': seed, 'index_key': 'image_num'},
            {'type': 'Sequence', 'first': seed, 'index_key': 'file_num'},
            {'type': 'Sequence', 'first': seed, 'index_key': 'obj_num_in_file'},
        ]
    )

    # TODO: I don't prefer this method of searching for multiband info
    # support single-band or multi-band observations of a scene
    if ("eval_variables" in galsim_config.keys()) and ("sfilter" in galsim_config["eval_variables"].keys()):
        nimages = len(galsim.config.GetFromConfig(galsim_config, "eval_variables.sfilter.items"))
    else:
        nimages = 1

    mbobs = ngmix.MultiBandObsList()
    for _i in range(nimages):
        image = galsim.config.BuildImage(galsim_config, _i, logger=logger)

        # draw images used for each ngmix Observation
        psf_size = 53  # TODO: would be good to specify in global config
        psf = draw_psf(galsim_config, psf_size, logger=logger)
        weight = draw_weight(galsim_config, logger=logger)
        noise = draw_noise(galsim_config, logger=logger)
        bmask = draw_bmask(galsim_config, logger=logger)
        ormask = draw_ormask(galsim_config, logger=logger)

        # construct the WCS
        # TODO: verify that this is the correct WCS to be using
        wcs = galsim.AffineTransform(
            galsim_config["pixel_scale"],
            0.0,
            0.0,
            galsim_config["pixel_scale"],
            origin=image.center,
        )

        # build the ngmix Observation of the PSF
        psf_jac = ngmix.jacobian.Jacobian(
            x=(psf.array.shape[1] - 1) / 2,
            y=(psf.array.shape[0] - 1) / 2,
            dudx=wcs.dudx,
            dudy=wcs.dudy,
            dvdx=wcs.dvdx,
            dvdy=wcs.dvdy,
        )
        psf_obs = ngmix.Observation(
            psf.array.copy(),
            jacobian=psf_jac,
        )

        # build the ngmix Observation of the image
        im_jac = ngmix.jacobian.Jacobian(
            x=(image.array.shape[1] - 1) / 2,
            y=(image.array.shape[0] - 1) / 2,
            dudx=wcs.dudx,
            dudy=wcs.dudy,
            dvdx=wcs.dvdx,
            dvdy=wcs.dvdy,
        )
        obs = ngmix.Observation(
            image.array.copy(),
            weight=weight.array.copy(),
            bmask=bmask.copy(),
            ormask=ormask.copy(),
            jacobian=im_jac,
            psf=psf_obs,
            noise=noise.array.copy(),
        )

        obslist = ngmix.ObsList()
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


def make_pair_config(config, g=0.02):
    """
    Create a pair of configs to simulate scenes sheared with equal and opposite
    shears for noise bias cancellation.
    """
    config_p = copy.deepcopy(config)
    config_m = copy.deepcopy(config)

    # TODO: this assumes we're applying the shear at the stamp stage (i.e., when doing
    # ring simulations. This won't work in general...
    # Though should (?) be ok for MixedScene from galsim_extra
    galsim.config.SetInConfig(config_p, "stamp.shear.g1", g)
    galsim.config.SetInConfig(config_m, "stamp.shear.g1", -g)

    # galsim.config.SetInConfig(config_p, "gal.shear.g1", g)
    # galsim.config.SetInConfig(config_m, "gal.shear.g1", -g)

    return config_p, config_m


def make_and_measure_pairs(config, galsim_config, seed, index, logger):
    """
    Build measurements of simulations
    """
    cosmic_shear = config["shear"]["g"]
    # galsim_config_p, galsim_config_m = make_pair_config(galsim_config, cosmic_shear)

    # TODO: multithread the p/m pieces in parallel?
    #       e.g., even rank do p, odd rank do m
    galsim.config.SetInConfig(galsim_config, "stamp.shear.g1", cosmic_shear)
    mbobs_p = observation_builder(config, galsim_config, seed=seed, logger=logger)
    galsim.config.SetInConfig(galsim_config, "stamp.shear.g1", -cosmic_shear)
    mbobs_m = observation_builder(config, galsim_config, seed=seed, logger=logger)

    # FIXME remove this -- just useful for some debugging
    # import matplotlib.pyplot as plt
    # for obslist_p, obslist_m in zip(mbobs_p, mbobs_m):
    #     for obs_p, obs_m in zip(obslist_p, obslist_m):
    #         fig, axs = plt.subplots(nrows=2, ncols=3)

    #         axs[0, 0].set_ylabel("$+\gamma$")
    #         axs[0, 0].imshow(obs_p.image, origin="lower")
    #         axs[0, 0].set_title("Image")
    #         axs[0, 1].imshow(obs_p.psf.image, origin="lower")
    #         axs[0, 1].set_title("PSF")
    #         axs[0, 2].imshow(obs_p.noise, origin="lower")
    #         axs[0, 2].set_title("Noise Realization")

    #         axs[1, 0].set_ylabel("$-\gamma$")
    #         axs[1, 0].imshow(obs_m.image, origin="lower")
    #         axs[1, 0].set_title("Image")
    #         axs[1, 1].imshow(obs_m.psf.image, origin="lower")
    #         axs[1, 1].set_title("PSF")
    #         axs[1, 2].imshow(obs_m.noise, origin="lower")
    #         axs[1, 2].set_title("Noise Realization")

    #         for ax in axs.ravel():
    #             ax.set_xticks([])
    #             ax.set_yticks([])
    # plt.show()

    # TODO Should these be the same rngs?
    mdet_rng_p = np.random.default_rng(seed)
    mdet_rng_m = np.random.default_rng(seed)

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng_p,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng_m,
    )

    return measure_pairs(config, res_p, res_m)


def _bootstrap(x1, y1, x2, y2, w, n_resample=1000):
    """
    Estimate the standard deviation via bootstrapping
    """
    rng = np.random.default_rng()  # TODO: seed this?

    n_sample = len(x1)

    m_bootstrap = np.empty(n_resample)
    c_bootstrap = np.empty(n_resample)

    for _i in range(n_resample):
        # perform bootstrap resampling
        _r = rng.choice(n_sample, size=n_resample, replace=True)  # resample indices
        _w = w[_r].copy()  # resample weights
        _w /= np.sum(_w)  # normalize resampled weights
        m_bootstrap[_i] = np.mean(y1[_r] * _w) / np.mean(x1[_r] * _w) - 1.  # compute the multiplicative bias of the resample
        c_bootstrap[_i] = np.mean(y2[_r] * _w) / np.mean(x2[_r] * _w)  # compute the additive bias of the resample

    m_est = np.mean(y1 * w) / np.mean(x1 * w) - 1
    m_var = np.var(m_bootstrap)
    c_est = np.mean(y2 * w) / np.mean(x2 * w)
    c_var = np.var(c_bootstrap)

    return (
        m_est, np.sqrt(m_var),
        c_est, np.sqrt(c_var),
    )


def _jackknife(x1, y1, x2, y2, w, n_resample=1000):
    """
    Estimate the standard deviation via the delete-m jackknife as defined in
    https://doi.org/10.1023/A:1008800423698
    """
    n = len(x1)  # sample size
    m = n // n_resample  # resample size
    _n = m * n_resample

    # estimators given all samples
    m_hat_n = np.mean(y1[:_n] * w[:_n]) / np.mean(x1[:_n] * w[:_n]) - 1
    c_hat_n = np.mean(y2[:_n] * w[:_n]) / np.mean(x2[:_n] * w[:_n])

    x1_j = np.empty(n_resample)
    y1_j = np.empty(n_resample)
    x2_j = np.empty(n_resample)
    y2_j = np.empty(n_resample)
    w_j = np.empty(n_resample)

    # precompute chunks to jackknife more efficiently
    for _i in range(n_resample):
        # perform jackknife resampling
        _r = slice(_i * m, (_i + 1) * m)  # resample indices
        w_j[_i] = np.sum(w[_r])
        x1_j[_i] = np.sum(x1[_r] * w[_r]) / w_j[_i]
        y1_j[_i] = np.sum(y1[_r] * w[_r]) / w_j[_i]
        x2_j[_i] = np.sum(x2[_r] * w[_r]) / w_j[_i]
        y2_j[_i] = np.sum(y2[_r] * w[_r]) / w_j[_i]

    # construct estimators of each observable based on subsamples each of size n - m
    m_hat_j = np.empty(n_resample)
    c_hat_j = np.empty(n_resample)

    for _i in range(n_resample):
        # perform jackknife resampling
        _r = ~(np.isin(np.arange(n_resample), _i))  # remove one chunk from the jackknife samples
        m_hat_j[_i] = np.sum(y1_j[_r] * w_j[_r]) / np.sum(x1_j[_r] * w_j[_r]) - 1
        c_hat_j[_i] = np.sum(y2_j[_r] * w_j[_r]) / np.sum(x2_j[_r] * w_j[_r])

    m_hat_J = n_resample * m_hat_n - np.sum((1 - w_j / np.sum(w_j)) * m_hat_j)
    c_hat_J = n_resample * c_hat_n - np.sum((1 - w_j / np.sum(w_j)) * c_hat_j)

    h_j = np.sum(w_j) / w_j
    m_tilde_j = h_j * m_hat_n - (h_j - 1) * m_hat_j
    c_tilde_j = h_j * c_hat_n - (h_j - 1) * c_hat_j
    m_var_J = np.sum(np.square(m_tilde_j - m_hat_J) / (h_j - 1)) / n_resample
    c_var_J = np.sum(np.square(c_tilde_j - c_hat_J) / (h_j - 1)) / n_resample

    return (
        m_hat_J, np.sqrt(m_var_J),
        c_hat_J, np.sqrt(c_var_J),
    )


def estimate_biases(data, calibration_shear, cosmic_shear, weights=None, method="bootstrap", n_resample=1000):
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
        wgt = len(res_p)

        data = []
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

                data.append(
                    tuple(list(pgm) + list(mgm) + [s2n_cut, 0 if ormask_cut else 1, -1, wgt])
                )

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

                data.append(
                    tuple(list(pgm) + list(mgm) + [s2n_cut, -1, mfrac_cut, wgt])
                )

        return data
    else:
        return None
