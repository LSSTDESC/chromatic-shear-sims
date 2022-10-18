"""
Utilities for running simulations
Some functions from https://github.com/beckermr/pizza-cutter-sims
"""

import copy

import yaml
import numpy as np

import fitsio
import galsim
import ngmix
import metadetect

import shear_bias_sims.image_lattice


ORMASK_CUTS = [True, False]
S2N_CUTS = [7, 8, 9, 10, 15, 20]
MFRAC_CUTS = [0, 1, 2, 5, 8, 10, 20, 50, 80, 100]


def _get_dtype():
    fkeys = ["g1p", "g1m", "g1", "g2p", "g2m", "g2"]
    ikeys = ["s2n_cut", "ormask_cut", "mfrac_cut"]
    dtype = []
    for key in fkeys:
        dtype.append((key, "f8"))
    for key in ikeys:
        dtype.append((key, "i4"))

    dtype.append(("weight", "f8"))

    return dtype


def _get_size(n_sims):
    return n_sims * len(S2N_CUTS) * (len(ORMASK_CUTS) + len(MFRAC_CUTS))


def _make_res_arrays(n_sims):
    dt = _get_dtype()
    n = _get_size(n_sims)
    return np.stack([np.zeros(n, dtype=dt), np.zeros(n, dtype=dt)], axis=-1)


def generate_arguments(config, galsim_config, rng, n, memmap_dict, logger):
    """
    Generate arguments for the measurement builder.
    """
    i = 0
    while i < n:
        seed = rng.integers(low=1, high=2**29)
        arg_dict = {
            "config": copy.deepcopy(config),
            "galsim_config": copy.deepcopy(galsim_config),
            "rng": np.random.default_rng(seed),
            "memmap_dict": memmap_dict,
            "idx": i,
            "logger": logger,
        }
        yield arg_dict
        i += 1


def observation_builder(config, rng, logger):
    """
    Build an ngmix Observation from a GalSim config dictionary.
    """
    # build the image
    image = galsim.config.BuildImage(config, logger=None)

    # build the PSF image
    psf_obj = galsim.config.BuildGSObject(config, "psf", logger=None)[0]
    psf_nx = 53  # N.B. this should must be odd as ngmix likes PSFs centered on a single pixel
    psf_ny = psf_nx
    psf = psf_obj.drawImage(nx=psf_nx, ny=psf_ny, scale=config["image"]["pixel_scale"])

    # build the weight image
    weight = galsim.Image(np.zeros((config["image_ysize"], config["image_xsize"])))
    galsim.config.noise.AddNoiseVariance(config, weight, logger=None)  # TODO: check if we need to do this with a "clean" config

    # build the noise image
    noise = galsim.Image(np.zeros((config["image_ysize"], config["image_xsize"])))
    galsim.config.noise.AddNoise(config, noise, logger=None)  # TODO: check if we need to do this with a "clean" config

    # build the bmask array
    # TODO: what is this anyways
    bmask = np.full(image.array.shape, int(0))

    # build the ormask array
    # TODO: what is this anyways
    ormask = np.full(image.array.shape, int(0))

    # construct the WCS
    # TODO: verify that this is the correct WCS to be using
    wcs = galsim.AffineTransform(
        config["pixel_scale"],
        0.0,
        0.0,
        config["pixel_scale"],
        origin=image.center,
    )

    # build the ngmix Observation of the PSF
    psf_cen = (psf.array.shape[0] - 1) / 2
    psf_jac = ngmix.jacobian.Jacobian(
        x=psf_cen,
        y=psf_cen,
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
    im_cen = (image.array.shape[0] - 1) / 2
    im_jac = ngmix.jacobian.Jacobian(
        x=im_cen,
        y=im_cen,
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

    return obs


def make_pair_config(config, g=0.02):
    """
    Create a pair of configs to simulate scenes sheared with equal and opposite
    shears for the noise bias cancellation algorithm.
    """
    config_p = copy.deepcopy(config)
    config_m = copy.deepcopy(config)

    config_p["stamp"]["shear"]["g1"] = g
    config_m["stamp"]["shear"]["g1"] = -g

    return config_p, config_m


def measurement_builder(config, galsim_config, rng, memmap_dict, idx, logger):
    """
    Build measurements of simulations and write to a memmap
    """
    cosmic_shear = config["shear"]["g"]
    config_p, config_m = make_pair_config(galsim_config, cosmic_shear)

    # TODO: multithread the p/m pieces in parallel?
    obs_p = observation_builder(config_p, rng, logger)
    obs_m = observation_builder(config_m, rng, logger)
    obslist_p = ngmix.ObsList()
    obslist_p.append(obs_p)
    obslist_m = ngmix.ObsList()
    obslist_m.append(obs_m)

    # TODO: this should really be grabbing an ObsList for each band
    mbobs_p = ngmix.MultiBandObsList()
    mbobs_p.append(obslist_p)
    mbobs_m = ngmix.MultiBandObsList()
    mbobs_m.append(obslist_m)

    # TODO: how do we handle all of the RNGs? when are they shared?
    mdet_seed = rng.integers(low=1, high=2**29)
    mdet_rng = np.random.default_rng(mdet_seed)

    res_p = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_p,
        mdet_rng,
    )

    res_m = metadetect.do_metadetect(
        config["metadetect"],
        mbobs_m,
        mdet_rng,
    )

    measurements = measure_pairs(config, res_p, res_m)

    # this pads the measurements with zeros to be the same size as expected by
    # the memmap
    full_measurements = _make_res_arrays(1)
    full_measurements[:len(measurements)] = measurements

    # TODO: a bit of a hack but works for now
    slice_length = _get_size(1)
    idx_start = idx * slice_length
    idx_stop = (idx + 1) * slice_length
    # logger.info(f"writing {memmap_dict['filename']}[{idx_start}:{idx_stop}]")
    memmap = np.memmap(**memmap_dict)
    memmap[idx_start:idx_stop] = full_measurements
    memmap.flush()

    return


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

def estimate_biases(meas_p, meas_m, calibration_shear, cosmic_shear, weights=None, method="bootstrap", n_resample=1000):
    """
    Estimate both additive and multiplicative biases with noise bias
    cancellation and bootstrapped standard deviations.
    """
    g1p = meas_p["g1"]
    R11p = (meas_p["g1p"] - meas_p["g1m"]) / (2 * calibration_shear)

    g1m = meas_m["g1"]
    R11m = (meas_m["g1p"] - meas_m["g1m"]) / (2 * calibration_shear)

    g2p = meas_p["g2"]
    R22p = (meas_p["g2p"] - meas_p["g2m"]) / (2 * calibration_shear)

    g2m = meas_m["g2"]
    R22m = (meas_m["g2p"] - meas_m["g2m"]) / (2 * calibration_shear)

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

    if method == "bootstrap":
        return _bootstrap(x1, y1, x2, y2, w, n_resample=n_resample)
    else:
        return _jackknife(x1, y1, x2, y2, w, n_resample=n_resample)


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

        # TODO: stack datap and datam into a single array of depth 2 here?
        dtype = _get_dtype()
        datap = []
        datam = []
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

                datap.append(
                    tuple(list(pgm) + [s2n_cut, 0 if ormask_cut else 1, -1, wgt])
                )
                datam.append(
                    tuple(list(mgm) + [s2n_cut, 0 if ormask_cut else 1, -1, wgt])
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

                datap.append(tuple(list(pgm) + [s2n_cut, -1, mfrac_cut, wgt]))
                datam.append(tuple(list(mgm) + [s2n_cut, -1, mfrac_cut, wgt]))

        return np.stack([np.array(datap, dtype=dtype), np.array(datam, dtype=dtype)], axis=-1)
    else:
        return np.stack([None, None], axis=-1)
