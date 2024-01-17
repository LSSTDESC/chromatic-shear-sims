from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import fitsio as fits
import tqdm
import joblib


def f2c(f_g, f_i, zp_g=28.38, zp_i=27.85):
    return -2.5 * np.log10(np.divide(f_g, f_i)) + zp_g - zp_i


def _mask(data, s2n_cut, ormask_cut, mfrac_cut):
    model = "wmom"
    t_ratio_cut = 1.2
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


def compute_e(infile, s2n_cut=10, ormask_cut=-1, mfrac_cut=100, resample=False, seed=None):
    f = fits.FITS(infile)

    d = f["co-noshear"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d = rng.choice(d, len(d), replace=True)
    mask = _mask(d, s2n_cut, ormask_cut, mfrac_cut)

    # NOSHEAR
    g = d["wmom_g"][mask]
    g1 = g[:, 0]
    g2 = g[:, 1]

    return np.array([np.nanmean(g1), np.nanmean(g2)])


def compute_R(infile, dg, s2n_cut=10, ormask_cut=-1, mfrac_cut=100, resample=False, seed=None):
    f = fits.FITS(infile)

    # 1p
    d_1p = f["co-1p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1p = rng.choice(d_1p, len(d_1p), replace=True)
    mask_1p = _mask(d_1p, s2n_cut, ormask_cut, mfrac_cut)
    g_1p = d_1p["wmom_g"][mask_1p]
    g1_1p = g_1p[:, 0]
    g2_1p = g_1p[:, 1]

    # 1m
    d_1m = f["co-1m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1m = rng.choice(d_1m, len(d_1m), replace=True)
    mask_1m = _mask(d_1m, s2n_cut, ormask_cut, mfrac_cut)
    g_1m = d_1m["wmom_g"][mask_1m]
    g1_1m = g_1m[:, 0]
    g2_1m = g_1m[:, 1]

    # 2p
    d_2p = f["co-2p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2p = rng.choice(d_2p, len(d_2p), replace=True)
    mask_2p = _mask(d_2p, s2n_cut, ormask_cut, mfrac_cut)
    g_2p = d_2p["wmom_g"][mask_2p]
    g1_2p = g_2p[:, 0]
    g2_2p = g_2p[:, 1]

    # 2m
    d_2m = f["co-2m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2m = rng.choice(d_2m, len(d_2m), replace=True)
    mask_2m = _mask(d_2m, s2n_cut, ormask_cut, mfrac_cut)
    g_2m = d_2m["wmom_g"][mask_2m]
    g1_2m = g_2m[:, 0]
    g2_2m = g_2m[:, 1]

    return np.array(
        [
            [
                (np.nanmean(g1_1p) - np.nanmean(g1_1m)) / (2 * dg),
                (np.nanmean(g2_1p) - np.nanmean(g2_1m)) / (2 * dg),
            ],
            [
                (np.nanmean(g1_2p) - np.nanmean(g1_2m)) / (2 * dg),
                (np.nanmean(g2_2p) - np.nanmean(g2_2m)) / (2 * dg),
            ],
        ]
    )


def compute_dRdc(infile, dg, dc, s2n_cut=10, ormask_cut=-1, mfrac_cut=100, resample=False, seed=None, alt=False):
    f = fits.FITS(infile)

    # c0

    ## NOSHEAR
    d_co = f["co-noshear"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_co = rng.choice(d_co, len(d_co), replace=True)
    mask_co = _mask(d_co, s2n_cut, ormask_cut, mfrac_cut)
    g_co = d_co["wmom_g"][mask_co]
    g1_co = g_co[:, 0]
    g2_co = g_co[:, 1]

    fl_co = d_co["wmom_band_flux"][mask_co]
    # c_co = -2.5 * np.log10(np.divide(fl_co[:, 0], fl_co[:, 2]))
    c_co = f2c(fl_co[:, 0], fl_co[:, 2])

    ## 1p
    d_1p_co = f["co-1p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1p_co = rng.choice(d_1p_co, len(d_1p_co), replace=True)
    mask_1p_co = _mask(d_1p_co, s2n_cut, ormask_cut, mfrac_cut)
    g_1p_co = d_1p_co["wmom_g"][mask_1p_co]
    g1_1p_co = g_1p_co[:, 0]
    g2_1p_co = g_1p_co[:, 1]

    fl_1p_co = d_1p_co["wmom_band_flux"][mask_1p_co]
    # c_1p_co = -2.5 * np.log10(np.divide(fl_1p_co[:, 0], fl_1p_co[:, 2]))
    c_1p_co = f2c(fl_1p_co[:, 0], fl_1p_co[:, 2])

    ## 1m
    d_1m_co = f["co-1m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1m_co = rng.choice(d_1m_co, len(d_1m_co), replace=True)
    mask_1m_co = _mask(d_1m_co, s2n_cut, ormask_cut, mfrac_cut)
    g_1m_co = d_1m_co["wmom_g"][mask_1m_co]
    g1_1m_co = g_1m_co[:, 0]
    g2_1m_co = g_1m_co[:, 1]

    fl_1m_co = d_1m_co["wmom_band_flux"][mask_1m_co]
    # c_1m_co = -2.5 * np.log10(np.divide(fl_1m_co[:, 0], fl_1m_co[:, 2]))
    c_1m_co = f2c(fl_1m_co[:, 0], fl_1m_co[:, 2])

    ## 2p
    d_2p_co = f["co-2p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2p_co = rng.choice(d_2p_co, len(d_2p_co), replace=True)
    mask_2p_co = _mask(d_2p_co, s2n_cut, ormask_cut, mfrac_cut)
    g_2p_co = d_2p_co["wmom_g"][mask_2p_co]
    g1_2p_co = g_2p_co[:, 0]
    g2_2p_co = g_2p_co[:, 1]

    fl_2p_co = d_2p_co["wmom_band_flux"][mask_2p_co]
    # c_2p_co = -2.5 * np.log10(np.divide(fl_2p_co[:, 0], fl_2p_co[:, 2]))
    c_2p_co = f2c(fl_2p_co[:, 0], fl_2p_co[:, 2])

    ## 2m
    d_2m_co = f["co-2m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2m_co = rng.choice(d_2m_co, len(d_2m_co), replace=True)
    mask_2m_co = _mask(d_2m_co, s2n_cut, ormask_cut, mfrac_cut)
    g_2m_co = d_2m_co["wmom_g"][mask_2m_co]
    g1_2m_co = g_2m_co[:, 0]
    g2_2m_co = g_2m_co[:, 1]

    fl_2m_co = d_2m_co["wmom_band_flux"][mask_2m_co]
    # c_2m_co = -2.5 * np.log10(np.divide(fl_2m_co[:, 0], fl_2m_co[:, 2]))
    c_2m_co = f2c(fl_2m_co[:, 0], fl_2m_co[:, 2])

    #--------------------------------------------------------------------------

    # cp

    ## NOSHEAR
    d_cp = f["cp-noshear"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_cp = rng.choice(d_cp, len(d_cp), replace=True)
    mask_cp = _mask(d_cp, s2n_cut, ormask_cut, mfrac_cut)
    g_cp = d_cp["wmom_g"][mask_cp]
    g1_cp = g_cp[:, 0]
    g2_cp = g_cp[:, 1]

    fl_cp = d_cp["wmom_band_flux"][mask_cp]
    # c_cp = -2.5 * np.log10(np.divide(fl_cp[:, 0], fl_cp[:, 2]))
    c_cp = f2c(fl_cp[:, 0], fl_cp[:, 2])

    ## 1p
    d_1p_cp = f["cp-1p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1p_cp = rng.choice(d_1p_cp, len(d_1p_cp), replace=True)
    mask_1p_cp = _mask(d_1p_cp, s2n_cut, ormask_cut, mfrac_cut)
    g_1p_cp = d_1p_cp["wmom_g"][mask_1p_cp]
    g1_1p_cp = g_1p_cp[:, 0]
    g2_1p_cp = g_1p_cp[:, 1]

    fl_1p_cp = d_1p_cp["wmom_band_flux"][mask_1p_cp]
    # c_1p_cp = -2.5 * np.log10(np.divide(fl_1p_cp[:, 0], fl_1p_cp[:, 2]))
    c_1p_cp = f2c(fl_1p_cp[:, 0], fl_1p_cp[:, 2])

    ## 1m
    d_1m_cp = f["cp-1m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1m_cp = rng.choice(d_1m_cp, len(d_1m_cp), replace=True)
    mask_1m_cp = _mask(d_1m_cp, s2n_cut, ormask_cut, mfrac_cut)
    g_1m_cp = d_1m_cp["wmom_g"][mask_1m_cp]
    g1_1m_cp = g_1m_cp[:, 0]
    g2_1m_cp = g_1m_cp[:, 1]

    fl_1m_cp = d_1m_cp["wmom_band_flux"][mask_1m_cp]
    # c_1m_cp = -2.5 * np.log10(np.divide(fl_1m_cp[:, 0], fl_1m_cp[:, 2]))
    c_1m_cp = f2c(fl_1m_cp[:, 0], fl_1m_cp[:, 2])

    ## 2p
    d_2p_cp = f["cp-2p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2p_cp = rng.choice(d_2p_cp, len(d_2p_cp), replace=True)
    mask_2p_cp = _mask(d_2p_cp, s2n_cut, ormask_cut, mfrac_cut)
    g_2p_cp = d_2p_cp["wmom_g"][mask_2p_cp]
    g1_2p_cp = g_2p_cp[:, 0]
    g2_2p_cp = g_2p_cp[:, 1]

    fl_2p_cp = d_2p_cp["wmom_band_flux"][mask_2p_cp]
    # c_2p_cp = -2.5 * np.log10(np.divide(fl_2p_cp[:, 0], fl_2p_cp[:, 2]))
    c_2p_cp = f2c(fl_2p_cp[:, 0], fl_2p_cp[:, 2])

    ## 2m
    d_2m_cp = f["cp-2m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2m_cp = rng.choice(d_2m_cp, len(d_2m_cp), replace=True)
    mask_2m_cp = _mask(d_2m_cp, s2n_cut, ormask_cut, mfrac_cut)
    g_2m_cp = d_2m_cp["wmom_g"][mask_2m_cp]
    g1_2m_cp = g_2m_cp[:, 0]
    g2_2m_cp = g_2m_cp[:, 1]

    fl_2m_cp = d_2m_cp["wmom_band_flux"][mask_2m_cp]
    # c_2m_cp = -2.5 * np.log10(np.divide(fl_2m_cp[:, 0], fl_2m_cp[:, 2]))
    c_2m_cp = f2c(fl_2m_cp[:, 0], fl_2m_cp[:, 2])

    #--------------------------------------------------------------------------

    # cm

    ## NOSHEAR
    d_cm = f["cm-noshear"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_cm = rng.choice(d_cm, len(d_cm), replace=True)
    mask_cm = _mask(d_cm, s2n_cut, ormask_cut, mfrac_cut)
    g_cm = d_cm["wmom_g"][mask_cm]
    g1_cm = g_cm[:, 0]
    g2_cm = g_cm[:, 1]

    fl_cm = d_cm["wmom_band_flux"][mask_cm]
    # c_cm = -2.5 * np.log10(np.divide(fl_cm[:, 0], fl_cm[:, 2]))
    c_cm = f2c(fl_cm[:, 0], fl_cm[:, 2])

    ## 1p
    d_1p_cm = f["cm-1p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1p_cm = rng.choice(d_1p_cm, len(d_1p_cm), replace=True)
    mask_1p_cm = _mask(d_1p_cm, s2n_cut, ormask_cut, mfrac_cut)
    g_1p_cm = d_1p_cm["wmom_g"][mask_1p_cm]
    g1_1p_cm = g_1p_cm[:, 0]
    g2_1p_cm = g_1p_cm[:, 1]

    fl_1p_cm = d_1p_cm["wmom_band_flux"][mask_1p_cm]
    # c_1p_cm = -2.5 * np.log10(np.divide(fl_1p_cm[:, 0], fl_1p_cm[:, 2]))
    c_1p_cm = f2c(fl_1p_cm[:, 0], fl_1p_cm[:, 2])

    ## 1m
    d_1m_cm = f["cm-1m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_1m_cm = rng.choice(d_1m_cm, len(d_1m_cm), replace=True)
    mask_1m_cm = _mask(d_1m_cm, s2n_cut, ormask_cut, mfrac_cut)
    g_1m_cm = d_1m_cm["wmom_g"][mask_1m_cm]
    g1_1m_cm = g_1m_cm[:, 0]
    g2_1m_cm = g_1m_cm[:, 1]

    fl_1m_cm = d_1m_cm["wmom_band_flux"][mask_1m_cm]
    # c_1m_cm = -2.5 * np.log10(np.divide(fl_1m_cm[:, 0], fl_1m_cm[:, 2]))
    c_1m_cm = f2c(fl_1m_cm[:, 0], fl_1m_cm[:, 2])

    ## 2p
    d_2p_cm = f["cm-2p"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2p_cm = rng.choice(d_2p_cm, len(d_2p_cm), replace=True)
    mask_2p_cm = _mask(d_2p_cm, s2n_cut, ormask_cut, mfrac_cut)
    g_2p_cm = d_2p_cm["wmom_g"][mask_2p_cm]
    g1_2p_cm = g_2p_cm[:, 0]
    g2_2p_cm = g_2p_cm[:, 1]

    fl_2p_cm = d_2p_cm["wmom_band_flux"][mask_2p_cm]
    # c_2p_cm = -2.5 * np.log10(np.divide(fl_2p_cm[:, 0], fl_2p_cm[:, 2]))
    c_2p_cm = f2c(fl_2p_cm[:, 0], fl_2p_cm[:, 2])

    ## 2m
    d_2m_cm = f["cm-2m"].read()
    if resample:
        rng = np.random.default_rng(seed)
        d_2m_cm = rng.choice(d_2m_cm, len(d_2m_cm), replace=True)
    mask_2m_cm = _mask(d_2m_cm, s2n_cut, ormask_cut, mfrac_cut)
    g_2m_cm = d_2m_cm["wmom_g"][mask_2m_cm]
    g1_2m_cm = g_2m_cm[:, 0]
    g2_2m_cm = g_2m_cm[:, 1]

    fl_2m_cm = d_2m_cm["wmom_band_flux"][mask_2m_cm]
    # c_2m_cm = -2.5 * np.log10(np.divide(fl_2m_cm[:, 0], fl_2m_cm[:, 2]))
    c_2m_cm = f2c(fl_2m_cm[:, 0], fl_2m_cm[:, 2])

    #--------------------------------------------------------------------------

    if not alt:
        return np.array(
            [
                [
                    (np.nanmean(g1_1p_cp * (c_1p_cp - MEDIAN)) - np.nanmean(g1_1m_cp * (c_1m_cp - MEDIAN)) - np.nanmean(g1_1p_cm * (c_1p_cm - MEDIAN)) + np.nanmean(g1_1m_cm * (c_1m_cm - MEDIAN))) / (2 * dg * 2 * dc),
                    (np.nanmean(g2_1p_cp * (c_1p_cp - MEDIAN)) - np.nanmean(g2_1m_cp * (c_1m_cp - MEDIAN)) - np.nanmean(g2_1p_cm * (c_1p_cm - MEDIAN)) + np.nanmean(g2_1m_cm * (c_1m_cm - MEDIAN))) / (2 * dg * 2 * dc),
                ],
                [
                    (np.nanmean(g1_2p_cp * (c_2p_cp - MEDIAN)) - np.nanmean(g1_2m_cp * (c_2m_cp - MEDIAN)) - np.nanmean(g1_2p_cm * (c_2p_cm - MEDIAN)) + np.nanmean(g1_2m_cm * (c_2m_cm - MEDIAN))) / (2 * dg * 2 * dc),
                    (np.nanmean(g2_2p_cp * (c_2p_cp - MEDIAN)) - np.nanmean(g2_2m_cp * (c_2m_cp - MEDIAN)) - np.nanmean(g2_2p_cm * (c_2p_cm - MEDIAN)) + np.nanmean(g2_2m_cm * (c_2m_cm - MEDIAN))) / (2 * dg * 2 * dc),
                ],
            ]
        )

    else:
        # more efficient formula from https://en.wikipedia.org/wiki/Finite_difference
        return np.array(
            [
                [
                    (np.nanmean(g1_1p_cp * (c_1p_cp - MEDIAN)) - np.nanmean(g1_1p_co * (c_1p_co - MEDIAN)) - np.nanmean(g1_cp * (c_cp - MEDIAN)) + 2 * np.nanmean(g1_co * (c_co - MEDIAN)) - np.nanmean(g1_1m_co * (c_1m_co - MEDIAN)) - np.nanmean(g1_cm * (c_cm - MEDIAN)) + np.nanmean(g1_1m_cm * (c_1m_cm - MEDIAN))) / (2 * dg * dc),
                    (np.nanmean(g2_1p_cp * (c_1p_cp - MEDIAN)) - np.nanmean(g2_1p_co * (c_1p_co - MEDIAN)) - np.nanmean(g2_cp * (c_cp - MEDIAN)) + 2 * np.nanmean(g2_co * (c_co - MEDIAN)) - np.nanmean(g2_1m_co * (c_1m_co - MEDIAN)) - np.nanmean(g2_cm * (c_cm - MEDIAN)) + np.nanmean(g2_1m_cm * (c_1m_cm - MEDIAN))) / (2 * dg * dc),
                ],
                [
                    (np.nanmean(g1_2p_cp * (c_2p_cp - MEDIAN)) - np.nanmean(g1_2p_co * (c_2p_co - MEDIAN)) - np.nanmean(g1_cp * (c_cp - MEDIAN)) + 2 * np.nanmean(g1_co * (c_co - MEDIAN)) - np.nanmean(g1_2m_co * (c_2m_co - MEDIAN)) - np.nanmean(g1_cm * (c_cm - MEDIAN)) + np.nanmean(g1_2m_cm * (c_2m_cm - MEDIAN))) / (2 * dg * dc),
                    (np.nanmean(g2_2p_cp * (c_2p_cp - MEDIAN)) - np.nanmean(g2_2p_co * (c_2p_co - MEDIAN)) - np.nanmean(g2_cp * (c_cp - MEDIAN)) + 2 * np.nanmean(g2_co * (c_co - MEDIAN)) - np.nanmean(g2_2m_co * (c_2m_co - MEDIAN)) - np.nanmean(g2_cm * (c_cm - MEDIAN)) + np.nanmean(g2_2m_cm * (c_2m_cm - MEDIAN))) / (2 * dg * dc),
                ],
            ]
        )


def compute_m(pfile, mfile, dg, dc, resample=False, seed=None):
    e_p = compute_e(pfile, resample=resample, seed=seed)
    e_m = compute_e(mfile, resample=resample, seed=seed)

    R_p = compute_R(pfile, dg, resample=resample, seed=seed)
    R_m = compute_R(mfile, dg, resample=resample, seed=seed)

    # dRdc_p = compute_dRdc(fp)
    # dRdc_m = compute_dRdc(fm)

    return (e_p[0] - e_m[0]) / (R_p[0, 0] + R_m[0, 0]) / 0.02 - 1


def compute_m_chromatic(pfile, mfile, dg, dc, resample=False, seed=None):
    e_p = compute_e(pfile, resample=resample, seed=seed)
    e_m = compute_e(mfile, resample=resample, seed=seed)

    R_p = compute_R(pfile, dg, resample=resample, seed=seed)
    R_m = compute_R(mfile, dg, resample=resample, seed=seed)

    dRdc_p = compute_dRdc(pfile, dg, dc, resample=resample, seed=seed)
    dRdc_m = compute_dRdc(mfile, dg, dc, resample=resample, seed=seed)

    # return (e_p[0] - e_m[0]) / (R_p[0, 0] + dRdc_p[0, 0] + R_m[0, 0] + dRdc_m[0, 0]) / 0.02 - 1
    return (
        (np.linalg.inv(R_p + dRdc_p) @ e_p)[0] / 2
        - (np.linalg.inv(R_m + dRdc_m) @ e_m)[0] / 2
    ) / 0.02 - 1


if __name__ == "__main__":
    dg = 0.01
    dc = 0.5
    # dc = (1.5550565719604492 - 0.2306804656982422) / 2.
    path = Path("/pscratch/sd/s/smau/drdc/mag-20-many")
    pfile = next(path.glob("*plus.fits")).as_posix()
    mfile = next(path.glob("*minus.fits")).as_posix()


    rng = np.random.default_rng(1)

    jobs = [
        joblib.delayed(compute_m)(pfile, mfile, dg, dc, resample=True, seed=seed)
        for seed in rng.integers(1, 2**32, 100)
    ]

    with joblib.Parallel(n_jobs=32, backend="loky", verbose=10) as par:
        m_bootstrap = par(jobs)

    m_mean = compute_m(pfile, mfile, dg, dc, resample=False)
    m_std = np.std(m_bootstrap)

    print("m = %0.3e +/- %0.3e [3-sigma]" % (m_mean, m_std * 3))

    rng = np.random.default_rng(1)

    jobs = [
        joblib.delayed(compute_m_chromatic)(pfile, mfile, dg, dc, resample=True, seed=seed)
        for seed in rng.integers(1, 2**32, 100)
    ]

    with joblib.Parallel(n_jobs=4, backend="loky", verbose=10) as par:
        m_bootstrap_chroma = par(jobs)

    m_mean_chroma = compute_m_chromatic(pfile, mfile, dg, dc, resample=False)
    m_std_chroma = np.std(m_bootstrap_chroma)

    print("m = %0.3e +/- %0.3e [3-sigma]" % (m_mean_chroma, m_std_chroma * 3))

    m_req = 0.002
    plt.axvspan(-m_req, m_req, fc="k", alpha=0.1)
    plt.axvline(4e-4, c="k", alpha=0.1, ls="--")
    plt.hist(m_bootstrap, histtype="step", label="R", ec="k")
    plt.hist(m_bootstrap_chroma, histtype="step", label="R \& dR/dc", ec="b")
    plt.axvline(m_mean, c="k")
    plt.axvline(m_mean_chroma, c="b")
    plt.legend()
    plt.xlabel("$m$")
    plt.show()
