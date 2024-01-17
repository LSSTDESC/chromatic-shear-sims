import argparse

import ngmix
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pyarrow.compute as pc
import pyarrow.dataset as ds
import tqdm

from chromatic_shear_bias.pipeline.pipeline import Pipeline


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}


# def f2c(f_g, f_i, zp_g=28.38, zp_i=27.85):
#     return -2.5 * np.log10(np.divide(f_g, f_i)) + zp_g - zp_i


# def _mask(data, s2n_cut, ormask_cut, mfrac_cut):
#     model = "wmom"
#     t_ratio_cut = 1.2
#     if "flags" in data.dtype.names:
#         flag_col = "flags"
#     else:
#         flag_col = model + "_flags"
# 
#     _cut_msk = (
#         (data[flag_col] == 0)
#         & (data[model + "_s2n"] > s2n_cut)
#         & (data[model + "_T_ratio"] > t_ratio_cut)
#     )
#     if ormask_cut:
#         _cut_msk = _cut_msk & (data["ormask"] == 0)
#     if mfrac_cut is not None:
#         _cut_msk = _cut_msk & (data["mfrac"] <= mfrac_cut)
#     return _cut_msk


def compute_e(batch):
    # NOSHEAR
    p_c1_g1_ns = np.nanmean(batch["p.c1.g1"])
    p_c1_g2_ns = np.nanmean(batch["p.c1.g2"])

    m_c1_g1_ns = np.nanmean(batch["m.c1.g1"])
    m_c1_g2_ns = np.nanmean(batch["m.c1.g2"])

    return np.array([p_c1_g1_ns, p_c1_g2_ns]), np.array([m_c1_g1_ns, m_c1_g2_ns])


def compute_R(batch, dg):

    p_c1_R11 = np.nanmean(pc.divide(pc.subtract(batch["p.c1.g1p"], batch["p.c1.g1m"]), 2 * dg))
    p_c1_R22 = np.nanmean(pc.divide(pc.subtract(batch["p.c1.g2p"], batch["p.c1.g2m"]), 2 * dg))

    m_c1_R11 = np.nanmean(pc.divide(pc.subtract(batch["m.c1.g1p"], batch["m.c1.g1m"]), 2 * dg))
    m_c1_R22 = np.nanmean(pc.divide(pc.subtract(batch["m.c1.g2p"], batch["m.c1.g2m"]), 2 * dg))

    return np.array([[p_c1_R11, 0], [0, p_c1_R22]]), np.array([[m_c1_R11, 0], [0, m_c1_R22]])


def compute_dRdc(batch, dg, dc, alt=False):
    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    p_c0_g1_ns = np.nanmean(batch["p.c0.g1"])
    p_c0_g2_ns = np.nanmean(batch["p.c0.g2"])
    p_c0_cg1_ns = np.nanmean(batch["p.c0.cg1"])
    p_c0_cg2_ns = np.nanmean(batch["p.c0.cg2"])

    # 1p
    p_c0_g1_p = np.nanmean(batch["p.c0.g1p"])
    p_c0_cg1_p = np.nanmean(batch["p.c0.cg1p"])

    # 1m
    p_c0_g1_m = np.nanmean(batch["p.c0.g1m"])
    p_c0_cg1_m = np.nanmean(batch["p.c0.cg1m"])

    # 2p
    p_c0_g2_p = np.nanmean(batch["p.c0.g2p"])
    p_c0_cg2_p = np.nanmean(batch["p.c0.cg2p"])

    # 2m
    p_c0_g2_m = np.nanmean(batch["p.c0.g2m"])
    p_c0_cg2_m = np.nanmean(batch["p.c0.cg2m"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    p_c1_g1_ns = np.nanmean(batch["p.c1.g1"])
    p_c1_g2_ns = np.nanmean(batch["p.c1.g2"])
    p_c1_cg1_ns = np.nanmean(batch["p.c1.cg1"])
    p_c1_cg2_ns = np.nanmean(batch["p.c1.cg2"])

    # 1p
    p_c1_g1_p = np.nanmean(batch["p.c1.g1p"])
    p_c1_cg1_p = np.nanmean(batch["p.c1.cg1p"])

    # 1m
    p_c1_g1_m = np.nanmean(batch["p.c1.g1m"])
    p_c1_cg1_m = np.nanmean(batch["p.c1.cg1m"])

    # 2p
    p_c1_g2_p = np.nanmean(batch["p.c1.g2p"])
    p_c1_cg2_p = np.nanmean(batch["p.c1.cg2p"])

    # 2m
    p_c1_g2_m = np.nanmean(batch["p.c1.g2m"])
    p_c1_cg2_m = np.nanmean(batch["p.c1.cg2m"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    p_c2_g1_ns = np.nanmean(batch["p.c2.g1"])
    p_c2_g2_ns = np.nanmean(batch["p.c2.g2"])
    p_c2_cg1_ns = np.nanmean(batch["p.c2.cg1"])
    p_c2_cg2_ns = np.nanmean(batch["p.c2.cg2"])

    # 1p
    p_c2_g1_p = np.nanmean(batch["p.c2.g1p"])
    p_c2_cg1_p = np.nanmean(batch["p.c2.cg1p"])

    # 1m
    p_c2_g1_m = np.nanmean(batch["p.c2.g1m"])
    p_c2_cg1_m = np.nanmean(batch["p.c2.cg1m"])

    # 2p
    p_c2_g2_p = np.nanmean(batch["p.c2.g2p"])
    p_c2_cg2_p = np.nanmean(batch["p.c2.cg2p"])

    # 2m
    p_c2_g2_m = np.nanmean(batch["p.c2.g2m"])
    p_c2_cg2_m = np.nanmean(batch["p.c2.cg2m"])

    #--------------------------------------------------------------------------

    if not alt:
        return (
            np.array(
                [[(p_c2_cg1_p - p_c2_cg1_m - p_c0_cg1_p + p_c0_cg1_m) / (2 * dg * 2 * dc), 0],
                 [0, (p_c2_cg2_p - p_c2_cg2_m - p_c0_cg2_p + p_c0_cg2_m) / (2 * dg * 2 * dc)]]
            ),
            np.array(
                [[(m_c2_cg1_p - m_c2_cg1_m - m_c0_cg1_p + m_c0_cg1_m) / (2 * dg * 2 * dc), 0],
                 [0, (m_c2_cg2_p - m_c2_cg2_m - m_c0_cg2_p + m_c0_cg2_m) / (2 * dg * 2 * dc)]]
            )
        )
    else:
        # more efficient formula from https://en.wikipedia.org/wiki/Finite_difference
        return (
            np.array(
                [[(p_c2_cg1_p - p_c1_cg1_p - p_c2_cg1_ns + 2 * p_c1_cg1_ns - p_c1_cg1_m - p_c0_cg1_ns + p_c0_cg1_m) / (2 * dg * dc), 0],
                 [0, (p_c2_cg2_p - p_c1_cg2_p - p_c2_cg2_ns + 2 * p_c1_cg2_ns - p_c1_cg2_m - p_c0_cg2_ns + p_c0_cg2_m / (2 * dg * dc))]]
            ),
            np.array(
                [[(m_c2_cg1_p - m_c1_cg1_p - m_c2_cg1_ns + 2 * m_c1_cg1_ns - m_c1_cg1_m - m_c0_cg1_ns + m_c0_cg1_m) / (2 * dg * dc), 0],
                 [0, (m_c2_cg2_p - m_c1_cg2_p - m_c2_cg2_ns + 2 * m_c1_cg2_ns - m_c1_cg2_m - m_c0_cg2_ns + m_c0_cg2_m / (2 * dg * dc))]]
            )
        )


def compute_m(batch, dg, dc):
    e_p, e_m = compute_e(batch)

    R_p, R_m = compute_R(batch, dg)

    return (e_p[0] - e_m[0]) / (R_p[0, 0] + R_m[0, 0]) / 0.02 - 1


def compute_m_chromatic(batch, dg, dc):
    e_p, e_m = compute_e(batch)

    R_p, R_m = compute_R(batch, dg)

    dRdc_p, dRdc_m = compute_dRdc(batch, dg, dc)

    return (e_p[0] - e_m[0]) / (R_p[0, 0] + dRdc_p[0, 0] + R_m[0, 0] + dRdc_m[0, 0]) / 0.02 - 1
    # return (
    #     (np.linalg.inv(R_p + dRdc_p) @ e_p)[0] / 2
    #     - (np.linalg.inv(R_m + dRdc_m) @ e_m)[0] / 2
    # ) / 0.02 - 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed [int]",
    )
    parser.add_argument(
        "--s2n-cut", type=int, default=10,
        help="Signal/noise cut [int; 10]",
    )
    parser.add_argument(
        "--ormask-cut", type=int, default=None,
        help="Cut to make on ormask. 0 indicates make a cut, 1 indicates no cut.",
    )
    parser.add_argument(
        "--mfrac-cut", type=int, default=None,
        help="Cut to make on mfrac. Given in percentages and comma separated. Cut keeps all objects less than the given value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n_resample",
        type=int,
        required=False,
        default=1000,
        help="Number of resample iterations"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of jobs to run [int; 1]",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = args.config
    seed = args.seed
    n_jobs = args.n_jobs

    pipeline = Pipeline(config)
    print("pipeline:", pipeline.name)
    print("seed:", seed)

    rng = np.random.default_rng(seed)

    measure_config = pipeline.config.get("measure")
    measure_type = measure_config.get("type")
    print("measure:", measure_type)

    if measure_type in CHROMATIC_MEASURES:
        colors_type = measure_config.get("colors")
        match colors_type:
            case "quantiles":
                chroma_colors = pipeline.galaxies.aggregate.get("quantiles")
            case "uniform":
                colors_min = pipeline.galaxies.aggregate.get("min_color")[0]
                colors_max = pipeline.galaxies.aggregate.get("max_color")[0]
                # TODO add config for number of colors here...
                chroma_colors = np.linspace(colors_min, colors_max, 3)
            case "centered":
                median = pipeline.galaxies.aggregate.get("median_color")
                chroma_colors = [median - 0.1, median, median + 0.1]
            case _:
                raise ValueError(f"Colors type {colors_type} not valid!")

    dg = ngmix.metacal.DEFAULT_STEP
    dc = (chroma_colors[2] - chroma_colors[0]) / 2.

    dataset = ds.dataset(args.output, format="arrow")
    predicate = \
        pc.field("s2n_cut") == args.s2n_cut
    if args.ormask_cut is not None:
        predicate &= (pc.field("ormask_cut") == args.ormask_cut)
    else:
        predicate &= (pc.field("ormask_cut") == -1)
    if args.mfrac_cut is not None:
        predicate &= (pc.field("mfrac_cut") == args.mfrac_cut)
    else:
        predicate &= (pc.field("mfrac_cut") == -1)

    batches = dataset.to_batches(filter=predicate)
    jobs = [
        joblib.delayed(compute_m)(batch, dg, dc)
        for batch in batches
    ]

    with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        m_chunks = parallel(jobs)

    m_bootstrap = []
    for i in tqdm.trange(args.n_resample, ncols=80):
        m_resample = rng.choice(m_chunks, size=len(m_chunks), replace=True)
        m_bootstrap.append(m_resample)
    m_bootstrap = np.array(m_bootstrap)

    m_mean = np.mean(m_bootstrap)
    m_std = np.std(m_bootstrap)

    print("m = %0.3e +/- %0.3e [3-sigma]" % (m_mean, m_std * 3))

    rng = np.random.default_rng(1)

    batches = dataset.to_batches(filter=predicate)
    jobs = [
        joblib.delayed(compute_m_chromatic)(batch, dg, dc)
        for batch in batches
    ]

    with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
        m_chunks_chroma = parallel(jobs)

    m_bootstrap_chroma = []
    for i in tqdm.trange(args.n_resample, ncols=80):
        m_resample_chroma = rng.choice(m_chunks_chroma, size=len(m_chunks_chroma), replace=True)
        m_bootstrap_chroma.append(m_resample_chroma)
    m_bootstrap_chroma = np.array(m_bootstrap_chroma)

    m_mean_chroma = np.mean(m_bootstrap_chroma)
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
