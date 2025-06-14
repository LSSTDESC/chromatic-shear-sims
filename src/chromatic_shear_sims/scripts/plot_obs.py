import argparse
import logging
import multiprocessing
import threading

import matplotlib as mpl
import numpy as np

from chromatic_shear_sims import utils
from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder

from . import log_util, name_util, plot_util


def _apply_selection(meas, model):
    if model == "wmom":
        tcut = 1.2
    else:
        tcut = 0.5

    s2n_cut = 10
    t_ratio_cut = tcut
    # mfrac_cut = 10
    # s2n_cut = 0
    # t_ratio_cut = 0
    mfrac_cut = 0
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

    o = meas["noshear"]
    q = _mask(o)
    ns = o[q]
    return ns


def plot_sim(mbobs, psf_mbobs, measure=None):
    bands = mbobs.meta.get("bands")

    if measure is not None:
        model = measure.config.get("model")
        meas = measure.run(mbobs, psf_mbobs)
        meas = _apply_selection(
            meas,
            model,
        )

    fig, axs = plot_util.subplots(
        2, len(bands),
        sharex="row",
        sharey="row",
    )

    norm_field = mpl.colors.Normalize()

    for i, band in enumerate(bands):
        obs = mbobs[i][0]
        image = obs.image
        weight_image = obs.weight

        psf_obs = psf_mbobs[i][0]
        psf_image = psf_obs.image

        axs[0, i].imshow(psf_image, origin="lower")
        axs[0, i].axvline((psf_image.shape[0] - 1) / 2, c="k", ls=":")
        axs[0, i].axhline((psf_image.shape[1] - 1) / 2, c="k", ls=":")

        axs[1, i].imshow(np.arcsinh(image * np.sqrt(weight_image)), origin="lower", norm=norm_field)

        if measure is not None:
            for j in range(len(meas)):
                axs[1, i].text(meas["sx_col"][j], meas["sx_row"][j], round(meas[model + "_s2n"][j]), c="r", ha="left", va="bottom")

        axs[0, i].set_title(band)

    axs[0, 0].set_ylabel("PSF")
    axs[1, 0].set_ylabel("Wide Field")

    return fig, axs


def plot_sim_pair(mbobs_dict, psf_mbobs, measure=None):
    plus_mbobs = mbobs_dict["plus"]
    minus_mbobs = mbobs_dict["minus"]

    bands = plus_mbobs.meta.get("bands")

    if measure is not None:
        model = measure.config.get("model")
        meas_p = measure.run(plus_mbobs, psf_mbobs)
        meas_p = _apply_selection(
            meas_p,
            model,
        )
        meas_m = measure.run(minus_mbobs, psf_mbobs)
        meas_m = _apply_selection(
            meas_m,
            model,
        )

    fig, axs = plot_util.subplots(
        3, len(bands),
        sharex="row",
        sharey="row",
    )

    norm_field = mpl.colors.Normalize()

    for i, band in enumerate(bands):
        plus_obs = plus_mbobs[i][0]
        plus_image = plus_obs.image
        plus_weight_image = plus_obs.weight

        minus_obs = minus_mbobs[i][0]
        minus_image = minus_obs.image
        minus_weight_image = minus_obs.weight

        psf_obs = psf_mbobs[i][0]
        psf_image = psf_obs.image

        axs[0, i].imshow(psf_image, origin="lower")
        axs[0, i].axvline((psf_image.shape[0] - 1) / 2, c="k", ls=":")
        axs[0, i].axhline((psf_image.shape[1] - 1) / 2, c="k", ls=":")

        axs[1, i].imshow(np.arcsinh(plus_image * np.sqrt(plus_weight_image)), origin="lower", norm=norm_field)

        axs[2, i].imshow(np.arcsinh(minus_image * np.sqrt(minus_weight_image)), origin="lower", norm=norm_field)

        if measure is not None:
            for j in range(len(meas_p)):
                axs[1, i].text(meas_p["sx_col"][j], meas_p["sx_row"][j], round(meas_p[model + "_s2n"][j]), c="r", ha="left", va="bottom")
            for j in range(len(meas_m)):
                axs[2, i].text(meas_m["sx_col"][j], meas_m["sx_row"][j], round(meas_m[model + "_s2n"][j]), c="r", ha="left", va="bottom")

        axs[0, i].set_title(band)

    axs[0, 0].set_ylabel("PSF")
    axs[1, 0].set_ylabel("plus")
    axs[2, 0].set_ylabel("minus")

    return fig, axs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
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
        "--n_sims",
        type=int,
        required=False,
        default=1,
        help="Number of sims to run [int; 1]"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of parallel jobs to run [int; 1]"
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="run detection",
    )
    parser.add_argument(
        "--pair",
        action="store_true",
        help="do shear pair",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    # parser.add_argument(
    #     "--display",
    #     action="store_true",
    #     help="display output",
    # )
    return parser.parse_args()


def main():
    args = get_args()
    log_level = log_util.get_level(args.log_level)
    logging.basicConfig(format=log_util.FORMAT, level=log_level)

    config_file = args.config
    config_name = name_util.get_config_name(config_file)
    simulation_builder = SimulationBuilder.from_yaml(config_file)
    if args.detect:
        measure = measurement.get_measure(
            **simulation_builder.config["measurement"].get("builder"),
        )
    else:
        measure = None

    context = multiprocessing.get_context("spawn")  # spawn is ~ fork-exec
    queue = context.Queue(-1)

    lp = threading.Thread(target=log_util.logger_thread, args=(queue,))
    lp.start()

    n_jobs = args.n_jobs
    n_sims = args.n_sims
    seed = args.seed
    seeds = utils.get_seeds(n_sims, seed=seed)

    if args.pair:
        sim_task = simulation_builder.make_sim_pair
        plot_task = plot_sim_pair
    else:
        sim_task = simulation_builder.make_sim
        plot_task = plot_sim

    with context.Pool(
        n_jobs,
        initializer=log_util.initializer,
        initargs=(queue, log_level),
        maxtasksperchild=max(1, n_sims // n_jobs),
    ) as pool:
        for i, (obs, psf) in enumerate(
            pool.imap(
                sim_task,
                seeds,
            )
        ):
            print(f"finished simulation {i + 1}/{n_sims}")
            psf_obs = simulation_builder.make_psf_obs(psf, color=0.8)
            fig, axs = plot_task(obs, psf_obs, measure=measure)

            figname = f"{config_name}-obs-{seeds[i]}.pdf"
            fig.savefig(figname)

    queue.put(None)
    lp.join()

    print("simulations completed")
