#!/usr/bin/env python

import argparse
import functools
import logging
import multiprocessing
import threading
import os

import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as ft
import pyarrow.parquet as pq
import yaml
from rich.progress import track

from chromatic_shear_sims import utils
# from chromatic_shear_sims.scripts import log_util, name_util, plot_util
# from chromatic_shear_sims.scripts import measure
from . import log_util, name_util, plot_util, measure



def task(aggregate_path, dg, color_index, resample=False, seed=None):
    aggregates = ft.read_table(aggregate_path)
    if resample:
        rng = np.random.default_rng(seed)
        resample_indices = rng.choice(len(aggregates), len(aggregates), replace=True)
        aggregates = aggregates.take(resample_indices)

    R_p, R_m = measure.compute_R_chromatic(aggregates, dg, color_index)
    return np.average([R_p, R_m], axis=0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "output",
        type=str,
        help="output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed [int]",
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
    parser.add_argument(
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    return parser.parse_args()


def main():
    args = get_args()

    log_level = log_util.get_level(args.log_level)
    logging.basicConfig(format=log_util.FORMAT, level=log_level)

    config_file = args.config
    seed = args.seed
    n_jobs = args.n_jobs
    n_resample = args.n_resample

    config_name = name_util.get_config_name(config_file)
    output_path = name_util.get_output_path(args.output, args.config)
    aggregate_path = name_util.get_aggregate_path(args.output, args.config)

    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)

    rng = np.random.default_rng(seed)

    with open(config_file) as fp:
        config = yaml.safe_load(fp)

    dg = ngmix.metacal.DEFAULT_STEP

    psf_colors = config["measurement"].get("colors")
    print(f"psf_colors: {psf_colors}")

    print(f"reading aggregates from {aggregate_path}")

    R_mean = {
        f"c{i}": None
        for i, color in enumerate(psf_colors)
    }
    for color_index, color in enumerate(psf_colors):
        color_key = f"c{color_index}"
        R_mean[color_key] = task(aggregate_path, dg, color_index)

    multiprocessing.set_start_method("spawn")
    queue = multiprocessing.Queue(-1)

    lp = threading.Thread(target=log_util.logger_thread, args=(queue,))
    lp.start()

    R_bootstrap = {
        f"c{i}": []
        for i, color in enumerate(psf_colors)
    }
    with multiprocessing.Pool(
        n_jobs,
        initializer=log_util.initializer,
        initargs=(queue, log_level),
        maxtasksperchild=max(1, n_resample // n_jobs),
        # context=ctx,
    ) as pool:

        for color_index, color in enumerate(psf_colors):
            color_key = f"c{color_index}"
            results = pool.imap(
                functools.partial(task, aggregate_path, dg, color_index, True),
                utils.get_seeds(n_resample, seed=seed)
            )

            for res in track(results, description="bootstrapping", total=n_resample):
                R_bootstrap[color_key].append(res)

    queue.put(None)
    lp.join()

    R_error = {}
    for _index, _bootstrap in R_bootstrap.items():
        R_bootstrap[_index] = np.array(_bootstrap)
        R_error[_index] = np.nanstd(R_bootstrap[_index], axis=0) * 3

    m_req = 2e-3

    fig, axs = plot_util.subplots(1, 1)

    axs.plot(
        psf_colors,
        [R[0, 0] for R in R_mean.values()],
        c="k",
    )
    # axs.errorbar(
    #     psf_colors,
    #     [R[0, 0] for R in R_mean.values()],
    #     [R[0, 0] for R in R_error.values()],
    # )
    axs.violinplot(
        [R[:, 0, 0] for R in R_bootstrap.values()],
        positions=psf_colors,
        widths=0.1,
    )
    axs.set_xlabel("$(g - i)_{PSF}$")
    axs.set_ylabel("$R_{11}$")

    # ax3 = fig.add_axes(
    #     divider.get_position(),
    #     axes_locator=divider.new_locator(nx=1, ny=3),
    #     sharex=ax1,
    # )
    # ax3.axvline(mean_color, ls="--", label="mean")
    # ax3.axvline(median_color, ls="-", label="median")
    # ax3.violinplot(
    #     all_R_11_bootstrap,
    #     positions=all_colors,
    #     widths=0.1,
    # )
    # ax3.set_ylabel("$R_{11}$")
    # ax3.xaxis.set_tick_params(labelbottom=False)

    # ax4 = fig.add_axes(
    #     divider.get_position(),
    #     axes_locator=divider.new_locator(nx=1, ny=4),
    #     sharex=ax1,
    # )
    # ax4.axvline(mean_color, ls="--", label="mean")
    # ax4.axvline(median_color, ls="-", label="median")
    # ax4.violinplot(
    #     all_color_p_bootstrap,
    #     positions=all_colors,
    #     widths=0.1,
    # )
    # ax4.violinplot(
    #     all_color_m_bootstrap,
    #     positions=all_colors,
    #     widths=0.1,
    # )
    # ax4.set_ylabel("$c$")
    # ax4.xaxis.set_tick_params(labelbottom=False)

    figname = f"{config_name}_drdc.pdf"
    fig.savefig(figname)


if __name__ == "__main__":
    main()
