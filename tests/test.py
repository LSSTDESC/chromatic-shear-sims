import argparse
import logging
import functools
import multiprocessing
import os
import threading
import time

import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import yaml
from rich.progress import track

from chromatic_shear_sims import utils
from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder

from chromatic_shear_sims.scripts import log_util
from chromatic_shear_sims.scripts import (
    run as run_script,
    aggregate as aggregate_script,
    measure as measure_script,
)


multiprocessing.set_start_method("spawn")


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
        help="Number of parallel jobs to run [int; 1]"
    )
    parser.add_argument(
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    return parser.parse_args()


def task(
    config_file,
    seed=None,
    n_sims=1,
    n_resample=1_000,
    n_jobs=1,
    log_level=1,
):
    log_level = log_util.get_level(log_level)
    logging.basicConfig(format=log_util.FORMAT, level=log_level)

    simulation_builder = SimulationBuilder.from_yaml(config_file)
    measure = measurement.get_measure(
        **simulation_builder.config["measurement"].get("builder"),
    )

    queue = multiprocessing.Queue(-1)

    lp = threading.Thread(target=log_util.logger_thread, args=(queue,))
    lp.start()

    seeds = utils.get_seeds(n_sims, seed=seed)

    with open(config_file) as fp:
        config = yaml.safe_load(fp)


    dg = ngmix.metacal.DEFAULT_STEP

    psf_colors = simulation_builder.config["measurement"].get("colors")
    psf_color_indices = config["measurement"].get("color_indices")
    dc = (psf_colors[psf_color_indices[2]] - psf_colors[psf_color_indices[0]]) / 2.
    color = psf_colors[psf_color_indices[1]]
    # print(f"psf_colors: {[psf_colors[i] for i in psf_color_indices]}")

    shear_steps = ["plus", "minus"]
    color_steps = [f"c{i}" for i, psf_color in enumerate(psf_colors)]
    mdet_steps = ["noshear", "1p", "1m", "2p", "2m"]

    data = {}
    for shear_step in shear_steps:
        data[shear_step] = {}
        for color_step in color_steps:
            data[shear_step][color_step] = {}
            for mdet_step in mdet_steps:
                data[shear_step][color_step][mdet_step] = []

    _start_time = time.time()
    with multiprocessing.Pool(
        n_jobs,
        initializer=log_util.initializer,
        initargs=(queue, log_level),
        maxtasksperchild=max(1, n_sims // n_jobs),
    ) as pool:
        # for i, batches in enumerate(
        for i, tables in enumerate(
            pool.imap(
                functools.partial(run_script.task, simulation_builder, measure),
                seeds,
            )
        ):
            _end_time = time.time()
            _elapsed_time = _end_time - _start_time
            # print(f"finished simulation {i + 1}/{n_sims} [{_elapsed_time / 60 :.0f} minutes elapsed]")
            for shear_step, color_tables in tables.items():
                for color_step, mdet_tables in color_tables.items():
                    for mdet_step, mdet_table in mdet_tables.items():
                        data[shear_step][color_step][mdet_step].append(mdet_table)

    # print(f"aggregating data")

    predicate = (
        (pc.field("pgauss_flags") == 0)
        & (pc.field("pgauss_s2n") > 10)
        & (pc.field("pgauss_T_ratio") > 0.5)
    )

    pre_aggregates = []
    for shear_step in shear_steps:
        for color_step in color_steps:
            for mdet_step in mdet_steps:
                _aggregate = aggregate_script.pre_aggregate(data[shear_step][color_step][mdet_step], predicate, colors=psf_colors, color_indices=psf_color_indices)
                _aggregate = _aggregate.append_column("mdet_step", pa.array([mdet_step for _ in range(_aggregate.num_rows)]))
                _aggregate = _aggregate.append_column("color_step", pa.array([color_step for _ in range(_aggregate.num_rows)]))
                _aggregate = _aggregate.append_column("shear_step", pa.array([shear_step for _ in range(_aggregate.num_rows)]))
                pre_aggregates.append(_aggregate)

    # print(f"pivoting aggregates")
    aggregates = aggregate_script.post_aggregate(pre_aggregates)

    (m_mean, c_mean), (m_mean_c1, c_mean_c1), (m_mean_c2, c_mean_c2) = measure_script.task(aggregates, dg, dc, color, psf_color_indices)

    m_bootstrap = []
    c_bootstrap = []
    m_bootstrap_c1 = []
    c_bootstrap_c1 = []
    m_bootstrap_c2 = []
    c_bootstrap_c2 = []

    # print(f"aggregating results from {n_resample} bootstrap resamples...")
    with multiprocessing.Pool(
        n_jobs,
        initializer=log_util.initializer,
        initargs=(queue, log_level),
        maxtasksperchild=max(1, n_resample // n_jobs),
    ) as pool:
        results = pool.imap(
            functools.partial(measure_script.task, aggregates, dg, dc, color, psf_color_indices, True),
            utils.get_seeds(n_resample, seed=seed)
        )

        for i, res in track(enumerate(results), description="bootstrapping", total=n_resample):
            (_m_bootstrap, _c_bootstrap), (_m_bootstrap_c1, _c_bootstrap_c1), (_m_bootstrap_c2, _c_bootstrap_c2) = res

            m_bootstrap.append(_m_bootstrap)
            c_bootstrap.append(_c_bootstrap)
            m_bootstrap_c1.append(_m_bootstrap_c1)
            c_bootstrap_c1.append(_c_bootstrap_c1)
            m_bootstrap_c2.append(_m_bootstrap_c2)
            c_bootstrap_c2.append(_c_bootstrap_c2)


    queue.put(None)
    lp.join()

    # print(f"finished processing bootstrap resamples")

    m_bootstrap = np.array(m_bootstrap)
    c_bootstrap = np.array(c_bootstrap)
    m_bootstrap_c1 = np.array(m_bootstrap_c1)
    c_bootstrap_c1 = np.array(c_bootstrap_c1)
    m_bootstrap_c2 = np.array(m_bootstrap_c2)
    c_bootstrap_c2 = np.array(c_bootstrap_c2)

    m_error = np.nanstd(m_bootstrap)
    c_error = np.nanstd(c_bootstrap)
    m_error_c1 = np.nanstd(m_bootstrap_c1)
    c_error_c1 = np.nanstd(c_bootstrap_c1)
    m_error_c2 = np.nanstd(m_bootstrap_c2)
    c_error_c2 = np.nanstd(c_bootstrap_c2)

    return (
        (m_mean, m_error),
        (c_mean, c_error),
        (m_mean_c1, m_error_c1),
        (c_mean_c1, c_error_c1),
        (m_mean_c2, m_error_c2),
        (c_mean_c2, c_error_c2),
    )


# def test_diffsky_btsettl_achromatic():
# def test_diffsky_btsettl_chromatic():
# def test_diffsky_simple_achromatic():
# def test_diffsky_simple_chromatic():
# def test_simple_btsettl_achromatic():
# def test_simple_btsettl_chromatic():


def test_simple_simple_achromatic():
    _config_file = os.path.join(
        os.path.dirname(__file__),
        "simple-simple-achromatic.yaml",
    )

    (
        (m_mean, m_error),
        (c_mean, c_error),
        (m_mean_c1, m_error_c1),
        (c_mean_c1, c_error_c1),
        (m_mean_c2, m_error_c2),
        (c_mean_c2, c_error_c2),
    ) = task(
        _config_file,
        n_sims=8,
        n_jobs=4,
    )
    assert np.abs(m_mean - 4e-4) < (m_error * 3)


def test_simple_simple_chromatic():
    _config_file = os.path.join(
        os.path.dirname(__file__),
        "simple-simple-chromatic.yaml",
    )

    (
        (m_mean, m_error),
        (c_mean, c_error),
        (m_mean_c1, m_error_c1),
        (c_mean_c1, c_error_c1),
        (m_mean_c2, m_error_c2),
        (c_mean_c2, c_error_c2),
    ) = task(
        _config_file,
        n_sims=8,
        n_jobs=4,
    )
    assert np.abs(m_mean - 4e-4) < (m_error * 3)


def main():
    args = get_args()

    (
        (m_mean, m_error),
        (c_mean, c_error),
        (m_mean_c1, m_error_c1),
        (c_mean_c1, c_error_c1),
        (m_mean_c2, m_error_c2),
        (c_mean_c2, c_error_c2),
    ) = task(
        args.config,
        seed=args.seed,
        n_sims=args.n_sims,
        n_resample=args.n_resample,
        n_jobs=args.n_jobs,
        log_level=args.log_level,
    )

    print(f"mdet (0): m = {m_mean:+0.3e} +/- {m_error * 3:0.3e} [3-sigma], c = {c_mean:+0.3e} +/- {c_error * 3:0.3e} [3-sigma]")
    print(f"drdc (1): m = {m_mean_c1:+0.3e} +/- {m_error_c1 * 3:0.3e} [3-sigma], c = {c_mean_c1:+0.3e} +/- {c_error_c1 * 3:0.3e} [3-sigma]")
    print(f"drdc (2): m = {m_mean_c2:+0.3e} +/- {m_error_c2 * 3:0.3e} [3-sigma], c = {c_mean_c2:+0.3e} +/- {c_error_c2 * 3:0.3e} [3-sigma]")


if __name__ == "__main__":
    main()
