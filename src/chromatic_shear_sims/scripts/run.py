import argparse
import logging
import functools
import multiprocessing
import threading
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chromatic_shear_sims import utils
from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder

from . import log_util, name_util


def measure_sim(mbobs, psf_mbobs, measure):
    bands = mbobs.meta.get("bands")

    meas = measure.run(mbobs, psf_mbobs)
    batches = measure.to_batches(meas)

    return batches


def measure_sim_pair(mbobs_dict, psf_mbobs, measure):
    plus_mbobs = mbobs_dict["plus"]
    minus_mbobs = mbobs_dict["minus"]

    bands = plus_mbobs.meta.get("bands")

    meas_p = measure.run(plus_mbobs, psf_mbobs)
    batches_p = measure.to_batches(meas_p)

    meas_m = measure.run(minus_mbobs, psf_mbobs)
    batches_m = measure.to_batches(meas_m)

    batches_dict = {
        "plus": batches_p,
        "minus": batches_m,
    }

    return batches_dict


def task(simulation_builder, measure, seed):
    obs_dict, psf = simulation_builder.make_sim_pair(seed)
    all_batches = []
    # for psf_color in [0.5, 0.8, 1.1]:
    psf_colors = simulation_builder.config["measurement"].get("colors")
    for i, psf_color in enumerate(psf_colors):
        color_step = f"c{i}"
        # psf_obs = simulation_builder.make_psf_obs(psf, color=psf_color)
        # batches = measure_sim(obs, psf_obs, measure)
        # table = pa.Table.from_batches(batches)
        # seed_array = pa.array([seeds[i] for _ in range(table.num_rows)])
        # color_array = pa.array([psf_color for _ in range(table.num_rows)])
        # table = table.append_column("seed", seed_array)
        # table = table.append_column("color_step", color_array)

        psf_obs = simulation_builder.make_psf_obs(psf, color=psf_color)
        batches_dict = measure_sim_pair(obs_dict, psf_obs, measure)
        for shear_step, batches in batches_dict.items():
            for batch in batches:
                seed_array = pa.array([seed for _ in range(batch.num_rows)])
                shear_array = pa.array([shear_step for _ in range(batch.num_rows)])
                color_array = pa.array([color_step for _ in range(batch.num_rows)])
                batch = batch.append_column("seed", seed_array)
                batch = batch.append_column("shear_step", shear_array)
                batch = batch.append_column("color_step", color_array)
                all_batches.append(batch)

    return all_batches



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
    simulation_builder = SimulationBuilder.from_yaml(config_file)
    measure = measurement.get_measure(
        **simulation_builder.config["measurement"].get("builder"),
    )

    multiprocessing.set_start_method("spawn")

    queue = multiprocessing.Queue(-1)

    lp = threading.Thread(target=log_util.logger_thread, args=(queue,))
    lp.start()

    n_jobs = args.n_jobs
    n_sims = args.n_sims
    seed = args.seed
    seeds = utils.get_seeds(n_sims, seed=seed)

    output_path = name_util.get_output_path(args.output, args.config)
    os.makedirs(output_path, exist_ok=True)
    run_path = name_util.get_run_path(args.output, args.config, seed)

    schema = measure.schema
    schema = schema.append(
        pa.field("seed", pa.int64()),
    )
    schema = schema.append(
        pa.field("shear_step", pa.string()),
    )
    schema = schema.append(
        pa.field("color_step", pa.string()),
    )

    with pq.ParquetWriter(run_path, schema=schema) as writer:
        with multiprocessing.Pool(
            n_jobs,
            initializer=log_util.initializer,
            initargs=(queue, log_level),
            maxtasksperchild=max(1, n_sims // n_jobs),
        ) as pool:
            for i, batches in enumerate(
                pool.imap(
                    # simulation_builder.make_sim,
                    # simulation_builder.make_sim_pair,
                    functools.partial(task, simulation_builder, measure),
                    seeds,
                )
            ):
                print(f"finished simulation {i + 1}/{n_sims}")
                for batch in batches:
                    writer.write_batch(batch)

    queue.put(None)
    lp.join()

    print("simulations completed")
