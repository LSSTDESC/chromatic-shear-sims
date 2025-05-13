import argparse
import logging
import functools
import multiprocessing
import os
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from chromatic_shear_sims import utils
from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder

from . import log_util, name_util


logger = logging.getLogger(__name__)


def measure_sim(mbobs, psf_mbobs, measure):
    bands = mbobs.meta.get("bands")

    meas = measure.run(mbobs, psf_mbobs)
    # batches = measure.to_batches(meas)
    table_dict = measure.to_table_dict(meas)

    # return batches
    return table_dict


def measure_sim_pair(mbobs_dict, psf_mbobs, measure):
    plus_mbobs = mbobs_dict["plus"]
    minus_mbobs = mbobs_dict["minus"]

    bands = plus_mbobs.meta.get("bands")

    meas_p = measure.run(plus_mbobs, psf_mbobs)
    # batches_p = measure.to_batches(meas_p)
    # table_p = measure.to_table(meas_p)
    table_p_dict = measure.to_table_dict(meas_p)

    meas_m = measure.run(minus_mbobs, psf_mbobs)
    # batches_m = measure.to_batches(meas_m)
    # table_m = measure.to_table(meas_m)
    table_m_dict = measure.to_table_dict(meas_m)

    # batches_dict = {
    #     "plus": batches_p,
    #     "minus": batches_m,
    # }
    # table_dict = {
    #     "plus": table_p,
    #     "minus": table_m,
    # }
    table_dicts = {
        "plus": table_p_dict,
        "minus": table_m_dict,
    }

    # return batches_dict
    # return table_dict
    return table_dicts


def task(simulation_builder, measure, seed):
    obs_dict, psf = simulation_builder.make_sim_pair(seed)
    # all_batches = []
    # tables = []
    tables = {}
    psf_colors = simulation_builder.config["measurement"].get("colors")
    psf_obs_dict = {
        f"c{i}": simulation_builder.make_psf_obs(psf, color=psf_color)
        for i, psf_color in enumerate(psf_colors)
    }

    for shear_step, obs in obs_dict.items():
        tables[shear_step] = {}
        for color_step, psf_obs in psf_obs_dict.items():
            tables[shear_step][color_step] = {}

            # psf_obs = simulation_builder.make_psf_obs(psf, color=psf_color)
            # batches = measure_sim(obs, psf_obs, measure)
            # table = pa.Table.from_batches(batches)
            # seed_array = pa.array([seeds[i] for _ in range(table.num_rows)])
            # color_array = pa.array([psf_color for _ in range(table.num_rows)])
            # table = table.append_column("seed", seed_array)
            # table = table.append_column("color_step", color_array)

            # psf_obs = simulation_builder.make_psf_obs(psf, color=psf_color)
            # batches_dict = measure_sim_pair(obs_dict, psf_obs, measure)
            # table_dict = measure_sim_pair(obs_dict, psf_obs, measure)
            # table_dicts = measure_sim_pair(obs_dict, psf_obs, measure)
            table_dict = measure_sim(obs, psf_obs, measure)

            # for shear_step, batches in batches_dict.items():
            #     # for batch in batches:
            #     #     batch = batch.append_column("seed", seed_array)
            #     #     batch = batch.append_column("shear_step", shear_array)
            #     #     batch = batch.append_column("color_step", color_array)
            #     #     all_batches.append(batch)
            #     table = pa.Table.from_batches(batches)
            #     seed_array = pa.array([seed for _ in range(table.num_rows)])
            #     shear_array = pa.array([shear_step for _ in range(table.num_rows)])
            #     color_array = pa.array([color_step for _ in range(table.num_rows)])
            #     table = table.append_column("seed", seed_array)
            #     table = table.append_column("shear_step", shear_array)
            #     table = table.append_column("color_step", color_array)
            #     for batch in table.to_batches():
            #         all_batches.append(batch)

            # for shear_step, table in table_dict.items():
            #     seed_array = pa.array([seed for _ in range(table.num_rows)])
            #     shear_array = pa.array([shear_step for _ in range(table.num_rows)])
            #     color_array = pa.array([color_step for _ in range(table.num_rows)])
            #     table = table.append_column("seed", seed_array)
            #     table = table.append_column("shear_step", shear_array)
            #     table = table.append_column("color_step", color_array)
            #     tables.append(table)

            for mdet_step, table in table_dict.items():
                seed_array = pa.array([seed for _ in range(table.num_rows)])
                table = table.append_column("seed", seed_array)
                tables[shear_step][color_step][mdet_step] = table

    # return all_batches
    # return pa.concat_tables(tables)
    return tables



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
    output_path = name_util.get_output_path(args.output, args.config)
    # run_path = name_util.get_run_path(args.output, args.config, seed)

    schema = measure.schema
    schema = schema.append(
        pa.field("seed", pa.int64()),
    )
    # schema = schema.append(
    #     pa.field("shear_step", pa.string()),
    # )
    # schema = schema.append(
    #     pa.field("color_step", pa.string()),
    # )

    psf_colors = simulation_builder.config["measurement"].get("colors")

    shear_steps = ["plus", "minus"]
    color_steps = [f"c{i}" for i, psf_color in enumerate(psf_colors)]
    mdet_steps = ["noshear", "1p", "1m", "2p", "2m"]

    writers = {}
    for shear_step in shear_steps:
        writers[shear_step] = {}
        for color_step in color_steps:
            writers[shear_step][color_step] = {}
            for mdet_step in mdet_steps:
                dataset_path = os.path.join(
                    output_path,
                    shear_step,
                    color_step,
                    mdet_step,
                )
                writer_path = os.path.join(
                    dataset_path,
                    f"{seed}.parquet",
                )
                os.makedirs(dataset_path, exist_ok=True)
                writers[shear_step][color_step][mdet_step] = pq.ParquetWriter(
                    writer_path,
                    schema=schema,
                )

    # with pq.ParquetWriter(run_path, schema=schema) as writer:
    #     with multiprocessing.Pool(
    #         n_jobs,
    #         initializer=log_util.initializer,
    #         initargs=(queue, log_level),
    #         maxtasksperchild=max(1, n_sims // n_jobs),
    #     ) as pool:
    #         # for i, batches in enumerate(
    #         for i, table in enumerate(
    #             pool.imap(
    #                 functools.partial(task, simulation_builder, measure),
    #                 seeds,
    #             )
    #         ):
    #             print(f"finished simulation {i + 1}/{n_sims}")
    #             # for batch in batches:
    #             #     writer.write_batch(batch)
    #             # writer.write_table(table)

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
                functools.partial(task, simulation_builder, measure),
                seeds,
            )
        ):
            _end_time = time.time()
            _elapsed_time = _end_time - _start_time
            print(f"finished simulation {i + 1}/{n_sims} [{_elapsed_time / 60 :.0f} minutes elapsed]")
            for shear_step, color_tables in tables.items():
                for color_step, mdet_tables in color_tables.items():
                    for mdet_step, mdet_table in mdet_tables.items():
                        writers[shear_step][color_step][mdet_step].write_table(mdet_table)


    for shear_step, shear_writers in writers.items():
        for color_step, color_writers in shear_writers.items():
            for mdet_step, mdet_writers in color_writers.items():
                writers[shear_step][color_step][mdet_step].close()


    # with multiprocessing.Pool(
    #     n_jobs,
    #     initializer=log_util.initializer,
    #     initargs=(queue, log_level),
    #     maxtasksperchild=max(1, n_sims // n_jobs),
    # ) as pool:
    #     for i, table in enumerate(
    #         pool.imap(
    #             functools.partial(task, simulation_builder, measure),
    #             seeds,
    #         )
    #     ):
    #         print(f"finished simulation {i + 1}/{n_sims}")
    #         pq.write_to_dataset(table, run_path, partitioning=["shear_step", "color_step", "mdet_step"])

    queue.put(None)
    lp.join()

    print("simulations completed")
