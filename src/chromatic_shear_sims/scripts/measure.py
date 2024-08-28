import argparse
import logging
from logging import handlers
import multiprocessing
import threading

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from chromatic_shear_sims import utils
from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder


import os
os.environ["THROUGHPUTS_DIR"] = "."
os.environ["DSPS_SSP_DATA"] = "dsps_ssp_data_singlemet.h5"


LOGGING_FORMAT = '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s'


def get_log_level(log_level):
    match log_level:
        case 0 | logging.ERROR:
            level = logging.ERROR
        case 1 | logging.WARNING:
            level = logging.WARNING
        case 2 | logging.INFO:
            level = logging.INFO
        case 3 | logging.DEBUG:
            level = logging.DEBUG
        case _:
            level = logging.INFO

    return level


# https://docs.python.org/3/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes
def logger_thread(queue):
    while True:
        record = queue.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def initializer(queue, log_level=None):
    queue_handler = handlers.QueueHandler(queue)
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.addHandler(queue_handler)
    logger.debug(f"spawning worker process")


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


def measure_sim(mbobs, psf_mbobs, measure):
    bands = mbobs.meta.get("bands")

    model = measure.config.get("model")
    meas = measure.run(mbobs, psf_mbobs)
    meas = _apply_selection(
        meas,
        model,
    )

    return meas


def measure_sim_pair(mbobs_dict, psf_mbobs, measure):
    plus_mbobs = mbobs_dict["plus"]
    minus_mbobs = mbobs_dict["minus"]

    bands = plus_mbobs.meta.get("bands")

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

    meas_dict = {
        "plus": meas_p,
        "minus": meas_m,
    }

    return meas_dict


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
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    return parser.parse_args()


def main():
    args = get_args()
    log_level = get_log_level(args.log_level)
    logging.basicConfig(format=LOGGING_FORMAT, level=log_level)

    config_file = args.config
    simulation_builder = SimulationBuilder.from_yaml(config_file)
    measure = measurement.get_measure(
        **simulation_builder.config["measurement"].get("builder"),
    )

    multiprocessing.set_start_method("spawn")

    queue = multiprocessing.Queue(-1)

    lp = threading.Thread(target=logger_thread, args=(queue,))
    lp.start()

    n_jobs = args.n_jobs
    n_sims = args.n_sims
    seed = args.seed
    seeds = utils.get_seeds(n_sims, seed=seed)

    with multiprocessing.Pool(
        n_jobs,
        initializer=initializer,
        initargs=(queue, log_level),
        maxtasksperchild=n_sims // n_jobs,
    ) as pool:
        for i, (obs, psf) in enumerate(
            pool.imap(
                simulation_builder.make_sim,
                # simulation_builder.make_sim_pair,
                seeds,
            )
        ):
            print(f"finished simulation {i + 1}/{n_sims}")
            for psf_color in [0.5, 0.8, 1.1]:
                psf_obs = simulation_builder.make_psf_obs(psf, color=psf_color)
                meas = measure_sim(obs, psf_obs, measure)
                # meas_dict = measure_sim_pair(obs, psf_obs, measure)
                print(meas)

    queue.put(None)
    lp.join()

    print("simulations completed")
