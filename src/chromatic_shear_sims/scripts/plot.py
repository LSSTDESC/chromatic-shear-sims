import argparse
import logging
from logging import handlers
import multiprocessing
import threading

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from chromatic_shear_sims import utils
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


def plot_sim(mbobs, psf_mbobs):
    bands = mbobs.meta.get("bands")

    fig, axs = plt.subplots(
        2, len(bands),
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )

    norm_field = mpl.colors.Normalize()

    for i, band in enumerate(bands):
        obs = mbobs[i][0]
        image = obs.image
        weight_image = obs.weight

        psf_obs = psf_mbobs[i][0]
        psf_image = psf_obs.image

        axs[0, i].imshow(psf_image, origin="lower")
        axs[0, i].axvline((psf_image.shape[0] - 1) / 2, c="gray", ls=":")
        axs[0, i].axhline((psf_image.shape[1] - 1) / 2, c="gray", ls=":")

        axs[1, i].imshow(np.arcsinh(image * np.sqrt(weight_image)), origin="lower", norm=norm_field)

        axs[0, i].set_title(band)

    axs[0, 0].set_ylabel("PSF")
    axs[1, 0].set_ylabel("Wide Field")

    plt.show()


def plot_sim_pair(mbobs_dict, psf_mbobs):
    plus_mbobs = mbobs_dict["plus"]
    minus_mbobs = mbobs_dict["minus"]
    bands = plus_mbobs.meta.get("bands")

    fig, axs = plt.subplots(
        3, len(bands),
        sharex="row",
        sharey="row",
        constrained_layout=True,
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
        axs[0, i].axvline((psf_image.shape[0] - 1) / 2, c="gray", ls=":")
        axs[0, i].axhline((psf_image.shape[1] - 1) / 2, c="gray", ls=":")

        axs[1, i].imshow(np.arcsinh(plus_image * np.sqrt(plus_weight_image)), origin="lower", norm=norm_field)

        axs[2, i].imshow(np.arcsinh(minus_image * np.sqrt(minus_weight_image)), origin="lower", norm=norm_field)

        axs[0, i].set_title(band)

    axs[0, 0].set_ylabel("PSF")
    axs[1, 0].set_ylabel("plus")
    axs[2, 0].set_ylabel("minus")

    plt.show()


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
    # parser.add_argument(
    #     "--detect",
    #     action="store_true",
    #     help="Whether to make detections",
    # )
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

    # while True:
    #     seed = utils.get_seed()
    #     sim = simulation_builder.make_obs(
    #         seed=seed,
    #     )
    #     plot_sim(sim)

    # while True:
    #     seed = utils.get_seed()
    #     sim_pair = simulation_builder.make_obs_pair(
    #         g1=0.02, g2=0.00,
    #         seed=seed,
    #     )
    #     plot_sim_pair(sim_pair)

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
        for i, (obs, psf_obs) in enumerate(
            pool.imap(
                simulation_builder.run_sim_pair,
                seeds,
            )
        ):
            print(f"finished simulation {i + 1}/{n_sims}")
            plot_sim_pair(obs, psf_obs)

    queue.put(None)
    lp.join()

    print("simulations completed")
