import argparse
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from chromatic_shear_sims import utils
from chromatic_shear_sims.simulation import SimulationBuilder
from chromatic_shear_sims.throughputs import load_throughputs


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


def plot_scene(scene):

    fig, axs = plt.subplots(
        2, 2,
        sharex="row",
        sharey="row",
        constrained_layout=True,
    )

    throughputs = load_throughputs(bands=["r"])

    wl = np.linspace(300, 1200, 1000)

    for galaxy in scene.galaxies:
        spec = galaxy.sed
        centroid = galaxy.calculateCentroid(throughputs["r"])
        axs[1, 0].plot(wl, spec(wl))
        axs[0, 0].scatter(centroid.x, centroid.y, s=12)

    for star in scene.stars:
        spec = star.sed
        centroid = star.calculateCentroid(throughputs["r"])
        axs[1, 1].plot(wl, spec(wl))
        axs[0, 1].scatter(centroid.x, centroid.y, s=12)

    axs[0, 0].set_xlabel("[$arcsec$]")
    axs[0, 0].set_ylabel("[$arcsec$]")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("$\lambda$ [$nm$]")
    axs[1, 0].set_ylabel("$f_{photons}$ [$photons/nm/cm^2/s$]")
    axs[0, 0].set_title("Galaxies")

    axs[0, 1].set_xlabel("[$arcsec$]")
    # axs[0, 1].set_ylabel("[$arcsec$]")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("$\lambda$ [$nm$]")
    # axs[1, 1].set_ylabel("$f_{photons}$ [$photons/nm/cm^2/s$]")
    axs[0, 1].set_title("Stars")

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

    n_sims = args.n_sims
    seed = args.seed
    seeds = utils.get_seeds(n_sims, seed=seed)

    for i, scene in enumerate(
        map(
            simulation_builder.make_scene,
            seeds,
        )
    ):
        print(f"finished simulation {i + 1}/{n_sims}")
        plot_scene(scene)

    print("simulations completed")
