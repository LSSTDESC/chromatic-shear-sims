import argparse
import logging

import numpy as np

from chromatic_shear_sims import utils
from chromatic_shear_sims.simulation import SimulationBuilder
from chromatic_shear_sims.throughputs import load_throughputs

from . import log_util, name_util, plot_util


def plot_scene(scene):

    fig, axs = plot_util.subplots(
        2, 2,
        sharex="row",
        sharey="row",
    )

    throughputs = load_throughputs(bands=["r"])

    wl = np.linspace(300, 1200, 1000)

    for (galaxy, position) in scene.galaxies:
        spec = galaxy.sed
        axs[1, 0].plot(wl, spec(wl))
        axs[0, 0].scatter(position.x, position.y, s=12)

    for (star, position) in scene.stars:
        spec = star.sed
        axs[1, 1].plot(wl, spec(wl))
        axs[0, 1].scatter(position.x, position.y, s=12)

    axs[0, 0].set_xlabel("[pixels]")
    axs[0, 0].set_ylabel("[pixels]")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_xlabel("$\lambda$ [$nm$]")
    axs[1, 0].set_ylabel("$f_{photons}$ [$photons/nm/cm^2/s$]")
    axs[0, 0].set_title("Galaxies")

    axs[0, 1].set_xlabel("[pixels]")
    # axs[0, 1].set_ylabel("[pixels]")
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_xlabel("$\lambda$ [$nm$]")
    # axs[1, 1].set_ylabel("$f_{photons}$ [$photons/nm/cm^2/s$]")
    axs[0, 1].set_title("Stars")

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
    config_name = name_util.get_config_name(config_file)
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
        fig, axs = plot_scene(scene)

        figname = f"{config_name}-scene-{seeds[i]}.pdf"
        fig.savefig(figname)

    print("simulations completed")
