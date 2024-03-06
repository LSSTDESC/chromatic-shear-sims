"""
"""

import functools
import logging
import os
from pathlib import Path
import time

import pyarrow.dataset as ds
import galsim


logger = logging.getLogger(__name__)


@functools.cache
def read_sed_file(file_name, wave_type, flux_type):
    return galsim.SED(file_name, wave_type, flux_type)


def build_star(sed_filename, sed_dir=None, mag=None, bandpass=None):
    _start_time = time.time()
    _standard_dict = {
        "lte*": "phoSimMLT",
        "bergeron*": "wDs",
        "k[mp]*": "kurucz",
    }
    wave_type = "Nm"
    flux_type = "flambda"
    # sed_filename = sed_filename.strip()  # FIXME
    # if not sed_filename.endswith(".gz"):
    #     # Some files are missing ".gz" in their suffix; if this is the case,
    #     # append to the current suffix
    #     sed_filename += ".gz"

    path_name = Path(sed_filename)
    for k, v in _standard_dict.items():
        matched = False
        if path_name.match(k):
            sed_path = Path(sed_dir) / v / path_name
            matched = True
            break  # we should only have one match

    if not matched:
        raise ValueError(
            f"Filename {sed_filename} does not match any known patterns in {sed_dir}"
        )

    if not sed_path.exists():
        raise ValueError(f"Filename {sed_filename} not found in {sed_dir}")

    sed_file = sed_path.as_posix()

    sed = read_sed_file(sed_file, wave_type, flux_type)

    # Use the catalog to recover normalization of SED
    if (mag is not None) and (bandpass is not None):
        sed = sed.withMagnitude(mag, bandpass)

    star = galsim.DeltaFunction() * sed

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"built star in {_elapsed_time} seconds")

    return star

class DC2Builder:
    def __init__(self, sed_dir=None, survey=None):
        self.sed_dir = sed_dir
        self.survey = survey
        self.columns = ["sedFilename", "rmag_obs"]

        logger.info(f"initializing DC2 builder with sed_dir: {self.sed_dir}")

    def build_stars(
        self,
        params,
    ):
        _start_time = time.time()
        stars = [
            build_star(
                params["sedFilename"][istar],
                sed_dir=self.sed_dir,
                mag=params["rmag_obs"][istar],
                bandpass=self.survey.bandpasses["r"],
            )
            for istar in range(len(params["sedFilename"]))
        ]
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        logger.info(f"built {len(stars)} stars in {_elapsed_time} seconds")

        return stars

class DC2Stars:
    def __init__(self, config, loader, sed_dir=None, survey=None):
        self.config = config
        self.sed_dir = sed_dir
        self.survey = survey
        self.loader = loader
        self.builder = DC2Builder(sed_dir=self.sed_dir, survey=self.survey)


    def __call__(self, color=None, tol=0.01):
        match color:
            case None:
                _star_params = self.loader.sample(
                    1,
                    columns=self.builder.columns,
                )
            case float() | int():
                # predicate=(pc.abs_checked(pc.field("gmag_obs") - pc.field("imag_obs") - color) < tol)
                predicate=pc.less(
                    pc.abs_checked(
                        pc.subtract(
                            pc.subtract(
                                pc.field("gmag_obs"),
                                pc.field("imag_obs")
                            ),
                            color
                        )
                    ),
                    tol
                )
                _star_params = self.loader.sample_with(
                    1,
                    columns=self.builder.columns,
                    predicate=predicate,
                )
            case _:
                raise ValueError(f"Color type not valid!")

        star_params = _star_params[0]
        star = self.builder.build_stars(star_params)
        return star


if __name__ == "__main__":
    import numpy as np
    import pyarrow.compute as pc
    from chromatic_shear_sims import surveys

    logging.basicConfig(level=logging.DEBUG)

    lsst = surveys.lsst
    lsst.load_bandpasses(os.environ.get("THROUGHPUT_DIR"))

    # SED_DIR = "/cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/sims_w_2020_15/stack/current/Linux64/sims_sed_library/2017.01.24/starSED/"
    SED_DIR = os.environ.get("SED_DIR")

    seed = None
    rng = np.random.default_rng(seed)

    n_stars = 10

    predicate = (
        (pc.field("imag") > pc.scalar(19))
        & (pc.field("imag") < pc.scalar(21))
        & pc.match_substring_regex(pc.field("sedFilename"), "^k[mp]*")
    )
    dc2builder = DC2Builder(
        SED_DIR,
        survey=lsst,
    )

    # dataset = ds.dataset("/pscratch/sd/s/smau/dc2_stellar_healpixel.arrow", format="arrow")
    # dataset = ds.dataset("/pscratch/sd/s/smau/dc2_stellar_healpixel_parquet")
    dataset = ds.dataset("/scratch/users/smau/dc2_stellar_healpixel_parquet")
    # count = dataset.count_rows(filter=predicate)
    scanner = dataset.scanner(filter=predicate)
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        # count,
        1000,
        size=10,
        replace=True,
        shuffle=True,
    )

    star_params = scanner.take(indices).to_pydict()

    stars = dc2builder.build_stars(star_params)

    print(f"|-----------------|---------------|-------------------|-------------|")
    print(f"|        r        |      g-i      |        g-i        |     g-i     |")
    print(f"|-----------------|---------------|-------------------|-------------|")
    print(f"|    cat |    obs |   cat |   obs | (obs - cat) / cat | |obs - cat| |")
    print(f"|-----------------|---------------|-------------------|-------------|")
    for i, star in enumerate(stars):
        mag_g = star.calculateMagnitude(lsst.bandpasses["g"])
        mag_r = star.calculateMagnitude(lsst.bandpasses["r"])
        mag_i = star.calculateMagnitude(lsst.bandpasses["i"])
        obs_color = mag_g - mag_i

        mag_g_cat = star_params["gmag_obs"][i]
        mag_r_cat = star_params["rmag_obs"][i]
        mag_i_cat = star_params["imag_obs"][i]
        color = mag_g_cat - mag_i_cat

        print(f"| {mag_r_cat:2.3f} | {mag_r:2.3f} | {color:2.3f} | {obs_color:2.3f} | {(obs_color - color) / color:2.14f} | {np.abs(obs_color - color)} |")

    print(f"|-----------------|---------------|-------------------|-------------|")

