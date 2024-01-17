"""
"""

import functools
from pathlib import Path

import pyarrow.dataset as ds
import galsim


_lsst_i = galsim.Bandpass("LSST_i.dat", "nm").withZeropoint("AB")


@functools.cache
def read_sed_file(file_name, wave_type, flux_type):
    return galsim.sed.SED(file_name, wave_type, flux_type)


def build_star(star_params, sed_dir, i_bandpass):
    _standard_dict = {
        "lte*": "phoSimMLT",
        "bergeron*": "wDs",
        "k[mp]*": "kurucz",
    }
    wave_type = "Nm"
    flux_type = "flambda"
    sed_filename = star_params.get("sedFilename")[0].strip()  # FIXME
    if not sed_filename.endswith(".gz"):
        # Some files are missing ".gz" in their suffix; if this is the case,
        # append to the current suffix
        sed_filename += ".gz"

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
    sed = sed.withMagnitude(star_params.get("imag")[0], i_bandpass)  # FIXME

    return galsim.DeltaFunction() * sed

class DC2Builder:
    def __init__(self, sed_dir=None):
        self.sed_dir = sed_dir
        self.lsst_i = galsim.Bandpass("LSST_i.dat", "nm").withZeropoint("AB")

        self.columns = ["sedFilename", "imag"]

    def build_star(
        self,
        params,
    ):
        star = build_star(params, self.sed_dir, self.lsst_i)

        return star


if __name__ == "__main__":
    import numpy as np
    import pyarrow.compute as pc

    # SED_DIR = "/cvmfs/sw.lsst.eu/linux-x86_64/lsst_sims/sims_w_2020_15/stack/current/Linux64/sims_sed_library/2017.01.24/starSED/"
    SED_DIR = "/pscratch/sd/s/smau/starSED/"

    seed = 42
    rng = np.random.default_rng(seed)

    n_stars = 10

    predicate = (
        (pc.field("imag") > pc.scalar(19))
        & (pc.field("imag") < pc.scalar(21))
        & pc.match_substring_regex(pc.field("sedFilename"), "^k[mp]*")
    )
    scanner = get_scanner("/pscratch/sd/s/smau/dc2_stellar_healpixel.arrow", predicate=predicate)
    num_rows = scanner.count_rows()

    star_indices = rng.choice(num_rows, n_stars, replace=True)
    star_params = scanner.take(star_indices).to_pylist()

    istars = rng.choice(n_stars, size=n_stars, replace=False, shuffle=True)
    stars = [
        build_star(
            star_params[istar],
            SED_DIR,
        )
        for istar in istars
    ]

    filters = {"u", "g", "r", "i", "z", "y"}
    bps = {
        f: galsim.Bandpass(f"LSST_{f}.dat", "nm").withZeropoint("AB")
        for f in filters
    }

    print(f"|     u |     g |     r |     i |     z |     y |")
    for star in stars:
        mag_u = star.calculateMagnitude(bps["u"])
        mag_g = star.calculateMagnitude(bps["g"])
        mag_r = star.calculateMagnitude(bps["r"])
        mag_i = star.calculateMagnitude(bps["i"])
        mag_z = star.calculateMagnitude(bps["z"])
        mag_y = star.calculateMagnitude(bps["y"])
        print(f"| {mag_u:2.2f} | {mag_g:2.2f} | {mag_r:2.2f} | {mag_i:2.2f} | {mag_z:2.2f} | {mag_y:2.2f} |")

