import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

import galsim

import dsps
from dsps.data_loaders import load_ssp_templates
from lsstdesc_diffsky import read_diffskypop_params
from lsstdesc_diffsky.defaults import OUTER_RIM_COSMO_PARAMS
from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import ALL_DIFFSKY_PNAMES
from lsstdesc_diffsky.io_utils import load_diffsky_params
from lsstdesc_diffsky.sed import calc_rest_sed_disk_bulge_knot_galpop


def make_gal(
     redshift,
     spheroidEllipticity1,
     spheroidEllipticity2,
     spheroidHalfLightRadiusArcsec,
     diskEllipticity1,
     diskEllipticity2,
     diskHalfLightRadiusArcsec,
     ssp_data,
     rest_sed_bulge,
     rest_sed_diffuse_disk,
     rest_sed_knot,
     mstar_total,
     mstar_bulge,
     mstar_diffuse_disk,
     mstar_knot,
     cosmo_params,
     n_knots=0,
     morphology="achromatic",
):
    """
    Create a galaxy from a diffsky catalog
    """
    bulge_ellipticity = galsim.Shear(
        g1=spheroidEllipticity1,
        g2=spheroidEllipticity2,
    )
    bulge = galsim.DeVaucouleurs(
        half_light_radius=spheroidHalfLightRadiusArcsec,
    ).shear(bulge_ellipticity)

    disk_ellipticity = galsim.Shear(
        g1=diskEllipticity1,
        g2=diskEllipticity2,
    )
    disk = galsim.Exponential(
        half_light_radius=diskHalfLightRadiusArcsec,
    ).shear(disk_ellipticity)

    if n_knots > 0:
        knots = galsim.RandomKnots(
            n_knots,
            profile=disk,
        )

    luminosity_distance = dsps.cosmology.luminosity_distance_to_z(
        redshift,
        cosmo_params.Om0,
        cosmo_params.w0,
        cosmo_params.wa,
        cosmo_params.h,
    )
    luminosity_area = 4 * np.pi * luminosity_distance**2

    wave_type = "angstrom"
    flux_type = "fnu"
    # DSPS units
    # import astropy.units as u
    # _wave_type = u.angstrom
    # _flux_type = u.Lsun / u.Hz / u.Mpc**2
    # flux_factor = (1 * _flux_type).to(galsim.SED._fnu).value
    flux_factor = 4.0204145742268754e-16

    bulge_sed_table = galsim.LookupTable(
        ssp_data.ssp_wave,
        rest_sed_bulge / luminosity_area * flux_factor,
    )
    bulge_sed = galsim.SED(bulge_sed_table, wave_type=wave_type, flux_type=flux_type)

    disk_sed_table = galsim.LookupTable(
        ssp_data.ssp_wave,
        rest_sed_diffuse_disk / luminosity_area * flux_factor,
    )
    disk_sed = galsim.SED(disk_sed_table, wave_type=wave_type, flux_type=flux_type)

    knot_sed_table = galsim.LookupTable(
        ssp_data.ssp_wave,
        rest_sed_knot / luminosity_area * flux_factor,
    )
    knot_sed = galsim.SED(knot_sed_table, wave_type=wave_type, flux_type=flux_type)

    match morphology:
        case "chromatic":
            if n_knots > 0:
                gal = bulge * bulge_sed + disk * disk_sed + knots * knot_sed
            else:
                # if no knots are drawn, it is important to reweight the disk
                # to preserve relative magnitudes
                gal = bulge * bulge_sed + disk * (disk_sed + knot_sed)
        case "achromatic":
            # decouple the morphology from the spectra
            m_total = mstar_total
            m_bulge = mstar_bulge
            m_disk = mstar_diffuse_disk
            m_knot = mstar_knot

            # note that we also thin the SED in this case
            if n_knots > 0:
                gal = (
                    (bulge * m_bulge + disk * m_disk + knots * m_knot) / m_total \
                    * (bulge_sed + disk_sed + knot_sed)
                )
            else:
                gal = (
                    (bulge * m_bulge + disk * (m_disk + m_knot)) / m_total \
                    * (bulge_sed + disk_sed + knot_sed)
                )
        case _:
            raise ValueError("Unrecognized morphology: %s" % morphology)

    gal = gal.atRedshift(redshift)

    return gal


def get_gal(
     rng,
     igal,
     mock,
     ssp_data,
     disk_bulge_sed_info,
     cosmo_params,
     n_knots=0,
     morphology="achromatic",
     rotate=True,
):
    """
    Make and return a galaxy from a mock
    """
    gal = make_gal(
        mock["redshift"][igal],
        mock["spheroidEllipticity1"][igal],
        mock["spheroidEllipticity2"][igal],
        mock["spheroidHalfLightRadiusArcsec"][igal],
        mock["diskEllipticity1"][igal],
        mock["diskEllipticity2"][igal],
        mock["diskHalfLightRadiusArcsec"][igal],
        ssp_data,
        disk_bulge_sed_info.rest_sed_bulge[igal],
        disk_bulge_sed_info.rest_sed_diffuse_disk[igal],
        disk_bulge_sed_info.rest_sed_knot[igal],
        disk_bulge_sed_info.mstar_total[igal],
        disk_bulge_sed_info.mstar_bulge[igal],
        disk_bulge_sed_info.mstar_diffuse_disk[igal],
        disk_bulge_sed_info.mstar_knot[igal],
        OUTER_RIM_COSMO_PARAMS,
        n_knots=n_knots,
        morphology=morphology,
    )

    if rotate:
        gal = gal.rotate(rng.uniform(0, 180) * galsim.degrees)

    return gal


class RomanRubinBuilder:
    def __init__(self, diffskypop_params=None, ssp_templates=None):
        self.diffskypop_params=diffskypop_params
        self.ssp_templates=ssp_templates

        self.all_diffskypop_params = read_diffskypop_params(self.diffskypop_params)
        self.ssp_data = load_ssp_templates(fn=ssp_templates)

        morph_columns = [
           "redshift",
           "spheroidEllipticity1",
           "spheroidEllipticity2",
           "spheroidHalfLightRadiusArcsec",
           "diskEllipticity1",
           "diskEllipticity2",
           "diskHalfLightRadiusArcsec",
        ]
        self.columns = list(set(morph_columns + ALL_DIFFSKY_PNAMES))

    def build_gals(
        self,
        params,
        n_knots=0,
        morphology="achromatic",
    ):
        diffsky_param_data = load_diffsky_params(params)
        args = (
            np.array(params['redshift']),
            diffsky_param_data.mah_params,
            diffsky_param_data.ms_params,
            diffsky_param_data.q_params,
            diffsky_param_data.fbulge_params,
            diffsky_param_data.fknot,
            self.ssp_data,
            self.all_diffskypop_params,
            OUTER_RIM_COSMO_PARAMS
        )

        disk_bulge_sed_info = calc_rest_sed_disk_bulge_knot_galpop(*args)

        # FIXME add rotate toggle
        gals = [
            make_gal(
                params["redshift"][igal],
                params["spheroidEllipticity1"][igal],
                params["spheroidEllipticity2"][igal],
                params["spheroidHalfLightRadiusArcsec"][igal],
                params["diskEllipticity1"][igal],
                params["diskEllipticity2"][igal],
                params["diskHalfLightRadiusArcsec"][igal],
                self.ssp_data,
                disk_bulge_sed_info.rest_sed_bulge[igal],
                disk_bulge_sed_info.rest_sed_diffuse_disk[igal],
                disk_bulge_sed_info.rest_sed_knot[igal],
                disk_bulge_sed_info.mstar_total[igal],
                disk_bulge_sed_info.mstar_bulge[igal],
                disk_bulge_sed_info.mstar_diffuse_disk[igal],
                disk_bulge_sed_info.mstar_knot[igal],
                OUTER_RIM_COSMO_PARAMS,
                n_knots=n_knots,
                morphology=morphology,
            )
            for igal in range(len(params["redshift"]))
        ]

        return gals


if __name__ == "__main__":


    seed = 42
    rng = np.random.default_rng(seed)

    n_gals = 10

    # define a sample predicate
    predicate = (
        (pc.field("LSST_obs_r") < pc.scalar(26))
        & (pc.field("LSST_obs_i") < pc.scalar(26))
        & (pc.field("LSST_obs_z") < pc.scalar(26))
    )
    scanner = get_scanner("/pscratch/sd/s/smau/roman_rubin_2023_v1.1.1_parquet", predicate=predicate)
    num_rows = scanner.count_rows()
    gal_indices = rng.choice(num_rows, n_gals, replace=True)
    mock = scanner.take(gal_indices).to_pydict()

    diffsky_param_data = load_diffsky_params(mock)
    all_diffskypop_params = read_diffskypop_params("roman_rubin_2023")
    ssp_data = load_ssp_templates(fn='/pscratch/sd/s/smau/dsps_ssp_data.h5')

    args = (
        np.array(mock['redshift']),
        diffsky_param_data.mah_params,
        diffsky_param_data.ms_params,
        diffsky_param_data.q_params,
        diffsky_param_data.fbulge_params,
        diffsky_param_data.fknot,
        ssp_data,
        all_diffskypop_params,
        OUTER_RIM_COSMO_PARAMS
    )

    disk_bulge_sed_info = calc_rest_sed_disk_bulge_knot_galpop(*args)

    igals = rng.choice(n_gals, size=n_gals, replace=False, shuffle=True)
    gals = [
        get_gal(
            rng,
            igal,
            mock,
            ssp_data,
            disk_bulge_sed_info,
            OUTER_RIM_COSMO_PARAMS,
            n_knots=0,
            morphology="achromatic",
            rotate=True,
        ) for igal in igals
    ]

    filters = {"u", "g", "r", "i", "z", "y"}
    bps = {
        f: galsim.Bandpass(f"LSST_{f}.dat", "nm").withZeropoint("AB")
        for f in filters
    }

    print(f"|     u |     g |     r |     i |     z |     y |")
    for gal in gals:
        mag_u = gal.calculateMagnitude(bps["u"])
        mag_g = gal.calculateMagnitude(bps["g"])
        mag_r = gal.calculateMagnitude(bps["r"])
        mag_i = gal.calculateMagnitude(bps["i"])
        mag_z = gal.calculateMagnitude(bps["z"])
        mag_y = gal.calculateMagnitude(bps["y"])
        print(f"| {mag_u:2.2f} | {mag_g:2.2f} | {mag_r:2.2f} | {mag_i:2.2f} | {mag_z:2.2f} | {mag_y:2.2f} |")

