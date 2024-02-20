import functools
import logging
import time

import galsim
import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

import dsps
# from dsps.data_loaders import load_ssp_templates
from lsstdesc_diffsky import read_diffskypop_params
from lsstdesc_diffsky.defaults import OUTER_RIM_COSMO_PARAMS
from lsstdesc_diffsky.io_utils.load_diffsky_healpixel import ALL_DIFFSKY_PNAMES
from lsstdesc_diffsky.io_utils import load_diffsky_params
# from lsstdesc_diffsky.sed import calc_rest_sed_disk_bulge_knot_galpop
from lsstdesc_diffsky.legacy.roman_rubin_2023.dsps.data_loaders.load_ssp_data import load_ssp_templates_singlemet
from lsstdesc_diffsky.legacy.roman_rubin_2023.dsps.data_loaders.defaults import SSPDataSingleMet
from lsstdesc_diffsky.sed.disk_bulge_sed_kernels_singlemet import calc_rest_sed_disk_bulge_knot_galpop


logger = logging.getLogger(__name__)


@functools.cache
def cached_read_diffskypop_params(mock_name):
    return read_diffskypop_params(mock_name)


@functools.cache
def cached_load_ssp_templates_singlemet(fn):
    return load_ssp_templates_singlemet(fn=fn)


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
    _start_time = time.time()
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

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.debug(f"built galaxy in {_elapsed_time:0.2f} seconds")

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
        rotation_angle = rng.uniform(0, 180) * galsim.degrees
        logger.debug(f"rotating galaxy by {rotation_angle}")
        gal = gal.rotate(rotation_angle)

    return gal


class RomanRubinBuilder:
    def __init__(self, diffskypop_params=None, ssp_templates=None, survey=None):
        self.diffskypop_params=diffskypop_params
        self.ssp_templates=ssp_templates

        logger.info(f"initializing roman rubin builder with diffskypop_params: {self.diffskypop_params}, ssp_templates: {self.ssp_templates}")

        # self.all_diffskypop_params = read_diffskypop_params(self.diffskypop_params)
        self.all_diffskypop_params = cached_read_diffskypop_params(self.diffskypop_params)
        # self.ssp_data = load_ssp_templates(fn=ssp_templates)
        # _ssp_data = load_ssp_templates_singlemet(fn=ssp_templates)
        _ssp_data = cached_load_ssp_templates_singlemet(fn=ssp_templates)
        if survey is not None:
            # remove porition of SED redder than redmost limit of all bandpasses;
            # note that we can't impose a lower limit due to redshifting
            _max = np.max([bandpass.red_limit for bandpass in survey.bandpasses.values()]) * 10
            _keep = (_ssp_data.ssp_wave < _max)
            logger.info(f"discarding {(~_keep).sum()} of {len(_keep)} wavelengths from templates")
            _ssp_wave = _ssp_data.ssp_wave[_keep]
            _ssp_flux = _ssp_data.ssp_flux[:, _keep]
            ssp_data = SSPDataSingleMet(_ssp_data.ssp_lg_age_gyr, _ssp_wave, _ssp_flux)
        else:
            ssp_data = _ssp_data
        self.ssp_data = ssp_data

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
        rotate=True,
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

        # # FIXME add rotate toggle
        # gals = [
        #     make_gal(
        #         params["redshift"][igal],
        #         params["spheroidEllipticity1"][igal],
        #         params["spheroidEllipticity2"][igal],
        #         params["spheroidHalfLightRadiusArcsec"][igal],
        #         params["diskEllipticity1"][igal],
        #         params["diskEllipticity2"][igal],
        #         params["diskHalfLightRadiusArcsec"][igal],
        #         self.ssp_data,
        #         disk_bulge_sed_info.rest_sed_bulge[igal],
        #         disk_bulge_sed_info.rest_sed_diffuse_disk[igal],
        #         disk_bulge_sed_info.rest_sed_knot[igal],
        #         disk_bulge_sed_info.mstar_total[igal],
        #         disk_bulge_sed_info.mstar_bulge[igal],
        #         disk_bulge_sed_info.mstar_diffuse_disk[igal],
        #         disk_bulge_sed_info.mstar_knot[igal],
        #         OUTER_RIM_COSMO_PARAMS,
        #         n_knots=n_knots,
        #         morphology=morphology,
        #     )
        #     for igal in range(len(params["redshift"]))
        # ]

        # FIXME seed rng
        logger.info(f"building galaxies with morphology: {morphology}, n_knots: {n_knots}, rotate: {rotate}")
        rng = np.random.default_rng()
        gals = [
            get_gal(
                rng,
                igal,
                params,
                self.ssp_data,
                disk_bulge_sed_info,
                OUTER_RIM_COSMO_PARAMS,
                n_knots=0,
                morphology="achromatic",
                rotate=rotate,
            )
            for igal in range(len(params["redshift"]))
        ]

        return gals


if __name__ == "__main__":
    from chromatic_shear_bias import surveys

    logging.basicConfig(level=logging.INFO)

    lsst = surveys.lsst

    seed = 42
    rng = np.random.default_rng(seed)

    n_gals = 10

    # define a sample predicate
    predicate = (
        (pc.field("LSST_obs_r") < pc.scalar(26))
        & (pc.field("LSST_obs_i") < pc.scalar(26))
        & (pc.field("LSST_obs_z") < pc.scalar(26))
    )
    romanrubinbuilder = RomanRubinBuilder(
        diffskypop_params="roman_rubin_2023",
        ssp_templates="/pscratch/sd/s/smau/dsps_ssp_data_singlemet.h5",
        survey=lsst,
    )
    columns = romanrubinbuilder.columns
    columns.append("LSST_obs_g")
    columns.append("LSST_obs_r")
    columns.append("LSST_obs_i")

    dataset = ds.dataset("/pscratch/sd/s/smau/roman_rubin_2023_v1.1.1_parquet")
    # count = dataset.count_rows(filter=predicate)
    scanner = dataset.scanner(columns=columns, filter=predicate)
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        # count,
        100,
        size=10,
        replace=True,
        shuffle=True,
    )

    gal_params = scanner.take(indices).to_pydict()

    gals = romanrubinbuilder.build_gals(gal_params)


    bps = lsst.bandpasses

    print(f"|-----------------|---------------|-------------------|-------------|")
    print(f"|        r        |      g-i      |        g-i        |     g-i     |")
    print(f"|-----------------|---------------|-------------------|-------------|")
    print(f"|    cat |    obs |   cat |   obs | (obs - cat) / cat | |obs - cat| |")
    print(f"|-----------------|---------------|-------------------|-------------|")
    for i, gal in enumerate(gals):
        mag_g = gal.calculateMagnitude(bps["g"])
        mag_r = gal.calculateMagnitude(bps["r"])
        mag_i = gal.calculateMagnitude(bps["i"])
        obs_color = mag_g - mag_i

        mag_g_cat = gal_params["LSST_obs_g"][i]
        mag_r_cat = gal_params["LSST_obs_r"][i]
        mag_i_cat = gal_params["LSST_obs_i"][i]
        color = mag_g_cat - mag_i_cat

        print(f"| {mag_r_cat:2.3f} | {mag_r:2.3f} | {color:2.3f} | {obs_color:2.3f} | {(obs_color - color) / color:2.14f} | {np.abs(obs_color - color)} |")

    print(f"|-----------------|---------------|-------------------|-------------|")
