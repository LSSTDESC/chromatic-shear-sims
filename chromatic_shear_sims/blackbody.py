import functools
import logging

import astropy.units as u
from astropy.constants import h, c, k_B
import galsim
import numpy as np


logger = logging.getLogger(__name__)


def blackbody_radiance(t, wl):
    """see https://en.wikipedia.org/wiki/Planck%27s_law#Different_forms
    Returns flambda
    """
    return (
        2 * h * c**2
        / (wl * u.nm)**5
        / (np.exp(h * c / (wl * u.nm * k_B * t * u.K)) - 1)
    ).to(u.erg / u.nm / u.cm**2 / u.s).value


def blackbody_sed(temp):
    return galsim.SED(
        functools.partial(blackbody_radiance, temp),
        wave_type="nm",
        flux_type="flambda"
    )


class BlackBody:
    def __init__(self, survey, t_min=2_000, t_max=30_000, nt=1_000):
        logger.info(f"initializing blackbody SED builder")
        self.survey = survey
        self._lut = None
        self.make_lut(t_min=t_min, t_max=t_max, nt=nt)

    def make_lut(self, t_min=2_000, t_max=30_000, nt=1_000):
        logger.info(f"making color-temp lookup table for blackbody SED")
        logger.debug(f"interpolating for temps in geom({t_min}, {t_max}, {nt})")
        bandpass_g = self.survey.bandpasses["g"]
        bandpass_i = self.survey.bandpasses["i"]

        temps = []
        colors = []

        # temp is exponentiated, so sample with log-spacing
        for temp in np.geomspace(t_min, t_max, nt):
            sed = galsim.SED(
                functools.partial(blackbody_radiance, temp), wave_type="nm", flux_type="flambda"
            )
            color = sed.calculateMagnitude(bandpass_g) - sed.calculateMagnitude(bandpass_i)
            logger.debug(f"(temp, color) = ({temp}, {color})")

            temps.append(temp)
            colors.append(color)

        # temp is exponentiated, so interpolate in log(temp)
        lut = galsim.LookupTable(colors, temps, x_log=False, f_log=True)

        self._lut = lut

    @property
    def lut(self):
        return self._lut

    def __call__(self, color):
        temp = self.lut(color)
        logger.debug(f"drawing blackbody SED with temperature {temp}")
        sed = blackbody_sed(temp)
        return sed


class BlackBodyStars:
    def __init__(self, config, survey):
        self.config = config
        self.survey = survey
        self.blackbody = BlackBody(self.survey)

    def __call__(self, color):
        sed = self.blackbody(color)
        return galsim.DeltaFunction() * sed

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from chromatic_shear_sims import surveys

    logging.basicConfig(level=logging.INFO)

    lsst = surveys.lsst
    lsst.load_bandpasses("/scratch/users/smau/baseline")

    bb = BlackBody(lsst, t_min=2_000, t_max=30_000, nt=1_000)

    # min_color': -0.5305055021049974, 'max_color': 3.566875820157911
    colors = np.linspace(-0.6, 3.6, 1000)
    true_colors = []
    meas_colors = []
    temps = []
    for color in colors:
        temp = bb.lut(color)
        sed = bb(color)
        meas_color = sed.calculateMagnitude(lsst.bandpasses["g"]) - sed.calculateMagnitude(lsst.bandpasses["i"])
        true_colors.append(color)
        meas_colors.append(meas_color)
        temps.append(temp)

    true_colors = np.array(true_colors)
    meas_colors = np.array(meas_colors)
    temps = np.array(temps)
    plt.scatter(true_colors, (meas_colors - true_colors) / true_colors, c=temps, cmap="twilight_shifted_r")
    plt.xlabel("true")
    plt.ylabel("(meas - true) / true")
    plt.colorbar()
    plt.show()

