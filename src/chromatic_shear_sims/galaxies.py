import logging

from chromatic_shear_sims import utils
from chromatic_shear_sims.stars import InterpolatedStarBuilder
from chromatic_shear_sims.throughputs import load_throughputs


logger = logging.getLogger(__name__)


class GalaxyBuilder:
    def __init__(self, entrypoint, **kwargs):
        self.model = utils.get_instance(entrypoint, **kwargs)
        self.name = self.model.name

    def __call__(self, galaxy_params, **kwargs):
        galaxy = self.model.get_galaxy(galaxy_params, **kwargs)
        return galaxy


class HybridGalaxyBuilder:
    def __init__(self, entrypoint, spectra_entrypoint=None, band_norm="r", band_1="g", band_2="i", **kwargs):
        self.morphology_model = utils.get_instance(entrypoint, **kwargs)

        self.band_1 = band_1
        self.band_2 = band_2
        self.band_norm = band_norm
        self.throughputs = load_throughputs(bands=[self.band_norm])

        self.spectra_builder = InterpolatedStarBuilder(
            entrypoint=spectra_entrypoint,
            band_1=self.band_1,
            band_2=self.band_2,
        )

        self.name = self.morphology_model.name + "-" + self.spectra_builder.name

    def __call__(self, morphology_params, obs_params, **kwargs):
        morphology = self.morphology_model.get_morphology(morphology_params)
        color = self.morphology_model.get_color(obs_params)
        spec = self.spectra_builder.get_spectrum(color)
        galaxy = (morphology * spec).withMagnitude(
            getattr(obs_params, f"LSST_obs_{self.band_norm}"),
            self.throughputs[self.band_norm],
        )

        return galaxy

