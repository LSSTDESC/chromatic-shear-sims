import logging

import galsim


logger = logging.getLogger(__name__)


class Scene:
    def __init__(self, galaxies=[], stars=[]):
        self._galaxies = list(galaxies)
        self._stars = list(stars)

    @property
    def galaxies(self):
        return self._galaxies

    @property
    def ngal(self):
        return len(self.galaxies)

    @property
    def stars(self):
        return self._stars

    @property
    def nstar(self):
        return len(self.stars)

    def with_shear(self, g1, g2):
        logger.info(f"making scene with shear g1={g1}, g2={g2}")
        shear = galsim.Shear(g1=g1, g2=g2)
        galaxies = [
            (galaxy.shear(shear), position.shear(shear))
            for (galaxy, position) in self.galaxies
        ]
        stars = self.stars
        return Scene(galaxies=galaxies, stars=stars)
