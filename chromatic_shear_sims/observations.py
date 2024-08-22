import copy
import logging

import galsim
import numpy as np
import ngmix

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)


def get_ormask(xsize, ysize):
    return np.full((xsize, ysize), int(0))


def get_bmask(xsize, ysize):
    return np.full((xsize, ysize), int(0))


def get_weight(xsize, ysize, noise_sigma):
    return np.full((xsize, ysize), 1 / noise_sigma ** 2)


def get_wcs(scale, center):
    return galsim.AffineTransform(
        scale,
        0.0,
        0.0,
        scale,
        origin=center,
    )


def get_jacobian(image, wcs):
    return ngmix.jacobian.Jacobian(
        x=(image.array.shape[1] - 1) / 2,
        y=(image.array.shape[0] - 1) / 2,
        dudx=wcs.dudx,
        dudy=wcs.dudy,
        dvdx=wcs.dvdx,
        dvdy=wcs.dvdy,
    )


def get_obs(
    throughput,
    scene,
    psf,
    psf_star,
    psf_image_builder,
    image_builder,
    sky_background,
    seed=None,
):
    grng = galsim.BaseDeviate(seed)

    image = image_builder.get_image()
    noise_image = image_builder.get_image()

    psf_image = psf_image_builder.get_image()

    noise_sigma = image_builder.get_noise_sigma(sky_background, throughput)
    noise = galsim.GaussianNoise(grng, sigma=noise_sigma)

    image.addNoise(noise)
    noise_image.addNoise(noise)

    for galaxy in scene.galaxies:
        observed = galsim.Convolve([psf.model, galaxy])
        observed.drawImage(
            throughput,
            image,
            add_to_image=True,
        )

    for star in scene.stars:
        observed = galsim.Convolve([psf.model, star])
        observed.drawImage(
            throughput,
            image,
            add_to_image=True,
        )

    psf_image = psf.draw_image(
        psf_star,
        throughput,
        psf_image,
    )

    ormask = get_ormask(image.nrow, image.ncol)
    bmask = get_bmask(image.nrow, image.ncol)
    weight = get_weight(image.nrow, image.ncol, noise_sigma)

    wcs = get_wcs(image.scale, image.center)
    im_jac = get_jacobian(image, wcs)
    psf_jac = get_jacobian(psf_image, wcs)

    psf_obs = ngmix.Observation(
        psf_image.array,
        jacobian=psf_jac
    )

    obs = ngmix.Observation(
        image.array,
        psf=psf_obs,
        noise=noise_image.array,
        weight=weight,
        ormask=ormask,
        bmask=bmask,
        jacobian=im_jac,
    )

    return obs


def get_mbobs(
    bands,
    throughputs,
    scene,
    psf,
    psf_star,
    psf_image_builder,
    image_builder,
    sky_background,
    seed=None,
):
    seeds = utils.get_seeds(len(bands), seed=seed)
    mbobs = ngmix.MultiBandObsList()
    for _seed, band in zip(seeds, bands):
        throughput = throughputs[band]
        obslist = ngmix.ObsList()
        obs = get_obs(
            throughput,
            scene,
            psf,
            psf_star,
            psf_image_builder,
            image_builder,
            sky_background,
            seed=_seed,
        )
        obslist.append(obs)
        mbobs.append(obslist)

    return mbobs


class ObservationBuilder:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_config(cls, observation_config):
        observation_config_copy = copy.deepcopy(observation_config)
        return cls(**observation_config_copy)


