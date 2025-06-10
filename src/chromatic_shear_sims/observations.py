import copy
import logging
import time

import galsim
import numpy as np
import ngmix

from chromatic_shear_sims import utils


logger = logging.getLogger(__name__)

gsparams = galsim.GSParams(
    maximum_fft_size=16384,
    kvalue_accuracy=1e-8,
    maxk_threshold=1e-5,
)


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


def get_psf_obs(
    throughput,
    psf,
    psf_star,
    psf_image_builder,
    seed=None,
):
    psf_image = psf_image_builder.get_image()

    psf_image = psf.draw_image(
        psf_star,
        throughput,
        psf_image,
    )

    wcs = get_wcs(psf_image.scale, psf_image.center)

    psf_jac = get_jacobian(psf_image, wcs)

    psf_obs = ngmix.Observation(
        psf_image.array,
        jacobian=psf_jac
    )

    return psf_obs

def get_psf_mbobs(
    bands,
    throughputs,
    psf,
    psf_star,
    psf_image_builder,
    seed=None,
):
    _start_time = time.time()

    mbobs = ngmix.MultiBandObsList()
    for band in bands:
        throughput = throughputs[band]
        obslist = ngmix.ObsList()
        _obs_start_time = time.time()
        obs = get_psf_obs(
            throughput,
            psf,
            psf_star,
            psf_image_builder,
            seed=seed,
        )
        obs.set_meta({
            "band": band,
        })
        _obs_end_time = time.time()
        _obs_elapsed_time = _obs_end_time - _obs_start_time
        logger.info(f"made {band}-band psf observation in {_obs_elapsed_time} seconds")
        obslist.append(obs)
        mbobs.append(obslist)

    mbobs.set_meta({
        "bands": bands,
    })

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.info(f"made {''.join(bands)}-multiband psf observation in {_elapsed_time} seconds")

    return mbobs


def get_obs(
    throughput,
    psf,
    scene,
    # psf_star,
    # psf_image_builder,
    image_builder,
    sky_background,
    seed=None,
):
    grng = galsim.BaseDeviate(seed)

    image = image_builder.get_image()
    noise_image = image_builder.get_image()

    noise_sigma = image_builder.get_noise_sigma(sky_background, throughput)
    noise = galsim.GaussianNoise(grng, sigma=noise_sigma)

    image.addNoise(noise)
    noise_image.addNoise(noise)

    if scene.ngal > 0:
        _start_time = time.time()
        for (galaxy, position) in scene.galaxies:
            observed = galsim.Convolve([psf.model, galaxy]).withGSParams(gsparams)
            # _stamp = observed.drawImage(
            #     throughput,
            #     scale=image.scale,
            #     center=image.center,
            #     offset=position,
            # )
            # _bounds = _stamp.bounds & image.bounds
            # image[_bounds] += _stamp[_bounds]
            observed = observed.shift(position.x * image.scale, position.y * image.scale)
            observed.drawImage(
                throughput,
                image,
                add_to_image=True,
            )
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        logger.info(f"drew {scene.ngal} galaxies in {_elapsed_time} seconds")

    if scene.nstar > 0:
        _start_time = time.time()
        for (star, position) in scene.stars:
            observed = galsim.Convolve([psf.model, star]).withGSParams(gsparams)
            # _stamp = observed.drawImage(
            #     throughput,
            #     scale=image.scale,
            #     center=image.center,
            #     offset=position,
            # )
            # _bounds = _stamp.bounds & image.bounds
            # image[_bounds] += _stamp[_bounds]
            observed = observed.shift(position.x * image.scale, position.y * image.scale)
            observed.drawImage(
                throughput,
                image,
                add_to_image=True,
            )
        _end_time = time.time()
        _elapsed_time = _end_time - _start_time
        logger.info(f"drew {scene.nstar} stars in {_elapsed_time} seconds")

    ormask = get_ormask(image.nrow, image.ncol)
    bmask = get_bmask(image.nrow, image.ncol)
    weight = get_weight(image.nrow, image.ncol, noise_sigma)

    wcs = get_wcs(image.scale, image.center)
    im_jac = get_jacobian(image, wcs)

    # psf_obs = get_psf_obs(
    #     throughput,
    #     psf,
    #     psf_star
    # )

    obs = ngmix.Observation(
        image.array,
        # psf=psf_obs,
        psf=None,
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
    psf,
    scene,
    # psf_star,
    # psf_image_builder,
    image_builder,
    sky_background,
    seed=None,
):
    _start_time = time.time()

    seeds = utils.get_seeds(len(bands), seed=seed)
    mbobs = ngmix.MultiBandObsList()
    for obs_seed, band in zip(seeds, bands):
        throughput = throughputs[band]
        obslist = ngmix.ObsList()
        _obs_start_time = time.time()
        obs = get_obs(
            throughput,
            psf,
            scene,
            # psf_star,
            # psf_image_builder,
            image_builder,
            sky_background,
            seed=obs_seed,
        )
        obs.set_meta({
            "band": band,
        })
        _obs_end_time = time.time()
        _obs_elapsed_time = _obs_end_time - _obs_start_time
        logger.info(f"made {band}-band observation in {_obs_elapsed_time} seconds")
        obslist.append(obs)
        mbobs.append(obslist)

    mbobs.set_meta({
        "bands": bands,
    })

    _end_time = time.time()
    _elapsed_time = _end_time - _start_time
    logger.info(f"made {''.join(bands)}-multiband observation in {_elapsed_time} seconds")

    return mbobs


def with_psf_obs(mbobs, psf_mbobs):
    mbobs_copy = copy.deepcopy(mbobs)
    psf_mbobs_copy = copy.deepcopy(psf_mbobs)
    for obslist, psf_obslist in zip(mbobs_copy, psf_mbobs_copy):
        for obs, psf_obs in zip(obslist, psf_obslist):
            obs.set_psf(psf_obs)

    return mbobs_copy
