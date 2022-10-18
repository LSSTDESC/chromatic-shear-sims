# Copyright (c) 2012-2022 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

import logging
import numpy as np

from galsim.config.image import ImageBuilder, FlattenNoiseVariance, RegisterImageType
from galsim.config.util import GetRNG
from galsim.config.value import ParseValue, GetAllParams
from galsim.config.stamp import BuildStamps, _ParseDType
from galsim.config.noise import AddSky, AddNoise
from galsim.errors import GalSimConfigError, GalSimConfigValueError
from galsim.image import Image
from galsim import random

# This file adds image type Lattice, which builds a larger image by tiling nx x ny individual
# postage stamps.

def build_lattice(full_xsize, full_ysize, nx, ny, v1, v2, rot=None):
    """
    Build a lattice from primitive translation vectors.
    Adapted from https://stackoverflow.com/a/6145068 and
    https://github.com/alexkaz2/hexalattice/blob/master/hexalattice/hexalattice.py
    """
    # ensure that the lattice vectors are normalized
    v1 /= np.sqrt(v1.dot(v1))
    v2 /= np.sqrt(v2.dot(v2))

    # compute the size of the lattice cells
    dx = full_xsize // nx
    dy = full_ysize // ny

    # first, create a square lattice that covers the full image
    xs = np.arange(-nx, nx + 1) * dx
    ys = np.arange(-ny, ny + 1) * dy
    x_square, y_square = np.meshgrid(xs, ys)

    # apply the lattice vectors to the lattice
    x_lattice = v1[0] * x_square + v2[0] * y_square
    y_lattice = v1[1] * x_square + v2[1] * y_square

    # construct the roation matrix and rotate the lattice points
    rotation = np.array(
        [
            [np.cos(np.radians(rot)), np.sin(np.radians(rot))],
            [-np.sin(np.radians(rot)), np.cos(np.radians(rot))],
        ]
    )
    xy_lattice_rot = np.stack(
        [x_lattice.reshape(-1), y_lattice.reshape(-1)],
        axis=-1,
    ) @ rotation.T
    x_lattice_rot, y_lattice_rot = np.split(xy_lattice_rot, 2, axis=1)

    # return x_lattice_rot, y_lattice_rot

    # remove points outside of the full image
    bounds_x = (-(full_xsize - 1) // 2, (full_xsize - 1) // 2)
    bounds_y = (-(full_ysize - 1) // 2, (full_ysize - 1) // 2)

    # TODO: make the border ratio configurable
    mask = (
        (x_lattice_rot > bounds_x[0] * 0.8)
        & (x_lattice_rot < bounds_x[1] * 0.8)
        & (y_lattice_rot > bounds_y[0] * 0.8)
        & (y_lattice_rot < bounds_y[1] * 0.8)
    )

    return x_lattice_rot[mask] + bounds_x[1], y_lattice_rot[mask] + bounds_y[1]


class LatticeImageBuilder(ImageBuilder):

    def setup(self, config, base, image_num, obj_num, ignore, logger):
        """Do the initialization and setup for building the image.

        This figures out the size that the image will be, but doesn't actually build it yet.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            ignore:     A list of parameters that are allowed to be in config that we can
                        ignore here. i.e. it won't be an error if these parameters are present.
            logger:     If given, a logger object to log progress.

        Returns:
            xsize, ysize
        """
        logger.debug('image %d: Building Lattice: image, obj = %d,%d',image_num,image_num,obj_num)

        extra_ignore = [ 'image_pos' ] # We create this below, so on subequent passes, we ignore it.
        req = { 'nx_tiles' : int , 'ny_tiles' : int }
        opt = { 'stamp_size' : int , 'stamp_xsize' : int , 'stamp_ysize' : int ,
                'border' : int , 'xborder' : int , 'yborder' : int , 'order' : str }
        params = GetAllParams(config, base, req=req, opt=opt, ignore=ignore+extra_ignore)[0]

        self.nx_tiles = params['nx_tiles']  # We'll need this again later, so save them in self.
        self.ny_tiles = params['ny_tiles']
        logger.debug('image %d: n_tiles = %d, %d',image_num,self.nx_tiles,self.ny_tiles)

        stamp_size = params.get('stamp_size',0)
        self.stamp_xsize = params.get('stamp_xsize',stamp_size)
        self.stamp_ysize = params.get('stamp_ysize',stamp_size)

        if (self.stamp_xsize <= 0) or (self.stamp_ysize <= 0):
            raise GalSimConfigError(
                "Both image.stamp_xsize and image.stamp_ysize need to be defined and > 0.")

        border = params.get("border",0)
        self.xborder = params.get("xborder",border)
        self.yborder = params.get("yborder",border)

        # Store the net grid spacing in the config dict as grid_xsize, grid_ysize for things like
        # PowerSpectrum that might want to know the grid spacing.
        base['grid_xsize'] = self.stamp_xsize + self.xborder
        base['grid_ysize'] = self.stamp_ysize + self.yborder

        # self.do_noise_in_stamps = self.xborder >= 0 and self.yborder >= 0
        # TODO: Note: if one of these is < 0 and the other is > 0, then
        #       this will add noise to the border region.  Not exactly the
        #       design, but I didn't bother to do the bookkeeping right to
        #       make the borders pure 0 in that case.

        full_xsize = (self.stamp_xsize + self.xborder) * self.nx_tiles - self.xborder
        full_ysize = (self.stamp_ysize + self.yborder) * self.ny_tiles - self.yborder

        # If image_force_xsize and image_force_ysize were set in config, make sure it matches.
        if ( ('image_force_xsize' in base and full_xsize != base['image_force_xsize']) or
             ('image_force_ysize' in base and full_ysize != base['image_force_ysize']) ):
            raise GalSimConfigError(
                "Unable to reconcile required image xsize and ysize with provided "
                "nx_tiles=%d, ny_tiles=%d, xborder=%d, yborder=%d\n"
                "Calculated full_size = (%d,%d) != required (%d,%d)."%(
                    self.nx_tiles, self.ny_tiles, self.xborder, self.yborder,
                    full_xsize, full_ysize, base['image_force_xsize'],base['image_force_ysize']))

        return full_xsize, full_ysize


    def buildImage(self, config, base, image_num, obj_num, logger):
        """
        Build an Image consisting of a tiled array of postage stamps.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            obj_num:    The first object number in the image.
            logger:     If given, a logger object to log progress.

        Returns:
            the final image and the current noise variance in the image as a tuple
        """
        full_xsize = base['image_xsize']
        full_ysize = base['image_ysize']
        wcs = base['wcs']

        dtype = _ParseDType(config, base)
        full_image = Image(full_xsize, full_ysize, dtype=dtype)
        full_image.setOrigin(base['image_origin'])
        full_image.wcs = wcs
        full_image.setZero()
        base['current_image'] = full_image

        # nobjects = self.nx_tiles * self.ny_tiles  # TODO: use scattered format for this

        # Make a list of ix,iy values according to the specified order:
        rng = np.random.default_rng()
        theta = rng.uniform(0, 360)  # TODO: grab from config
        v1 = np.asarray([1, 0], dtype=float)  # TODO: grab from config
        v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)  # TODO: grab from config
        x_lattice, y_lattice = build_lattice(full_xsize-1, full_ysize-1, self.nx_tiles, self.ny_tiles, v1, v2, theta)
        nobjects = len(x_lattice)

        # Define a 'image_pos' field so the stamps can set their position appropriately in case
        # we need it for PowerSpectum or NFWHalo.
        config['image_pos'] = {
            'type' : 'XY',
            'x' : { 'type' : 'List',
                    'items' : x_lattice
                  },
            'y' : { 'type' : 'List',
                    'items' : y_lattice
                  }
        }

        stamps, current_vars = BuildStamps(
                nobjects, base, logger=logger, obj_num=obj_num,
                xsize=self.stamp_xsize, ysize=self.stamp_ysize, do_noise=False)

        base['index_key'] = 'image_num'

        for k in range(nobjects):
            # This is our signal that the object was skipped.
            if stamps[k] is None: continue
            bounds = stamps[k].bounds & full_image.bounds
            logger.debug('image %d: full bounds = %s',image_num,str(full_image.bounds))
            logger.debug('image %d: stamp %d bounds = %s',image_num,k,str(stamps[k].bounds))
            logger.debug('image %d: Overlap = %s',image_num,str(bounds))
            if bounds.isDefined():
                full_image[bounds] += stamps[k][bounds]
            else:
                logger.info(
                    "Object centered at (%d,%d) is entirely off the main image, "
                    "whose bounds are (%d,%d,%d,%d)."%(
                        stamps[k].center.x, stamps[k].center.y,
                        full_image.bounds.xmin, full_image.bounds.xmax,
                        full_image.bounds.ymin, full_image.bounds.ymax))

        # Bring the image so far up to a flat noise variance
        current_var = FlattenNoiseVariance(
                base, full_image, stamps, current_vars, logger)

        return full_image, current_var

    def makeTasks(self, config, base, jobs, logger):
        """Turn a list of jobs into a list of tasks.

        Here we just have one job per task.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            jobs:       A list of jobs to split up into tasks.  Each job in the list is a
                        dict of parameters that includes 'image_num' and 'obj_num'.
            logger:     If given, a logger object to log progress.

        Returns:
            a list of tasks
        """
        return [ [ (job, k) ] for k, job in enumerate(jobs) ]

    def addNoise(self, image, config, base, image_num, obj_num, current_var, logger):
        """Add the final noise to a Lattice image

        Parameters:
            image:          The image onto which to add the noise.
            config:         The configuration dict for the image field.
            base:           The base configuration dict.
            image_num:      The current image number.
            obj_num:        The first object number in the image.
            current_var:    The current noise variance in each postage stamps.
            logger:         If given, a logger object to log progress.
        """
        base['current_noise_image'] = base['current_image']
        AddSky(base,image)
        AddNoise(base,image,current_var,logger)

    def getNObj(self, config, base, image_num, logger=None):
        """Get the number of objects that will be built for this image.

        Parameters:
            config:     The configuration dict for the image field.
            base:       The base configuration dict.
            image_num:  The current image number.
            logger:     If given, a logger object to log progress.

        Returns:
            the number of objects
        """
        orig_index_key = base.get('index_key',None)
        base['index_key'] = 'image_num'
        base['image_num'] = image_num

        if 'nx_tiles' not in config or 'ny_tiles' not in config:
            raise GalSimConfigError(
                "Attributes nx_tiles and ny_tiles are required for image.type = Lattice")
        nx = ParseValue(config,'nx_tiles',base,int)[0]
        ny = ParseValue(config,'ny_tiles',base,int)[0]
        base['index_key'] = orig_index_key
        return nx*ny

# Register this as a valid image type
RegisterImageType('Lattice', LatticeImageBuilder())
