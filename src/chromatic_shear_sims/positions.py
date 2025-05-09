import copy
import logging

import galsim
import numpy as np


logger = logging.getLogger(__name__)


def _build_lattice(
    xsize,
    ysize,
    separation,
    v1,
    v2,
    rotation_angle=None,
    border=0,
):
    """
    Build a lattice from primitive translation vectors.
    Method adapted from https://stackoverflow.com/a/6145068 and
    https://github.com/alexkaz2/hexalattice/blob/master/hexalattice/hexalattice.py
    """
    # ensure that the lattice vectors are normalized
    v1 /= np.sqrt(v1.dot(v1))
    v2 /= np.sqrt(v2.dot(v2))

    # first, create a square lattice that covers the full image
    xs = np.arange(-xsize // 2, xsize // 2 + 1) * separation
    ys = np.arange(-ysize // 2, ysize // 2 + 1) * separation
    x_square, y_square = np.meshgrid(xs, ys)

    # apply the lattice vectors to the lattice
    x_lattice = v1[0] * x_square + v2[0] * y_square
    y_lattice = v1[1] * x_square + v2[1] * y_square

    # construct the rotation matrix and rotate the lattice points
    rotation = np.asarray(
        [
            [np.cos(np.radians(rotation_angle)), -np.sin(np.radians(rotation_angle))],
            [np.sin(np.radians(rotation_angle)), np.cos(np.radians(rotation_angle))],
        ]
    )
    xy_lattice_rot = (
        np.stack(
            [x_lattice.reshape(-1), y_lattice.reshape(-1)],
            axis=-1,
        )
        @ rotation.T
    )
    x_lattice_rot, y_lattice_rot = np.split(xy_lattice_rot, 2, axis=1)

    # remove points outside of the full image
    bounds_x = (-(xsize - 1) // 2, (xsize - 1) // 2)
    bounds_y = (-(ysize - 1) // 2, (ysize - 1) // 2)

    # remove points according to the border
    mask = (
        (x_lattice_rot > bounds_x[0] + border)
        & (x_lattice_rot < bounds_x[1] - border)
        & (y_lattice_rot > bounds_y[0] + border)
        & (y_lattice_rot < bounds_y[1] - border)
    )

    return x_lattice_rot[mask], y_lattice_rot[mask]


def _to_scene(xs, ys):
    """
    Convert pixel coordinates to GalSim positions
    """
    return [
        galsim.PositionD(
            x=x,
            y=y,
        )
        for (x, y) in zip(xs, ys)
    ]


def _dither(xs, ys, dither_scale, seed=None):
    """
    Apply a random uniform dithering to the pixel positions
    """
    rng = np.random.default_rng(seed)
    dither_x = [
        x + rng.uniform(-dither_scale, dither_scale)
        for x in xs
    ]
    dither_y = [
        y + rng.uniform(-dither_scale, dither_scale)
        for y in ys
    ]
    return (dither_x, dither_y)


def _get_single_pos():
    xs = [0]
    ys = [0]
    return (xs, ys)


def _get_random_pos(n, xsize, ysize, border=0, seed=None):
    rng = np.random.default_rng(seed)
    N = rng.poisson(n)
    xs = rng.uniform(
        -xsize / 2 + border,
        xsize / 2 - border,
        N,
    )
    ys = rng.uniform(
        -ysize / 2 + border,
        ysize / 2 - border,
        N,
    )
    return (xs, ys)


def _get_hex_pos(separation, xsize, ysize, border=0, seed=None):
    rng = np.random.default_rng(seed)
    v1 = np.asarray([1, 0], dtype=float)
    v2 = np.asarray([np.cos(np.radians(120)), np.sin(np.radians(120))], dtype=float)
    rotation_angle = rng.uniform(0, 360)
    xs, ys = _build_lattice(
        xsize,
        ysize,
        separation,
        v1,
        v2,
        rotation_angle,
        border,
    )
    return (xs, ys)


def get_positions(
    scene_type,
    n=None,
    separation=None,
    xsize=None,
    ysize=None,
    dither=None,
    border=0,
    seed=None,
):
    match scene_type:
        case "single":
            xs, ys = _get_single_pos()
        case "random":
            xs, ys = _get_random_pos(n, xsize, ysize, border=border, seed=seed)
        case "hex":
            xs, ys = _get_hex_pos(separation, xsize, ysize, border=border, seed=seed)
        case "none":
            xs = []
            ys = []
        case _:
            raise ValueError(f"Scene type {scene_type} not valid!")

    if dither is not None:
        xs, ys = _dither(xs, ys, dither, seed=seed)

    scene_positions = _to_scene(xs, ys)

    return scene_positions


def get_rotations(n, seed=None):
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 360, n)
    return [
        angle * galsim.degrees
        for angle in angles
    ]


class PositionBuilder:
    def __init__(self, position_type, position_kwargs, shear_scene=False):
        # positions = get_positions(*args, **kwargs)
        # self.positions = positions
        self.position_type = position_type
        self.position_kwargs = position_kwargs
        self.shear_scene = shear_scene

    @classmethod
    def from_config(cls, position_config):
        position_config_copy = copy.deepcopy(position_config)
        position_type = position_config_copy.pop("type")
        shear_scene = position_config_copy.pop("shear", False)
        return cls(position_type, position_config_copy, shear_scene=shear_scene)

    def get_positions(self, seed=None):
        return get_positions(
            self.position_type,
            **self.position_kwargs,
            seed=seed,
        )

    def get_rotations_for(self, iterable, seed=None):
        n = len(iterable)
        return get_rotations(
            n,
            seed=seed,
        )
