import argparse
import functools
import logging
import multiprocessing
import threading
import os

import ngmix
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as ft
import yaml
from rich.progress import track

from chromatic_shear_sims import utils
from . import log_util, name_util, plot_util


def task(aggregate_path, dg, dc, color, color_indices, resample=False, seed=None):
    # aggregates = ft.read_table(aggregate_path)
    aggregates = ds.dataset(aggregate_path).to_table()
    if resample:
        rng = np.random.default_rng(seed)
        resample_indices = rng.choice(len(aggregates), len(aggregates), replace=True)
        aggregates = aggregates.take(resample_indices)
    m_bootstrap, c_bootstrap = compute_bias(aggregates, dg, dc, color_indices=color_indices)
    m_bootstrap_c1, c_bootstrap_c1 = compute_bias_chromatic(aggregates, dg, dc, color, color_indices=color_indices, order=1)
    m_bootstrap_c2, c_bootstrap_c2 = compute_bias_chromatic(aggregates, dg, dc, color, color_indices=color_indices, order=2)
    return (
        (m_bootstrap, c_bootstrap),
        (m_bootstrap_c1, c_bootstrap_c1),
        (m_bootstrap_c2, c_bootstrap_c2)
    )


def weighted_average(array, weights=None):
    if weights is not None:
        wavg = pc.divide(pc.sum(pc.multiply(array, weights)), pc.sum(weights))
    else:
        wavg = pc.mean(array)

    return wavg


def compute_e_chromatic(results, color_index=None):
    color_key = f"c{color_index}"

    # NOSHEAR
    p_c1_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key}:mdet_step=noshear:weight"]).as_py()

    m_c1_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key}:mdet_step=noshear:weight"]).as_py()

    return np.array([p_c1_e1_noshear, p_c1_e2_noshear]), np.array([m_c1_e1_noshear, m_c1_e2_noshear])


def compute_R_chromatic(results, dg, color_index=None):
    color_key = f"c{color_index}"

    p_c1_R11 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=1p:e1"], results[f"shear_step=plus:color_step={color_key}:mdet_step=1p:weight"]),
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=1m:e1"], results[f"shear_step=plus:color_step={color_key}:mdet_step=1m:weight"])
    ), 2 * dg).as_py()
    p_c1_R12 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=2p:e1"], results[f"shear_step=plus:color_step={color_key}:mdet_step=2p:weight"]),
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=2m:e1"], results[f"shear_step=plus:color_step={color_key}:mdet_step=2m:weight"])
    ), 2 * dg).as_py()
    p_c1_R21 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=1p:e2"], results[f"shear_step=plus:color_step={color_key}:mdet_step=1p:weight"]),
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=1m:e2"], results[f"shear_step=plus:color_step={color_key}:mdet_step=1m:weight"])
    ), 2 * dg).as_py()
    p_c1_R22 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=2p:e2"], results[f"shear_step=plus:color_step={color_key}:mdet_step=2p:weight"]),
        weighted_average(results[f"shear_step=plus:color_step={color_key}:mdet_step=2m:e2"], results[f"shear_step=plus:color_step={color_key}:mdet_step=2m:weight"])
    ), 2 * dg).as_py()

    m_c1_R11 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=1p:e1"], results[f"shear_step=minus:color_step={color_key}:mdet_step=1p:weight"]),
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=1m:e1"], results[f"shear_step=minus:color_step={color_key}:mdet_step=1m:weight"])
    ), 2 * dg).as_py()
    m_c1_R12 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=2p:e1"], results[f"shear_step=minus:color_step={color_key}:mdet_step=2p:weight"]),
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=2m:e1"], results[f"shear_step=minus:color_step={color_key}:mdet_step=2m:weight"])
    ), 2 * dg).as_py()
    m_c1_R21 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=1p:e2"], results[f"shear_step=minus:color_step={color_key}:mdet_step=1p:weight"]),
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=1m:e2"], results[f"shear_step=minus:color_step={color_key}:mdet_step=1m:weight"])
    ), 2 * dg).as_py()
    m_c1_R22 = pc.divide(pc.subtract(
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=2p:e2"], results[f"shear_step=minus:color_step={color_key}:mdet_step=2p:weight"]),
        weighted_average(results[f"shear_step=minus:color_step={color_key}:mdet_step=2m:e2"], results[f"shear_step=minus:color_step={color_key}:mdet_step=2m:weight"])
    ), 2 * dg).as_py()

    return np.array([[p_c1_R11, p_c1_R12], [p_c1_R21, p_c1_R22]]), np.array([[m_c1_R11, m_c1_R12], [m_c1_R21, m_c1_R22]])


def compute_de(results, dg, dc, color, color_indices=None, order=1):
    color_key_0 = f"c{color_indices[0]}"
    color_key_1 = f"c{color_indices[1]}"
    color_key_2 = f"c{color_indices[2]}"

    # c0
    p_c0_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e1c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e1c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e2c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e2c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e1cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e1cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e2cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e2cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()

    m_c0_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e1c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e1c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e2c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e2c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e1cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e1cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e2cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e2cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()

    # c1
    p_c1_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e1c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e1c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e2c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e1cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e1cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e2cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()

    m_c1_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e1c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e1c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e2c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e1cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e1cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e2cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()

    # c2
    p_c2_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e1c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e1c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e2c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e2c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e1cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e1cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e2cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e2cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()

    m_c2_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e1c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e1c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e2c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e2c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e1cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e1cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e2cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e2cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()

    match order:
        case 1:
            p_de_1 = (
                p_c2_e1c_noshear - p_c0_e1c_noshear - color * (
                    p_c2_e1_noshear - p_c0_e1_noshear
                )
            ) / (2 * dc)
            p_de_2 = (
                p_c2_e2c_noshear - p_c0_e2c_noshear - color * (
                    p_c2_e1_noshear - p_c0_e1_noshear
                )
            ) / (2 * dc)

            m_de_1 = (
                m_c2_e1c_noshear - m_c0_e1c_noshear - color * (
                    m_c2_e1_noshear - m_c0_e1_noshear
                )
            ) / (2 * dc)
            m_de_2 = (
                m_c2_e2c_noshear - m_c0_e2c_noshear - color * (
                    m_c2_e1_noshear - m_c0_e1_noshear
                )
            ) / (2 * dc)
        case 2:
            p_de_1 = (
                p_c2_e1c_noshear - p_c0_e1c_noshear - color * (
                    p_c2_e1_noshear - p_c0_e1_noshear
                )
            ) / (2 * dc) + (
                p_c2_e1cc_noshear - 2 * p_c1_e1cc_noshear + p_c0_e1cc_noshear  \
                - 2 * color * (p_c2_e1c_noshear - 2 * p_c1_e1c_noshear + p_c0_e1c_noshear)  \
                + color**2 * (p_c2_e1_noshear - 2 * p_c1_e1_noshear + p_c0_e1_noshear)
            ) / (2 * dc**2)
            p_de_2 = (
                p_c2_e2c_noshear - p_c0_e2c_noshear - color * (
                    p_c2_e1_noshear - p_c0_e1_noshear
                )
            ) / (2 * dc) + (
                p_c2_e2cc_noshear - 2 * p_c1_e2cc_noshear + p_c0_e2cc_noshear  \
                - 2 * color * (p_c2_e2c_noshear - 2 * p_c1_e2c_noshear + p_c0_e2c_noshear)  \
                + color**2 * (p_c2_e2_noshear - 2 * p_c1_e2_noshear + p_c0_e2_noshear)
            ) / (2 * dc**2)

            m_de_1 = (
                m_c2_e1c_noshear - m_c0_e1c_noshear - color * (
                    m_c2_e1_noshear - m_c0_e1_noshear
                )
            ) / (2 * dc) + (
                m_c2_e1cc_noshear - 2 * m_c1_e1cc_noshear + m_c0_e1cc_noshear  \
                - 2 * color * (m_c2_e1c_noshear - 2 * m_c1_e1c_noshear + m_c0_e1c_noshear)  \
                + color**2 * (m_c2_e1_noshear - 2 * m_c1_e1_noshear + m_c0_e1_noshear)
            ) / (2 * dc**2)
            m_de_2 = (
                m_c2_e2c_noshear - m_c0_e2c_noshear - color * (
                    m_c2_e1_noshear - m_c0_e1_noshear
                )
            ) / (2 * dc) + (
                m_c2_e2cc_noshear - 2 * m_c1_e2cc_noshear + m_c0_e2cc_noshear  \
                - 2 * color * (m_c2_e2c_noshear - 2 * m_c1_e2c_noshear + m_c0_e2c_noshear)  \
                + color**2 * (m_c2_e2_noshear - 2 * m_c1_e2_noshear + m_c0_e2_noshear)
            ) / (2 * dc**2)

    return np.array([p_de_1, p_de_2]), np.array([m_de_1, m_de_2])


def compute_dR(results, dg, dc, color, color_indices=None, order=1):
    color_key_0 = f"c{color_indices[0]}"
    color_key_1 = f"c{color_indices[1]}"
    color_key_2 = f"c{color_indices[2]}"

    #---------------------------------------------------------------------------
    # plus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    p_c0_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e1c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e1c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e2c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e2c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e1cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e1cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    p_c0_e2cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:e2cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()

    # 1p
    p_c0_e1_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:e1"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    p_c0_e2_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:e2"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    p_c0_e1c_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:e1c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    p_c0_e2c_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:e2c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    p_c0_e1cc_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:e1cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    p_c0_e2cc_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:e2cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()

    # 1m
    p_c0_e1_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:e1"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    p_c0_e2_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:e2"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    p_c0_e1c_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:e1c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    p_c0_e2c_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:e2c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    p_c0_e1cc_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:e1cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    p_c0_e2cc_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:e2cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()

    # 2p
    p_c0_e1_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:e1"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    p_c0_e2_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:e2"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    p_c0_e1c_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:e1c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    p_c0_e2c_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:e2c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    p_c0_e1cc_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:e1cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    p_c0_e2cc_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:e2cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()

    # 2m
    p_c0_e1_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:e1"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    p_c0_e2_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:e2"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    p_c0_e1c_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:e1c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    p_c0_e2c_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:e2c"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    p_c0_e1cc_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:e1cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    p_c0_e2cc_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:e2cc"], results[f"shear_step=plus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    p_c1_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e1c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e1c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e2c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e1cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e1cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    p_c1_e2cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:e2cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()

    # 1p
    p_c1_e1_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:e1"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    p_c1_e2_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:e2"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    p_c1_e1c_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:e1c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    p_c1_e2c_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:e2c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    p_c1_e1cc_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:e1cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    p_c1_e2cc_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:e2cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()

    # 1m
    p_c1_e1_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:e1"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    p_c1_e2_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:e2"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    p_c1_e1c_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:e1c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    p_c1_e2c_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:e2c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    p_c1_e1cc_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:e1cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    p_c1_e2cc_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:e2cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()

    # 2p
    p_c1_e1_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:e1"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    p_c1_e2_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:e2"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    p_c1_e1c_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:e1c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    p_c1_e2c_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:e2c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    p_c1_e1cc_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:e1cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    p_c1_e2cc_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:e2cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()

    # 2m
    p_c1_e1_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:e1"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    p_c1_e2_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:e2"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    p_c1_e1c_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:e1c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    p_c1_e2c_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:e2c"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    p_c1_e1cc_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:e1cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    p_c1_e2cc_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:e2cc"], results[f"shear_step=plus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    p_c2_e1_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e1"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e2_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e2"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e1c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e1c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e2c_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e2c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e1cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e1cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    p_c2_e2cc_noshear = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:e2cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()

    # 1p
    p_c2_e1_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:e1"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    p_c2_e2_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:e2"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    p_c2_e1c_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:e1c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    p_c2_e2c_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:e2c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    p_c2_e1cc_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:e1cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    p_c2_e2cc_1p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:e2cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()

    # 1m
    p_c2_e1_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:e1"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    p_c2_e2_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:e2"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    p_c2_e1c_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:e1c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    p_c2_e2c_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:e2c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    p_c2_e1cc_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:e1cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    p_c2_e2cc_1m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:e2cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()

    # 2p
    p_c2_e1_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:e1"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    p_c2_e2_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:e2"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    p_c2_e1c_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:e1c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    p_c2_e2c_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:e2c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    p_c2_e1cc_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:e1cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    p_c2_e2cc_2p = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:e2cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()

    # 2m
    p_c2_e1_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:e1"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    p_c2_e2_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:e2"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    p_c2_e1c_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:e1c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    p_c2_e2c_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:e2c"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    p_c2_e1cc_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:e1cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    p_c2_e2cc_2m = weighted_average(results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:e2cc"], results[f"shear_step=plus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()

    #---------------------------------------------------------------------------
    # minus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    m_c0_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e1c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e1c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e2c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e2c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e1cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e1cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()
    m_c0_e2cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:e2cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=noshear:weight"]).as_py()

    # 1p
    m_c0_e1_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:e1"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    m_c0_e2_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:e2"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    m_c0_e1c_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:e1c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    m_c0_e2c_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:e2c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    m_c0_e1cc_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:e1cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()
    m_c0_e2cc_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:e2cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1p:weight"]).as_py()

    # 1m
    m_c0_e1_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:e1"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    m_c0_e2_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:e2"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    m_c0_e1c_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:e1c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    m_c0_e2c_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:e2c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    m_c0_e1cc_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:e1cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()
    m_c0_e2cc_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:e2cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=1m:weight"]).as_py()

    # 2p
    m_c0_e1_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:e1"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    m_c0_e2_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:e2"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    m_c0_e1c_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:e1c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    m_c0_e2c_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:e2c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    m_c0_e1cc_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:e1cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()
    m_c0_e2cc_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:e2cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2p:weight"]).as_py()

    # 2m
    m_c0_e1_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:e1"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    m_c0_e2_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:e2"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    m_c0_e1c_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:e1c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    m_c0_e2c_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:e2c"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    m_c0_e1cc_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:e1cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()
    m_c0_e2cc_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:e2cc"], results[f"shear_step=minus:color_step={color_key_0}:mdet_step=2m:weight"]).as_py()

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    m_c1_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e1c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e1c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e2c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e1cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e1cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()
    m_c1_e2cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:e2cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=noshear:weight"]).as_py()

    # 1p
    m_c1_e1_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:e1"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    m_c1_e2_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:e2"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    m_c1_e1c_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:e1c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    m_c1_e2c_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:e2c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    m_c1_e1cc_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:e1cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()
    m_c1_e2cc_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:e2cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1p:weight"]).as_py()

    # 1m
    m_c1_e1_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:e1"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    m_c1_e2_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:e2"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    m_c1_e1c_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:e1c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    m_c1_e2c_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:e2c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    m_c1_e1cc_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:e1cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()
    m_c1_e2cc_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:e2cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=1m:weight"]).as_py()

    # 2p
    m_c1_e1_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:e1"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    m_c1_e2_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:e2"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    m_c1_e1c_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:e1c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    m_c1_e2c_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:e2c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    m_c1_e1cc_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:e1cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()
    m_c1_e2cc_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:e2cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2p:weight"]).as_py()

    # 2m
    m_c1_e1_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:e1"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    m_c1_e2_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:e2"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    m_c1_e1c_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:e1c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    m_c1_e2c_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:e2c"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    m_c1_e1cc_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:e1cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()
    m_c1_e2cc_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:e2cc"], results[f"shear_step=minus:color_step={color_key_1}:mdet_step=2m:weight"]).as_py()

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    m_c2_e1_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e1"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e2_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e2"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e1c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e1c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e2c_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e2c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e1cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e1cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()
    m_c2_e2cc_noshear = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:e2cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=noshear:weight"]).as_py()

    # 1p
    m_c2_e1_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:e1"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    m_c2_e2_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:e2"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    m_c2_e1c_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:e1c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    m_c2_e2c_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:e2c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    m_c2_e1cc_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:e1cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()
    m_c2_e2cc_1p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:e2cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1p:weight"]).as_py()

    # 1m
    m_c2_e1_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:e1"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    m_c2_e2_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:e2"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    m_c2_e1c_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:e1c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    m_c2_e2c_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:e2c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    m_c2_e1cc_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:e1cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()
    m_c2_e2cc_1m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:e2cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=1m:weight"]).as_py()

    # 2p
    m_c2_e1_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:e1"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    m_c2_e2_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:e2"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    m_c2_e1c_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:e1c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    m_c2_e2c_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:e2c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    m_c2_e1cc_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:e1cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()
    m_c2_e2cc_2p = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:e2cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2p:weight"]).as_py()

    # 2m
    m_c2_e1_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:e1"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    m_c2_e2_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:e2"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    m_c2_e1c_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:e1c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    m_c2_e2c_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:e2c"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    m_c2_e1cc_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:e1cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()
    m_c2_e2cc_2m = weighted_average(results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:e2cc"], results[f"shear_step=minus:color_step={color_key_2}:mdet_step=2m:weight"]).as_py()

    #---------------------------------------------------------------------------

    match order:
        case 1:
            dR11_p = (
                p_c2_e1c_1p - p_c2_e1c_1m - p_c0_e1c_1p + p_c0_e1c_1m - color * (
                    p_c2_e1_1p - p_c2_e1_1m - p_c0_e1_1p + p_c0_e1_1m
                )
            ) / (2 * dg * 2 * dc)
            dR12_p = (
                p_c2_e1c_2p - p_c2_e1c_2m - p_c0_e1c_2p + p_c0_e1c_2m - color * (
                    p_c2_e1_2p - p_c2_e1_2m - p_c0_e1_2p + p_c0_e1_2m
                )
            ) / (2 * dg * 2 * dc)
            dR21_p = (
                p_c2_e2c_1p - p_c2_e2c_1m - p_c0_e2c_1p + p_c0_e2c_1m - color * (
                    p_c2_e2_1p - p_c2_e2_1m - p_c0_e2_1p + p_c0_e2_1m
                )
            ) / (2 * dg * 2 * dc)
            dR22_p = (
                p_c2_e2c_2p - p_c2_e2c_2m - p_c0_e2c_2p + p_c0_e2c_2m - color * (
                    p_c2_e2_2p - p_c2_e2_2m - p_c0_e2_2p + p_c0_e2_2m
                )
            ) / (2 * dg * 2 * dc)

            dR11_m = (
                m_c2_e1c_1p - m_c2_e1c_1m - m_c0_e1c_1p + m_c0_e1c_1m - color * (
                    m_c2_e1_1p - m_c2_e1_1m - m_c0_e1_1p + m_c0_e1_1m
                )
            ) / (2 * dg * 2 * dc)
            dR12_m = (
                m_c2_e1c_2p - m_c2_e1c_2m - m_c0_e1c_2p + m_c0_e1c_2m - color * (
                    m_c2_e1_2p - m_c2_e1_2m - m_c0_e1_2p + m_c0_e1_2m
                )
            ) / (2 * dg * 2 * dc)
            dR21_m = (
                m_c2_e2c_1p - m_c2_e2c_1m - m_c0_e2c_1p + m_c0_e2c_1m - color * (
                    m_c2_e2_1p - m_c2_e2_1m - m_c0_e2_1p + m_c0_e2_1m
                )
            ) / (2 * dg * 2 * dc)
            dR22_m = (
                m_c2_e2c_2p - m_c2_e2c_2m - m_c0_e2c_2p + m_c0_e2c_2m - color * (
                    m_c2_e2_2p - m_c2_e2_2m - m_c0_e2_2p + m_c0_e2_2m
                )
            ) / (2 * dg * 2 * dc)
        case 2:
            dR11_p = (
                p_c2_e1c_1p - p_c2_e1c_1m - p_c0_e1c_1p + p_c0_e1c_1m - color * (
                    p_c2_e1_1p - p_c2_e1_1m - p_c0_e1_1p + p_c0_e1_1m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e1cc_1p - p_c2_e1cc_1m - 2 * (p_c1_e1cc_1p - p_c1_e1cc_1m) + p_c0_e1cc_1p - p_c0_e1cc_1m \
                - 2 * color * (p_c2_e1c_1p - p_c2_e1c_1m - 2 * (p_c1_e1c_1p - p_c1_e1c_1m) + p_c0_e1c_1p - p_c0_e1c_1m) \
                + color**2 * (p_c2_e1_1p - p_c2_e1_1m - 2 * (p_c1_e1_1p - p_c1_e1_1m) + p_c0_e1_1p - p_c0_e1_1m)
            ) / (2 * dg * 2 * dc**2)
            dR12_p = (
                p_c2_e1c_2p - p_c2_e1c_2m - p_c0_e1c_2p + p_c0_e1c_2m - color * (
                    p_c2_e1_2p - p_c2_e1_2m - p_c0_e1_2p + p_c0_e1_2m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e1cc_2p - p_c2_e1cc_2m - 2 * (p_c1_e1cc_2p - p_c1_e1cc_2m) + p_c0_e1cc_2p - p_c0_e1cc_2m \
                - 2 * color * (p_c2_e1c_2p - p_c2_e1c_2m - 2 * (p_c1_e1c_2p - p_c1_e1c_2m) + p_c0_e1c_2p - p_c0_e1c_2m) \
                + color**2 * (p_c2_e1_2p - p_c2_e1_2m - 2 * (p_c1_e1_2p - p_c1_e1_2m) + p_c0_e1_2p - p_c0_e1_2m)
            ) / (2 * dg * 2 * dc**2)
            dR21_p = (
                p_c2_e2c_1p - p_c2_e2c_1m - p_c0_e2c_1p + p_c0_e2c_1m - color * (
                    p_c2_e2_1p - p_c2_e2_1m - p_c0_e2_1p + p_c0_e2_1m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e2cc_1p - p_c2_e2cc_1m - 2 * (p_c1_e2cc_1p - p_c1_e2cc_1m) + p_c0_e2cc_1p - p_c0_e2cc_1m \
                - 2 * color * (p_c2_e2c_1p - p_c2_e2c_1m - 2 * (p_c1_e2c_1p - p_c1_e2c_1m) + p_c0_e2c_1p - p_c0_e2c_1m) \
                + color**2 * (p_c2_e2_1p - p_c2_e2_1m - 2 * (p_c1_e2_1p - p_c1_e2_1m) + p_c0_e2_1p - p_c0_e2_1m)
            ) / (2 * dg * 2 * dc**2)
            dR22_p = (
                p_c2_e2c_2p - p_c2_e2c_2m - p_c0_e2c_2p + p_c0_e2c_2m - color * (
                    p_c2_e2_2p - p_c2_e2_2m - p_c0_e2_2p + p_c0_e2_2m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e2cc_2p - p_c2_e2cc_2m - 2 * (p_c1_e2cc_2p - p_c1_e2cc_2m) + p_c0_e2cc_2p - p_c0_e2cc_2m \
                - 2 * color * (p_c2_e2c_2p - p_c2_e2c_2m - 2 * (p_c1_e2c_2p - p_c1_e2c_2m) + p_c0_e2c_2p - p_c0_e2c_2m) \
                + color**2 * (p_c2_e2_2p - p_c2_e2_2m - 2 * (p_c1_e2_2p - p_c1_e2_2m) + p_c0_e2_2p - p_c0_e2_2m)
            ) / (2 * dg * 2 * dc**2)

            dR11_m = (
                m_c2_e1c_1p - m_c2_e1c_1m - m_c0_e1c_1p + m_c0_e1c_1m - color * (
                    m_c2_e1_1p - m_c2_e1_1m - m_c0_e1_1p + m_c0_e1_1m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e1cc_1p - m_c2_e1cc_1m - 2 * (m_c1_e1cc_1p - m_c1_e1cc_1m) + m_c0_e1cc_1p - m_c0_e1cc_1m \
                - 2 * color * (m_c2_e1c_1p - m_c2_e1c_1m - 2 * (m_c1_e1c_1p - m_c1_e1c_1m) + m_c0_e1c_1p - m_c0_e1c_1m) \
                + color**2 * (m_c2_e1_1p - m_c2_e1_1m - 2 * (m_c1_e1_1p - m_c1_e1_1m) + m_c0_e1_1p - m_c0_e1_1m)
            ) / (2 * dg * 2 * dc**2)
            dR12_m = (
                m_c2_e1c_2p - m_c2_e1c_2m - m_c0_e1c_2p + m_c0_e1c_2m - color * (
                    m_c2_e1_2p - m_c2_e1_2m - m_c0_e1_2p + m_c0_e1_2m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e1cc_2p - m_c2_e1cc_2m - 2 * (m_c1_e1cc_2p - m_c1_e1cc_2m) + m_c0_e1cc_2p - m_c0_e1cc_2m \
                - 2 * color * (m_c2_e1c_2p - m_c2_e1c_2m - 2 * (m_c1_e1c_2p - m_c1_e1c_2m) + m_c0_e1c_2p - m_c0_e1c_2m) \
                + color**2 * (m_c2_e1_2p - m_c2_e1_2m - 2 * (m_c1_e1_2p - m_c1_e1_2m) + m_c0_e1_2p - m_c0_e1_2m)
            ) / (2 * dg * 2 * dc**2)
            dR21_m = (
                m_c2_e2c_1p - m_c2_e2c_1m - m_c0_e2c_1p + m_c0_e2c_1m - color * (
                    m_c2_e2_1p - m_c2_e2_1m - m_c0_e2_1p + m_c0_e2_1m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e2cc_1p - m_c2_e2cc_1m - 2 * (m_c1_e2cc_1p - m_c1_e2cc_1m) + m_c0_e2cc_1p - m_c0_e2cc_1m \
                - 2 * color * (m_c2_e2c_1p - m_c2_e2c_1m - 2 * (m_c1_e2c_1p - m_c1_e2c_1m) + m_c0_e2c_1p - m_c0_e2c_1m) \
                + color**2 * (m_c2_e2_1p - m_c2_e2_1m - 2 * (m_c1_e2_1p - m_c1_e2_1m) + m_c0_e2_1p - m_c0_e2_1m)
            ) / (2 * dg * 2 * dc**2)
            dR22_m = (
                m_c2_e2c_2p - m_c2_e2c_2m - m_c0_e2c_2p + m_c0_e2c_2m - color * (
                    m_c2_e2_2p - m_c2_e2_2m - m_c0_e2_2p + m_c0_e2_2m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e2cc_2p - m_c2_e2cc_2m - 2 * (m_c1_e2cc_2p - m_c1_e2cc_2m) + m_c0_e2cc_2p - m_c0_e2cc_2m \
                - 2 * color * (m_c2_e2c_2p - m_c2_e2c_2m - 2 * (m_c1_e2c_2p - m_c1_e2c_2m) + m_c0_e2c_2p - m_c0_e2c_2m) \
                + color**2 * (m_c2_e2_2p - m_c2_e2_2m - 2 * (m_c1_e2_2p - m_c1_e2_2m) + m_c0_e2_2p - m_c0_e2_2m)
            ) / (2 * dg * 2 * dc**2)

    return (
        np.array([
            [dR11_p, dR12_p],
            [dR21_p, dR22_p],
        ]),
        np.array([
            [dR11_m, dR12_m],
            [dR21_m, dR22_m],
        ]),
    )


def compute_bias(batch, dg, dc, color_indices=None):
    e_p, e_m = compute_e_chromatic(batch, color_index=color_indices[1])
    R_p, R_m = compute_R_chromatic(batch, dg, color_index=color_indices[1])

    # g_p = np.linalg.inv(R_p) @ e_p
    # g_m = np.linalg.inv(R_m) @ e_m

    g_p = e_p / np.diag(R_p)
    g_m = e_m / np.diag(R_m)

    m = (g_p - g_m)[0] / 2 / 0.02 - 1

    c = (g_p + g_m)[1] / 2

    return m, c


def compute_bias_chromatic(batch, dg, dc, color, color_indices=None, order=1):
    e_p, e_m = compute_e_chromatic(batch, color_index=color_indices[1])
    R_p, R_m = compute_R_chromatic(batch, dg, color_index=color_indices[1])

    de_p, de_m = compute_de(batch, dg, dc, color, color_indices=color_indices, order=order)
    dR_p, dR_m = compute_dR(batch, dg, dc, color, color_indices=color_indices, order=order)

    # g_p = np.linalg.inv(R_p + dR_p) @ (e_p + de_p)
    # g_m = np.linalg.inv(R_m + dR_m) @ (e_m + de_m)

    g_p = (e_p + de_p) / np.diag(R_p + dR_p)
    g_m = (e_m + de_m) / np.diag(R_m + dR_m)

    m = (g_p - g_m)[0] / 2 / 0.02 - 1

    c = (g_p + g_m)[1] / 2

    return m, c


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "output",
        type=str,
        help="output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed [int]",
    )
    parser.add_argument(
        "--n_resample",
        type=int,
        required=False,
        default=1000,
        help="Number of resample iterations"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of jobs to run [int; 1]",
    )
    parser.add_argument(
        "--log_level",
        type=int,
        required=False,
        default=2,
        help="logging level [int; 2]",
    )
    return parser.parse_args()


def main():
    args = get_args()

    log_level = log_util.get_level(args.log_level)
    logging.basicConfig(format=log_util.FORMAT, level=log_level)

    config_file = args.config
    seed = args.seed
    n_jobs = args.n_jobs
    n_resample = args.n_resample

    config_name = name_util.get_config_name(config_file)
    output_path = name_util.get_output_path(args.output, args.config)
    aggregate_path = name_util.get_aggregate_path(args.output, args.config)

    pa.set_cpu_count(1)
    pa.set_io_thread_count(2)

    rng = np.random.default_rng(seed)

    with open(config_file) as fp:
        config = yaml.safe_load(fp)

    dg = ngmix.metacal.DEFAULT_STEP

    psf_colors = config["measurement"].get("colors")
    psf_color_indices = config["measurement"].get("color_indices")
    dc = (psf_colors[psf_color_indices[2]] - psf_colors[psf_color_indices[0]]) / 2.
    color = psf_colors[psf_color_indices[1]]
    print(f"psf_colors: {[psf_colors[i] for i in psf_color_indices]}")

    print(f"reading aggregates from {aggregate_path}")

    # aggregates = ft.read_table(aggregate_path)
    # m_mean, c_mean = compute_bias(aggregates, dg, dc, color_indices=psf_color_indices)
    # m_mean_c1, c_mean_c1 = compute_bias_chromatic(aggregates, dg, dc, color, color_indices=psf_color_indices, order=1)
    # m_mean_c2, c_mean_c2 = compute_bias_chromatic(aggregates, dg, dc, color, color_indices=psf_color_indices, order=2)
    (m_mean, c_mean), (m_mean_c1, c_mean_c1), (m_mean_c2, c_mean_c2) = task(aggregate_path, dg, dc, color, psf_color_indices)

    multiprocessing.set_start_method("spawn")
    # ctx = multiprocessing.get_context("spawn")  # spawn is ~ fork-exec
    # queue = ctx.Queue(-1)
    queue = multiprocessing.Queue(-1)

    lp = threading.Thread(target=log_util.logger_thread, args=(queue,))
    lp.start()

    m_bootstrap = []
    c_bootstrap = []
    m_bootstrap_c1 = []
    c_bootstrap_c1 = []
    m_bootstrap_c2 = []
    c_bootstrap_c2 = []

    print(f"aggregating results from {n_resample} bootstrap resamples...")
    with multiprocessing.Pool(
        n_jobs,
        initializer=log_util.initializer,
        initargs=(queue, log_level),
        maxtasksperchild=max(1, n_resample // n_jobs),
        # context=ctx,
    ) as pool:
        results = pool.imap(
            functools.partial(task, aggregate_path, dg, dc, color, psf_color_indices, True),
            utils.get_seeds(n_resample, seed=seed)
        )

        for i, res in track(enumerate(results), description="bootstrapping", total=n_resample):
            # _m_bootstrap, _c_bootstrap = compute_bias(res, dg, dc, color_indices=psf_color_indices)
            # _m_bootstrap_c1, _c_bootstrap_c1 = compute_bias_chromatic(res, dg, dc, color, color_indices=psf_color_indices, order=1)
            # _m_bootstrap_c2, _c_bootstrap_c2 = compute_bias_chromatic(res, dg, dc, color, color_indices=psf_color_indices, order=2)
            (_m_bootstrap, _c_bootstrap), (_m_bootstrap_c1, _c_bootstrap_c1), (_m_bootstrap_c2, _c_bootstrap_c2) = res

            m_bootstrap.append(_m_bootstrap)
            c_bootstrap.append(_c_bootstrap)
            m_bootstrap_c1.append(_m_bootstrap_c1)
            c_bootstrap_c1.append(_c_bootstrap_c1)
            m_bootstrap_c2.append(_m_bootstrap_c2)
            c_bootstrap_c2.append(_c_bootstrap_c2)

    queue.put(None)
    lp.join()

    print(f"finished processing bootstrap resamples")


    m_bootstrap = np.array(m_bootstrap)
    c_bootstrap = np.array(c_bootstrap)
    m_bootstrap_c1 = np.array(m_bootstrap_c1)
    c_bootstrap_c1 = np.array(c_bootstrap_c1)
    m_bootstrap_c2 = np.array(m_bootstrap_c2)
    c_bootstrap_c2 = np.array(c_bootstrap_c2)

    # report 3 standard devations as error
    m_error = np.nanstd(m_bootstrap)
    c_error = np.nanstd(c_bootstrap)
    m_error_c1 = np.nanstd(m_bootstrap_c1)
    c_error_c1 = np.nanstd(c_bootstrap_c1)
    m_error_c2 = np.nanstd(m_bootstrap_c2)
    c_error_c2 = np.nanstd(c_bootstrap_c2)

    outfile = f"{config_name}_colors-{psf_color_indices[0]}-{psf_color_indices[1]}-{psf_color_indices[2]}.txt"
    with open(outfile, "w") as fp:
        fp.write(f"order, m_mean, m_error, c_mean, c_error\n")
        fp.write(f"0, {m_mean}, {m_error}, {c_mean}, {c_error}\n")
        fp.write(f"1, {m_mean_c1}, {m_error_c1}, {c_mean_c1}, {c_error_c1}\n")
        fp.write(f"2, {m_mean_c2}, {m_error_c2}, {c_mean_c2}, {c_error_c2}\n")

    print(f"mdet (0): m = {m_mean:0.3e} +/- {m_error * 3:0.3e} [3-sigma], c = {c_mean:0.3e} +/- {c_error * 3:0.3e} [3-sigma]")
    print(f"drdc (1): m = {m_mean_c1:0.3e} +/- {m_error_c1 * 3:0.3e} [3-sigma], c = {c_mean_c1:0.3e} +/- {c_error_c1 * 3:0.3e} [3-sigma]")
    print(f"drdc (2): m = {m_mean_c2:0.3e} +/- {m_error_c2 * 3:0.3e} [3-sigma], c = {c_mean_c2:0.3e} +/- {c_error_c2 * 3:0.3e} [3-sigma]")

    m_req = 2e-3

    m_min = np.min([m_bootstrap.min(), m_bootstrap_c1.min(), m_bootstrap_c2.min()])
    m_max = np.max([m_bootstrap.max(), m_bootstrap_c1.max(), m_bootstrap_c2.max()])
    nbins = 50
    m_bin_edges = np.linspace(m_min, m_max, nbins + 1)
    # m_bin_edges = np.arange(m_min, m_max, 5e-4)
    m_bootstrap_hist, _ = np.histogram(m_bootstrap, bins=m_bin_edges)
    m_bootstrap_c1_hist, _ = np.histogram(m_bootstrap_c1, bins=m_bin_edges)
    m_bootstrap_c2_hist, _ = np.histogram(m_bootstrap_c2, bins=m_bin_edges)

    # zeroth-order chromatic correction

    fig, ax = plot_util.subplots(1, 1)

    ax.axvspan(-m_req, m_req, fc="k", alpha=0.1)
    ax.axvline(4e-4, c="k", alpha=0.1, ls="--")
    ax.stairs(m_bootstrap_hist, m_bin_edges, ec="k")
    ax.axvline(m_mean, c="k")
    ax.set_xlabel("$m$")
    ax.set_title(config_name)

    figname = f"{config_name}_m-0_colors-{psf_color_indices[0]}-{psf_color_indices[1]}-{psf_color_indices[2]}.pdf"
    fig.savefig(figname)

    # first-order chromatic correction


    fig, ax = plot_util.subplots(1, 1)

    ax.axvspan(-m_req, m_req, fc="k", alpha=0.1)
    ax.axvline(4e-4, c="k", alpha=0.1, ls="--")
    ax.stairs(m_bootstrap_hist, m_bin_edges, ec="k", label="0")
    ax.axvline(m_mean, c="k")
    ax.stairs(m_bootstrap_c1_hist, m_bin_edges, ec="b", label="1")
    ax.axvline(m_mean_c1, c="b")
    ax.set_xlabel("$m$")
    ax.legend(loc="upper right")
    ax.set_title(config_name)

    figname = f"{config_name}_m-1_colors-{psf_color_indices[0]}-{psf_color_indices[1]}-{psf_color_indices[2]}.pdf"
    fig.savefig(figname)

    # second-order chromatic correction

    fig, ax = plot_util.subplots(1, 1)

    ax.axvspan(-m_req, m_req, fc="k", alpha=0.1)
    ax.axvline(4e-4, c="k", alpha=0.1, ls="--")
    ax.stairs(m_bootstrap_hist, m_bin_edges, ec="k", label="0")
    ax.axvline(m_mean, c="k")
    ax.stairs(m_bootstrap_c1_hist, m_bin_edges, ec="b", label="1")
    ax.axvline(m_mean_c1, c="b")
    ax.stairs(m_bootstrap_c2_hist, m_bin_edges, ec="r", label="2")
    ax.axvline(m_mean_c2, c="r")
    ax.set_xlabel("$m$")
    ax.legend(loc="upper right")
    ax.set_title(config_name)

    figname = f"{config_name}_m-2_colors-{psf_color_indices[0]}-{psf_color_indices[1]}-{psf_color_indices[2]}.pdf"
    fig.savefig(figname)


if __name__ == "__main__":
    main()
