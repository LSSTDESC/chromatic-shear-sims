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
import pyarrow.feather as ft
from pyarrow import acero
from rich.progress import track

from chromatic_shear_sims import measurement
from chromatic_shear_sims.simulation import SimulationBuilder

from . import log_util, name_util, plot_util


def compute_e(results):
    # NOSHEAR
    p_e1_ns = np.average(results["plus"]["noshear"]["e1"])
    p_e2_ns = np.average(results["plus"]["noshear"]["e2"])

    m_e1_ns = np.average(results["minus"]["noshear"]["e1"])
    m_e2_ns = np.average(results["minus"]["noshear"]["e2"])

    return np.array([p_e1_ns, p_e2_ns]), np.array([m_e1_ns, m_e2_ns])


def compute_e_chromatic(results):
    # NOSHEAR
    p_c1_e1_ns = np.average(results["plus"]["c1"]["noshear"]["e1"])
    p_c1_e2_ns = np.average(results["plus"]["c1"]["noshear"]["e2"])

    m_c1_e1_ns = np.average(results["minus"]["c1"]["noshear"]["e1"])
    m_c1_e2_ns = np.average(results["minus"]["c1"]["noshear"]["e2"])

    return np.array([p_c1_e1_ns, p_c1_e2_ns]), np.array([m_c1_e1_ns, m_c1_e2_ns])


def compute_e_step(results, shear_step, color_step):
    e1 = np.average(results[shear_step][color_step]["noshear"]["e1"])
    e2 = np.average(results[shear_step][color_step]["noshear"]["e2"])

    return np.array([e1, e2])


def compute_R(results, dg):

    p_R11 = np.average(pc.divide(pc.subtract(
        results["plus"]["1p"]["e1"],
        results["plus"]["1m"]["e1"]
    ), 2 * dg))
    p_R12 = np.average(pc.divide(pc.subtract(
        results["plus"]["2p"]["e1"],
        results["plus"]["2m"]["e1"]
    ), 2 * dg))
    p_R21 = np.average(pc.divide(pc.subtract(
        results["plus"]["1p"]["e2"],
        results["plus"]["1m"]["e2"]
    ), 2 * dg))
    p_R22 = np.average(pc.divide(pc.subtract(
        results["plus"]["2p"]["e2"],
        results["plus"]["2m"]["e2"]
    ), 2 * dg))

    m_R11 = np.average(pc.divide(pc.subtract(
        results["minus"]["1p"]["e1"],
        results["minus"]["1m"]["e1"]
    ), 2 * dg))
    m_R12 = np.average(pc.divide(pc.subtract(
        results["minus"]["2p"]["e1"],
        results["minus"]["2m"]["e1"]
    ), 2 * dg))
    m_R21 = np.average(pc.divide(pc.subtract(
        results["minus"]["1p"]["e2"],
        results["minus"]["1m"]["e2"]
    ), 2 * dg))
    m_R22 = np.average(pc.divide(pc.subtract(
        results["minus"]["2p"]["e2"],
        results["minus"]["2m"]["e2"]
    ), 2 * dg))

    return np.array([[p_R11, p_R12], [p_R21, p_R22]]), np.array([[m_R11, m_R12], [m_R21, m_R22]])


def compute_R_chromatic(results, dg):

    p_c1_R11 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["1p"]["e1"],
        results["plus"]["c1"]["1m"]["e1"]
    ), 2 * dg))
    p_c1_R12 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["2p"]["e1"],
        results["plus"]["c1"]["2m"]["e1"]
    ), 2 * dg))
    p_c1_R21 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["1p"]["e2"],
        results["plus"]["c1"]["1m"]["e2"]
    ), 2 * dg))
    p_c1_R22 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["2p"]["e2"],
        results["plus"]["c1"]["2m"]["e2"]
    ), 2 * dg))

    m_c1_R11 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["1p"]["e1"],
        results["minus"]["c1"]["1m"]["e1"]
    ), 2 * dg))
    m_c1_R12 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["2p"]["e1"],
        results["minus"]["c1"]["2m"]["e1"]
    ), 2 * dg))
    m_c1_R21 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["1p"]["e2"],
        results["minus"]["c1"]["1m"]["e2"]
    ), 2 * dg))
    m_c1_R22 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["2p"]["e2"],
        results["minus"]["c1"]["2m"]["e2"]
    ), 2 * dg))

    return np.array([[p_c1_R11, p_c1_R12], [p_c1_R21, p_c1_R22]]), np.array([[m_c1_R11, m_c1_R12], [m_c1_R21, m_c1_R22]])


def compute_R_step(results, dg, shear_step, color_step):
    R11 = np.average(pc.divide(pc.subtract(
        results[shear_step][color_step]["1p"]["e1"],
        results[shear_step][color_step]["1m"]["e1"]
    ), 2 * dg))
    R12 = np.average(pc.divide(pc.subtract(
        results[shear_step][color_step]["2p"]["e1"],
        results[shear_step][color_step]["2m"]["e1"]
    ), 2 * dg))
    R21 = np.average(pc.divide(pc.subtract(
        results[shear_step][color_step]["1p"]["e2"],
        results[shear_step][color_step]["1m"]["e2"]
    ), 2 * dg))
    R22 = np.average(pc.divide(pc.subtract(
        results[shear_step][color_step]["2p"]["e2"],
        results[shear_step][color_step]["2m"]["e2"]
    ), 2 * dg))

    return np.array([[R11, R12], [R21, R22]])


def compute_R_var(results, dg):

    p_c1_R11 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["1p"]["var_e1"],
        results["plus"]["c1"]["1m"]["var_e1"]
    ), 2 * dg))
    p_c1_R12 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["2p"]["var_e1"],
        results["plus"]["c1"]["2m"]["var_e1"]
    ), 2 * dg))
    p_c1_R21 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["1p"]["var_e2"],
        results["plus"]["c1"]["1m"]["var_e2"]
    ), 2 * dg))
    p_c1_R22 = np.average(pc.divide(pc.subtract(
        results["plus"]["c1"]["2p"]["var_e2"],
        results["plus"]["c1"]["2m"]["var_e2"]
    ), 2 * dg))

    m_c1_R11 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["1p"]["var_e1"],
        results["minus"]["c1"]["1m"]["var_e1"]
    ), 2 * dg))
    m_c1_R12 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["2p"]["var_e1"],
        results["minus"]["c1"]["2m"]["var_e1"]
    ), 2 * dg))
    m_c1_R21 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["1p"]["var_e2"],
        results["minus"]["c1"]["1m"]["var_e2"]
    ), 2 * dg))
    m_c1_R22 = np.average(pc.divide(pc.subtract(
        results["minus"]["c1"]["2p"]["var_e2"],
        results["minus"]["c1"]["2m"]["var_e2"]
    ), 2 * dg))

    return np.array([[p_c1_R11, p_c1_R12], [p_c1_R21, p_c1_R22]]), np.array([[m_c1_R11, m_c1_R12], [m_c1_R21, m_c1_R22]])


def compute_de(results, dg, dc, color, order=1):
    # c0
    p_c0_e1_ns = np.average(results["plus"]["c0"]["noshear"]["e1"])
    p_c0_e2_ns = np.average(results["plus"]["c0"]["noshear"]["e2"])
    p_c0_e1c_ns = np.average(results["plus"]["c0"]["noshear"]["e1c"])
    p_c0_e2c_ns = np.average(results["plus"]["c0"]["noshear"]["e2c"])
    p_c0_e1cc_ns = np.average(results["plus"]["c0"]["noshear"]["e1cc"])
    p_c0_e2cc_ns = np.average(results["plus"]["c0"]["noshear"]["e2cc"])

    m_c0_e1_ns = np.average(results["minus"]["c0"]["noshear"]["e1"])
    m_c0_e2_ns = np.average(results["minus"]["c0"]["noshear"]["e2"])
    m_c0_e1c_ns = np.average(results["minus"]["c0"]["noshear"]["e1c"])
    m_c0_e2c_ns = np.average(results["minus"]["c0"]["noshear"]["e2c"])
    m_c0_e1cc_ns = np.average(results["minus"]["c0"]["noshear"]["e1cc"])
    m_c0_e2cc_ns = np.average(results["minus"]["c0"]["noshear"]["e2cc"])

    # c1
    p_c1_e1_ns = np.average(results["plus"]["c1"]["noshear"]["e1"])
    p_c1_e2_ns = np.average(results["plus"]["c1"]["noshear"]["e2"])
    p_c1_e1c_ns = np.average(results["plus"]["c1"]["noshear"]["e1c"])
    p_c1_e2c_ns = np.average(results["plus"]["c1"]["noshear"]["e2c"])
    p_c1_e1cc_ns = np.average(results["plus"]["c1"]["noshear"]["e1cc"])
    p_c1_e2cc_ns = np.average(results["plus"]["c1"]["noshear"]["e2cc"])

    m_c1_e1_ns = np.average(results["minus"]["c1"]["noshear"]["e1"])
    m_c1_e2_ns = np.average(results["minus"]["c1"]["noshear"]["e2"])
    m_c1_e1c_ns = np.average(results["minus"]["c1"]["noshear"]["e1c"])
    m_c1_e2c_ns = np.average(results["minus"]["c1"]["noshear"]["e2c"])
    m_c1_e1cc_ns = np.average(results["minus"]["c1"]["noshear"]["e1cc"])
    m_c1_e2cc_ns = np.average(results["minus"]["c1"]["noshear"]["e2cc"])

    # c2
    p_c2_e1_ns = np.average(results["plus"]["c2"]["noshear"]["e1"])
    p_c2_e2_ns = np.average(results["plus"]["c2"]["noshear"]["e2"])
    p_c2_e1c_ns = np.average(results["plus"]["c2"]["noshear"]["e1c"])
    p_c2_e2c_ns = np.average(results["plus"]["c2"]["noshear"]["e2c"])
    p_c2_e1cc_ns = np.average(results["plus"]["c2"]["noshear"]["e1cc"])
    p_c2_e2cc_ns = np.average(results["plus"]["c2"]["noshear"]["e2cc"])

    m_c2_e1_ns = np.average(results["minus"]["c2"]["noshear"]["e1"])
    m_c2_e2_ns = np.average(results["minus"]["c2"]["noshear"]["e2"])
    m_c2_e1c_ns = np.average(results["minus"]["c2"]["noshear"]["e1c"])
    m_c2_e2c_ns = np.average(results["minus"]["c2"]["noshear"]["e2c"])
    m_c2_e1cc_ns = np.average(results["minus"]["c2"]["noshear"]["e1cc"])
    m_c2_e2cc_ns = np.average(results["minus"]["c2"]["noshear"]["e2cc"])

    # p_dedc_1 = (p_c2_e1dc_ns - p_c0_e1dc_ns) / (2 * dc)
    # p_dedc_2 = (p_c2_e2dc_ns - p_c0_e2dc_ns) / (2 * dc)

    # m_dedc_1 = (m_c2_e1dc_ns - m_c0_e1dc_ns) / (2 * dc)
    # m_dedc_2 = (m_c2_e2dc_ns - m_c0_e2dc_ns) / (2 * dc)

    match order:
        case 1:
            p_de_1 = (
                p_c2_e1c_ns - p_c0_e1c_ns - color * (
                    p_c2_e1_ns - p_c0_e1_ns
                )
            ) / (2 * dc)
            p_de_2 = (
                p_c2_e2c_ns - p_c0_e2c_ns - color * (
                    p_c2_e1_ns - p_c0_e1_ns
                )
            ) / (2 * dc)

            m_de_1 = (
                m_c2_e1c_ns - m_c0_e1c_ns - color * (
                    m_c2_e1_ns - m_c0_e1_ns
                )
            ) / (2 * dc)
            m_de_2 = (
                m_c2_e2c_ns - m_c0_e2c_ns - color * (
                    m_c2_e1_ns - m_c0_e1_ns
                )
            ) / (2 * dc)
        case 2:
            p_de_1 = (
                p_c2_e1c_ns - p_c0_e1c_ns - color * (
                    p_c2_e1_ns - p_c0_e1_ns
                )
            ) / (2 * dc) + (
                p_c2_e1cc_ns - 2 * p_c1_e1cc_ns + p_c0_e1cc_ns  \
                - 2 * color * (p_c2_e1c_ns - 2 * p_c1_e1c_ns + p_c0_e1c_ns)  \
                + color**2 * (p_c2_e1_ns - 2 * p_c1_e1_ns + p_c0_e1_ns)
            ) / (2 * dc**2)
            p_de_2 = (
                p_c2_e2c_ns - p_c0_e2c_ns - color * (
                    p_c2_e1_ns - p_c0_e1_ns
                )
            ) / (2 * dc) + (
                p_c2_e2cc_ns - 2 * p_c1_e2cc_ns + p_c0_e2cc_ns  \
                - 2 * color * (p_c2_e2c_ns - 2 * p_c1_e2c_ns + p_c0_e2c_ns)  \
                + color**2 * (p_c2_e2_ns - 2 * p_c1_e2_ns + p_c0_e2_ns)
            ) / (2 * dc**2)

            m_de_1 = (
                m_c2_e1c_ns - m_c0_e1c_ns - color * (
                    m_c2_e1_ns - m_c0_e1_ns
                )
            ) / (2 * dc) + (
                m_c2_e1cc_ns - 2 * m_c1_e1cc_ns + m_c0_e1cc_ns  \
                - 2 * color * (m_c2_e1c_ns - 2 * m_c1_e1c_ns + m_c0_e1c_ns)  \
                + color**2 * (m_c2_e1_ns - 2 * m_c1_e1_ns + m_c0_e1_ns)
            ) / (2 * dc**2)
            m_de_2 = (
                m_c2_e2c_ns - m_c0_e2c_ns - color * (
                    m_c2_e1_ns - m_c0_e1_ns
                )
            ) / (2 * dc) + (
                m_c2_e2cc_ns - 2 * m_c1_e2cc_ns + m_c0_e2cc_ns  \
                - 2 * color * (m_c2_e2c_ns - 2 * m_c1_e2c_ns + m_c0_e2c_ns)  \
                + color**2 * (m_c2_e2_ns - 2 * m_c1_e2_ns + m_c0_e2_ns)
            ) / (2 * dc**2)

    return np.array([p_de_1, p_de_2]), np.array([m_de_1, m_de_2])


def compute_dR(results, dg, dc, color, order=1):

    #---------------------------------------------------------------------------
    # plus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    p_c0_e1_ns = np.average(results["plus"]["c0"]["noshear"]["e1"])
    p_c0_e2_ns = np.average(results["plus"]["c0"]["noshear"]["e2"])
    p_c0_e1c_ns = np.average(results["plus"]["c0"]["noshear"]["e1c"])
    p_c0_e2c_ns = np.average(results["plus"]["c0"]["noshear"]["e2c"])
    p_c0_e1cc_ns = np.average(results["plus"]["c0"]["noshear"]["e1cc"])
    p_c0_e2cc_ns = np.average(results["plus"]["c0"]["noshear"]["e2cc"])
    p_c0_e1dc_ns = np.average(results["plus"]["c0"]["noshear"]["e1dc"])
    p_c0_e2dc_ns = np.average(results["plus"]["c0"]["noshear"]["e2dc"])
    p_c0_e1dc2_ns = np.average(results["plus"]["c0"]["noshear"]["e1dc2"])
    p_c0_e2dc2_ns = np.average(results["plus"]["c0"]["noshear"]["e2dc2"])

    # 1p
    p_c0_e1_1p = np.average(results["plus"]["c0"]["1p"]["e1"])
    p_c0_e2_1p = np.average(results["plus"]["c0"]["1p"]["e2"])
    p_c0_e1c_1p = np.average(results["plus"]["c0"]["1p"]["e1c"])
    p_c0_e2c_1p = np.average(results["plus"]["c0"]["1p"]["e2c"])
    p_c0_e1cc_1p = np.average(results["plus"]["c0"]["1p"]["e1cc"])
    p_c0_e2cc_1p = np.average(results["plus"]["c0"]["1p"]["e2cc"])
    p_c0_e1dc_1p = np.average(results["plus"]["c0"]["1p"]["e1dc"])
    p_c0_e2dc_1p = np.average(results["plus"]["c0"]["1p"]["e2dc"])
    p_c0_e1dc2_1p = np.average(results["plus"]["c0"]["1p"]["e1dc2"])
    p_c0_e2dc2_1p = np.average(results["plus"]["c0"]["1p"]["e2dc2"])

    # 1m
    p_c0_e1_1m = np.average(results["plus"]["c0"]["1m"]["e1"])
    p_c0_e2_1m = np.average(results["plus"]["c0"]["1m"]["e2"])
    p_c0_e1c_1m = np.average(results["plus"]["c0"]["1m"]["e1c"])
    p_c0_e2c_1m = np.average(results["plus"]["c0"]["1m"]["e2c"])
    p_c0_e1cc_1m = np.average(results["plus"]["c0"]["1m"]["e1cc"])
    p_c0_e2cc_1m = np.average(results["plus"]["c0"]["1m"]["e2cc"])
    p_c0_e1dc_1m = np.average(results["plus"]["c0"]["1m"]["e1dc"])
    p_c0_e2dc_1m = np.average(results["plus"]["c0"]["1m"]["e2dc"])
    p_c0_e1dc2_1m = np.average(results["plus"]["c0"]["1m"]["e1dc2"])
    p_c0_e2dc2_1m = np.average(results["plus"]["c0"]["1m"]["e2dc2"])

    # 2p
    p_c0_e1_2p = np.average(results["plus"]["c0"]["2p"]["e1"])
    p_c0_e2_2p = np.average(results["plus"]["c0"]["2p"]["e2"])
    p_c0_e1c_2p = np.average(results["plus"]["c0"]["2p"]["e1c"])
    p_c0_e2c_2p = np.average(results["plus"]["c0"]["2p"]["e2c"])
    p_c0_e1cc_2p = np.average(results["plus"]["c0"]["2p"]["e1cc"])
    p_c0_e2cc_2p = np.average(results["plus"]["c0"]["2p"]["e2cc"])
    p_c0_e1dc_2p = np.average(results["plus"]["c0"]["2p"]["e1dc"])
    p_c0_e2dc_2p = np.average(results["plus"]["c0"]["2p"]["e2dc"])
    p_c0_e1dc2_2p = np.average(results["plus"]["c0"]["2p"]["e1dc2"])
    p_c0_e2dc2_2p = np.average(results["plus"]["c0"]["2p"]["e2dc2"])

    # 2m
    p_c0_e1_2m = np.average(results["plus"]["c0"]["2m"]["e1"])
    p_c0_e2_2m = np.average(results["plus"]["c0"]["2m"]["e2"])
    p_c0_e1c_2m = np.average(results["plus"]["c0"]["2m"]["e1c"])
    p_c0_e2c_2m = np.average(results["plus"]["c0"]["2m"]["e2c"])
    p_c0_e1cc_2m = np.average(results["plus"]["c0"]["2m"]["e1cc"])
    p_c0_e2cc_2m = np.average(results["plus"]["c0"]["2m"]["e2cc"])
    p_c0_e1dc_2m = np.average(results["plus"]["c0"]["2m"]["e1dc"])
    p_c0_e2dc_2m = np.average(results["plus"]["c0"]["2m"]["e2dc"])
    p_c0_e1dc2_2m = np.average(results["plus"]["c0"]["2m"]["e1dc2"])
    p_c0_e2dc2_2m = np.average(results["plus"]["c0"]["2m"]["e2dc2"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    p_c1_e1_ns = np.average(results["plus"]["c1"]["noshear"]["e1"])
    p_c1_e2_ns = np.average(results["plus"]["c1"]["noshear"]["e2"])
    p_c1_e1c_ns = np.average(results["plus"]["c1"]["noshear"]["e1c"])
    p_c1_e2c_ns = np.average(results["plus"]["c1"]["noshear"]["e2c"])
    p_c1_e1cc_ns = np.average(results["plus"]["c1"]["noshear"]["e1cc"])
    p_c1_e2cc_ns = np.average(results["plus"]["c1"]["noshear"]["e2cc"])
    p_c1_e1dc_ns = np.average(results["plus"]["c1"]["noshear"]["e1dc"])
    p_c1_e2dc_ns = np.average(results["plus"]["c1"]["noshear"]["e2dc"])
    p_c1_e1dc2_ns = np.average(results["plus"]["c1"]["noshear"]["e1dc2"])
    p_c1_e2dc2_ns = np.average(results["plus"]["c1"]["noshear"]["e2dc2"])

    # 1p
    p_c1_e1_1p = np.average(results["plus"]["c1"]["1p"]["e1"])
    p_c1_e2_1p = np.average(results["plus"]["c1"]["1p"]["e2"])
    p_c1_e1c_1p = np.average(results["plus"]["c1"]["1p"]["e1c"])
    p_c1_e2c_1p = np.average(results["plus"]["c1"]["1p"]["e2c"])
    p_c1_e1cc_1p = np.average(results["plus"]["c1"]["1p"]["e1cc"])
    p_c1_e2cc_1p = np.average(results["plus"]["c1"]["1p"]["e2cc"])
    p_c1_e1dc_1p = np.average(results["plus"]["c1"]["1p"]["e1dc"])
    p_c1_e2dc_1p = np.average(results["plus"]["c1"]["1p"]["e2dc"])
    p_c1_e1dc2_1p = np.average(results["plus"]["c1"]["1p"]["e1dc2"])
    p_c1_e2dc2_1p = np.average(results["plus"]["c1"]["1p"]["e2dc2"])

    # 1m
    p_c1_e1_1m = np.average(results["plus"]["c1"]["1m"]["e1"])
    p_c1_e2_1m = np.average(results["plus"]["c1"]["1m"]["e2"])
    p_c1_e1c_1m = np.average(results["plus"]["c1"]["1m"]["e1c"])
    p_c1_e2c_1m = np.average(results["plus"]["c1"]["1m"]["e2c"])
    p_c1_e1cc_1m = np.average(results["plus"]["c1"]["1m"]["e1cc"])
    p_c1_e2cc_1m = np.average(results["plus"]["c1"]["1m"]["e2cc"])
    p_c1_e1dc_1m = np.average(results["plus"]["c1"]["1m"]["e1dc"])
    p_c1_e2dc_1m = np.average(results["plus"]["c1"]["1m"]["e2dc"])
    p_c1_e1dc2_1m = np.average(results["plus"]["c1"]["1m"]["e1dc2"])
    p_c1_e2dc2_1m = np.average(results["plus"]["c1"]["1m"]["e2dc2"])

    # 2p
    p_c1_e1_2p = np.average(results["plus"]["c1"]["2p"]["e1"])
    p_c1_e2_2p = np.average(results["plus"]["c1"]["2p"]["e2"])
    p_c1_e1c_2p = np.average(results["plus"]["c1"]["2p"]["e1c"])
    p_c1_e2c_2p = np.average(results["plus"]["c1"]["2p"]["e2c"])
    p_c1_e1cc_2p = np.average(results["plus"]["c1"]["2p"]["e1cc"])
    p_c1_e2cc_2p = np.average(results["plus"]["c1"]["2p"]["e2cc"])
    p_c1_e1dc_2p = np.average(results["plus"]["c1"]["2p"]["e1dc"])
    p_c1_e2dc_2p = np.average(results["plus"]["c1"]["2p"]["e2dc"])
    p_c1_e1dc2_2p = np.average(results["plus"]["c1"]["2p"]["e1dc2"])
    p_c1_e2dc2_2p = np.average(results["plus"]["c1"]["2p"]["e2dc2"])

    # 2m
    p_c1_e1_2m = np.average(results["plus"]["c1"]["2m"]["e1"])
    p_c1_e2_2m = np.average(results["plus"]["c1"]["2m"]["e2"])
    p_c1_e1c_2m = np.average(results["plus"]["c1"]["2m"]["e1c"])
    p_c1_e2c_2m = np.average(results["plus"]["c1"]["2m"]["e2c"])
    p_c1_e1cc_2m = np.average(results["plus"]["c1"]["2m"]["e1cc"])
    p_c1_e2cc_2m = np.average(results["plus"]["c1"]["2m"]["e2cc"])
    p_c1_e1dc_2m = np.average(results["plus"]["c1"]["2m"]["e1dc"])
    p_c1_e2dc_2m = np.average(results["plus"]["c1"]["2m"]["e2dc"])
    p_c1_e1dc2_2m = np.average(results["plus"]["c1"]["2m"]["e1dc2"])
    p_c1_e2dc2_2m = np.average(results["plus"]["c1"]["2m"]["e2dc2"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    p_c2_e1_ns = np.average(results["plus"]["c2"]["noshear"]["e1"])
    p_c2_e2_ns = np.average(results["plus"]["c2"]["noshear"]["e2"])
    p_c2_e1c_ns = np.average(results["plus"]["c2"]["noshear"]["e1c"])
    p_c2_e2c_ns = np.average(results["plus"]["c2"]["noshear"]["e2c"])
    p_c2_e1cc_ns = np.average(results["plus"]["c2"]["noshear"]["e1cc"])
    p_c2_e2cc_ns = np.average(results["plus"]["c2"]["noshear"]["e2cc"])
    p_c2_e1dc_ns = np.average(results["plus"]["c2"]["noshear"]["e1dc"])
    p_c2_e2dc_ns = np.average(results["plus"]["c2"]["noshear"]["e2dc"])
    p_c2_e1dc2_ns = np.average(results["plus"]["c2"]["noshear"]["e1dc2"])
    p_c2_e2dc2_ns = np.average(results["plus"]["c2"]["noshear"]["e2dc2"])

    # 1p
    p_c2_e1_1p = np.average(results["plus"]["c2"]["1p"]["e1"])
    p_c2_e2_1p = np.average(results["plus"]["c2"]["1p"]["e2"])
    p_c2_e1c_1p = np.average(results["plus"]["c2"]["1p"]["e1c"])
    p_c2_e2c_1p = np.average(results["plus"]["c2"]["1p"]["e2c"])
    p_c2_e1cc_1p = np.average(results["plus"]["c2"]["1p"]["e1cc"])
    p_c2_e2cc_1p = np.average(results["plus"]["c2"]["1p"]["e2cc"])
    p_c2_e1dc_1p = np.average(results["plus"]["c2"]["1p"]["e1dc"])
    p_c2_e2dc_1p = np.average(results["plus"]["c2"]["1p"]["e2dc"])
    p_c2_e1dc2_1p = np.average(results["plus"]["c2"]["1p"]["e1dc2"])
    p_c2_e2dc2_1p = np.average(results["plus"]["c2"]["1p"]["e2dc2"])

    # 1m
    p_c2_e1_1m = np.average(results["plus"]["c2"]["1m"]["e1"])
    p_c2_e2_1m = np.average(results["plus"]["c2"]["1m"]["e2"])
    p_c2_e1c_1m = np.average(results["plus"]["c2"]["1m"]["e1c"])
    p_c2_e2c_1m = np.average(results["plus"]["c2"]["1m"]["e2c"])
    p_c2_e1cc_1m = np.average(results["plus"]["c2"]["1m"]["e1cc"])
    p_c2_e2cc_1m = np.average(results["plus"]["c2"]["1m"]["e2cc"])
    p_c2_e1dc_1m = np.average(results["plus"]["c2"]["1m"]["e1dc"])
    p_c2_e2dc_1m = np.average(results["plus"]["c2"]["1m"]["e2dc"])
    p_c2_e1dc2_1m = np.average(results["plus"]["c2"]["1m"]["e1dc2"])
    p_c2_e2dc2_1m = np.average(results["plus"]["c2"]["1m"]["e2dc2"])

    # 2p
    p_c2_e1_2p = np.average(results["plus"]["c2"]["2p"]["e1"])
    p_c2_e2_2p = np.average(results["plus"]["c2"]["2p"]["e2"])
    p_c2_e1c_2p = np.average(results["plus"]["c2"]["2p"]["e1c"])
    p_c2_e2c_2p = np.average(results["plus"]["c2"]["2p"]["e2c"])
    p_c2_e1cc_2p = np.average(results["plus"]["c2"]["2p"]["e1cc"])
    p_c2_e2cc_2p = np.average(results["plus"]["c2"]["2p"]["e2cc"])
    p_c2_e1dc_2p = np.average(results["plus"]["c2"]["2p"]["e1dc"])
    p_c2_e2dc_2p = np.average(results["plus"]["c2"]["2p"]["e2dc"])
    p_c2_e1dc2_2p = np.average(results["plus"]["c2"]["2p"]["e1dc2"])
    p_c2_e2dc2_2p = np.average(results["plus"]["c2"]["2p"]["e2dc2"])

    # 2m
    p_c2_e1_2m = np.average(results["plus"]["c2"]["2m"]["e1"])
    p_c2_e2_2m = np.average(results["plus"]["c2"]["2m"]["e2"])
    p_c2_e1c_2m = np.average(results["plus"]["c2"]["2m"]["e1c"])
    p_c2_e2c_2m = np.average(results["plus"]["c2"]["2m"]["e2c"])
    p_c2_e1cc_2m = np.average(results["plus"]["c2"]["2m"]["e1cc"])
    p_c2_e2cc_2m = np.average(results["plus"]["c2"]["2m"]["e2cc"])
    p_c2_e1dc_2m = np.average(results["plus"]["c2"]["2m"]["e1dc"])
    p_c2_e2dc_2m = np.average(results["plus"]["c2"]["2m"]["e2dc"])
    p_c2_e1dc2_2m = np.average(results["plus"]["c2"]["2m"]["e1dc2"])
    p_c2_e2dc2_2m = np.average(results["plus"]["c2"]["2m"]["e2dc2"])

    #---------------------------------------------------------------------------
    # minus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    m_c0_e1_ns = np.average(results["minus"]["c0"]["noshear"]["e1"])
    m_c0_e2_ns = np.average(results["minus"]["c0"]["noshear"]["e2"])
    m_c0_e1c_ns = np.average(results["minus"]["c0"]["noshear"]["e1c"])
    m_c0_e2c_ns = np.average(results["minus"]["c0"]["noshear"]["e2c"])
    m_c0_e1cc_ns = np.average(results["minus"]["c0"]["noshear"]["e1cc"])
    m_c0_e2cc_ns = np.average(results["minus"]["c0"]["noshear"]["e2cc"])
    m_c0_e1dc_ns = np.average(results["minus"]["c0"]["noshear"]["e1dc"])
    m_c0_e2dc_ns = np.average(results["minus"]["c0"]["noshear"]["e2dc"])
    m_c0_e1dc2_ns = np.average(results["minus"]["c0"]["noshear"]["e1dc2"])
    m_c0_e2dc2_ns = np.average(results["minus"]["c0"]["noshear"]["e2dc2"])

    # 1p
    m_c0_e1_1p = np.average(results["minus"]["c0"]["1p"]["e1"])
    m_c0_e2_1p = np.average(results["minus"]["c0"]["1p"]["e2"])
    m_c0_e1c_1p = np.average(results["minus"]["c0"]["1p"]["e1c"])
    m_c0_e2c_1p = np.average(results["minus"]["c0"]["1p"]["e2c"])
    m_c0_e1cc_1p = np.average(results["minus"]["c0"]["1p"]["e1cc"])
    m_c0_e2cc_1p = np.average(results["minus"]["c0"]["1p"]["e2cc"])
    m_c0_e1dc_1p = np.average(results["minus"]["c0"]["1p"]["e1dc"])
    m_c0_e2dc_1p = np.average(results["minus"]["c0"]["1p"]["e2dc"])
    m_c0_e1dc2_1p = np.average(results["minus"]["c0"]["1p"]["e1dc2"])
    m_c0_e2dc2_1p = np.average(results["minus"]["c0"]["1p"]["e2dc2"])

    # 1m
    m_c0_e1_1m = np.average(results["minus"]["c0"]["1m"]["e1"])
    m_c0_e2_1m = np.average(results["minus"]["c0"]["1m"]["e2"])
    m_c0_e1c_1m = np.average(results["minus"]["c0"]["1m"]["e1c"])
    m_c0_e2c_1m = np.average(results["minus"]["c0"]["1m"]["e2c"])
    m_c0_e1cc_1m = np.average(results["minus"]["c0"]["1m"]["e1cc"])
    m_c0_e2cc_1m = np.average(results["minus"]["c0"]["1m"]["e2cc"])
    m_c0_e1dc_1m = np.average(results["minus"]["c0"]["1m"]["e1dc"])
    m_c0_e2dc_1m = np.average(results["minus"]["c0"]["1m"]["e2dc"])
    m_c0_e1dc2_1m = np.average(results["minus"]["c0"]["1m"]["e1dc2"])
    m_c0_e2dc2_1m = np.average(results["minus"]["c0"]["1m"]["e2dc2"])

    # 2p
    m_c0_e1_2p = np.average(results["minus"]["c0"]["2p"]["e1"])
    m_c0_e2_2p = np.average(results["minus"]["c0"]["2p"]["e2"])
    m_c0_e1c_2p = np.average(results["minus"]["c0"]["2p"]["e1c"])
    m_c0_e2c_2p = np.average(results["minus"]["c0"]["2p"]["e2c"])
    m_c0_e1cc_2p = np.average(results["minus"]["c0"]["2p"]["e1cc"])
    m_c0_e2cc_2p = np.average(results["minus"]["c0"]["2p"]["e2cc"])
    m_c0_e1dc_2p = np.average(results["minus"]["c0"]["2p"]["e1dc"])
    m_c0_e2dc_2p = np.average(results["minus"]["c0"]["2p"]["e2dc"])
    m_c0_e1dc2_2p = np.average(results["minus"]["c0"]["2p"]["e1dc2"])
    m_c0_e2dc2_2p = np.average(results["minus"]["c0"]["2p"]["e2dc2"])

    # 2m
    m_c0_e1_2m = np.average(results["minus"]["c0"]["2m"]["e1"])
    m_c0_e2_2m = np.average(results["minus"]["c0"]["2m"]["e2"])
    m_c0_e1c_2m = np.average(results["minus"]["c0"]["2m"]["e1c"])
    m_c0_e2c_2m = np.average(results["minus"]["c0"]["2m"]["e2c"])
    m_c0_e1cc_2m = np.average(results["minus"]["c0"]["2m"]["e1cc"])
    m_c0_e2cc_2m = np.average(results["minus"]["c0"]["2m"]["e2cc"])
    m_c0_e1dc_2m = np.average(results["minus"]["c0"]["2m"]["e1dc"])
    m_c0_e2dc_2m = np.average(results["minus"]["c0"]["2m"]["e2dc"])
    m_c0_e1dc2_2m = np.average(results["minus"]["c0"]["2m"]["e1dc2"])
    m_c0_e2dc2_2m = np.average(results["minus"]["c0"]["2m"]["e2dc2"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    m_c1_e1_ns = np.average(results["minus"]["c1"]["noshear"]["e1"])
    m_c1_e2_ns = np.average(results["minus"]["c1"]["noshear"]["e2"])
    m_c1_e1c_ns = np.average(results["minus"]["c1"]["noshear"]["e1c"])
    m_c1_e2c_ns = np.average(results["minus"]["c1"]["noshear"]["e2c"])
    m_c1_e1cc_ns = np.average(results["minus"]["c1"]["noshear"]["e1cc"])
    m_c1_e2cc_ns = np.average(results["minus"]["c1"]["noshear"]["e2cc"])
    m_c1_e1dc_ns = np.average(results["minus"]["c1"]["noshear"]["e1dc"])
    m_c1_e2dc_ns = np.average(results["minus"]["c1"]["noshear"]["e2dc"])
    m_c1_e1dc2_ns = np.average(results["minus"]["c1"]["noshear"]["e1dc2"])
    m_c1_e2dc2_ns = np.average(results["minus"]["c1"]["noshear"]["e2dc2"])

    # 1p
    m_c1_e1_1p = np.average(results["minus"]["c1"]["1p"]["e1"])
    m_c1_e2_1p = np.average(results["minus"]["c1"]["1p"]["e2"])
    m_c1_e1c_1p = np.average(results["minus"]["c1"]["1p"]["e1c"])
    m_c1_e2c_1p = np.average(results["minus"]["c1"]["1p"]["e2c"])
    m_c1_e1cc_1p = np.average(results["minus"]["c1"]["1p"]["e1cc"])
    m_c1_e2cc_1p = np.average(results["minus"]["c1"]["1p"]["e2cc"])
    m_c1_e1dc_1p = np.average(results["minus"]["c1"]["1p"]["e1dc"])
    m_c1_e2dc_1p = np.average(results["minus"]["c1"]["1p"]["e2dc"])
    m_c1_e1dc2_1p = np.average(results["minus"]["c1"]["1p"]["e1dc2"])
    m_c1_e2dc2_1p = np.average(results["minus"]["c1"]["1p"]["e2dc2"])

    # 1m
    m_c1_e1_1m = np.average(results["minus"]["c1"]["1m"]["e1"])
    m_c1_e2_1m = np.average(results["minus"]["c1"]["1m"]["e2"])
    m_c1_e1c_1m = np.average(results["minus"]["c1"]["1m"]["e1c"])
    m_c1_e2c_1m = np.average(results["minus"]["c1"]["1m"]["e2c"])
    m_c1_e1cc_1m = np.average(results["minus"]["c1"]["1m"]["e1cc"])
    m_c1_e2cc_1m = np.average(results["minus"]["c1"]["1m"]["e2cc"])
    m_c1_e1dc_1m = np.average(results["minus"]["c1"]["1m"]["e1dc"])
    m_c1_e2dc_1m = np.average(results["minus"]["c1"]["1m"]["e2dc"])
    m_c1_e1dc2_1m = np.average(results["minus"]["c1"]["1m"]["e1dc2"])
    m_c1_e2dc2_1m = np.average(results["minus"]["c1"]["1m"]["e2dc2"])

    # 2p
    m_c1_e1_2p = np.average(results["minus"]["c1"]["2p"]["e1"])
    m_c1_e2_2p = np.average(results["minus"]["c1"]["2p"]["e2"])
    m_c1_e1c_2p = np.average(results["minus"]["c1"]["2p"]["e1c"])
    m_c1_e2c_2p = np.average(results["minus"]["c1"]["2p"]["e2c"])
    m_c1_e1cc_2p = np.average(results["minus"]["c1"]["2p"]["e1cc"])
    m_c1_e2cc_2p = np.average(results["minus"]["c1"]["2p"]["e2cc"])
    m_c1_e1dc_2p = np.average(results["minus"]["c1"]["2p"]["e1dc"])
    m_c1_e2dc_2p = np.average(results["minus"]["c1"]["2p"]["e2dc"])
    m_c1_e1dc2_2p = np.average(results["minus"]["c1"]["2p"]["e1dc2"])
    m_c1_e2dc2_2p = np.average(results["minus"]["c1"]["2p"]["e2dc2"])

    # 2m
    m_c1_e1_2m = np.average(results["minus"]["c1"]["2m"]["e1"])
    m_c1_e2_2m = np.average(results["minus"]["c1"]["2m"]["e2"])
    m_c1_e1c_2m = np.average(results["minus"]["c1"]["2m"]["e1c"])
    m_c1_e2c_2m = np.average(results["minus"]["c1"]["2m"]["e2c"])
    m_c1_e1cc_2m = np.average(results["minus"]["c1"]["2m"]["e1cc"])
    m_c1_e2cc_2m = np.average(results["minus"]["c1"]["2m"]["e2cc"])
    m_c1_e1dc_2m = np.average(results["minus"]["c1"]["2m"]["e1dc"])
    m_c1_e2dc_2m = np.average(results["minus"]["c1"]["2m"]["e2dc"])
    m_c1_e1dc2_2m = np.average(results["minus"]["c1"]["2m"]["e1dc2"])
    m_c1_e2dc2_2m = np.average(results["minus"]["c1"]["2m"]["e2dc2"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    m_c2_e1_ns = np.average(results["minus"]["c2"]["noshear"]["e1"])
    m_c2_e2_ns = np.average(results["minus"]["c2"]["noshear"]["e2"])
    m_c2_e1c_ns = np.average(results["minus"]["c2"]["noshear"]["e1c"])
    m_c2_e2c_ns = np.average(results["minus"]["c2"]["noshear"]["e2c"])
    m_c2_e1cc_ns = np.average(results["minus"]["c2"]["noshear"]["e1cc"])
    m_c2_e2cc_ns = np.average(results["minus"]["c2"]["noshear"]["e2cc"])
    m_c2_e1dc_ns = np.average(results["minus"]["c2"]["noshear"]["e1dc"])
    m_c2_e2dc_ns = np.average(results["minus"]["c2"]["noshear"]["e2dc"])
    m_c2_e1dc2_ns = np.average(results["minus"]["c2"]["noshear"]["e1dc2"])
    m_c2_e2dc2_ns = np.average(results["minus"]["c2"]["noshear"]["e2dc2"])

    # 1p
    m_c2_e1_1p = np.average(results["minus"]["c2"]["1p"]["e1"])
    m_c2_e2_1p = np.average(results["minus"]["c2"]["1p"]["e2"])
    m_c2_e1c_1p = np.average(results["minus"]["c2"]["1p"]["e1c"])
    m_c2_e2c_1p = np.average(results["minus"]["c2"]["1p"]["e2c"])
    m_c2_e1cc_1p = np.average(results["minus"]["c2"]["1p"]["e1cc"])
    m_c2_e2cc_1p = np.average(results["minus"]["c2"]["1p"]["e2cc"])
    m_c2_e1dc_1p = np.average(results["minus"]["c2"]["1p"]["e1dc"])
    m_c2_e2dc_1p = np.average(results["minus"]["c2"]["1p"]["e2dc"])
    m_c2_e1dc2_1p = np.average(results["minus"]["c2"]["1p"]["e1dc2"])
    m_c2_e2dc2_1p = np.average(results["minus"]["c2"]["1p"]["e2dc2"])

    # 1m
    m_c2_e1_1m = np.average(results["minus"]["c2"]["1m"]["e1"])
    m_c2_e2_1m = np.average(results["minus"]["c2"]["1m"]["e2"])
    m_c2_e1c_1m = np.average(results["minus"]["c2"]["1m"]["e1c"])
    m_c2_e2c_1m = np.average(results["minus"]["c2"]["1m"]["e2c"])
    m_c2_e1cc_1m = np.average(results["minus"]["c2"]["1m"]["e1cc"])
    m_c2_e2cc_1m = np.average(results["minus"]["c2"]["1m"]["e2cc"])
    m_c2_e1dc_1m = np.average(results["minus"]["c2"]["1m"]["e1dc"])
    m_c2_e2dc_1m = np.average(results["minus"]["c2"]["1m"]["e2dc"])
    m_c2_e1dc2_1m = np.average(results["minus"]["c2"]["1m"]["e1dc2"])
    m_c2_e2dc2_1m = np.average(results["minus"]["c2"]["1m"]["e2dc2"])

    # 2p
    m_c2_e1_2p = np.average(results["minus"]["c2"]["2p"]["e1"])
    m_c2_e2_2p = np.average(results["minus"]["c2"]["2p"]["e2"])
    m_c2_e1c_2p = np.average(results["minus"]["c2"]["2p"]["e1c"])
    m_c2_e2c_2p = np.average(results["minus"]["c2"]["2p"]["e2c"])
    m_c2_e1cc_2p = np.average(results["minus"]["c2"]["2p"]["e1cc"])
    m_c2_e2cc_2p = np.average(results["minus"]["c2"]["2p"]["e2cc"])
    m_c2_e1dc_2p = np.average(results["minus"]["c2"]["2p"]["e1dc"])
    m_c2_e2dc_2p = np.average(results["minus"]["c2"]["2p"]["e2dc"])
    m_c2_e1dc2_2p = np.average(results["minus"]["c2"]["2p"]["e1dc2"])
    m_c2_e2dc2_2p = np.average(results["minus"]["c2"]["2p"]["e2dc2"])

    # 2m
    m_c2_e1_2m = np.average(results["minus"]["c2"]["2m"]["e1"])
    m_c2_e2_2m = np.average(results["minus"]["c2"]["2m"]["e2"])
    m_c2_e1c_2m = np.average(results["minus"]["c2"]["2m"]["e1c"])
    m_c2_e2c_2m = np.average(results["minus"]["c2"]["2m"]["e2c"])
    m_c2_e1cc_2m = np.average(results["minus"]["c2"]["2m"]["e1cc"])
    m_c2_e2cc_2m = np.average(results["minus"]["c2"]["2m"]["e2cc"])
    m_c2_e1dc_2m = np.average(results["minus"]["c2"]["2m"]["e1dc"])
    m_c2_e2dc_2m = np.average(results["minus"]["c2"]["2m"]["e2dc"])
    m_c2_e1dc2_2m = np.average(results["minus"]["c2"]["2m"]["e1dc2"])
    m_c2_e2dc2_2m = np.average(results["minus"]["c2"]["2m"]["e2dc2"])

    #---------------------------------------------------------------------------

    # dR11_p = (p_c2_e1c_1p - p_c2_e1c_1m - p_c0_e1c_1p + p_c0_e1c_1m) / (2 * dg * 2 * dc)
    # dR12_p = (p_c2_e1c_2p - p_c2_e1c_2m - p_c0_e1c_2p + p_c0_e1c_2m) / (2 * dg * 2 * dc)
    # dR21_p = (p_c2_e2c_1p - p_c2_e2c_1m - p_c0_e2c_1p + p_c0_e2c_1m) / (2 * dg * 2 * dc)
    # dR22_p = (p_c2_e2c_2p - p_c2_e2c_2m - p_c0_e2c_2p + p_c0_e2c_2m) / (2 * dg * 2 * dc)

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
                p_c2_e1dc2_1p - p_c2_e1dc2_1m - 2 * (p_c1_e1dc2_1p - p_c1_e1dc2_1m) + p_c0_e1dc2_1p - p_c0_e1dc2_1m
                # p_c2_e1cc_1p - p_c2_e1cc_1m - 2 * (p_c1_e1cc_1p - p_c1_e1cc_1m) + p_c0_e1cc_1p - p_c0_e1cc_1m \
                # - 2 * color * (p_c2_e1c_1p - p_c2_e1c_1m - 2 * (p_c1_e1c_1p - p_c1_e1c_1m) + p_c0_e1c_1p - p_c0_e1c_1m) \
                # + color**2 * (p_c2_e1_1p - p_c2_e1_1m - 2 * (p_c1_e1_1p - p_c1_e1_1m) + p_c0_e1_1p - p_c0_e1_1m)
            ) / (2 * dg * 2 * dc**2)
            dR12_p = (
                p_c2_e1c_2p - p_c2_e1c_2m - p_c0_e1c_2p + p_c0_e1c_2m - color * (
                    p_c2_e1_2p - p_c2_e1_2m - p_c0_e1_2p + p_c0_e1_2m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e1dc2_2p - p_c2_e1dc2_2m - 2 * (p_c1_e1dc2_2p - p_c1_e1dc2_2m) + p_c0_e1dc2_2p - p_c0_e1dc2_2m
                # p_c2_e1cc_2p - p_c2_e1cc_2m - 2 * (p_c1_e1cc_2p - p_c1_e1cc_2m) + p_c0_e1cc_2p - p_c0_e1cc_2m \
                # - 2 * color * (p_c2_e1c_2p - p_c2_e1c_2m - 2 * (p_c1_e1c_2p - p_c1_e1c_2m) + p_c0_e1c_2p - p_c0_e1c_2m) \
                # + color**2 * (p_c2_e1_2p - p_c2_e1_2m - 2 * (p_c1_e1_2p - p_c1_e1_2m) + p_c0_e1_2p - p_c0_e1_2m)
            ) / (2 * dg * 2 * dc**2)
            dR21_p = (
                p_c2_e2c_1p - p_c2_e2c_1m - p_c0_e2c_1p + p_c0_e2c_1m - color * (
                    p_c2_e2_1p - p_c2_e2_1m - p_c0_e2_1p + p_c0_e2_1m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e2dc2_1p - p_c2_e2dc2_1m - 2 * (p_c1_e2dc2_1p - p_c1_e2dc2_1m) + p_c0_e2dc2_1p - p_c0_e2dc2_1m
                # p_c2_e2cc_1p - p_c2_e2cc_1m - 2 * (p_c1_e2cc_1p - p_c1_e2cc_1m) + p_c0_e2cc_1p - p_c0_e2cc_1m \
                # - 2 * color * (p_c2_e2c_1p - p_c2_e2c_1m - 2 * (p_c1_e2c_1p - p_c1_e2c_1m) + p_c0_e2c_1p - p_c0_e2c_1m) \
                # + color**2 * (p_c2_e2_1p - p_c2_e2_1m - 2 * (p_c1_e2_1p - p_c1_e2_1m) + p_c0_e2_1p - p_c0_e2_1m)
            ) / (2 * dg * 2 * dc**2)
            dR22_p = (
                p_c2_e2c_2p - p_c2_e2c_2m - p_c0_e2c_2p + p_c0_e2c_2m - color * (
                    p_c2_e2_2p - p_c2_e2_2m - p_c0_e2_2p + p_c0_e2_2m
                )
            ) / (2 * dg * 2 * dc) + (
                p_c2_e2dc2_2p - p_c2_e2dc2_2m - 2 * (p_c1_e2dc2_2p - p_c1_e2dc2_2m) + p_c0_e2dc2_2p - p_c0_e2dc2_2m
                # p_c2_e2cc_2p - p_c2_e2cc_2m - 2 * (p_c1_e2cc_2p - p_c1_e2cc_2m) + p_c0_e2cc_2p - p_c0_e2cc_2m \
                # - 2 * color * (p_c2_e2c_2p - p_c2_e2c_2m - 2 * (p_c1_e2c_2p - p_c1_e2c_2m) + p_c0_e2c_2p - p_c0_e2c_2m) \
                # + color**2 * (p_c2_e2_2p - p_c2_e2_2m - 2 * (p_c1_e2_2p - p_c1_e2_2m) + p_c0_e2_2p - p_c0_e2_2m)
            ) / (2 * dg * 2 * dc**2)

            dR11_m = (
                m_c2_e1c_1p - m_c2_e1c_1m - m_c0_e1c_1p + m_c0_e1c_1m - color * (
                    m_c2_e1_1p - m_c2_e1_1m - m_c0_e1_1p + m_c0_e1_1m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e1dc2_1p - m_c2_e1dc2_1m - 2 * (m_c1_e1dc2_1p - m_c1_e1dc2_1m) + m_c0_e1dc2_1p - m_c0_e1dc2_1m
                # m_c2_e1cc_1p - m_c2_e1cc_1m - 2 * (m_c1_e1cc_1p - m_c1_e1cc_1m) + m_c0_e1cc_1p - m_c0_e1cc_1m \
                # - 2 * color * (m_c2_e1c_1p - m_c2_e1c_1m - 2 * (m_c1_e1c_1p - m_c1_e1c_1m) + m_c0_e1c_1p - m_c0_e1c_1m) \
                # + color**2 * (m_c2_e1_1p - m_c2_e1_1m - 2 * (m_c1_e1_1p - m_c1_e1_1m) + m_c0_e1_1p - m_c0_e1_1m)
            ) / (2 * dg * 2 * dc**2)
            dR12_m = (
                m_c2_e1c_2p - m_c2_e1c_2m - m_c0_e1c_2p + m_c0_e1c_2m - color * (
                    m_c2_e1_2p - m_c2_e1_2m - m_c0_e1_2p + m_c0_e1_2m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e1dc2_2p - m_c2_e1dc2_2m - 2 * (m_c1_e1dc2_2p - m_c1_e1dc2_2m) + m_c0_e1dc2_2p - m_c0_e1dc2_2m
                # m_c2_e1cc_2p - m_c2_e1cc_2m - 2 * (m_c1_e1cc_2p - m_c1_e1cc_2m) + m_c0_e1cc_2p - m_c0_e1cc_2m \
                # - 2 * color * (m_c2_e1c_2p - m_c2_e1c_2m - 2 * (m_c1_e1c_2p - m_c1_e1c_2m) + m_c0_e1c_2p - m_c0_e1c_2m) \
                # + color**2 * (m_c2_e1_2p - m_c2_e1_2m - 2 * (m_c1_e1_2p - m_c1_e1_2m) + m_c0_e1_2p - m_c0_e1_2m)
            ) / (2 * dg * 2 * dc**2)
            dR21_m = (
                m_c2_e2c_1p - m_c2_e2c_1m - m_c0_e2c_1p + m_c0_e2c_1m - color * (
                    m_c2_e2_1p - m_c2_e2_1m - m_c0_e2_1p + m_c0_e2_1m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e2dc2_1p - m_c2_e2dc2_1m - 2 * (m_c1_e2dc2_1p - m_c1_e2dc2_1m) + m_c0_e2dc2_1p - m_c0_e2dc2_1m
                # m_c2_e2cc_1p - m_c2_e2cc_1m - 2 * (m_c1_e2cc_1p - m_c1_e2cc_1m) + m_c0_e2cc_1p - m_c0_e2cc_1m \
                # - 2 * color * (m_c2_e2c_1p - m_c2_e2c_1m - 2 * (m_c1_e2c_1p - m_c1_e2c_1m) + m_c0_e2c_1p - m_c0_e2c_1m) \
                # + color**2 * (m_c2_e2_1p - m_c2_e2_1m - 2 * (m_c1_e2_1p - m_c1_e2_1m) + m_c0_e2_1p - m_c0_e2_1m)
            ) / (2 * dg * 2 * dc**2)
            dR22_m = (
                m_c2_e2c_2p - m_c2_e2c_2m - m_c0_e2c_2p + m_c0_e2c_2m - color * (
                    m_c2_e2_2p - m_c2_e2_2m - m_c0_e2_2p + m_c0_e2_2m
                )
            ) / (2 * dg * 2 * dc) + (
                m_c2_e2dc2_2p - m_c2_e2dc2_2m - 2 * (m_c1_e2dc2_2p - m_c1_e2dc2_2m) + m_c0_e2dc2_2p - m_c0_e2dc2_2m
                # m_c2_e2cc_2p - m_c2_e2cc_2m - 2 * (m_c1_e2cc_2p - m_c1_e2cc_2m) + m_c0_e2cc_2p - m_c0_e2cc_2m \
                # - 2 * color * (m_c2_e2c_2p - m_c2_e2c_2m - 2 * (m_c1_e2c_2p - m_c1_e2c_2m) + m_c0_e2c_2p - m_c0_e2c_2m) \
                # + color**2 * (m_c2_e2_2p - m_c2_e2_2m - 2 * (m_c1_e2_2p - m_c1_e2_2m) + m_c0_e2_2p - m_c0_e2_2m)
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


def compute_bias(batch, dg, dc):
    # e_p, e_m = compute_e(batch)
    # R_p, R_m = compute_R(batch, dg)

    e_p, e_m = compute_e_chromatic(batch)
    R_p, R_m = compute_R_chromatic(batch, dg)

    g_p = np.linalg.inv(R_p) @ e_p
    g_m = np.linalg.inv(R_m) @ e_m

    m = (g_p - g_m)[0] / 2 / 0.02 - 1

    c = (g_p + g_m)[1] / 2

    # return g_p, g_m, m, c
    return m, c

    # AE = (e_p + e_m) / 2
    # DE = e_p - e_m
    # AR = (R_p + R_m) / 2

    # m = (np.linalg.inv(AR) @ DE / 2 / 0.02 - 1)[0]
    # c = (np.linalg.inv(AR) @ AE)[1]

    # return m, c


def compute_bias_chromatic(batch, dg, dc, color, order=1):
    e_p, e_m = compute_e_chromatic(batch)
    R_p, R_m = compute_R_chromatic(batch, dg)

    de_p, de_m = compute_de(batch, dg, dc, color, order=order)
    dR_p, dR_m = compute_dR(batch, dg, dc, color, order=order)

    g_p = np.linalg.inv(R_p + dR_p) @ (e_p + de_p)
    g_m = np.linalg.inv(R_m + dR_m) @ (e_m + de_m)

    m = (g_p - g_m)[0] / 2 / 0.02 - 1

    c = (g_p + g_m)[1] / 2

    return m, c

    # AE = ((e_p - dedc_p) + (e_m - dedc_m)) / 2
    # DE = (e_p - dedc_p) - (e_m - dedc_m)
    # AR = ((R_p + dRdc_p) + (R_m + dRdc_m)) / 2

    # m = (np.linalg.inv(AR) @ DE / 2 / 0.02 - 1)[0]
    # c = (np.linalg.inv(AR) @ AE)[1]

    # return m, c


def pivot_aggregates(res, seeds=None):
    """
    This more or less implements a pivot wider operation via successive
    application of filters
    """
    table_source_node = acero.Declaration(
        "table_source",
        acero.TableSourceNodeOptions(
            res
        ),
    )

    aggregates = {}
    for shear_step in ["plus", "minus"]:
        aggregates[shear_step] = {}
        for mdet_step in ["noshear", "1p", "1m", "2p", "2m"]:
            # print(f"aggregating {shear_step}:{mdet_step}")
            predicate = (
                (pc.field("shear_step") == shear_step)
                & (pc.field("mdet_step") == mdet_step)
            )
            if seeds is not None:
                predicate &= pc.is_in(pc.field("seed"), seeds)
            post_filter_node = acero.Declaration(
                "filter",
                acero.FilterNodeOptions(
                    predicate,
                ),
            )
            weight_project_node = acero.Declaration(
                "project",
                acero.ProjectNodeOptions(
                    [
                        pc.field("count"),
                        pc.multiply(pc.field("mean_e1"), pc.field("count")),
                        pc.multiply(pc.field("mean_e2"), pc.field("count")),
                        pc.multiply(pc.field("mean_c"), pc.field("count")),
                        pc.multiply(pc.field("mean_e1e2"), pc.field("count")),
                        pc.multiply(pc.field("mean_e1c"), pc.field("count")),
                        pc.multiply(pc.field("mean_e2c"), pc.field("count")),
                        pc.multiply(pc.field("mean_e1cc"), pc.field("count")),
                        pc.multiply(pc.field("mean_e2cc"), pc.field("count")),
                        pc.multiply(pc.field("mean_e1dc"), pc.field("count")),
                        pc.multiply(pc.field("mean_e2dc"), pc.field("count")),
                        pc.multiply(pc.field("mean_e1dc2"), pc.field("count")),
                        pc.multiply(pc.field("mean_e2dc2"), pc.field("count")),
                        # pc.multiply(pc.field("var_e1"), pc.field("count")),
                        # pc.multiply(pc.field("var_e2"), pc.field("count")),
                        # pc.multiply(pc.field("var_c"), pc.field("count")),
                        # pc.multiply(pc.field("var_e1e2"), pc.field("count")),
                        # pc.multiply(pc.field("var_e1c"), pc.field("count")),
                        # pc.multiply(pc.field("var_e2c"), pc.field("count")),
                        # pc.multiply(pc.field("var_e1dc"), pc.field("count")),
                        # pc.multiply(pc.field("var_e2dc"), pc.field("count")),
                    ],
                    names=[
                        "count",
                        "weighted_mean_e1",
                        "weighted_mean_e2",
                        "weighted_mean_c",
                        "weighted_mean_e1e2",
                        "weighted_mean_e1c",
                        "weighted_mean_e2c",
                        "weighted_mean_e1cc",
                        "weighted_mean_e2cc",
                        "weighted_mean_e1dc",
                        "weighted_mean_e2dc",
                        "weighted_mean_e1dc2",
                        "weighted_mean_e2dc2",
                        # "weighted_var_e1",
                        # "weighted_var_e2",
                        # "weighted_var_c",
                        # "weighted_var_e1e2",
                        # "weighted_var_e1c",
                        # "weighted_var_e2c",
                        # "weighted_var_e1dc",
                        # "weighted_var_e2dc",
                    ],
                ),
            )
            post_aggregate_node = acero.Declaration(
                "aggregate",
                acero.AggregateNodeOptions(
                    [
                        ("count", "sum", None, "sum_count"),
                        ("weighted_mean_e1", "sum", None, "sum_mean_e1"),
                        ("weighted_mean_e2", "sum", None, "sum_mean_e2"),
                        ("weighted_mean_c", "sum", None, "sum_mean_c"),
                        ("weighted_mean_e1e2", "sum", None, "sum_mean_e1e2"),
                        ("weighted_mean_e1c", "sum", None, "sum_mean_e1c"),
                        ("weighted_mean_e2c", "sum", None, "sum_mean_e2c"),
                        ("weighted_mean_e1cc", "sum", None, "sum_mean_e1cc"),
                        ("weighted_mean_e2cc", "sum", None, "sum_mean_e2cc"),
                        ("weighted_mean_e1dc", "sum", None, "sum_mean_e1dc"),
                        ("weighted_mean_e2dc", "sum", None, "sum_mean_e2dc"),
                        ("weighted_mean_e1dc2", "sum", None, "sum_mean_e1dc2"),
                        ("weighted_mean_e2dc2", "sum", None, "sum_mean_e2dc2"),
                        # ("weighted_var_e1", "sum", None, "sum_var_e1"),
                        # ("weighted_var_e2", "sum", None, "sum_var_e2"),
                        # ("weighted_var_c", "sum", None, "sum_var_c"),
                        # ("weighted_var_e1e2", "sum", None, "sum_var_e1e2"),
                        # ("weighted_var_e1c", "sum", None, "sum_var_e1c"),
                        # ("weighted_var_e2c", "sum", None, "sum_var_e2c"),
                        # ("weighted_var_e1dc", "sum", None, "sum_var_e1dc"),
                        # ("weighted_var_e2dc", "sum", None, "sum_var_e2dc"),
                    ],
                ),
            )
            post_project_node = acero.Declaration(
                "project",
                acero.ProjectNodeOptions(
                    [
                        pc.divide(pc.field("sum_mean_e1"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e2"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_c"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e1e2"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e1c"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e2c"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e1cc"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e2cc"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e1dc"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e2dc"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e1dc2"), pc.field("sum_count")),
                        pc.divide(pc.field("sum_mean_e2dc2"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e1"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e2"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_c"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e1e2"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e1c"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e2c"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e1dc"), pc.field("sum_count")),
                        # pc.divide(pc.field("sum_var_e2dc"), pc.field("sum_count")),
                    ],
                    names=[
                        "e1",
                        "e2",
                        "c",
                        "e1e2",
                        "e1c",
                        "e2c",
                        "e1cc",
                        "e2cc",
                        "e1dc",
                        "e2dc",
                        "e1dc2",
                        "e2dc2",
                        # "var_e1",
                        # "var_e2",
                        # "var_c",
                        # "var_e1e2",
                        # "var_e1c",
                        # "var_e2c",
                        # "var_e1dc",
                        # "var_e2dc",
                    ],
                ),
            )
            reproject_node = acero.Declaration(
                "project",
                acero.ProjectNodeOptions(
                    [
                        pc.field("e1"),
                        pc.field("e2"),
                        pc.field("c"),
                        pc.field("e1e2"),
                        pc.field("e1c"),
                        pc.field("e2c"),
                        pc.field("e1cc"),
                        pc.field("e2cc"),
                        pc.field("e1dc"),
                        pc.field("e2dc"),
                        pc.field("e1dc2"),
                        pc.field("e2dc2"),
                        # pc.field("var_e1"),
                        # pc.field("var_e2"),
                        # pc.field("var_c"),
                        # pc.field("var_e1e2"),
                        # pc.field("var_e1c"),
                        # pc.field("var_e2c"),
                        # pc.field("var_e1dc"),
                        # pc.field("var_e2dc"),
                        # pc.subtract(
                        #     pc.field("e1e2"),
                        #     pc.multiply(
                        #         pc.field("e1"),
                        #         pc.field("e2"),
                        #     ),
                        # ),
                        # pc.subtract(
                        #     pc.field("e1c"),
                        #     pc.multiply(
                        #         pc.field("e1"),
                        #         pc.field("c"),
                        #     ),
                        # ),
                        # pc.subtract(
                        #     pc.field("e2c"),
                        #     pc.multiply(
                        #         pc.field("e2"),
                        #         pc.field("c"),
                        #     ),
                        # ),
                    ],
                    names=[
                        "e1",
                        "e2",
                        "c",
                        "e1e2",
                        "e1c",
                        "e2c",
                        "e1cc",
                        "e2cc",
                        "e1dc",
                        "e2dc",
                        "e1dc2",
                        "e2dc2",
                        # "var_e1",
                        # "var_e2",
                        # "var_c",
                        # "var_e1e2",
                        # "var_e1c",
                        # "var_e2c",
                        # "var_e1dc",
                        # "var_e2dc",
                        # "cov_e1e2",
                        # "cov_e1c",
                        # "cov_e2c",
                    ],
                ),
            )
            seq = [
                table_source_node,
                post_filter_node,
                weight_project_node,
                post_aggregate_node,
                post_project_node,
                reproject_node,
            ]
            plan = acero.Declaration.from_sequence(seq)
            pivot = plan.to_table(use_threads=True)
            aggregates[shear_step][mdet_step] = pivot

    return aggregates


def pivot_aggregates_chromatic(res, seeds=None):
    """
    This more or less implements a pivot wider operation via successive
    application of filters
    """
    table_source_node = acero.Declaration(
        "table_source",
        acero.TableSourceNodeOptions(
            res
        ),
    )

    aggregates = {}
    for shear_step in ["plus", "minus"]:
        aggregates[shear_step] = {}
        for color_step in ["c0", "c1", "c2"]:
            aggregates[shear_step][color_step] = {}
            for mdet_step in ["noshear", "1p", "1m", "2p", "2m"]:
                # print(f"aggregating {shear_step}:{color_step}:{mdet_step}")
                predicate = (
                    (pc.field("shear_step") == shear_step)
                    & (pc.field("color_step") == color_step)
                    & (pc.field("mdet_step") == mdet_step)
                )
                if seeds is not None:
                    predicate &= pc.is_in(pc.field("seed"), seeds)
                post_filter_node = acero.Declaration(
                    "filter",
                    acero.FilterNodeOptions(
                        predicate,
                    ),
                )
                weight_project_node = acero.Declaration(
                    "project",
                    acero.ProjectNodeOptions(
                        [
                            pc.field("count"),
                            pc.multiply(pc.field("mean_e1"), pc.field("count")),
                            pc.multiply(pc.field("mean_e2"), pc.field("count")),
                            pc.multiply(pc.field("mean_c"), pc.field("count")),
                            pc.multiply(pc.field("mean_e1e2"), pc.field("count")),
                            pc.multiply(pc.field("mean_e1c"), pc.field("count")),
                            pc.multiply(pc.field("mean_e2c"), pc.field("count")),
                            pc.multiply(pc.field("mean_e1cc"), pc.field("count")),
                            pc.multiply(pc.field("mean_e2cc"), pc.field("count")),
                            pc.multiply(pc.field("mean_e1dc"), pc.field("count")),
                            pc.multiply(pc.field("mean_e2dc"), pc.field("count")),
                            pc.multiply(pc.field("mean_e1dc2"), pc.field("count")),
                            pc.multiply(pc.field("mean_e2dc2"), pc.field("count")),
                            # pc.multiply(pc.field("var_e1"), pc.field("count")),
                            # pc.multiply(pc.field("var_e2"), pc.field("count")),
                            # pc.multiply(pc.field("var_c"), pc.field("count")),
                            # pc.multiply(pc.field("var_e1e2"), pc.field("count")),
                            # pc.multiply(pc.field("var_e1c"), pc.field("count")),
                            # pc.multiply(pc.field("var_e2c"), pc.field("count")),
                            # pc.multiply(pc.field("var_e1dc"), pc.field("count")),
                            # pc.multiply(pc.field("var_e2dc"), pc.field("count")),
                        ],
                        names=[
                            "count",
                            "weighted_mean_e1",
                            "weighted_mean_e2",
                            "weighted_mean_c",
                            "weighted_mean_e1e2",
                            "weighted_mean_e1c",
                            "weighted_mean_e2c",
                            "weighted_mean_e1cc",
                            "weighted_mean_e2cc",
                            "weighted_mean_e1dc",
                            "weighted_mean_e2dc",
                            "weighted_mean_e1dc2",
                            "weighted_mean_e2dc2",
                            # "weighted_var_e1",
                            # "weighted_var_e2",
                            # "weighted_var_c",
                            # "weighted_var_e1e2",
                            # "weighted_var_e1c",
                            # "weighted_var_e2c",
                            # "weighted_var_e1dc",
                            # "weighted_var_e2dc",
                        ],
                    ),
                )
                post_aggregate_node = acero.Declaration(
                    "aggregate",
                    acero.AggregateNodeOptions(
                        [
                            ("count", "sum", None, "sum_count"),
                            ("weighted_mean_e1", "sum", None, "sum_mean_e1"),
                            ("weighted_mean_e2", "sum", None, "sum_mean_e2"),
                            ("weighted_mean_c", "sum", None, "sum_mean_c"),
                            ("weighted_mean_e1e2", "sum", None, "sum_mean_e1e2"),
                            ("weighted_mean_e1c", "sum", None, "sum_mean_e1c"),
                            ("weighted_mean_e2c", "sum", None, "sum_mean_e2c"),
                            ("weighted_mean_e1cc", "sum", None, "sum_mean_e1cc"),
                            ("weighted_mean_e2cc", "sum", None, "sum_mean_e2cc"),
                            ("weighted_mean_e1dc", "sum", None, "sum_mean_e1dc"),
                            ("weighted_mean_e2dc", "sum", None, "sum_mean_e2dc"),
                            ("weighted_mean_e1dc2", "sum", None, "sum_mean_e1dc2"),
                            ("weighted_mean_e2dc2", "sum", None, "sum_mean_e2dc2"),
                            # ("weighted_var_e1", "sum", None, "sum_var_e1"),
                            # ("weighted_var_e2", "sum", None, "sum_var_e2"),
                            # ("weighted_var_c", "sum", None, "sum_var_c"),
                            # ("weighted_var_e1e2", "sum", None, "sum_var_e1e2"),
                            # ("weighted_var_e1c", "sum", None, "sum_var_e1c"),
                            # ("weighted_var_e2c", "sum", None, "sum_var_e2c"),
                            # ("weighted_var_e1dc", "sum", None, "sum_var_e1dc"),
                            # ("weighted_var_e2dc", "sum", None, "sum_var_e2dc"),
                        ],
                    ),
                )
                post_project_node = acero.Declaration(
                    "project",
                    acero.ProjectNodeOptions(
                        [
                            pc.divide(pc.field("sum_mean_e1"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e2"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_c"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e1e2"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e1c"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e2c"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e1cc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e2cc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e1dc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e2dc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e1dc2"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e2dc2"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e1"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e2"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_c"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e1e2"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e1c"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e2c"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e1dc"), pc.field("sum_count")),
                            # pc.divide(pc.field("sum_var_e2dc"), pc.field("sum_count")),
                        ],
                        names=[
                            "e1",
                            "e2",
                            "c",
                            "e1e2",
                            "e1c",
                            "e2c",
                            "e1cc",
                            "e2cc",
                            "e1dc",
                            "e2dc",
                            "e1dc2",
                            "e2dc2",
                            # "var_e1",
                            # "var_e2",
                            # "var_c",
                            # "var_e1e2",
                            # "var_e1c",
                            # "var_e2c",
                            # "var_e1dc",
                            # "var_e2dc",
                        ],
                    ),
                )
                reproject_node = acero.Declaration(
                    "project",
                    acero.ProjectNodeOptions(
                        [
                            pc.field("e1"),
                            pc.field("e2"),
                            pc.field("c"),
                            pc.field("e1e2"),
                            pc.field("e1c"),
                            pc.field("e2c"),
                            pc.field("e1cc"),
                            pc.field("e2cc"),
                            pc.field("e1dc"),
                            pc.field("e2dc"),
                            pc.field("e1dc2"),
                            pc.field("e2dc2"),
                            # pc.field("var_e1"),
                            # pc.field("var_e2"),
                            # pc.field("var_c"),
                            # pc.field("var_e1e2"),
                            # pc.field("var_e1c"),
                            # pc.field("var_e2c"),
                            # pc.field("var_e1dc"),
                            # pc.field("var_e2dc"),
                            # pc.subtract(
                            #     pc.field("e1e2"),
                            #     pc.multiply(
                            #         pc.field("e1"),
                            #         pc.field("e2"),
                            #     ),
                            # ),
                            # pc.subtract(
                            #     pc.field("e1c"),
                            #     pc.multiply(
                            #         pc.field("e1"),
                            #         pc.field("c"),
                            #     ),
                            # ),
                            # pc.subtract(
                            #     pc.field("e2c"),
                            #     pc.multiply(
                            #         pc.field("e2"),
                            #         pc.field("c"),
                            #     ),
                            # ),
                        ],
                        names=[
                            "e1",
                            "e2",
                            "c",
                            "e1e2",
                            "e1c",
                            "e2c",
                            "e1cc",
                            "e2cc",
                            "e1dc",
                            "e2dc",
                            "e1dc2",
                            "e2dc2",
                            # "var_e1",
                            # "var_e2",
                            # "var_c",
                            # "var_e1e2",
                            # "var_e1c",
                            # "var_e2c",
                            # "var_e1dc",
                            # "var_e2dc",
                            # "cov_e1e2",
                            # "cov_e1c",
                            # "cov_e2c",
                        ],
                    ),
                )
                seq = [
                    table_source_node,
                    post_filter_node,
                    weight_project_node,
                    post_aggregate_node,
                    post_project_node,
                    reproject_node,
                ]
                plan = acero.Declaration.from_sequence(seq)
                pivot = plan.to_table(use_threads=True)
                aggregates[shear_step][color_step][mdet_step] = pivot

    return aggregates


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
    output_path = args.output

    config_name = name_util.get_config_name(config_file)
    aggregate_path = name_util.get_aggregate_path(args.output, args.config)

    pa.set_cpu_count(n_jobs)
    pa.set_io_thread_count(2 * n_jobs)

    rng = np.random.default_rng(seed)

    simulation_builder = SimulationBuilder.from_yaml(config_file)
    measure = measurement.get_measure(
        **simulation_builder.config["measurement"].get("builder"),
    )

    dg = ngmix.metacal.DEFAULT_STEP

    psf_colors = simulation_builder.config["measurement"].get("colors")
    dc = psf_colors[1] - psf_colors[0]
    color = psf_colors[1]

    print(f"reading aggregates from {aggregate_path}")
    aggregates = ft.read_table(
        aggregate_path,
    )
    aggregates.validate(full=True)

    seeds = np.sort(np.unique(aggregates["seed"]))

    print(f"aggregating results")
    results = pivot_aggregates_chromatic(aggregates)

    m_mean, c_mean = compute_bias(results, dg, dc)
    m_mean_c1, c_mean_c1 = compute_bias_chromatic(results, dg, dc, color, order=1)
    m_mean_c2, c_mean_c2 = compute_bias_chromatic(results, dg, dc, color, order=2)

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
            functools.partial(pivot_aggregates_chromatic, aggregates),
            (
                pa.array(rng.choice(seeds, size=len(seeds), replace=True))
                for _ in range(n_resample)
            ),
        )

        for i, res in track(enumerate(results), description="bootstrapping", total=n_resample):
            _m_bootstrap, _c_bootstrap = compute_bias(res, dg, dc)
            _m_bootstrap_c1, _c_bootstrap_c1 = compute_bias_chromatic(res, dg, dc, color, order=1)
            _m_bootstrap_c2, _c_bootstrap_c2 = compute_bias_chromatic(res, dg, dc, color, order=2)

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

    # report 3 standard deviations as error
    m_error = np.nanstd(m_bootstrap) * 3
    c_error = np.nanstd(c_bootstrap) * 3
    m_error_c1 = np.nanstd(m_bootstrap_c1) * 3
    c_error_c1 = np.nanstd(c_bootstrap_c1) * 3
    m_error_c2 = np.nanstd(m_bootstrap_c2) * 3
    c_error_c2 = np.nanstd(c_bootstrap_c2) * 3

    print(f"mdet (0): m = {m_mean:0.3e} +/- {m_error:0.3e} [3-sigma], c = {c_mean:0.3e} +/- {c_error:0.3e} [3-sigma]")
    print(f"drdc (1): m = {m_mean_c1:0.3e} +/- {m_error_c1:0.3e} [3-sigma], c = {c_mean_c1:0.3e} +/- {c_error_c1:0.3e} [3-sigma]")
    print(f"drdc (2): m = {m_mean_c2:0.3e} +/- {m_error_c2:0.3e} [3-sigma], c = {c_mean_c2:0.3e} +/- {c_error_c2:0.3e} [3-sigma]")

    m_req = 2e-3

    m_min = np.min([m_bootstrap.min(), m_bootstrap_c1.min(), m_bootstrap_c2.min()])
    m_max = np.max([m_bootstrap.max(), m_bootstrap_c1.max(), m_bootstrap_c2.max()])
    m_bin_edges = np.linspace(m_min, m_max, 50)
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

    figname = f"{config_name}-m-0.pdf"
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

    figname = f"{config_name}-m-1.pdf"
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

    figname = f"{config_name}-m-2.pdf"
    fig.savefig(figname)


if __name__ == "__main__":
    main()
