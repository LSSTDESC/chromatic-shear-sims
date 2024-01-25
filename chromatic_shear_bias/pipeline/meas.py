import argparse

import ngmix
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pyarrow import acero
import tqdm

from chromatic_shear_bias.pipeline.pipeline import Pipeline


CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}


def compute_e(results):
    # NOSHEAR
    p_c1_g1_ns = np.nanmean(results["plus"]["c1"]["noshear"]["g1"])
    p_c1_g2_ns = np.nanmean(results["plus"]["c1"]["noshear"]["g2"])

    m_c1_g1_ns = np.nanmean(results["minus"]["c1"]["noshear"]["g1"])
    m_c1_g2_ns = np.nanmean(results["minus"]["c1"]["noshear"]["g2"])

    return np.array([p_c1_g1_ns, p_c1_g2_ns]), np.array([m_c1_g1_ns, m_c1_g2_ns])


def compute_R(results, dg):

    p_c1_R11 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c1"]["1p"]["g1"],
        results["plus"]["c1"]["1m"]["g1"]
    ), 2 * dg))
    p_c1_R12 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c1"]["2p"]["g1"],
        results["plus"]["c1"]["2m"]["g1"]
    ), 2 * dg))
    p_c1_R21 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c1"]["1p"]["g2"],
        results["plus"]["c1"]["1m"]["g2"]
    ), 2 * dg))
    p_c1_R22 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c1"]["2p"]["g2"],
        results["plus"]["c1"]["2m"]["g2"]
    ), 2 * dg))

    m_c1_R11 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c1"]["1p"]["g1"],
        results["minus"]["c1"]["1m"]["g1"]
    ), 2 * dg))
    m_c1_R12 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c1"]["2p"]["g1"],
        results["minus"]["c1"]["2m"]["g1"]
    ), 2 * dg))
    m_c1_R21 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c1"]["1p"]["g2"],
        results["minus"]["c1"]["1m"]["g2"]
    ), 2 * dg))
    m_c1_R22 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c1"]["2p"]["g2"],
        results["minus"]["c1"]["2m"]["g2"]
    ), 2 * dg))

    return np.array([[p_c1_R11, p_c1_R12], [p_c1_R21, p_c1_R22]]), np.array([[m_c1_R11, m_c1_R12], [m_c1_R21, m_c1_R22]])


def compute_dRdc_direct(results, dg, dc, color):
    # c0
    p_c0_R11 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c0"]["1p"]["g1"],
        results["plus"]["c0"]["1m"]["g1"]
    ), 2 * dg))
    p_c0_R12 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c0"]["2p"]["g1"],
        results["plus"]["c0"]["2m"]["g1"]
    ), 2 * dg))
    p_c0_R21 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c0"]["1p"]["g2"],
        results["plus"]["c0"]["1m"]["g2"]
    ), 2 * dg))
    p_c0_R22 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c0"]["2p"]["g2"],
        results["plus"]["c0"]["2m"]["g2"]
    ), 2 * dg))

    p_c0_R = np.array([[p_c0_R11, p_c0_R12], [p_c0_R21, p_c0_R22]])

    m_c0_R11 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c0"]["1p"]["g1"],
        results["minus"]["c0"]["1m"]["g1"]
    ), 2 * dg))
    m_c0_R12 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c0"]["2p"]["g1"],
        results["minus"]["c0"]["2m"]["g1"]
    ), 2 * dg))
    m_c0_R21 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c0"]["1p"]["g2"],
        results["minus"]["c0"]["1m"]["g2"]
    ), 2 * dg))
    m_c0_R22 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c0"]["2p"]["g2"],
        results["minus"]["c0"]["2m"]["g2"]
    ), 2 * dg))

    m_c0_R = np.array([[m_c0_R11, m_c0_R12], [m_c0_R21, m_c0_R22]])

    # c2
    p_c2_R11 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c2"]["1p"]["g1"],
        results["plus"]["c2"]["1m"]["g1"]
    ), 2 * dg))
    p_c2_R12 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c2"]["2p"]["g1"],
        results["plus"]["c2"]["2m"]["g1"]
    ), 2 * dg))
    p_c2_R21 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c2"]["1p"]["g2"],
        results["plus"]["c2"]["1m"]["g2"]
    ), 2 * dg))
    p_c2_R22 = np.nanmean(pc.divide(pc.subtract(
        results["plus"]["c2"]["2p"]["g2"],
        results["plus"]["c2"]["2m"]["g2"]
    ), 2 * dg))

    p_c2_R = np.array([[p_c2_R11, p_c2_R12], [p_c2_R21, p_c2_R22]])

    m_c2_R11 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c2"]["1p"]["g1"],
        results["minus"]["c2"]["1m"]["g1"]
    ), 2 * dg))
    m_c2_R12 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c2"]["2p"]["g1"],
        results["minus"]["c2"]["2m"]["g1"]
    ), 2 * dg))
    m_c2_R21 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c2"]["1p"]["g2"],
        results["minus"]["c2"]["1m"]["g2"]
    ), 2 * dg))
    m_c2_R22 = np.nanmean(pc.divide(pc.subtract(
        results["minus"]["c2"]["2p"]["g2"],
        results["minus"]["c2"]["2m"]["g2"]
    ), 2 * dg))


    m_c2_R = np.array([[m_c2_R11, m_c2_R12], [m_c2_R21, m_c2_R22]])

    p_c1 = np.nanmean(results["plus"]["c1"]["noshear"]["color"])
    m_c1 = np.nanmean(results["minus"]["c1"]["noshear"]["color"])

    p_dRdc = (p_c2_R - p_c0_R) / (2 * dc)
    m_dRdc = (m_c2_R - m_c0_R) / (2 * dc)

    return p_dRdc * (p_c1 - color), m_dRdc * (m_c1 - color)


def compute_dRdc(results, dg, dc, alt=False):

    #---------------------------------------------------------------------------
    # plus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    p_c0_g1_ns = np.nanmean(results["plus"]["c0"]["noshear"]["g1"])
    p_c0_g1c_ns = np.nanmean(results["plus"]["c0"]["noshear"]["g1c"])
    p_c0_g2_ns = np.nanmean(results["plus"]["c0"]["noshear"]["g2"])
    p_c0_g2c_ns = np.nanmean(results["plus"]["c0"]["noshear"]["g2c"])

    # 1p
    p_c0_g1_1p = np.nanmean(results["plus"]["c0"]["1p"]["g1"])
    p_c0_g1c_1p = np.nanmean(results["plus"]["c0"]["1p"]["g1c"])
    p_c0_g2_1p = np.nanmean(results["plus"]["c0"]["1p"]["g2"])
    p_c0_g2c_1p = np.nanmean(results["plus"]["c0"]["1p"]["g2c"])

    # 1m
    p_c0_g1_1m = np.nanmean(results["plus"]["c0"]["1m"]["g1"])
    p_c0_g1c_1m = np.nanmean(results["plus"]["c0"]["1m"]["g1c"])
    p_c0_g2_1m = np.nanmean(results["plus"]["c0"]["1m"]["g2"])
    p_c0_g2c_1m = np.nanmean(results["plus"]["c0"]["1m"]["g2c"])

    # 2p
    p_c0_g1_2p = np.nanmean(results["plus"]["c0"]["2p"]["g1"])
    p_c0_g1c_2p = np.nanmean(results["plus"]["c0"]["2p"]["g1c"])
    p_c0_g2_2p = np.nanmean(results["plus"]["c0"]["2p"]["g2"])
    p_c0_g2c_2p = np.nanmean(results["plus"]["c0"]["2p"]["g2c"])

    # 2m
    p_c0_g1_2m = np.nanmean(results["plus"]["c0"]["2m"]["g1"])
    p_c0_g1c_2m = np.nanmean(results["plus"]["c0"]["2m"]["g1c"])
    p_c0_g2_2m = np.nanmean(results["plus"]["c0"]["2m"]["g2"])
    p_c0_g2c_2m = np.nanmean(results["plus"]["c0"]["2m"]["g2c"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    p_c1_g1_ns = np.nanmean(results["plus"]["c1"]["noshear"]["g1"])
    p_c1_g1c_ns = np.nanmean(results["plus"]["c1"]["noshear"]["g1c"])
    p_c1_g2_ns = np.nanmean(results["plus"]["c1"]["noshear"]["g2"])
    p_c1_g2c_ns = np.nanmean(results["plus"]["c1"]["noshear"]["g2c"])

    # 1p
    p_c1_g1_1p = np.nanmean(results["plus"]["c1"]["1p"]["g1"])
    p_c1_g1c_1p = np.nanmean(results["plus"]["c1"]["1p"]["g1c"])
    p_c1_g2_1p = np.nanmean(results["plus"]["c1"]["1p"]["g2"])
    p_c1_g2c_1p = np.nanmean(results["plus"]["c1"]["1p"]["g2c"])

    # 1m
    p_c1_g1_1m = np.nanmean(results["plus"]["c1"]["1m"]["g1"])
    p_c1_g1c_1m = np.nanmean(results["plus"]["c1"]["1m"]["g1c"])
    p_c1_g2_1m = np.nanmean(results["plus"]["c1"]["1m"]["g2"])
    p_c1_g2c_1m = np.nanmean(results["plus"]["c1"]["1m"]["g2c"])

    # 2p
    p_c1_g1_2p = np.nanmean(results["plus"]["c1"]["2p"]["g1"])
    p_c1_g1c_2p = np.nanmean(results["plus"]["c1"]["2p"]["g1c"])
    p_c1_g2_2p = np.nanmean(results["plus"]["c1"]["2p"]["g2"])
    p_c1_g2c_2p = np.nanmean(results["plus"]["c1"]["2p"]["g2c"])

    # 2m
    p_c1_g1_2m = np.nanmean(results["plus"]["c1"]["2m"]["g1"])
    p_c1_g1c_2m = np.nanmean(results["plus"]["c1"]["2m"]["g1c"])
    p_c1_g2_2m = np.nanmean(results["plus"]["c1"]["2m"]["g2"])
    p_c1_g2c_2m = np.nanmean(results["plus"]["c1"]["2m"]["g2c"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    p_c2_g1_ns = np.nanmean(results["plus"]["c2"]["noshear"]["g1"])
    p_c2_g2_ns = np.nanmean(results["plus"]["c2"]["noshear"]["g2"])
    p_c2_g1c_ns = np.nanmean(results["plus"]["c2"]["noshear"]["g1c"])
    p_c2_g2c_ns = np.nanmean(results["plus"]["c2"]["noshear"]["g2c"])

    # 1p
    p_c2_g1_1p = np.nanmean(results["plus"]["c2"]["1p"]["g1"])
    p_c2_g1c_1p = np.nanmean(results["plus"]["c2"]["1p"]["g1c"])
    p_c2_g2_1p = np.nanmean(results["plus"]["c2"]["1p"]["g2"])
    p_c2_g2c_1p = np.nanmean(results["plus"]["c2"]["1p"]["g2c"])

    # 1m
    p_c2_g1_1m = np.nanmean(results["plus"]["c2"]["1m"]["g1"])
    p_c2_g1c_1m = np.nanmean(results["plus"]["c2"]["1m"]["g1c"])
    p_c2_g2_1m = np.nanmean(results["plus"]["c2"]["1m"]["g2"])
    p_c2_g2c_1m = np.nanmean(results["plus"]["c2"]["1m"]["g2c"])

    # 2p
    p_c2_g1_2p = np.nanmean(results["plus"]["c2"]["2p"]["g1"])
    p_c2_g1c_2p = np.nanmean(results["plus"]["c2"]["2p"]["g1c"])
    p_c2_g2_2p = np.nanmean(results["plus"]["c2"]["2p"]["g2"])
    p_c2_g2c_2p = np.nanmean(results["plus"]["c2"]["2p"]["g2c"])

    # 2m
    p_c2_g1_2m = np.nanmean(results["plus"]["c2"]["2m"]["g1"])
    p_c2_g1c_2m = np.nanmean(results["plus"]["c2"]["2m"]["g1c"])
    p_c2_g2_2m = np.nanmean(results["plus"]["c2"]["2m"]["g2"])
    p_c2_g2c_2m = np.nanmean(results["plus"]["c2"]["2m"]["g2c"])

    #---------------------------------------------------------------------------
    # minus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    m_c0_g1_ns = np.nanmean(results["minus"]["c0"]["noshear"]["g1"])
    m_c0_g1c_ns = np.nanmean(results["minus"]["c0"]["noshear"]["g1c"])
    m_c0_g2_ns = np.nanmean(results["minus"]["c0"]["noshear"]["g2"])
    m_c0_g2c_ns = np.nanmean(results["minus"]["c0"]["noshear"]["g2c"])

    # 1p
    m_c0_g1_1p = np.nanmean(results["minus"]["c0"]["1p"]["g1"])
    m_c0_g1c_1p = np.nanmean(results["minus"]["c0"]["1p"]["g1c"])
    m_c0_g2_1p = np.nanmean(results["minus"]["c0"]["1p"]["g2"])
    m_c0_g2c_1p = np.nanmean(results["minus"]["c0"]["1p"]["g2c"])

    # 1m
    m_c0_g1_1m = np.nanmean(results["minus"]["c0"]["1m"]["g1"])
    m_c0_g1c_1m = np.nanmean(results["minus"]["c0"]["1m"]["g1c"])
    m_c0_g2_1m = np.nanmean(results["minus"]["c0"]["1m"]["g2"])
    m_c0_g2c_1m = np.nanmean(results["minus"]["c0"]["1m"]["g2c"])

    # 2p
    m_c0_g1_2p = np.nanmean(results["minus"]["c0"]["2p"]["g1"])
    m_c0_g1c_2p = np.nanmean(results["minus"]["c0"]["2p"]["g1c"])
    m_c0_g2_2p = np.nanmean(results["minus"]["c0"]["2p"]["g2"])
    m_c0_g2c_2p = np.nanmean(results["minus"]["c0"]["2p"]["g2c"])

    # 2m
    m_c0_g1_2m = np.nanmean(results["minus"]["c0"]["2m"]["g1"])
    m_c0_g1c_2m = np.nanmean(results["minus"]["c0"]["2m"]["g1c"])
    m_c0_g2_2m = np.nanmean(results["minus"]["c0"]["2m"]["g2"])
    m_c0_g2c_2m = np.nanmean(results["minus"]["c0"]["2m"]["g2c"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    m_c1_g1_ns = np.nanmean(results["minus"]["c1"]["noshear"]["g1"])
    m_c1_g1c_ns = np.nanmean(results["minus"]["c1"]["noshear"]["g1c"])
    m_c1_g2_ns = np.nanmean(results["minus"]["c1"]["noshear"]["g2"])
    m_c1_g2c_ns = np.nanmean(results["minus"]["c1"]["noshear"]["g2c"])

    # 1p
    m_c1_g1_1p = np.nanmean(results["minus"]["c1"]["1p"]["g1"])
    m_c1_g1c_1p = np.nanmean(results["minus"]["c1"]["1p"]["g1c"])
    m_c1_g2_1p = np.nanmean(results["minus"]["c1"]["1p"]["g2"])
    m_c1_g2c_1p = np.nanmean(results["minus"]["c1"]["1p"]["g2c"])

    # 1m
    m_c1_g1_1m = np.nanmean(results["minus"]["c1"]["1m"]["g1"])
    m_c1_g1c_1m = np.nanmean(results["minus"]["c1"]["1m"]["g1c"])
    m_c1_g2_1m = np.nanmean(results["minus"]["c1"]["1m"]["g2"])
    m_c1_g2c_1m = np.nanmean(results["minus"]["c1"]["1m"]["g2c"])

    # 2p
    m_c1_g1_2p = np.nanmean(results["minus"]["c1"]["2p"]["g1"])
    m_c1_g1c_2p = np.nanmean(results["minus"]["c1"]["2p"]["g1c"])
    m_c1_g2_2p = np.nanmean(results["minus"]["c1"]["2p"]["g2"])
    m_c1_g2c_2p = np.nanmean(results["minus"]["c1"]["2p"]["g2c"])

    # 2m
    m_c1_g1_2m = np.nanmean(results["minus"]["c1"]["2m"]["g1"])
    m_c1_g1c_2m = np.nanmean(results["minus"]["c1"]["2m"]["g1c"])
    m_c1_g2_2m = np.nanmean(results["minus"]["c1"]["2m"]["g2"])
    m_c1_g2c_2m = np.nanmean(results["minus"]["c1"]["2m"]["g2c"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    m_c2_g1_ns = np.nanmean(results["minus"]["c2"]["noshear"]["g1"])
    m_c2_g1c_ns = np.nanmean(results["minus"]["c2"]["noshear"]["g1c"])
    m_c2_g2_ns = np.nanmean(results["minus"]["c2"]["noshear"]["g2"])
    m_c2_g2c_ns = np.nanmean(results["minus"]["c2"]["noshear"]["g2c"])

    # 1p
    m_c2_g1_1p = np.nanmean(results["minus"]["c2"]["1p"]["g1"])
    m_c2_g1c_1p = np.nanmean(results["minus"]["c2"]["1p"]["g1c"])
    m_c2_g2_1p = np.nanmean(results["minus"]["c2"]["1p"]["g2"])
    m_c2_g2c_1p = np.nanmean(results["minus"]["c2"]["1p"]["g2c"])

    # 1m
    m_c2_g1_1m = np.nanmean(results["minus"]["c2"]["1m"]["g1"])
    m_c2_g1c_1m = np.nanmean(results["minus"]["c2"]["1m"]["g1c"])
    m_c2_g2_1m = np.nanmean(results["minus"]["c2"]["1m"]["g2"])
    m_c2_g2c_1m = np.nanmean(results["minus"]["c2"]["1m"]["g2c"])

    # 2p
    m_c2_g1_2p = np.nanmean(results["minus"]["c2"]["2p"]["g1"])
    m_c2_g1c_2p = np.nanmean(results["minus"]["c2"]["2p"]["g1c"])
    m_c2_g2_2p = np.nanmean(results["minus"]["c2"]["2p"]["g2"])
    m_c2_g2c_2p = np.nanmean(results["minus"]["c2"]["2p"]["g2c"])

    # 2m
    m_c2_g1_2m = np.nanmean(results["minus"]["c2"]["2m"]["g1"])
    m_c2_g1c_2m = np.nanmean(results["minus"]["c2"]["2m"]["g1c"])
    m_c2_g2_2m = np.nanmean(results["minus"]["c2"]["2m"]["g2"])
    m_c2_g2c_2m = np.nanmean(results["minus"]["c2"]["2m"]["g2c"])

    #---------------------------------------------------------------------------

    if not alt:
        return (
            np.array(
                [
                    [
                        (p_c2_g1c_1p - p_c2_g1c_1m - p_c0_g1c_1p + p_c0_g1c_1m) / (2 * dg * 2 * dc),
                        (p_c2_g1c_2p - p_c2_g1c_2m - p_c0_g1c_2p + p_c0_g1c_2m) / (2 * dg * 2 * dc),
                    ],
                    [
                        (p_c2_g2c_1p - p_c2_g2c_1m - p_c0_g2c_1p + p_c0_g2c_1m) / (2 * dg * 2 * dc),
                        (p_c2_g2c_2p - p_c2_g2c_2m - p_c0_g2c_2p + p_c0_g2c_2m) / (2 * dg * 2 * dc),
                    ],
                ]
            ),
            np.array(
                [
                    [
                        (m_c2_g1c_1p - m_c2_g1c_1m - m_c0_g1c_1p + m_c0_g1c_1m) / (2 * dg * 2 * dc),
                        (m_c2_g1c_2p - m_c2_g1c_2m - m_c0_g1c_2p + m_c0_g1c_2m) / (2 * dg * 2 * dc),
                    ],
                    [
                        (m_c2_g2c_1p - m_c2_g2c_1m - m_c0_g2c_1p + m_c0_g2c_1m) / (2 * dg * 2 * dc),
                        (m_c2_g2c_2p - m_c2_g2c_2m - m_c0_g2c_2p + m_c0_g2c_2m) / (2 * dg * 2 * dc),
                    ],
                ]
            )
        )
    else:
        # more efficient formula from https://en.wikipedia.org/wiki/Finite_difference
        return (
            np.array(
                [
                    [
                        (p_c2_g1c_1p - p_c1_g1c_1p - p_c2_g1c_ns + 2 * p_c1_g1c_ns - p_c1_g1c_1m - p_c0_g1c_ns + p_c0_g1c_1m) / (2 * dg * dc),
                        (p_c2_g1c_2p - p_c1_g1c_2p - p_c2_g1c_ns + 2 * p_c1_g1c_ns - p_c1_g1c_2m - p_c0_g1c_ns + p_c0_g1c_2m) / (2 * dg * dc),
                    ],
                    [
                        (p_c2_g2c_1p - p_c1_g2c_1p - p_c2_g2c_ns + 2 * p_c1_g2c_ns - p_c1_g2c_1m - p_c0_g2c_ns + p_c0_g2c_1m) / (2 * dg * dc),
                        (p_c2_g2c_2p - p_c1_g2c_2p - p_c2_g2c_ns + 2 * p_c1_g2c_ns - p_c1_g2c_2m - p_c0_g2c_ns + p_c0_g2c_2m / (2 * dg * dc)),
                    ],
                ],
            ),
            np.array(
                [
                    [
                        (m_c2_g1c_1p - m_c1_g1c_1p - m_c2_g1c_ns + 2 * m_c1_g1c_ns - m_c1_g1c_1m - m_c0_g1c_ns + m_c0_g1c_1m) / (2 * dg * dc),
                        (m_c2_g1c_2p - m_c1_g1c_2p - m_c2_g1c_ns + 2 * m_c1_g1c_ns - m_c1_g1c_2m - m_c0_g1c_ns + m_c0_g1c_2m) / (2 * dg * dc),
                    ],
                    [
                        (m_c2_g2c_1p - m_c1_g2c_1p - m_c2_g2c_ns + 2 * m_c1_g2c_ns - m_c1_g2c_1m - m_c0_g2c_ns + m_c0_g2c_1m) / (2 * dg * dc),
                        (m_c2_g2c_2p - m_c1_g2c_2p - m_c2_g2c_ns + 2 * m_c1_g2c_ns - m_c1_g2c_2m - m_c0_g2c_ns + m_c0_g2c_2m / (2 * dg * dc)),
                    ],
                ],
            )
        )


def compute_m(results, dg, dc):
    e_p, e_m = compute_e(results)

    R_p, R_m = compute_R(results, dg)

    # return (e_p[0] - e_m[0]) / (R_p[0, 0] + R_m[0, 0]) / 0.02 - 1
    return (
        np.linalg.inv(R_p) @ e_p / 2
        - np.linalg.inv(R_m) @ e_m / 2
    )[0] / 0.02 - 1


def compute_m_chromatic_direct(batch, dg, dc, color):
    e_p, e_m = compute_e(batch)

    R_p, R_m = compute_R(batch, dg)

    dRdc_p, dRdc_m = compute_dRdc_direct(batch, dg, dc, color)

    # return (e_p[0] - e_m[0]) / (R_p[0, 0] + dRdc_p[0, 0] + R_m[0, 0] + dRdc_m[0, 0]) / 0.02 - 1
    return (
        np.linalg.inv(R_p + dRdc_p) @ e_p / 2
        - np.linalg.inv(R_m + dRdc_m) @ e_m / 2
    )[0] / 0.02 - 1


def compute_m_chromatic(batch, dg, dc, alt=False):
    e_p, e_m = compute_e(batch)

    R_p, R_m = compute_R(batch, dg)

    dRdc_p, dRdc_m = compute_dRdc(batch, dg, dc, alt=alt)

    # return (e_p[0] - e_m[0]) / (R_p[0, 0] + dRdc_p[0, 0] + R_m[0, 0] + dRdc_m[0, 0]) / 0.02 - 1
    # return (e_p[0] / (R_p[0, 0] + dRdc_p[0, 0]) / 2 -  e_m[0] / (R_m[0, 0] + dRdc_m[0, 0]) / 2)  / 0.02 - 1
    return (
        np.linalg.inv(R_p + dRdc_p) @ e_p / 2
        - np.linalg.inv(R_m + dRdc_m) @ e_m / 2
    )[0] / 0.02 - 1


def pre_aggregate(dataset, predicate, color):
    """
    Aggregate measurements at the image level to accelerate bootstrapping
    """
    import galsim
    zp_0 = galsim.Bandpass("LSST_g.dat", wave_type="nm").withZeropoint("AB").zeropoint
    zp_2 = galsim.Bandpass("LSST_i.dat", wave_type="nm").withZeropoint("AB").zeropoint

    scan_node = acero.Declaration(
        "scan",
        acero.ScanNodeOptions(
            dataset,
            # columns=projection,
            filter=predicate,
        ),
    )
    filter_node = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(
            predicate,
        ),
    )
    pre_project_node = acero.Declaration(
        "project",
        acero.ProjectNodeOptions(
            [
                pc.field("seed"),
                pc.field("shear"),
                pc.field("color_step"),
                pc.field("mdet_step"),
                pc.list_element(pc.field("pgauss_g"), 0),
                pc.list_element(pc.field("pgauss_g"), 1),
                pc.add(
                    pc.multiply(
                        pc.scalar(-2.5),
                        pc.log10(
                            pc.divide(
                                pc.list_element(pc.field("pgauss_band_flux"), 0),
                                pc.list_element(pc.field("pgauss_band_flux"), 2)
                            )
                        )
                    ),
                    pc.subtract(pc.scalar(zp_0), pc.scalar(zp_2))
                ),
                pc.multiply(
                    pc.list_element(pc.field("pgauss_g"), 0),
                    pc.subtract(
                        pc.add(
                            pc.multiply(
                                pc.scalar(-2.5),
                                pc.log10(
                                    pc.divide(
                                        pc.list_element(pc.field("pgauss_band_flux"), 0),
                                        pc.list_element(pc.field("pgauss_band_flux"), 2)
                                    )
                                )
                            ),
                            pc.subtract(pc.scalar(zp_0), pc.scalar(zp_2))
                        ),
                        pc.scalar(color),
                    ),
                ),
                pc.multiply(
                    pc.list_element(pc.field("pgauss_g"), 1),
                    pc.subtract(
                        pc.add(
                            pc.multiply(
                                pc.scalar(-2.5),
                                pc.log10(
                                    pc.divide(
                                        pc.list_element(pc.field("pgauss_band_flux"), 0),
                                        pc.list_element(pc.field("pgauss_band_flux"), 2)
                                    )
                                )
                            ),
                            pc.subtract(pc.scalar(zp_0), pc.scalar(zp_2))
                        ),
                        pc.scalar(color),
                    ),
                ),
            ],
            names=[
                "seed",
                "shear",
                "color_step",
                "mdet_step",
                "g1",
                "g2",
                "color",
                "g1c",
                "g2c",
            ],
        )
    )
    pre_aggregate_node = acero.Declaration(
        "aggregate",
        acero.AggregateNodeOptions(
            [
                ("seed", "hash_count", None, "count"),
                ("g1", "hash_mean", None, "mean_g1"),
                ("g2", "hash_mean", None, "mean_g2"),
                ("g1c", "hash_mean", None, "mean_g1c"),
                ("g2c", "hash_mean", None, "mean_g2c"),
                ("color", "hash_mean", None, "mean_color"),
            ],
            keys=["seed", "shear", "color_step", "mdet_step"],
        )
    )
    # FIXME is there a cleaner way to address null colors?
    post_filter_node = acero.Declaration(
        "filter",
        acero.FilterNodeOptions(
            pc.is_finite(pc.field("mean_color")),
        ),
    )
    seq = [
        scan_node,
        filter_node,
        pre_project_node,
        pre_aggregate_node,
        post_filter_node,
    ]
    plan = acero.Declaration.from_sequence(seq)
    print(plan)
    res = plan.to_table(use_threads=True)

    return res


# def pivot_aggregates(res):
#     """
#     This more or less implements a pivot wider operation via successive
#     application of filters
#     """
#     seeds = np.sort(np.unique(res["seed"]))
# 
#     table_source_node = acero.Declaration(
#         "table_source",
#         acero.TableSourceNodeOptions(
#             res
#         ),
#     )
# 
#     results = []
#     for seed in tqdm.tqdm(seeds, ncols=80):
#         aggregates = {}
#         for shear_step in ["plus", "minus"]:
#             aggregates[shear_step] = {}
#             for color_step in ["c0", "c1", "c2"]:
#                 aggregates[shear_step][color_step] = {}
#                 for mdet_step in ["noshear", "1p", "1m", "2p", "2m"]:
#                     post_filter_node = acero.Declaration(
#                         "filter",
#                         acero.FilterNodeOptions(
#                             (pc.field("seed") == seed) \
#                             & (pc.field("shear") == shear_step) \
#                             & (pc.field("color_step") == color_step) \
#                             & (pc.field("mdet_step") == mdet_step)
#                         ),
#                     )
#                     post_aggregate_node = acero.Declaration(
#                         "aggregate",
#                         acero.AggregateNodeOptions(
#                             [
#                                 ("count", "sum", None, "count"),
#                                 ("mean_g1", "mean", None, "g1"),
#                                 ("mean_g2", "mean", None, "g2"),
#                                 ("mean_g1c", "mean", None, "g1c"),
#                                 ("mean_g2c", "mean", None, "g2c"),
#                                 ("mean_color", "mean", None, "color"),
#                             ],
#                         ),
#                     )
#                     seq = [
#                         table_source_node,
#                         post_filter_node,
#                         post_aggregate_node,
#                     ]
#                     plan = acero.Declaration.from_sequence(seq)
#                     pivot = plan.to_table(use_threads=True)
#                     aggregates[shear_step][color_step][mdet_step] = pivot
# 
#         results.append(aggregates)
# 
#     return results

def pivot_aggregates(res, seeds):
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
                post_filter_node = acero.Declaration(
                    "filter",
                    acero.FilterNodeOptions(
                        (pc.is_in(pc.field("seed"), seeds)) \
                        & (pc.field("shear") == shear_step) \
                        & (pc.field("color_step") == color_step) \
                        & (pc.field("mdet_step") == mdet_step)
                    ),
                )
                post_aggregate_node = acero.Declaration(
                    "aggregate",
                    acero.AggregateNodeOptions(
                        [
                            ("count", "sum", None, "count"),
                            ("mean_g1", "mean", None, "g1"),
                            ("mean_g2", "mean", None, "g2"),
                            ("mean_g1c", "mean", None, "g1c"),
                            ("mean_g2c", "mean", None, "g2c"),
                            ("mean_color", "mean", None, "color"),
                        ],
                    ),
                )
                seq = [
                    table_source_node,
                    post_filter_node,
                    post_aggregate_node,
                ]
                plan = acero.Declaration.from_sequence(seq)
                pivot = plan.to_table(use_threads=True)
                aggregates[shear_step][color_step][mdet_step] = pivot

    return aggregates


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="configuration file [yaml]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=None,
        help="RNG seed [int]",
    )
    parser.add_argument(
        "--s2n-cut", type=int, default=10,
        help="Signal/noise cut [int; 10]",
    )
    parser.add_argument(
        "--ormask-cut", type=int, default=None,
        help="Cut to make on ormask. 0 indicates make a cut, 1 indicates no cut.",
    )
    parser.add_argument(
        "--mfrac-cut", type=float, default=None,
        help="Cut to make on mfrac. Given in percentages and comma separated. Cut keeps all objects less than the given value.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--n_resample",
        type=int,
        required=False,
        default=1000,
        help="Number of resample iterations"
    )
    # parser.add_argument(
    #     "--n_jobs",
    #     type=int,
    #     required=False,
    #     default=1,
    #     help="Number of jobs to run [int; 1]",
    # )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    config = args.config
    seed = args.seed
    # n_jobs = args.n_jobs

    pa.set_cpu_count(32)
    pa.set_io_thread_count(32)

    pipeline = Pipeline(config)
    print("pipeline:", pipeline.name)
    print("seed:", seed)
    pipeline.load()

    rng = np.random.default_rng(seed)

    measure_config = pipeline.config.get("measure")
    measure_type = measure_config.get("type")
    print("measure:", measure_type)

    if measure_type in CHROMATIC_MEASURES:
        colors_type = measure_config.get("colors")
        match colors_type:
            case "quantiles":
                chroma_colors = pipeline.galaxies.aggregate.get("quantiles")
            case "uniform":
                colors_min = pipeline.galaxies.aggregate.get("min_color")[0]
                colors_max = pipeline.galaxies.aggregate.get("max_color")[0]
                # TODO add config for number of colors here...
                chroma_colors = np.linspace(colors_min, colors_max, 3)
            case "centered":
                median = pipeline.galaxies.aggregate.get("median_color")[0]
                chroma_colors = [median - 0.1, median, median + 0.1]
            case _:
                raise ValueError(f"Colors type {colors_type} not valid!")

    dg = ngmix.metacal.DEFAULT_STEP
    dc = (chroma_colors[2] - chroma_colors[0]) / 2.
    color = chroma_colors[1]

    dataset = ds.dataset(args.output, format="arrow")

    predicate = \
        (pc.field("pgauss_flags") == 0) \
        & (pc.field("pgauss_s2n") > args.s2n_cut) \
        & (pc.field("pgauss_T_ratio") > 0.5)

    aggregates = pre_aggregate(dataset, predicate, color)
    # results = pivot_aggregates(aggregates)

    seeds = np.sort(np.unique(aggregates["seed"]))
    all_seeds = pa.array(seeds)

    results = pivot_aggregates(aggregates, all_seeds)
    m_mean = compute_m(results, dg, dc)
    # m_mean_chroma = compute_m_chromatic(results, dg, dc)
    m_mean_chroma = compute_m_chromatic_direct(results, dg, dc, color)

    m_bootstrap = []
    m_bootstrap_chroma = []
    for i in tqdm.trange(args.n_resample, ncols=80):
        _seeds = pa.array(rng.choice(seeds, size=len(seeds), replace=True))
        _results = pivot_aggregates(aggregates, _seeds)

        _m_bootstrap = compute_m(_results, dg, dc)
        # _m_bootstrap_chroma = compute_m_chromatic(_results, dg, dc)
        _m_bootstrap_chroma = compute_m_chromatic_direct(_results, dg, dc, color)

        m_bootstrap.append(_m_bootstrap)
        m_bootstrap_chroma.append(_m_bootstrap_chroma)

    m_bootstrap = np.array(m_bootstrap)
    m_bootstrap_chroma = np.array(m_bootstrap_chroma)


    # jobs = [
    #     joblib.delayed(compute_m)(res, dg, dc)
    #     for res in results
    # ]

    # with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
    #     m_chunks = parallel(jobs)

    # m_bootstrap = []
    # for i in tqdm.trange(args.n_resample, ncols=80):
    #     m_resample = rng.choice(m_chunks, size=len(m_chunks), replace=True)
    #     m_bootstrap.append(np.nanmean(m_resample))
    # m_bootstrap = np.array(m_bootstrap)

    # m_mean = np.nanmean(m_bootstrap)
    m_error = np.nanstd(m_bootstrap)

    print("mdet: m = %0.3e +/- %0.3e [3-sigma]" % (m_mean, m_error * 3))

    # batches = dataset.to_batches(filter=predicate)
    # jobs = [
    #     joblib.delayed(compute_m_chromatic)(res, dg, dc, alt=False)
    #     # joblib.delayed(compute_m_chromatic_direct)(res, dg, dc, color)
    #     for res in results
    # ]

    # with joblib.Parallel(n_jobs=n_jobs, verbose=10) as parallel:
    #     m_chunks_chroma = parallel(jobs)

    # m_bootstrap_chroma = []
    # for i in tqdm.trange(args.n_resample, ncols=80):
    #     m_resample_chroma = rng.choice(m_chunks_chroma, size=len(m_chunks_chroma), replace=True)
    #     m_bootstrap_chroma.append(np.nanmean(m_resample_chroma))
    # m_bootstrap_chroma = np.array(m_bootstrap_chroma)

    # m_mean_chroma = np.nanmean(m_bootstrap_chroma)
    m_error_chroma = np.nanstd(m_bootstrap_chroma)

    print("drdc: m = %0.3e +/- %0.3e [3-sigma]" % (m_mean_chroma, m_error_chroma * 3))

    m_req = 2e-3
    plt.axvspan(-m_req, m_req, fc="k", alpha=0.1)
    plt.axvline(4e-4, c="k", alpha=0.1, ls="--")
    plt.hist(m_bootstrap, histtype="step", label="R", ec="k")
    plt.hist(m_bootstrap_chroma, histtype="step", label="R & dR/dc", ec="b")
    plt.axvline(m_mean, c="k")
    plt.axvline(m_mean_chroma, c="b")
    plt.legend()
    plt.xlabel("$m$")
    plt.show()
