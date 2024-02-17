import argparse
import logging
import os

import ngmix
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as ft
import pyarrow.parquet as pq
from pyarrow import acero
import tqdm

from chromatic_shear_bias.pipeline.pipeline import Pipeline
from chromatic_shear_bias.pipeline import logging_config


logger = logging.getLogger(__name__)

CHROMATIC_MEASURES = {
    "chromatic_metadetect",
    "drdc",
}


def compute_e(results):
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


def compute_R_corr(results, shear_step, i, j, dg, dc, color):
    # return np.average(
    #     pc.add(
    #         pc.divide(
    #             pc.subtract(
    #                 results[shear_step]["c1"][f"{j}p"][f"e{i}"],
    #                 results[shear_step]["c1"][f"{j}m"][f"e{i}"]
    #             ),
    #             2 * dg
    #         ),
    #         pc.subtract(
    #             pc.divide(
    #                 pc.subtract(
    #                     pc.subtract(
    #                         results[shear_step]["c2"][f"{j}p"]["e1c"],
    #                         results[shear_step]["c2"][f"{j}m"]["e1c"]
    #                     ),
    #                     pc.subtract(
    #                         results[shear_step]["c0"][f"{j}p"]["e1c"],
    #                         results[shear_step]["c0"][f"{j}m"]["e1c"]
    #                     ),
    #                 ),
    #                 2 * dg * 2 * dc
    #             ),
    #             pc.multiply(
    #                 color,
    #                 pc.divide(
    #                     pc.subtract(
    #                         pc.subtract(
    #                             results[shear_step]["c2"][f"{j}p"][f"e{i}"],
    #                             results[shear_step]["c2"][f"{j}m"][f"e{i}"]
    #                         ),
    #                         pc.subtract(
    #                             results[shear_step]["c0"][f"{j}p"][f"e{i}"],
    #                             results[shear_step]["c0"][f"{j}m"][f"e{i}"]
    #                         ),
    #                     ),
    #                     2 * dg * 2 * dc
    #                 ),
    #             ),
    #         ),
    #     )
    # )
    return np.average(
        pc.add(
            pc.divide(
                pc.subtract(
                    results[shear_step]["c1"][f"{j}p"][f"e{i}"],
                    results[shear_step]["c1"][f"{j}m"][f"e{i}"]
                ),
                2 * dg
            ),
            pc.multiply(
                pc.divide(
                    pc.subtract(
                        pc.subtract(
                            results[shear_step]["c2"][f"{j}p"]["e1"],
                            results[shear_step]["c2"][f"{j}m"]["e1"]
                        ),
                        pc.subtract(
                            results[shear_step]["c0"][f"{j}p"]["e1"],
                            results[shear_step]["c0"][f"{j}m"]["e1"]
                        ),
                    ),
                    2 * dg * 2 * dc
                ),
                pc.subtract(
                    results[shear_step]["c1"]["noshear"]["color"],
                    color,
                ),
            ),
        )
    )




def compute_dedc(results, dg, dc):
    # c0
    p_c0_e1dc_ns = np.average(results["plus"]["c0"]["noshear"]["e1dc"])
    p_c0_e2dc_ns = np.average(results["plus"]["c0"]["noshear"]["e2dc"])

    m_c0_e1dc_ns = np.average(results["minus"]["c0"]["noshear"]["e1dc"])
    m_c0_e2dc_ns = np.average(results["minus"]["c0"]["noshear"]["e2dc"])

    # c2
    p_c2_e1dc_ns = np.average(results["plus"]["c2"]["noshear"]["e1dc"])
    p_c2_e2dc_ns = np.average(results["plus"]["c2"]["noshear"]["e2dc"])

    m_c2_e1dc_ns = np.average(results["minus"]["c2"]["noshear"]["e1dc"])
    m_c2_e2dc_ns = np.average(results["minus"]["c2"]["noshear"]["e2dc"])

    p_dedc_1 = (p_c2_e1dc_ns - p_c0_e1dc_ns) / (2 * dc)
    p_dedc_2 = (p_c2_e2dc_ns - p_c0_e2dc_ns) / (2 * dc)

    m_dedc_1 = (m_c2_e1dc_ns - m_c0_e1dc_ns) / (2 * dc)
    m_dedc_2 = (m_c2_e2dc_ns - m_c0_e2dc_ns) / (2 * dc)

    return np.array([p_dedc_1, p_dedc_2]), np.array([m_dedc_1, m_dedc_2])


def compute_dedc_factored(results, dg, dc, color):
    # c0
    p_c0_e1_ns = np.average(results["plus"]["c0"]["noshear"]["e1"])
    p_c0_e2_ns = np.average(results["plus"]["c0"]["noshear"]["e2"])

    m_c0_e1_ns = np.average(results["minus"]["c0"]["noshear"]["e1"])
    m_c0_e2_ns = np.average(results["minus"]["c0"]["noshear"]["e2"])

    # c2
    p_c2_e1_ns = np.average(results["plus"]["c2"]["noshear"]["e1"])
    p_c2_e2_ns = np.average(results["plus"]["c2"]["noshear"]["e2"])

    m_c2_e1_ns = np.average(results["minus"]["c2"]["noshear"]["e1"])
    m_c2_e2_ns = np.average(results["minus"]["c2"]["noshear"]["e2"])

    p_dedc_1 = (p_c2_e1_ns - p_c0_e1_ns) / (2 * dc)
    p_dedc_2 = (p_c2_e2_ns - p_c0_e2_ns) / (2 * dc)

    m_dedc_1 = (m_c2_e1_ns - m_c0_e1_ns) / (2 * dc)
    m_dedc_2 = (m_c2_e2_ns - m_c0_e2_ns) / (2 * dc)

    p_c1 = np.average(results["plus"]["c1"]["noshear"]["c"])
    m_c1 = np.average(results["minus"]["c1"]["noshear"]["c"])

    return np.array([p_dedc_1 * (p_c1 - color), p_dedc_2 * (p_c1 - color)]), np.array([m_dedc_1 * (m_c1 - color), m_dedc_2 * (m_c1 - color)])


def compute_dRdc_factored(results, dg, dc, color):
    # c0
    p_c0_R11 = np.average(pc.divide(pc.subtract(
        results["plus"]["c0"]["1p"]["e1"],
        results["plus"]["c0"]["1m"]["e1"]
    ), 2 * dg))
    p_c0_R12 = np.average(pc.divide(pc.subtract(
        results["plus"]["c0"]["2p"]["e1"],
        results["plus"]["c0"]["2m"]["e1"]
    ), 2 * dg))
    p_c0_R21 = np.average(pc.divide(pc.subtract(
        results["plus"]["c0"]["1p"]["e2"],
        results["plus"]["c0"]["1m"]["e2"]
    ), 2 * dg))
    p_c0_R22 = np.average(pc.divide(pc.subtract(
        results["plus"]["c0"]["2p"]["e2"],
        results["plus"]["c0"]["2m"]["e2"]
    ), 2 * dg))

    p_c0_R = np.array([[p_c0_R11, p_c0_R12], [p_c0_R21, p_c0_R22]])

    m_c0_R11 = np.average(pc.divide(pc.subtract(
        results["minus"]["c0"]["1p"]["e1"],
        results["minus"]["c0"]["1m"]["e1"]
    ), 2 * dg))
    m_c0_R12 = np.average(pc.divide(pc.subtract(
        results["minus"]["c0"]["2p"]["e1"],
        results["minus"]["c0"]["2m"]["e1"]
    ), 2 * dg))
    m_c0_R21 = np.average(pc.divide(pc.subtract(
        results["minus"]["c0"]["1p"]["e2"],
        results["minus"]["c0"]["1m"]["e2"]
    ), 2 * dg))
    m_c0_R22 = np.average(pc.divide(pc.subtract(
        results["minus"]["c0"]["2p"]["e2"],
        results["minus"]["c0"]["2m"]["e2"]
    ), 2 * dg))

    m_c0_R = np.array([[m_c0_R11, m_c0_R12], [m_c0_R21, m_c0_R22]])

    # c2
    p_c2_R11 = np.average(pc.divide(pc.subtract(
        results["plus"]["c2"]["1p"]["e1"],
        results["plus"]["c2"]["1m"]["e1"]
    ), 2 * dg))
    p_c2_R12 = np.average(pc.divide(pc.subtract(
        results["plus"]["c2"]["2p"]["e1"],
        results["plus"]["c2"]["2m"]["e1"]
    ), 2 * dg))
    p_c2_R21 = np.average(pc.divide(pc.subtract(
        results["plus"]["c2"]["1p"]["e2"],
        results["plus"]["c2"]["1m"]["e2"]
    ), 2 * dg))
    p_c2_R22 = np.average(pc.divide(pc.subtract(
        results["plus"]["c2"]["2p"]["e2"],
        results["plus"]["c2"]["2m"]["e2"]
    ), 2 * dg))

    p_c2_R = np.array([[p_c2_R11, p_c2_R12], [p_c2_R21, p_c2_R22]])

    m_c2_R11 = np.average(pc.divide(pc.subtract(
        results["minus"]["c2"]["1p"]["e1"],
        results["minus"]["c2"]["1m"]["e1"]
    ), 2 * dg))
    m_c2_R12 = np.average(pc.divide(pc.subtract(
        results["minus"]["c2"]["2p"]["e1"],
        results["minus"]["c2"]["2m"]["e1"]
    ), 2 * dg))
    m_c2_R21 = np.average(pc.divide(pc.subtract(
        results["minus"]["c2"]["1p"]["e2"],
        results["minus"]["c2"]["1m"]["e2"]
    ), 2 * dg))
    m_c2_R22 = np.average(pc.divide(pc.subtract(
        results["minus"]["c2"]["2p"]["e2"],
        results["minus"]["c2"]["2m"]["e2"]
    ), 2 * dg))


    m_c2_R = np.array([[m_c2_R11, m_c2_R12], [m_c2_R21, m_c2_R22]])

    p_c1 = np.average(results["plus"]["c1"]["noshear"]["c"])
    m_c1 = np.average(results["minus"]["c1"]["noshear"]["c"])

    p_dRdc = (p_c2_R - p_c0_R) / (2 * dc)
    m_dRdc = (m_c2_R - m_c0_R) / (2 * dc)

    return p_dRdc * (p_c1 - color), m_dRdc * (m_c1 - color)


def compute_dRdc(results, dg, dc, alt=False):

    #---------------------------------------------------------------------------
    # plus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    p_c0_e1_ns = np.average(results["plus"]["c0"]["noshear"]["e1"])
    p_c0_e1dc_ns = np.average(results["plus"]["c0"]["noshear"]["e1dc"])
    p_c0_e2_ns = np.average(results["plus"]["c0"]["noshear"]["e2"])
    p_c0_e2dc_ns = np.average(results["plus"]["c0"]["noshear"]["e2dc"])

    # 1p
    p_c0_e1_1p = np.average(results["plus"]["c0"]["1p"]["e1"])
    p_c0_e1dc_1p = np.average(results["plus"]["c0"]["1p"]["e1dc"])
    p_c0_e2_1p = np.average(results["plus"]["c0"]["1p"]["e2"])
    p_c0_e2dc_1p = np.average(results["plus"]["c0"]["1p"]["e2dc"])

    # 1m
    p_c0_e1_1m = np.average(results["plus"]["c0"]["1m"]["e1"])
    p_c0_e1dc_1m = np.average(results["plus"]["c0"]["1m"]["e1dc"])
    p_c0_e2_1m = np.average(results["plus"]["c0"]["1m"]["e2"])
    p_c0_e2dc_1m = np.average(results["plus"]["c0"]["1m"]["e2dc"])

    # 2p
    p_c0_e1_2p = np.average(results["plus"]["c0"]["2p"]["e1"])
    p_c0_e1dc_2p = np.average(results["plus"]["c0"]["2p"]["e1dc"])
    p_c0_e2_2p = np.average(results["plus"]["c0"]["2p"]["e2"])
    p_c0_e2dc_2p = np.average(results["plus"]["c0"]["2p"]["e2dc"])

    # 2m
    p_c0_e1_2m = np.average(results["plus"]["c0"]["2m"]["e1"])
    p_c0_e1dc_2m = np.average(results["plus"]["c0"]["2m"]["e1dc"])
    p_c0_e2_2m = np.average(results["plus"]["c0"]["2m"]["e2"])
    p_c0_e2dc_2m = np.average(results["plus"]["c0"]["2m"]["e2dc"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    p_c1_e1_ns = np.average(results["plus"]["c1"]["noshear"]["e1"])
    p_c1_e1dc_ns = np.average(results["plus"]["c1"]["noshear"]["e1dc"])
    p_c1_e2_ns = np.average(results["plus"]["c1"]["noshear"]["e2"])
    p_c1_e2dc_ns = np.average(results["plus"]["c1"]["noshear"]["e2dc"])

    # 1p
    p_c1_e1_1p = np.average(results["plus"]["c1"]["1p"]["e1"])
    p_c1_e1dc_1p = np.average(results["plus"]["c1"]["1p"]["e1dc"])
    p_c1_e2_1p = np.average(results["plus"]["c1"]["1p"]["e2"])
    p_c1_e2dc_1p = np.average(results["plus"]["c1"]["1p"]["e2dc"])

    # 1m
    p_c1_e1_1m = np.average(results["plus"]["c1"]["1m"]["e1"])
    p_c1_e1dc_1m = np.average(results["plus"]["c1"]["1m"]["e1dc"])
    p_c1_e2_1m = np.average(results["plus"]["c1"]["1m"]["e2"])
    p_c1_e2dc_1m = np.average(results["plus"]["c1"]["1m"]["e2dc"])

    # 2p
    p_c1_e1_2p = np.average(results["plus"]["c1"]["2p"]["e1"])
    p_c1_e1dc_2p = np.average(results["plus"]["c1"]["2p"]["e1dc"])
    p_c1_e2_2p = np.average(results["plus"]["c1"]["2p"]["e2"])
    p_c1_e2dc_2p = np.average(results["plus"]["c1"]["2p"]["e2dc"])

    # 2m
    p_c1_e1_2m = np.average(results["plus"]["c1"]["2m"]["e1"])
    p_c1_e1dc_2m = np.average(results["plus"]["c1"]["2m"]["e1dc"])
    p_c1_e2_2m = np.average(results["plus"]["c1"]["2m"]["e2"])
    p_c1_e2dc_2m = np.average(results["plus"]["c1"]["2m"]["e2dc"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    p_c2_e1_ns = np.average(results["plus"]["c2"]["noshear"]["e1"])
    p_c2_e2_ns = np.average(results["plus"]["c2"]["noshear"]["e2"])
    p_c2_e1dc_ns = np.average(results["plus"]["c2"]["noshear"]["e1dc"])
    p_c2_e2dc_ns = np.average(results["plus"]["c2"]["noshear"]["e2dc"])

    # 1p
    p_c2_e1_1p = np.average(results["plus"]["c2"]["1p"]["e1"])
    p_c2_e1dc_1p = np.average(results["plus"]["c2"]["1p"]["e1dc"])
    p_c2_e2_1p = np.average(results["plus"]["c2"]["1p"]["e2"])
    p_c2_e2dc_1p = np.average(results["plus"]["c2"]["1p"]["e2dc"])

    # 1m
    p_c2_e1_1m = np.average(results["plus"]["c2"]["1m"]["e1"])
    p_c2_e1dc_1m = np.average(results["plus"]["c2"]["1m"]["e1dc"])
    p_c2_e2_1m = np.average(results["plus"]["c2"]["1m"]["e2"])
    p_c2_e2dc_1m = np.average(results["plus"]["c2"]["1m"]["e2dc"])

    # 2p
    p_c2_e1_2p = np.average(results["plus"]["c2"]["2p"]["e1"])
    p_c2_e1dc_2p = np.average(results["plus"]["c2"]["2p"]["e1dc"])
    p_c2_e2_2p = np.average(results["plus"]["c2"]["2p"]["e2"])
    p_c2_e2dc_2p = np.average(results["plus"]["c2"]["2p"]["e2dc"])

    # 2m
    p_c2_e1_2m = np.average(results["plus"]["c2"]["2m"]["e1"])
    p_c2_e1dc_2m = np.average(results["plus"]["c2"]["2m"]["e1dc"])
    p_c2_e2_2m = np.average(results["plus"]["c2"]["2m"]["e2"])
    p_c2_e2dc_2m = np.average(results["plus"]["c2"]["2m"]["e2dc"])

    #---------------------------------------------------------------------------
    # minus
    #---------------------------------------------------------------------------

    # c0 -----------------------------------------------------------------------

    # NOSHEAR
    m_c0_e1_ns = np.average(results["minus"]["c0"]["noshear"]["e1"])
    m_c0_e1dc_ns = np.average(results["minus"]["c0"]["noshear"]["e1dc"])
    m_c0_e2_ns = np.average(results["minus"]["c0"]["noshear"]["e2"])
    m_c0_e2dc_ns = np.average(results["minus"]["c0"]["noshear"]["e2dc"])

    # 1p
    m_c0_e1_1p = np.average(results["minus"]["c0"]["1p"]["e1"])
    m_c0_e1dc_1p = np.average(results["minus"]["c0"]["1p"]["e1dc"])
    m_c0_e2_1p = np.average(results["minus"]["c0"]["1p"]["e2"])
    m_c0_e2dc_1p = np.average(results["minus"]["c0"]["1p"]["e2dc"])

    # 1m
    m_c0_e1_1m = np.average(results["minus"]["c0"]["1m"]["e1"])
    m_c0_e1dc_1m = np.average(results["minus"]["c0"]["1m"]["e1dc"])
    m_c0_e2_1m = np.average(results["minus"]["c0"]["1m"]["e2"])
    m_c0_e2dc_1m = np.average(results["minus"]["c0"]["1m"]["e2dc"])

    # 2p
    m_c0_e1_2p = np.average(results["minus"]["c0"]["2p"]["e1"])
    m_c0_e1dc_2p = np.average(results["minus"]["c0"]["2p"]["e1dc"])
    m_c0_e2_2p = np.average(results["minus"]["c0"]["2p"]["e2"])
    m_c0_e2dc_2p = np.average(results["minus"]["c0"]["2p"]["e2dc"])

    # 2m
    m_c0_e1_2m = np.average(results["minus"]["c0"]["2m"]["e1"])
    m_c0_e1dc_2m = np.average(results["minus"]["c0"]["2m"]["e1dc"])
    m_c0_e2_2m = np.average(results["minus"]["c0"]["2m"]["e2"])
    m_c0_e2dc_2m = np.average(results["minus"]["c0"]["2m"]["e2dc"])

    # c1 -----------------------------------------------------------------------

    # NOSHEAR
    m_c1_e1_ns = np.average(results["minus"]["c1"]["noshear"]["e1"])
    m_c1_e1dc_ns = np.average(results["minus"]["c1"]["noshear"]["e1dc"])
    m_c1_e2_ns = np.average(results["minus"]["c1"]["noshear"]["e2"])
    m_c1_e2dc_ns = np.average(results["minus"]["c1"]["noshear"]["e2dc"])

    # 1p
    m_c1_e1_1p = np.average(results["minus"]["c1"]["1p"]["e1"])
    m_c1_e1dc_1p = np.average(results["minus"]["c1"]["1p"]["e1dc"])
    m_c1_e2_1p = np.average(results["minus"]["c1"]["1p"]["e2"])
    m_c1_e2dc_1p = np.average(results["minus"]["c1"]["1p"]["e2dc"])

    # 1m
    m_c1_e1_1m = np.average(results["minus"]["c1"]["1m"]["e1"])
    m_c1_e1dc_1m = np.average(results["minus"]["c1"]["1m"]["e1dc"])
    m_c1_e2_1m = np.average(results["minus"]["c1"]["1m"]["e2"])
    m_c1_e2dc_1m = np.average(results["minus"]["c1"]["1m"]["e2dc"])

    # 2p
    m_c1_e1_2p = np.average(results["minus"]["c1"]["2p"]["e1"])
    m_c1_e1dc_2p = np.average(results["minus"]["c1"]["2p"]["e1dc"])
    m_c1_e2_2p = np.average(results["minus"]["c1"]["2p"]["e2"])
    m_c1_e2dc_2p = np.average(results["minus"]["c1"]["2p"]["e2dc"])

    # 2m
    m_c1_e1_2m = np.average(results["minus"]["c1"]["2m"]["e1"])
    m_c1_e1dc_2m = np.average(results["minus"]["c1"]["2m"]["e1dc"])
    m_c1_e2_2m = np.average(results["minus"]["c1"]["2m"]["e2"])
    m_c1_e2dc_2m = np.average(results["minus"]["c1"]["2m"]["e2dc"])

    # c2 -----------------------------------------------------------------------

    # NOSHEAR
    m_c2_e1_ns = np.average(results["minus"]["c2"]["noshear"]["e1"])
    m_c2_e1dc_ns = np.average(results["minus"]["c2"]["noshear"]["e1dc"])
    m_c2_e2_ns = np.average(results["minus"]["c2"]["noshear"]["e2"])
    m_c2_e2dc_ns = np.average(results["minus"]["c2"]["noshear"]["e2dc"])

    # 1p
    m_c2_e1_1p = np.average(results["minus"]["c2"]["1p"]["e1"])
    m_c2_e1dc_1p = np.average(results["minus"]["c2"]["1p"]["e1dc"])
    m_c2_e2_1p = np.average(results["minus"]["c2"]["1p"]["e2"])
    m_c2_e2dc_1p = np.average(results["minus"]["c2"]["1p"]["e2dc"])

    # 1m
    m_c2_e1_1m = np.average(results["minus"]["c2"]["1m"]["e1"])
    m_c2_e1dc_1m = np.average(results["minus"]["c2"]["1m"]["e1dc"])
    m_c2_e2_1m = np.average(results["minus"]["c2"]["1m"]["e2"])
    m_c2_e2dc_1m = np.average(results["minus"]["c2"]["1m"]["e2dc"])

    # 2p
    m_c2_e1_2p = np.average(results["minus"]["c2"]["2p"]["e1"])
    m_c2_e1dc_2p = np.average(results["minus"]["c2"]["2p"]["e1dc"])
    m_c2_e2_2p = np.average(results["minus"]["c2"]["2p"]["e2"])
    m_c2_e2dc_2p = np.average(results["minus"]["c2"]["2p"]["e2dc"])

    # 2m
    m_c2_e1_2m = np.average(results["minus"]["c2"]["2m"]["e1"])
    m_c2_e1dc_2m = np.average(results["minus"]["c2"]["2m"]["e1dc"])
    m_c2_e2_2m = np.average(results["minus"]["c2"]["2m"]["e2"])
    m_c2_e2dc_2m = np.average(results["minus"]["c2"]["2m"]["e2dc"])

    #---------------------------------------------------------------------------

    if alt == False:
        return (
            np.array(
                [
                    [
                        (p_c2_e1dc_1p - p_c2_e1dc_1m - p_c0_e1dc_1p + p_c0_e1dc_1m) / (2 * dg * 2 * dc),
                        (p_c2_e1dc_2p - p_c2_e1dc_2m - p_c0_e1dc_2p + p_c0_e1dc_2m) / (2 * dg * 2 * dc),
                    ],
                    [
                        (p_c2_e2dc_1p - p_c2_e2dc_1m - p_c0_e2dc_1p + p_c0_e2dc_1m) / (2 * dg * 2 * dc),
                        (p_c2_e2dc_2p - p_c2_e2dc_2m - p_c0_e2dc_2p + p_c0_e2dc_2m) / (2 * dg * 2 * dc),
                    ],
                ]
            ),
            np.array(
                [
                    [
                        (m_c2_e1dc_1p - m_c2_e1dc_1m - m_c0_e1dc_1p + m_c0_e1dc_1m) / (2 * dg * 2 * dc),
                        (m_c2_e1dc_2p - m_c2_e1dc_2m - m_c0_e1dc_2p + m_c0_e1dc_2m) / (2 * dg * 2 * dc),
                    ],
                    [
                        (m_c2_e2dc_1p - m_c2_e2dc_1m - m_c0_e2dc_1p + m_c0_e2dc_1m) / (2 * dg * 2 * dc),
                        (m_c2_e2dc_2p - m_c2_e2dc_2m - m_c0_e2dc_2p + m_c0_e2dc_2m) / (2 * dg * 2 * dc),
                    ],
                ]
            )
        )
        # return (
        #     np.array(
        #         [
        #             [
        #                 (p_c2_e1dc_1p - p_c2_e1dc_1m - p_c0_e1dc_1p + p_c0_e1dc_1m) / (2 * dg * 2 * dc),
        #                 (p_c2_e1dc_2p - p_c2_e1dc_2m - p_c0_e1dc_2p + p_c0_e1dc_2m) / (2 * dg * 2 * dc),
        #             ],
        #             [
        #                 (p_c2_e2dc_1p - p_c2_e2dc_1m - p_c0_e2dc_1p + p_c0_e2dc_1m) / (2 * dg * 2 * dc),
        #                 (p_c2_e2dc_2p - p_c2_e2dc_2m - p_c0_e2dc_2p + p_c0_e2dc_2m) / (2 * dg * 2 * dc),
        #             ],
        #         ]
        #     ),
        #     np.array(
        #         [
        #             [
        #                 (m_c2_e1dc_1p - m_c2_e1dc_1m - m_c0_e1dc_1p + m_c0_e1dc_1m) / (2 * dg * 2 * dc),
        #                 (m_c2_e1dc_2p - m_c2_e1dc_2m - m_c0_e1dc_2p + m_c0_e1dc_2m) / (2 * dg * 2 * dc),
        #             ],
        #             [
        #                 (m_c2_e2dc_1p - m_c2_e2dc_1m - m_c0_e2dc_1p + m_c0_e2dc_1m) / (2 * dg * 2 * dc),
        #                 (m_c2_e2dc_2p - m_c2_e2dc_2m - m_c0_e2dc_2p + m_c0_e2dc_2m) / (2 * dg * 2 * dc),
        #             ],
        #         ]
        #     )
        # )
    else:
        # more efficient formula from https://en.wikipedia.org/wiki/Finite_difference
        return (
            np.array(
                [
                    [
                        (p_c2_e1dc_1p - p_c1_e1dc_1p - p_c2_e1dc_ns + 2 * p_c1_e1dc_ns - p_c1_e1dc_1m - p_c0_e1dc_ns + p_c0_e1dc_1m) / (2 * dg * dc),
                        (p_c2_e1dc_2p - p_c1_e1dc_2p - p_c2_e1dc_ns + 2 * p_c1_e1dc_ns - p_c1_e1dc_2m - p_c0_e1dc_ns + p_c0_e1dc_2m) / (2 * dg * dc),
                    ],
                    [
                        (p_c2_e2dc_1p - p_c1_e2dc_1p - p_c2_e2dc_ns + 2 * p_c1_e2dc_ns - p_c1_e2dc_1m - p_c0_e2dc_ns + p_c0_e2dc_1m) / (2 * dg * dc),
                        (p_c2_e2dc_2p - p_c1_e2dc_2p - p_c2_e2dc_ns + 2 * p_c1_e2dc_ns - p_c1_e2dc_2m - p_c0_e2dc_ns + p_c0_e2dc_2m / (2 * dg * dc)),
                    ],
                ],
            ),
            np.array(
                [
                    [
                        (m_c2_e1dc_1p - m_c1_e1dc_1p - m_c2_e1dc_ns + 2 * m_c1_e1dc_ns - m_c1_e1dc_1m - m_c0_e1dc_ns + m_c0_e1dc_1m) / (2 * dg * dc),
                        (m_c2_e1dc_2p - m_c1_e1dc_2p - m_c2_e1dc_ns + 2 * m_c1_e1dc_ns - m_c1_e1dc_2m - m_c0_e1dc_ns + m_c0_e1dc_2m) / (2 * dg * dc),
                    ],
                    [
                        (m_c2_e2dc_1p - m_c1_e2dc_1p - m_c2_e2dc_ns + 2 * m_c1_e2dc_ns - m_c1_e2dc_1m - m_c0_e2dc_ns + m_c0_e2dc_1m) / (2 * dg * dc),
                        (m_c2_e2dc_2p - m_c1_e2dc_2p - m_c2_e2dc_ns + 2 * m_c1_e2dc_ns - m_c1_e2dc_2m - m_c0_e2dc_ns + m_c0_e2dc_2m / (2 * dg * dc)),
                    ],
                ],
            )
        )


def compute_bias(results, dg, dc):
    e_p, e_m = compute_e(results)

    R_p, R_m = compute_R(results, dg)

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


def compute_bias_chromatic_testing(batch, dg, dc):
    e_p, e_m = compute_e(batch)

    R_p, R_m = compute_R(batch, dg)

    dedc_p, dedc_m = compute_dedc(batch, dg, dc)
    dRdc_p, dRdc_m = compute_dRdc(batch, dg, dc)

    g_p = np.linalg.inv(R_p) @ e_p + (-np.linalg.inv(R_p) @ dRdc_p @ np.linalg.inv(R_p) @ e_p + np.linalg.inv(R_p) @ dedc_p)
    g_m = np.linalg.inv(R_m) @ e_m + (-np.linalg.inv(R_p) @ dRdc_p @ np.linalg.inv(R_p) @ e_p + np.linalg.inv(R_p) @ dedc_p)

    m = (g_p - g_m)[0] / 2 / 0.02 - 1

    c = (g_p + g_m)[1] / 2

    # return g_p, g_m, m, c
    return m, c


def compute_bias_chromatic(batch, dg, dc):
    e_p, e_m = compute_e(batch)

    R_p, R_m = compute_R(batch, dg)

    dedc_p, dedc_m = compute_dedc(batch, dg, dc)
    dRdc_p, dRdc_m = compute_dRdc(batch, dg, dc)

    g_p = np.linalg.inv(R_p + dRdc_p) @ (e_p - dedc_p)
    g_m = np.linalg.inv(R_m + dRdc_m) @ (e_m - dedc_m)

    m = (g_p - g_m)[0] / 2 / 0.02 - 1

    c = (g_p + g_m)[1] / 2

    # return g_p, g_m, m, c
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
        for color_step in ["c0", "c1", "c2"]:
            aggregates[shear_step][color_step] = {}
            for mdet_step in ["noshear", "1p", "1m", "2p", "2m"]:
                predicate = (
                    (pc.field("shear") == shear_step)
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
                            pc.multiply(pc.field("mean_e1dc"), pc.field("count")),
                            pc.multiply(pc.field("mean_e2dc"), pc.field("count")),
                            pc.multiply(pc.field("var_e1"), pc.field("count")),
                            pc.multiply(pc.field("var_e2"), pc.field("count")),
                            pc.multiply(pc.field("var_c"), pc.field("count")),
                            pc.multiply(pc.field("var_e1e2"), pc.field("count")),
                            pc.multiply(pc.field("var_e1c"), pc.field("count")),
                            pc.multiply(pc.field("var_e2c"), pc.field("count")),
                            pc.multiply(pc.field("var_e1dc"), pc.field("count")),
                            pc.multiply(pc.field("var_e2dc"), pc.field("count")),
                        ],
                        names=[
                            "count",
                            "weighted_mean_e1",
                            "weighted_mean_e2",
                            "weighted_mean_c",
                            "weighted_mean_e1e2",
                            "weighted_mean_e1c",
                            "weighted_mean_e2c",
                            "weighted_mean_e1dc",
                            "weighted_mean_e2dc",
                            "weighted_var_e1",
                            "weighted_var_e2",
                            "weighted_var_c",
                            "weighted_var_e1e2",
                            "weighted_var_e1c",
                            "weighted_var_e2c",
                            "weighted_var_e1dc",
                            "weighted_var_e2dc",
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
                            ("weighted_mean_e1dc", "sum", None, "sum_mean_e1dc"),
                            ("weighted_mean_e2dc", "sum", None, "sum_mean_e2dc"),
                            ("weighted_var_e1", "sum", None, "sum_var_e1"),
                            ("weighted_var_e2", "sum", None, "sum_var_e2"),
                            ("weighted_var_c", "sum", None, "sum_var_c"),
                            ("weighted_var_e1e2", "sum", None, "sum_var_e1e2"),
                            ("weighted_var_e1c", "sum", None, "sum_var_e1c"),
                            ("weighted_var_e2c", "sum", None, "sum_var_e2c"),
                            ("weighted_var_e1dc", "sum", None, "sum_var_e1dc"),
                            ("weighted_var_e2dc", "sum", None, "sum_var_e2dc"),
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
                            pc.divide(pc.field("sum_mean_e1dc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_mean_e2dc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e1"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e2"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_c"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e1e2"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e1c"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e2c"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e1dc"), pc.field("sum_count")),
                            pc.divide(pc.field("sum_var_e2dc"), pc.field("sum_count")),
                        ],
                        names=[
                            "e1",
                            "e2",
                            "c",
                            "e1e2",
                            "e1c",
                            "e2c",
                            "e1dc",
                            "e2dc",
                            "var_e1",
                            "var_e2",
                            "var_c",
                            "var_e1e2",
                            "var_e1c",
                            "var_e2c",
                            "var_e1dc",
                            "var_e2dc",
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
                            pc.field("e1dc"),
                            pc.field("e2dc"),
                            pc.field("var_e1"),
                            pc.field("var_e2"),
                            pc.field("var_c"),
                            pc.field("var_e1e2"),
                            pc.field("var_e1c"),
                            pc.field("var_e2c"),
                            pc.field("var_e1dc"),
                            pc.field("var_e2dc"),
                            pc.subtract(
                                pc.field("e1e2"),
                                pc.multiply(
                                    pc.field("e1"),
                                    pc.field("e2"),
                                ),
                            ),
                            pc.subtract(
                                pc.field("e1c"),
                                pc.multiply(
                                    pc.field("e1"),
                                    pc.field("c"),
                                ),
                            ),
                            pc.subtract(
                                pc.field("e2c"),
                                pc.multiply(
                                    pc.field("e2"),
                                    pc.field("c"),
                                ),
                            ),
                        ],
                        names=[
                            "e1",
                            "e2",
                            "c",
                            "e1e2",
                            "e1c",
                            "e2c",
                            "e1dc",
                            "e2dc",
                            "var_e1",
                            "var_e2",
                            "var_c",
                            "var_e1e2",
                            "var_e1c",
                            "var_e2c",
                            "var_e1dc",
                            "var_e2dc",
                            "cov_e1e2",
                            "cov_e1c",
                            "cov_e2c",
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

    logger_config = logging_config.defaults
    log_level = logging_config.get_level(args.log_level)
    logging.basicConfig(level=log_level, **logging_config.defaults)

    logger.info(f"{vars(args)}")

    config = args.config
    seed = args.seed
    n_jobs = args.n_jobs
    output = args.output

    pa.set_cpu_count(n_jobs)
    # pa.set_io_thread_count(n_jobs)
    pa.set_io_thread_count(1)

    pipeline = Pipeline(config)
    pipeline.load()

    rng = np.random.default_rng(seed)

    dg = ngmix.metacal.DEFAULT_STEP

    colors = pipeline.get_colors()
    if colors:
        color = colors[1]
        dc = -(colors[2] - colors[0]) / 2.
    else:
        color = None
        dc = None

    aggregate_path = os.path.join(
        output,
        f"{pipeline.name}_aggregates.feather",
    )
    logger.info(f"reading aggregates from {aggregate_path}")
    aggregates = ft.read_table(
        aggregate_path,
    )
    aggregates.validate(full=True)

    seeds = np.sort(np.unique(aggregates["seed"]))

    results = pivot_aggregates(aggregates)

    # g_p, g_m, m_mean, c_mean = compute_bias(results, dg, dc)
    # g_p_chroma, g_m_chroma, m_mean_chroma, c_mean_chroma = compute_bias_chromatic(results, dg, dc, color)
    # # g_p_chroma, g_m_chroma, m_mean_chroma, c_mean_chroma = compute_bias_chromatic_testing(results, dg, dc, color)
    m_mean, c_mean = compute_bias(results, dg, dc)
    m_mean_chroma, c_mean_chroma = compute_bias_chromatic(results, dg, dc)

    jobs = []
    for i in tqdm.trange(args.n_resample, ncols=80):
        _seeds = pa.array(rng.choice(seeds, size=len(seeds), replace=True))
        jobs.append(joblib.delayed(pivot_aggregates)(aggregates, seeds=_seeds))

    _results = joblib.Parallel(n_jobs=n_jobs, verbose=10, return_as="generator")(jobs)

    # g_p_bootstrap = []
    # g_m_bootstrap = []
    # g_p_bootstrap_chroma = []
    # g_m_bootstrap_chroma = []
    m_bootstrap = []
    m_bootstrap_chroma = []
    c_bootstrap = []
    c_bootstrap_chroma = []
    for _res in _results:
        # _g_p_bootstrap, _g_m_bootstrap, _m_bootstrap, _c_bootstrap = compute_bias(_res, dg, dc)
        # _g_p_bootstrap_chroma, _g_m_bootstrap_chroma, _m_bootstrap_chroma, _c_bootstrap_chroma = compute_bias_chromatic(_res, dg, dc, color)
        # # _g_p_bootstrap_chroma, _g_m_bootstrap_chroma, _m_bootstrap_chroma, _c_bootstrap_chroma = compute_bias_chromatic_testing(_res, dg, dc, color)
        _m_bootstrap, _c_bootstrap = compute_bias(_res, dg, dc)
        _m_bootstrap_chroma, _c_bootstrap_chroma = compute_bias_chromatic(_res, dg, dc)

        # g_p_bootstrap.append(_g_p_bootstrap)
        # g_m_bootstrap.append(_g_m_bootstrap)
        # g_p_bootstrap_chroma.append(_g_p_bootstrap_chroma)
        # g_m_bootstrap_chroma.append(_g_m_bootstrap_chroma)
        m_bootstrap.append(_m_bootstrap)
        m_bootstrap_chroma.append(_m_bootstrap_chroma)
        c_bootstrap.append(_c_bootstrap)
        c_bootstrap_chroma.append(_c_bootstrap_chroma)

    # g_p_bootstrap = np.array(g_p_bootstrap)
    # g_m_bootstrap = np.array(g_m_bootstrap)
    # g_p_bootstrap_chroma = np.array(g_p_bootstrap_chroma)
    # g_m_bootstrap_chroma = np.array(g_m_bootstrap_chroma)
    m_bootstrap = np.array(m_bootstrap)
    m_bootstrap_chroma = np.array(m_bootstrap_chroma)
    c_bootstrap = np.array(c_bootstrap)
    c_bootstrap_chroma = np.array(c_bootstrap_chroma)

    # report 3 standard deviations as error
    m_error = np.nanstd(m_bootstrap) * 3
    m_error_chroma = np.nanstd(m_bootstrap_chroma) * 3
    c_error = np.nanstd(c_bootstrap) * 3
    c_error_chroma = np.nanstd(c_bootstrap_chroma) * 3

    # print(f"mdet -- plus: {g_p} -- minus: {g_m}")
    print(f"mdet: m = {m_mean:0.3e} +/- {m_error:0.3e} [3-sigma], c = {c_mean:0.3e} +/- {c_error:0.3e} [3-sigma]")
    # print(f"drdc -- plus: {g_p_chroma} -- minus: {g_m_chroma}")
    print(f"drdc: m = {m_mean_chroma:0.3e} +/- {m_error_chroma:0.3e} [3-sigma], c = {c_mean_chroma:0.3e} +/- {c_error_chroma:0.3e} [3-sigma]")

    m_req = 2e-3

    fig, axs = plt.subplots(2, 2)

    # axs[0, 0].axvline(0.02, c="k", alpha=0.1, ls="--")
    # axs[0, 0].hist(g_p_bootstrap[:, 0], histtype="step", label=r"$R$", ec="k")
    # axs[0, 0].axvline(g_p[0], c="k")
    # axs[0, 0].hist(g_p_bootstrap_chroma[:, 0], histtype="step", label=r"$R$ \& $\partial R / \partial c$", ec="b")
    # axs[0, 0].axvline(g_p_chroma[0], c="b")
    # # axs[0, 0].legend()
    # axs[0, 0].set_xlabel("$g_1^+$")

    # axs[0, 1].axvline(-0.02, c="k", alpha=0.1, ls="--")
    # axs[0, 1].hist(g_m_bootstrap[:, 0], histtype="step", label=r"$R$", ec="k")
    # axs[0, 1].axvline(g_m[0], c="k")
    # axs[0, 1].hist(g_m_bootstrap_chroma[:, 0], histtype="step", label=r"$R$ \& $\partial R / \partial c$", ec="b")
    # axs[0, 1].axvline(g_m_chroma[0], c="b")
    # # axs[0, 1].legend()
    # axs[0, 1].set_xlabel("$g_1^-$")


    axs[1, 0].axvspan(-m_req, m_req, fc="k", alpha=0.1)
    axs[1, 0].axvline(4e-4, c="k", alpha=0.1, ls="--")
    axs[1, 0].hist(m_bootstrap, histtype="step", label=r"$R$", ec="k")
    axs[1, 0].axvline(m_mean, c="k")
    axs[1, 0].hist(m_bootstrap_chroma, histtype="step", label=r"$R$ \& $\partial R / \partial c$", ec="b")
    axs[1, 0].axvline(m_mean_chroma, c="b")
    # axs[1, 0].legend()
    axs[1, 0].set_xlabel("$m$")

    axs[1, 1].hist(c_bootstrap, histtype="step", label=r"$R$", ec="k")
    axs[1, 1].axvline(c_mean, c="k")
    axs[1, 1].hist(c_bootstrap_chroma, histtype="step", label=r"$R$ \& $\partial R / \partial c$", ec="b")
    axs[1, 1].axvline(c_mean_chroma, c="b")
    axs[1, 1].legend()
    axs[1, 1].set_xlabel("$c$")

    plt.savefig("out.pdf")

    plt.show()


if __name__ == "__main__":
    main()
