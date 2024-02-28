"""\
src / detectors.py

--------------------------------------------------------------------------------

Aditya Marathe

Tools for detector transformations, and getting the beam direction inside of a 
detector. Directly taken (with some minor changes) from Prof. Nichol's code.
"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals


__all__ = [
    'FD_R_OFFSET_VEC',
    'FD_M_ROTATIONAL_MAT',
    'FD_M_ROTATIONAL_INV_MAT',
    'ND_R_OFFSET_VEC',
    'ND_M_ROTATIONAL_MAT',
    'ND_M_ROTATIONAL_INV_MAT',
    'FD_BLOCK_CENTRE',
    'ND_BLOCK_CENTRE',
    'POINT_S',
    'MC_ZERO',
    'POINT_100',
    'POINT_600',
    'POINT_800',
    'conv_fd_to_numi',
    'conv_numi_to_fd',
    'conv_nd_to_numi',
    'conv_numi_to_nd',
    'calculate_beam_direction_at_fd',
    'calculate_beam_direction_at_nd'
]

import numpy as np
import numpy.typing as npt


# Constants --------------------------------------------------------------------


# Far-detector transform

FD_R_OFFSET_VEC = np.array(
    [57.666820, -51777.408547, -808852.640542]
)

FD_M_ROTATIONAL_MAT = np.array(
    [
        [0.99990622174    , 0.000472822894411, 0.0136866417481],
        [0.000472822894411, 0.997616062703   , -0.0690070132296],
        [-0.0136866417481 , 0.0690070132296  , 0.997522284444]
    ]
)

FD_M_ROTATIONAL_INV_MAT = np.linalg.inv(FD_M_ROTATIONAL_MAT)

# Near-detector transform

ND_R_OFFSET_VEC = np.array(
    [2.269447, 61.001882, -991.131313]
) 

ND_M_ROTATIONAL_MAT = np.array(
    [
        [0.99990   , 3.0533E-6 , 1.4112E-2],
        [8.2300E-4 , 0.99831   , 5.8097E-2],
        [-1.4088E-2, -5.8103E-2, 0.99821]
    ]
)

ND_M_ROTATIONAL_INV_MAT = np.linalg.inv(ND_M_ROTATIONAL_MAT)

# Block centres from the survey measurements

FD_BLOCK_CENTRE = np.array(
    [11037.705436, -4164.619757, 810452.13544]
)

ND_BLOCK_CENTRE = np.array(
    [11.793636, -2.928869, 999.282492]
)

POINT_S = np.array(   # TODO: Is that what this variable refers to?
    [0, 0, 170.399278]
)

MC_ZERO = np.array([0, 0, 0])

POINT_100 = np.array([0, 0, 100])
POINT_600 = np.array([0, 0, 600])
POINT_800 = np.array([0, 0, 800])


# Helper functions -------------------------------------------------------------


def conv_fd_to_numi(fd_r_vec: npt.NDArray) -> npt.NDArray:
    """\
    Converts from FD to NuMI coordinates.
    
    Args
    ----
    fd_r_vec: npt.NDArray
        Position vector (at FD) to be transformed.

    Returns
    -------
    npt.NDArray
        The transformed position vector at NuMI.
    """
    return np.matmul(FD_M_ROTATIONAL_MAT, fd_r_vec - FD_R_OFFSET_VEC)


def conv_numi_to_fd(numi_r_vec: npt.NDArray) -> npt.NDArray:
    """\
    Converts from NuMI to FD coordinates.
    
    Args
    ----
    numi_r_vec: npt.NDArray
        Position vector (at NuMI) to be transformed.

    Returns
    -------
    npt.NDArray
        The transformed position vector at FD.
    """
    return np.matmul(FD_M_ROTATIONAL_INV_MAT, numi_r_vec) + FD_R_OFFSET_VEC


def conv_nd_to_numi(nd_r_vec: npt.NDArray) -> npt.NDArray:
    """\
    Converts from ND to NuMI coordinates.
    
    Args
    ----
    nd_r_vec: npt.NDArray
        Position vector (at ND) to be transformed.

    Returns
    -------
    npt.NDArray
        The transformed position vector at NuMI.
    """
    return np.matmul(ND_M_ROTATIONAL_MAT, nd_r_vec - ND_R_OFFSET_VEC)


def conv_numi_to_nd(numi_r_vec: npt.NDArray) -> npt.NDArray:
    """\
    Converts from NuMI to ND coordinates.
    
    Args
    ----
    numi_r_vec: npt.NDArray
        Position vector (at NuMI) to be transformed.

    Returns
    -------
    npt.NDArray
        The transformed position vector at ND.
    """
    return np.matmul(ND_M_ROTATIONAL_INV_MAT, numi_r_vec) + ND_R_OFFSET_VEC


# Core -------------------------------------------------------------------------


def calculate_beam_direction_at_fd(
        fd_r_vec: npt.NDArray,
        normalised: bool = True
    ) -> npt.NDArray:
    """\
    Calculates the beam direction vector at a given FD point.

    Args
    ----
    fd_r_vec: npt.NDArray
        Position vector inside of the FD.

    normalised: bool
        Normalises the output vector if set to `True`. Defaults to `True`.

    Returns
    -------
    npt.NDArray
        The beam direction (unit) vector at a point in the FD.
    """

    fd_r_centre_from_point = FD_BLOCK_CENTRE - fd_r_vec

    fd_centre_dist = np.linalg.norm(fd_r_centre_from_point)

    fd_r_centre_from_point_unit = (
        fd_r_centre_from_point / np.linalg.norm(fd_r_centre_from_point)
    )

    fd_r_beam_front = conv_numi_to_fd(
        POINT_S + (fd_centre_dist - 29.942) * fd_r_centre_from_point_unit
    )

    fd_r_beam_back = conv_numi_to_fd(
        POINT_S + (fd_centre_dist + 29.942) * fd_r_centre_from_point_unit
    )

    fd_r_beam_dir = fd_r_beam_back - fd_r_beam_front

    if normalised:
        return fd_r_beam_dir / np.linalg.norm(fd_r_beam_dir)

    return fd_r_beam_dir


def calculate_beam_direction_at_nd(
        nd_r_vec: npt.NDArray,
        normalised: bool = True
    ) -> npt.NDArray:
    """\
    Calculates the beam direction vector at a given ND point.

    Args
    ----
    nd_r_vec: npt.NDArray
        Position vector inside of the ND.

    normalised: bool
        Normalises the output vector if set to `True`. Defaults to `True`.

    Returns
    -------
    npt.NDArray
        The beam direction (unit) vector at a point in the ND.
    """
    nd_r_centre_from_point = ND_BLOCK_CENTRE - nd_r_vec

    nd_centre_dist = np.linalg.norm(nd_r_centre_from_point)

    nd_r_centre_from_point_unit = (
        nd_r_centre_from_point / np.linalg.norm(nd_r_centre_from_point)
    )

    nd_r_beam_front = conv_numi_to_nd(
        POINT_S + (nd_centre_dist - 29.942) * nd_r_centre_from_point_unit
    )

    nd_r_beam_back = conv_numi_to_nd(
        POINT_S + (nd_centre_dist + 29.942) * nd_r_centre_from_point_unit
    )

    nd_r_beam_dir = nd_r_beam_back - nd_r_beam_front
    
    if normalised:
        return nd_r_beam_dir / np.linalg.norm(nd_r_beam_dir)

    return nd_r_beam_dir
