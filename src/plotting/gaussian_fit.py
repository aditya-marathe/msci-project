"""\
src / plotting / gaussian_fit.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

__all__ = [
    'fit_to_gaussian'
]

import numpy as np
import numpy.typing as npt

from scipy.optimize import curve_fit


def _gaussian(
        x: npt.NDArray,
        amp: int | float,
        mean: int | float,
        std: int | float
    ) -> npt.NDArray:
    """
    Typical gaussian function with parameters amplitude, mean, and standard
    deviation. Used as the fit function.
    """
    return amp * np.exp(-(x - mean)**2 / (2 * std**2))


def fit_to_gaussian(
        x: npt.NDArray,
        x_obs: npt.NDArray,
        y_obs: npt.NDArray,
        mean: int | float,
        std: int | float
    ) -> tuple[npt.NDArray, dict[str, tuple[float, float]]]:
    """\
    Fit a gaussian to given data.
    """
    params, cov_matrix = curve_fit(_gaussian, x_obs, y_obs, p0=[1, mean, std])

    param_errors = np.sqrt(np.diag(cov_matrix))

    return _gaussian(x, *params), {
        'Amp': (params[0], param_errors[0]),
        'Mean': (params[1], param_errors[1]),
        'StdDev': (params[2], param_errors[2])
    }
