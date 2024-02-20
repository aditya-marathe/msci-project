"""\
src / ana / spectrum.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ana.data import *

from ana.var import *


class Binning:
    STANDARD_ENERGY = np.linspace(0, 10, 100 + 1)
    HALF_ENERGY = np.linspace(0, 5, 50 + 1)


class Spectrum:
    """\
    Spectrum
    --------

    This class stores a histogram contaning the number of events per bin for a 
    given variable `Var`, and also storing the exposure (protons on target).
    """
    def __init__(
            self, 
            var: "Var",
            binning: npt.NDArray
        ) -> None:
        """\
        
        """
        self._var = var
        self._pot = Var(branch='rec.mc.nu.beam', var='potnum')

        self.binning = binning

    def get(
            self, 
            data: "NOvAData"
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """\

        """
        values, edges = np.histogram(self._var(data=data), bins=self.binning)
        centres = (edges[:-1] + edges[1:]) / 2
        return (values, edges, centres), self._pot(data=data)

    def __call__(
            self,
            data: "NOvAdata"
        ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        """
        return self.get(data=data)