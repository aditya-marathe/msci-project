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
            data: "NOvAData", 
            bining: npt.NDArray
        ) -> None:
        self._var_data = var(data=data)
        # self._pot = Var(branch=, var=)