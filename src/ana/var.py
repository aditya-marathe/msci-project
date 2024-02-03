"""\
src / ana / var.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import h5py

import pandas as pd

if TYPE_CHECKING:
    from ana.data import NOvAData


class Var:
    """\
    A variable inside of the NOvA dataset.
    """
    def __init__(self, branch: str, var: str) -> None:
        """\
        `Var` constructor.

        Args
        ----
        branch: str
            The branch name or "parent key" inside the HDF5 database.

        var: str
            The variable name or "child key" inside the HDF5 databse.
        """
        self._branch = branch
        self._var = var

    def get_df(self, df: pd.DataFrame) -> pd.Series:
        """\
        Gets the data for this variable from a NOvA Pandas `DataFrame` / table.

        Args
        ----
        df: pd.DataFrame
            Pandas data frame of the NOvA data.

        Returns
        -------
        pd.Series
            The sample of data for this variable in the given NOvA data.
        """
        return df[self._branch + '.' + self._var]

    def get_h5(self, h5file: h5py.File) -> npt.NDArray:
        """\
        Gets the data for this variable from a NOvA dataset.

        Args
        ----
        h5file: h5py.File
            An HDF5 database of the NOvA data.

        Returns
        -------
        npt.NDArray
            The sample of data for this variable in the given NOvA data.
        """
        return _get_var(h5file, self._branch, self._var)

    def get(self, data: "NOvAData") -> pd.Series:
        """\
        Gets the data for this variable from a NOvA dataset.

        Args
        ----
        data: NOvAData
            A NOvAData object.

        Returns
        -------
        pd.Series
            The sample of data for this variable in the given NOvA data.
        """
        return self.get_df(data.table)

    def __call__(self, data: "NOvAData") -> pd.Series:
        """\
        Making the class callable, for some syntactic sugar.
        """
        return self.get(data)

    def __str__(self) -> str:
        return f'{self._branch}.{self._var}'

    def __repr__(self) -> str:
        return f'Var(name=\'{self._branch}.{self._var}\')'
