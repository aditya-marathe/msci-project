"""
src / data.py
--------------------------------------------------------------------------------

Aditya Marathe

"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement

__all__ = [
    'NOvAData'
]

import numpy as np
import numpy.typing as npt

import h5py

import pandas as pd


# Constants --------------------------------------------------------------------


_VARS_TO_EXTRACT = {
    
    # Energy predictor branch

    'rec.energy.numu': [
        'E',  # Old energy predictor
        'calccE',
        'hadcalE',
        'hadtrkE',
        'lstmmuon',  # LSTM energy predictor
        'lstmnu',
        'regcvnhadE',
        'trkccE',
        'recomuonE'
    ],
    'rec.energy.numu.hadclust': [
        'calE',
        'nhit'
    ],

    # Selector (PID) branch

    'rec.sel.contain': [
        'kalfwdcell',
        'kalbakcell',
        'cosfwdcell',
        'cosbakcell'
    ],
    'rec.sel.cosrej': [
        'numucontpid2020fhc'
    ],
    'rec.sel.cvnloosepreselptp': [
        'cosmicid',
        'ncid',
        'nueid',
        'numuid',
        'nutauid'
    ],
    'rec.sel.remid': [
        'pid'
    ],
    'rec.sel.scann': [
        'scpid'
    ],
    'rec.sel.veto': [
        'keep'
    ],

    # Slice branch

    'rec.slc': [
        'calE',  # Calorimetric energy of all hits in this slice
        'firstplane',
        'lastplane',
        'nhit',
        'ncontplanes'
    ],

    # Beam spill branch

    'rec.spill.cosmiccvn': [
        'passSel'
    ],

    # Track branch

    'rec.trk.cosmic': [
        'ntracks'
    ],
    'rec.trk.kalman': [
        'idxremid',
        'ntracks'
    ],
    'rec.trk.kalman.tracks': [
        'dir.x',  # --> Needed for calculating kinematics
        'dir.y',
        'dir.z',
        'nhit',
        'nhitx',
        'nhity',
        'calE',  # Calorimetric energy of all hits in the prong
        'nplane',
        'maxplanecont'
    ],

    # MC Truth branch

    'rec.mc': [
        'nnu'
    ],
    'rec.mc.nu': [
        'pdg',
        'mode',
        'iscc'
    ],
    'rec.mc.nu.beam': [
        'potnum'  # Number of POT (Protons On Target)
    ]
}

_DIFF_LEN_BRANCHES = [
    'rec.mc.nu',
    'rec.mc.nu.beam',
    'rec.training.cvnmaps',
    'rec.training.trainingdata',
    'rec.trk.kalman.tracks'
]

_TABLE_INDEX_NAMES = [
    'run', 
    'subrun', 
    'cycle', 
    'batch', 
    'evt', 
    'subevt'
]

_TABLE_COL_FILLNA_MAP = {
    'rec.mc.nu.pdg': 0,
    'rec.mc.nu.mode': -1,
    'rec.mc.nu.iscc': 0,
    'rec.mc.nu.beam.potnum': 0
}


# Helper functions -------------------------------------------------------------


def _get_var(h5file: h5py.File, branch: str, var: str) -> npt.NDArray:
    """\
    Get a variable from a NOvA HDF5 database.

    Args
    ----
    h5file: h5py.File
        An open `File` object in read mode.

    branch: str
        The branch name or "parent key" inside the HDF5 database.

    var: str
        The variable name or "child key" inside the HDF5 databse.

    Returns
    -------
    npt.NDArray
        A 1-dimensional array of the target variable data.
    """
    return h5file[branch][var][:].flatten()


def _get_multi_index(
        h5file: h5py.File, 
        ref_branch: str, 
        unique: bool = False
    ) -> pd.MultiIndex | tuple[pd.MultiIndex, npt.NDArray]:
    """\
    Creates the multi-index for a Pandas data frame.

    Args
    ----
    ref_branch: str
        A branch name or "parent key" inside the HDF5 database, which will be 
        used as a reference for creating the multi-index.

    unique: bool
        Set to `True` if the branch contains repeated rows that share the same 
        index. Defaults to `False`.

    Returns
    -------
    pd.MultiIndex
        A Pandas `MultiIndex` object.
    """
    unique_indices = None

    indices = [
        _get_var(h5file, ref_branch, name) for name in _TABLE_INDEX_NAMES
    ]
    concat_indices = np.column_stack(indices)

    if unique:
        concat_indices, unique_indices = np.unique(
            concat_indices, 
            return_inverse=True, 
            axis=0
        )

    concat_indices = concat_indices.tolist()

    if unique:
        return (
            pd.MultiIndex.from_tuples(
                tuples=concat_indices,
                name=_TABLE_INDEX_NAMES
            ),
            unique_indices
        )

    return pd.MultiIndex.from_tuples(
        tuples=concat_indices,
        name=_TABLE_INDEX_NAMES
    )


def _get_branch_df(
    h5file: h5py.File,
    branch: str, 
    vars: list[str], 
    flatten: bool
    ) -> pd.DataFrame:
    """\
    Creates a small Pandas `DataFrame` for the given branch.

    Args
    ----
    h5file: h5py.File
        An open `File` object in read mode.

    branch: str
        The branch name or "parent key" inside the HDF5 database.

    var: str
        The variable name or "child key" inside the HDF5 databse.

    flatten: bool
        If set to `True`, the column data will be flattened to a 1-dimensional
        array to reverse the effect of using `np.split`. If left as `False`, it 
        will store rows with the same index as a NumPy array.

    Returns
    -------
    pd.DataFrame
        A Pandas `DataFrame` object containing data of the given branch and 
        branch variables.
    """
    multi_index, unique_indices = _get_multi_index(h5file, branch, unique=True)

    mini_table = pd.DataFrame(index=multi_index)

    for var in vars:
        var_values = _get_var(h5file, branch, var)
        split_indices = np.where(
            unique_indices[:-1] != unique_indices[1:]
        )[0] + 1
        col_values = np.split(var_values, split_indices)

        if flatten:
            col_values = np.asarray(col_values).flatten()
        
        col_name = branch + '.' + var
        mini_table[col_name] = col_values

    return mini_table


# Class(es) --------------------------------------------------------------------


class NOvAData:
    """\
    Used to load and store a sample of the simulated NOvA dataset.

    This class has the following roles:

        - Loading one/more copymerged HDF5 NOvA databases into Python. 
            + They are saved into a Pandas `DataFrame` / "table".
            + If more than one file is loaded, then their tables get merged.
            + Each table is indexed by the run, subrun, cycle, batch, event, and
              subevent.

        - Plotting Spectrum data in a standardised style.
            + Also, able to save high-resolution images of the plots.

    This class aims to make the following easier to implement:

        - Applying the latest NOvA cuts / any custom cuts.

    """
    def __init__(self, h5dirs: list[str]) -> None:
        """\
        NOvA data object constructor.

        Args
        ----
        h5dir: list
            List of copymerged HDF5 files.
        """
        tables = list()

        for h5dir in h5dirs:
            with h5py.File(name=h5dir, mode='r') as h5file:
                # So, CAFAna stores information about the run, subrun, cycle of 
                # the MC simulation, batch of the MC simulation, event, subevent 
                # in the standard record header (SRHeader). But, in the HDF5 
                # format, each "branch" (except for `rec.sel.scann`) has these 
                # stored individually inside of each branch. So, the idea is 
                # that we want to group these together and merge all the 
                # branches nicely into a Pandas `DataFrame` object.

                table = pd.DataFrame(
                    index=_get_multi_index(h5file, 'rec.energy.numu')
                )

                # Fill the table
                for branch in h5file.keys():
                    # Special case: If the variable in question has a different
                    # number of data points (e.g. variables in the neutrino MC 
                    # branch) then we would have to use a different method to
                    # extract data from this column.                    
                    if branch in _DIFF_LEN_BRANCHES:
                        vars = _VARS_TO_EXTRACT.get(branch, [])

                        if len(vars) > 0:
                            mini_table = _get_branch_df(
                                h5file=h5file, 
                                branch=branch, 
                                vars=vars,
                                flatten=branch.startswith('rec.mc')
                            )

                            table = pd.merge(
                                table,
                                mini_table, 
                                left_index=True, 
                                right_index=True, 
                                how='outer'
                            )

                        continue
                    
                    # Normal case: When the variable has the expected column 
                    # length.
                    for var in _VARS_TO_EXTRACT.get(branch, []):
                        col_name = branch + '.' + var
                        table[col_name] = _get_var(h5file, branch, var)

            tables.append(table)

        self.table = pd.concat(tables)

        # Account for any NaN values in the table
        for column_name, fill_value in _TABLE_COL_FILLNA_MAP.items():
            if column_name in self.table.columns:
                self.table[column_name].fillna(fill_value, inplace=True)        
