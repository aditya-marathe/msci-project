"""\
src / ana / data.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement

__all__ = [
    'EVENT_TYPE_MAP',
    'VARS_TO_SCALE',
    'NOvAData'
]

import pathlib

import numpy as np
import numpy.typing as npt

import h5py

import pandas as pd

from sklearn.model_selection import train_test_split

# Local

import detectors
import transforms as transforms_

# TODO: Some of the functions here can be parallelised easily, could save time
#       for much large datasets, so something to consier for future updates...


# Constants --------------------------------------------------------------------

EVENT_TYPE_MAP = {
    0: 'Background',
    1: 'NuMu CC',
    2: 'ANuMu CC',
    3: 'ENumU CC',
    4: 'NC'
}

_VARS_TO_EXTRACT = {
    
    # Energy estimator branch

    'rec.energy.numu': [
        'E',  # Old energy estimator
        'calccE',
        'hadcalE',
        'hadtrkE',
        'lstmmuon',  # LSTM energy estimator
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
        'ntracks',
        'ntracks2d'
    ],
    'rec.trk.kalman.tracks': [
        'dir.x',  # --> Needed for calculating kinematics
        'dir.y',
        'dir.z',
        'start.x',
        'start.y',
        'start.z',
        'len',
        'muonid',
        'rempid',
        'nhit',
        'nhitx',
        'nhity',
        'calE',  # Calorimetric energy of all hits in the prong
        'overlapE',
        'nplane',
        'maxplanecont'
    ],

    # MC Truth branch

    'rec.mc': [
        'nnu'
    ],
    'rec.mc.nu': [
        'E',
        'pdg',
        'mode',
        'iscc'
    ],
    # 'rec.mc.nu.beam': [  --> Not sure how to use this, and it is not required
    #                          for my analysis anyway!
    #     'potnum'  # Number of POT (Protons On Target)
    # ]
}

VARS_TO_SCALE = {
    'rec.energy.numu.E',
    'rec.energy.numu.calccE',
    'rec.energy.numu.hadcalE',
    'rec.energy.numu.hadtrkE',
    'rec.energy.numu.lstmmuon',
    'rec.energy.numu.lstmnu',
    'rec.energy.numu.regcvnhadE',
    'rec.energy.numu.trkccE',
    'rec.energy.numu.recomuonE',

    'rec.energy.numu.hadclust.calE',
    # 'rec.energy.numu.hadclust.nhit',

    # 'rec.sel.contain.kalfwdcell',
    # 'rec.sel.contain.kalbakcell',
    # 'rec.sel.contain.cosfwdcell',
    # 'rec.sel.contain.cosbakcell',

    'rec.slc.calE',
    # 'rec.slc.firstplane',
    # 'rec.slc.lastplane',
    # 'rec.slc.nhit',
    # 'rec.slc.ncontplanes',

    # 'rec.trk.cosmic.ntracks',

    # 'rec.trk.kalman.ntracks',
    # 'rec.trk.kalman.ntracks2d',

    'rec.mc.nu.E'
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
    'rec.mc.nu.E': -5,  # Negative value will be taken out by data transforms...
    'rec.mc.nu.pdg': 0,
    'rec.mc.nu.mode': -1,  # -1 means an unknown interaction mode
    'rec.mc.nu.iscc': 0,
    'rec.mc.nu.beam.potnum': 0
}


# Helper functions -------------------------------------------------------------

def _log(message: str) -> None:
    """\
    Prints a log message to the consle.
    
    Args
    ----
    message: str
        Log message.
    """
    if NOvAData.verbose:
        print(f'NOvAData | {message}')


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
    return h5file[branch][var][:].flatten()  # type: ignore


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
                names=_TABLE_INDEX_NAMES
            ),
            unique_indices  # type: ignore
        )

    return pd.MultiIndex.from_tuples(
        tuples=concat_indices,
        names=_TABLE_INDEX_NAMES
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


# Class ------------------------------------------------------------------------


# Notes: There is probably a way to implement this by inheriting the Pandas
# `DataFrame` class, but this is good enough for now...


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

    verbose: bool = True

    def __init__(self, table: pd.DataFrame) -> None:
        """\
        `NOvAData` constructor.

        Args
        ----
        table: pd.DataFrame
            Table containing NOvA data.
        """
        self.table: pd.DataFrame = table
        self._remove_nansense_data()

        _log(f'Initialised {str(self)}.')

    @staticmethod
    def init_from_copymerge_h5(h5dirs: list[str | pathlib.Path]) -> 'NOvAData':
        """\
        Alternative `NOvAData` constructor.

        Args
        ----
        h5dir: list
            List of copymerged HDF5 files.

        Returns
        -------
        NoVAData
            An initialised `NOvAData` object.
        """

        tables = list()

        _log('Loading tables from copymerged HDF5 files...')

        for i_dir, h5dir in enumerate(h5dirs):
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
                        vars_ = _VARS_TO_EXTRACT.get(branch, [])

                        if len(vars_) > 0:
                            mini_table = _get_branch_df(
                                h5file=h5file,
                                branch=branch,
                                vars=vars_,
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

            progress = f'{i_dir + 1} / {len(h5dirs)}'
            _log(f'Loaded table from copymerged HDF5 files ({progress}).')

        concat_table = pd.concat(tables)

        return NOvAData(table=concat_table)

    @staticmethod
    def init_from_saved_h5(h5dirs: list[str | pathlib.Path]) -> 'NOvAData':
        """\
        Alternative `NOvAData` constructor used to load table(s) saved using the
        `NOvAData.save_table` method.

        Args
        ----
        h5dir: list
            List of the saved HDF5 files.

        Returns
        -------
        NoVAData
            An initialised `NOvAData` object.
        """
        tables = list()

        _log('Loading tables from saved HDF5 file(s)...')

        for i_dir, h5dir in enumerate(h5dirs):
            tables.append(
                pd.read_hdf(h5dir, key='table')
            )
            progress = f'{i_dir + 1} / {len(h5dirs)}'
            _log(f'Loaded table from saved HDF5 file(s) ({progress}).')
        
        concat_table = pd.concat(tables)

        return NOvAData(table=concat_table)

    def _remove_nansense_data(self) -> None:
        """\
        Helper function: Removes unwanted NaN records in the table.
        """
        # Account for any NaN values in the table
        for column_name, fill_value in _TABLE_COL_FILLNA_MAP.items():
            if column_name in self.table.columns:
                self.table[column_name] = self.table[column_name].fillna(
                    fill_value
                )

        # So, sometimes events can be a bit too messy to reconstruct, and we
        # may encounter some rows the the energy estimator branch which may have
        # NaN values... these can be simply dropped.
        # Notes: This could also be added to the quality cuts...
        self.table = self.table.dropna(
            subset=[
                n for n in self.table.columns if n.startswith('rec.energy')
            ]
        )

    def fill_ana_flags(
            self,
            inplace: bool = False
        ) -> 'NOvAData' | None:  # type: ignore
        """\
        Function fills the table with flags to streamline data analysis.

        Args
        ----
        inplace: bool
            If `True`, this method adds the new columns to this table, if 
            `False` then it returns a new `NOvAData` with the updated table. 
            Defaults to `False`.

        Returns
        -------
        NOvAData | None
            If `inplace` was set to `False` a new `NOvAData` object is returned.
        """
        table_copy = self.table.copy()

        is_nu_event = table_copy['rec.mc.nnu'] > 0

        # Event interaction

        table_copy['ana.mc.flag.isCC'] = (
            table_copy['rec.mc.nu.iscc'] * is_nu_event
        )
        table_copy['ana.mc.flag.isNC'] = (
            (table_copy['ana.mc.flag.isCC'] < 1) * is_nu_event
        )

        # Specific event interactions

        table_copy['ana.mc.flag.isNuMu'] = (
            table_copy['rec.mc.nu.pdg'] == 14
        )
        table_copy['ana.mc.flag.isANuMu'] = (
            table_copy['rec.mc.nu.pdg'] == -14
        )

        table_copy['ana.mc.flag.isNuE'] = (
            table_copy['rec.mc.nu.pdg'] == 12
        )
        table_copy['ana.mc.flag.isANuE'] = (
            table_copy['rec.mc.nu.pdg'] == -12
        )

        table_copy['ana.mc.flag.isNuMuCC'] = (
            table_copy['ana.mc.flag.isNuMu'] 
                * table_copy['ana.mc.flag.isCC']
        )
        table_copy['ana.mc.flag.isANuMuCC'] = (
            table_copy['ana.mc.flag.isANuMu'] 
                * table_copy['ana.mc.flag.isCC']
        )

        table_copy['ana.mc.flag.isNuECC'] = (
            table_copy['ana.mc.flag.isNuE'] 
                * table_copy['ana.mc.flag.isCC']
        )
        table_copy['ana.mc.flag.isANuECC'] = (
            table_copy['ana.mc.flag.isANuE'] 
                * table_copy['ana.mc.flag.isCC']
        )
        
        _log('Filled MC truth flags.')
        
        # Replace with the updated table. This is done to ensure that there
        # are no unfinished operations (due to exceptions) becasue that
        # could result in weird half filled rows/columns.

        if inplace:
            self.table = table_copy
            return

        return NOvAData(table=table_copy)

    def fill_ana_track_kinematics(
            self,
            inplace: bool = False
        ) -> 'NOvAData' | None:  # type: ignore
        """\
        Fill the table with the calculated track kinematics.

        Args
        ----
        inplace: bool
            If `True`, this method adds the new columns to this table, if 
            `False` then it returns a new `NOvAData` with the updated table. 
            Defaults to `False`.

        Returns
        -------
        NOvAData | None
            If `inplace` was set to `False` a new `NOvAData` object is returned.
        """
        table_copy = self.table.copy()

        fd_r_beam_dir = detectors.calculate_beam_direction_at_fd(
            detectors.POINT_S
        )

        table_copy['ana.trk.kalman.tracks.cosBeam'] = (
            table_copy['rec.trk.kalman.tracks.dir.x'] * fd_r_beam_dir[0]
            + table_copy['rec.trk.kalman.tracks.dir.y'] * fd_r_beam_dir[1]
            + table_copy['rec.trk.kalman.tracks.dir.z'] * fd_r_beam_dir[2]
        )

        table_copy['ana.trk.kalman.tracks.PtToPmu'] = (
            1 - table_copy['ana.trk.kalman.tracks.cosBeam']**2
        )**0.5

        table_copy['ana.trk.kalman.tracks.Pmu'] = np.sqrt(
            table_copy['rec.energy.numu.lstmmuon']**2 - (105.658E-3)**2
        )
        table_copy['ana.trk.kalman.tracks.Pt'] = (
            table_copy['ana.trk.kalman.tracks.PtToPmu']
            * table_copy['ana.trk.kalman.tracks.Pmu']
        )

        table_copy['ana.trk.kalman.tracks.Qsquared'] = (
            2 * table_copy['rec.energy.numu.lstmnu']
            * (
                table_copy['rec.energy.numu.lstmmuon']
                - table_copy['ana.trk.kalman.tracks.Pmu']
                * table_copy['ana.trk.kalman.tracks.cosBeam']
            )
            - (105.658E-3)**2
        )
        table_copy['ana.trk.kalman.tracks.W'] = (
            (
                0.938272**2 + 2 * 0.938272 * (
                    table_copy['rec.energy.numu.lstmnu']
                    - table_copy['rec.energy.numu.lstmmuon']
                ) - table_copy['ana.trk.kalman.tracks.Qsquared']
            )**0.5
        )

        _log('Filled track kinematics.')
    
        if inplace:
            self.table = table_copy
            return

        return NOvAData(table=table_copy)

    def fill_categorical(
            self,
            inplace: bool = False
        ) -> 'NOvAData' | None:
        """\
        Function fills the table with categorical columns such as the event 
        type - particularly useful for classification problems.

        Args
        ----
        inplace: bool
            If `True`, this method adds the new columns to this table, if 
            `False` then it returns a new `NOvAData` with the updated table. 
            Defaults to `False`.

        Returns
        -------
        NOvAData | None
            If `inplace` was set to `False` a new `NOvAData` object is returned.
        """
        table_copy = self.table.copy()

        table_copy['ana.cat.event_type'] = (
            self.table['ana.mc.flag.isNuMuCC'] 
            + 2 * self.table['ana.mc.flag.isANuMuCC'] 
            + 3 * self.table['ana.mc.flag.isNuECC'] 
            + 4 * self.table['ana.mc.flag.isNC']
            # 0 := Other background e.g., cosmic events...
        )

        _log('Filled categorical data.')

        if inplace:
            self.table = table_copy
            return

        return NOvAData(table=table_copy)

    def apply_transforms(
            self,
            transforms: list[str],
            inplace: bool = False
        ) -> 'NOvAData' | None:
        """\
        Apply transformations using given functions.
    
        Args
        ----
        transforms: list[str]
            List of names of the transformation functions from the 
            `src/transforms.py` module.
        inplace: bool
            If `True`, this method applies the transformations to this table, if 
            `False` then it returns a new `NOvAData` with the applied 
            transformations. Defaults to `False`.

        Returns
        -------
        NOvAData | None
            If `inplace` was set to `False` a new `NOvAData` object is returned.
        """
        table_copy = self.table.copy()

        for tf_name in transforms:
            tf = getattr(transforms_, tf_name, None)
            if tf and callable(tf):
                table_copy = tf(table_copy)
                _log(
                    f'Applied a transform which {transforms_.get_tf_info(tf)}'
                )
            else:
                _log(f'Unable to apply transform `{tf_name}`.')

        if inplace:
            self.table = table_copy
            return

        return NOvAData(table=table_copy)

    def reset_index(self, inplace: bool = False) -> 'NOvAData' | None:
        """\
        Resets the index from `MutiIndex` based on the 'run', 'subrun', etc. to
        an simple `Index` - can save some memory, and make it easier to ac.
        """
        table = self.table.reset_index(drop=True, inplace=inplace)

        if (not inplace) and table is not None:
            return NOvAData(table=table)

    def train_test_split(
            self,
            x_cols: list[str],
            y_cols: list[str],
            test_size: float = 0.2,
            shuffle: bool = False,
            random_state: np.random.RandomState | None = None
        ) -> dict[str, pd.DataFrame | pd.Series]:
        """\
        Wraps behaviour of Sci-kit Learn's `train_test_split`.

        Args
        ----
        x_cols: list[str]
            Names of the features (x-variable - used for training).
        y_cols: list[str]
            Name of the (label) data (y-variable - that will be predicted).
        test_size: float
            Percentage of the table events used for testing. Defaults to 0.2.
        shuffle: bool
            Shuffles the table rows if `True`. Defaults to `False`.
        random_state: np.random.RandomState | None
            Random state used for shuffling the data. Defaults to `None`.
        
        Returns
        -------
        dict[str, pd.DataFrame | pd.Series] 
        | tuple[dict[str, pd.DataFrame | pd.Series], StandardScaler]
            A dictionary containing the x, y training and testing data indicated
            by the keys: 'XTrain', 'YTrain', 'XTest', and 'YTest'.
        """
        table = self.table.reset_index(drop=True).copy()  # type: ignore

        train_table, test_table = train_test_split(
            table,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

        y_train = train_table[y_cols]
        y_test = test_table[y_cols]

        return {
            'XTrain': train_table[x_cols],
            'YTrain': y_train,
            'XTest': test_table[x_cols],
            'YTest': y_test
        }

    def save_table(self, h5dir: str | pathlib.Path) -> None:
        """\
        Saves the current table as an HDF5 file.

        Args
        ----
        h5dir: str | pathlib.Path
            Filename (optionally with an included path/directory) for saving the
            table. Must end with '.h5' to indicate a file in HDF5 format!
        """
        self.table.to_hdf(
            path_or_buf=h5dir,
            key='table',
            format='fixed',  # For fast read/write operations.
            data_columns=True
        )

        _log('Saved table to HDF5.')

    def get_scalable_vars(self) -> list[str]:
        """
        Gets the variables (columns) which are scalable in this dataset.

        Returns
        -------
        list[str]
            List of the column names that are scalable.
        """
        cols_set = set(self.table.columns)
        return list(cols_set.intersection(VARS_TO_SCALE))

    def copy(self) -> 'NOvAData':
        """
        Returns a copy of this `NOvAData` object.

        Returns
        -------
        NOvAData
            Copied `NOvAData` object.
        """
        return NOvAData(table=self.table.copy(deep=True))

    def __str__(self) -> str:
        n_features = len(self.table.columns)
        return f'NOvAData(features={n_features}, events={len(self.table):_})'
    
    def __repr__(self) -> str:
        return str(self)
