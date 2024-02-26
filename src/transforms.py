"""\
src / transforms.py

--------------------------------------------------------------------------------

Aditya Marathe

"""

from typing import Callable as _Callable

import numpy as _np

import pandas as _pd


def get_tf_info(tf: _Callable[..., _pd.DataFrame], lower: bool = True) -> str:
    """\
    Get information about the transformation being made, given that the function
    has a docstring.

    Args
    ----
    tf: Callable[..., pd.DataFrame]
        Any function that behaves like a transform and returns a `DataFrame`.
    lower: bool
        Make the first character lowercase for embedding.
    """
    output = '[No information found, please add a docstring for this transform]'

    if tf.__doc__:
        output = tf.__doc__.split('\n')[-2].removeprefix('    ')

        if lower:
            output = output[0].lower() + output[1:]

        if output[-1] != '.':
            output = output + '.'  # Puncuation is important!

    return output


def tf_290124_positive_energies(df: _pd.DataFrame) -> _pd.DataFrame:
    """\
    Transform: 29/01/2024
    ---------------------
    Cuts out the negative energies caused by issues with the reco. models.
    """
    targets = [
        'rec.energy.numu.hadclust.calE',
        'rec.energy.numu.hadtrkE',
        'rec.energy.numu.lstmmuon',
        'rec.energy.numu.lstmnu',
    ]

    df_copy = df.copy()

    for target in targets:
        df_copy = df_copy[df_copy[target] > 0.]

    return df_copy


def tf_290124_numu_energy(df: _pd.DataFrame) -> _pd.DataFrame:
    """\
    Transform: 29/01/2024
    ---------------------
    Restricts the muon-neutrino energy to be between 0 and 5 GeV.
    """
    targets = [
        'trueEnu',
        'rec.energy.numu.E',
        'rec.energy.numu.lstmnu'
    ]

    df_copy = df.copy()

    for target in targets:
        df_copy = df_copy[(df_copy[target] > 0.) & (df_copy[target] < 5.)]

    return df_copy


def tf_290124_valid_pid(df: _pd.DataFrame) -> _pd.DataFrame:
    """\
    Transform: 29/01/2024
    ---------------------
    Ensures that the PID score is between 0 and 1.
    """
    targets = [
        'rec.sel.cvnloosepreselptp.cosmicid',
        'rec.sel.cvnloosepreselptp.ncid',
        'rec.sel.cvnloosepreselptp.nueid',
        'rec.sel.cvnloosepreselptp.numuid',
        'rec.sel.remid.pid',
        'rec.sel.scann.scpid'
    ]

    df_copy = df.copy()

    for target in targets:
        df_copy[(df_copy[target] < 0) & (df_copy[target] > 1)] = 0.
   
    return df_copy


def tf_050224_max_prongs(
        df: _pd.DataFrame,
        max_prongs: int = 5
    ) -> _pd.DataFrame:
    """\
    Transform: 05/02/24
    -------------------
    Limits the number of max prongs to a certain amount.
    """
    df_copy = df.copy()

    return df_copy[df_copy['rec.trk.kalman.ntracks'] <= max_prongs]


def tf_050224_add_padding(
        df: _pd.DataFrame,
        max_prongs: int = 5
    ) -> _pd.DataFrame:
    """\
    Transform 05/02/24
    ------------------
    Note: This transform can only be called after `tf_050224_add_padding`.
    Adds padding ([0, 0, ...]) to make prong variables of equal length.
    """
    df_copy = df.copy()

    targets = [
        'rec.trk.kalman.tracks.dir.x',
        'rec.trk.kalman.tracks.dir.y',
        'rec.trk.kalman.tracks.dir.z',
        'rec.trk.kalman.tracks.start.x',
        'rec.trk.kalman.tracks.start.y',
        'rec.trk.kalman.tracks.start.z',
        'rec.trk.kalman.tracks.len',
        'rec.trk.kalman.tracks.muonid',
        'rec.trk.kalman.tracks.rempid',
        'rec.trk.kalman.tracks.nhit',
        'rec.trk.kalman.tracks.nhitx',
        'rec.trk.kalman.tracks.nhity',
        'rec.trk.kalman.tracks.calE',
        'rec.trk.kalman.tracks.overlapE',
        'rec.trk.kalman.tracks.nplane',
        'rec.trk.kalman.tracks.maxplanecont',
        'ana.trk.kalman.tracks.cosBeam',
        'ana.trk.kalman.tracks.PtToPmu',
        'ana.trk.kalman.tracks.Pt',
        'ana.trk.kalman.tracks.Qsquared',
        'ana.trk.kalman.tracks.W'
    ]


    def add_padding(row):
        for target in targets:
            padding = max_prongs - len(row[target])
            row[target] = _np.concatenate((row[target], [0] * padding))
        return row

    return df_copy.apply(add_padding, axis=1)  # type: ignore


def tf_120224_numu_energy(df: _pd.DataFrame) -> _pd.DataFrame:
    """\
    Transform: 12/02/2024
    ---------------------
    Restricts the muon-neutrino energy to be between 0 and 5 GeV.
    """
    targets = [
        'rec.mc.nu.E',
        'rec.energy.numu.E',
        'rec.energy.numu.lstmnu'
    ]

    df_copy = df.copy()

    for target in targets:
        df_copy = df_copy[(df_copy[target] > 0.) & (df_copy[target] < 5.)]

    return df_copy


def tf_120224_first_prong(df: _pd.DataFrame) -> _pd.DataFrame:
    """\
    Transform 12/02/24
    ------------------
    Only keeps data for the first prong.
    """
    df_copy = df.copy()
   
    targets = [
        'ana.trk.kalman.tracks.cosBeam', 
	    'ana.trk.kalman.tracks.PtToPmu',
        'ana.trk.kalman.tracks.Pt',
        'ana.trk.kalman.tracks.Qsquared',
        'ana.trk.kalman.tracks.W'
    ]

    for target in targets:
        df_copy[target] = df_copy[target].apply(
            lambda row: row[0] if len(row) > 0 else float('nan')
        )

    return df_copy
