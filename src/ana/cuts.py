"""\
src / ana / cuts.py
--------------------------------------------------------------------------------

Aditya Marathe
"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

__all__ = [
    'CutFuncType',
    'Cuts',
    'cut_numu_detector_quality',
    'cut_numu_data_quality',
    'cut_numu_2020_pid',
    'cut_numu_containment',
    'cut_numu_cosmic_rej',
    'cut_numu_veto'
]

from typing import Callable
from typing import TypeAlias

import pandas as pd

CutFuncType: TypeAlias = Callable[..., pd.Series]


# Predefined cut functions
# ---------------------------------------------------------------------------- #


def cut_numu_detector_quality(df: pd.DataFrame) -> pd.Series:
    """
    Cut / NuMu / Detector Quality

    Detector quality cuts to ensure that detector is functioning correctly,
    therefore, the collected data is reliable.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    a = df['rec.energy.numu.trkccE'] > 0
    b = df['rec.sel.remid.pid'] > 0
    c = df['rec.slc.nhit'] > 20
    d = df['rec.slc.ncontplanes'] > 4
    e = df['rec.trk.cosmic.ntracks'] > 0
    return a & b & c & d & e


def cut_numu_data_quality(df: pd.DataFrame) -> pd.Series:
    """\
    Cut / NuMu / Data Quality

    Several models (ML or otherwise) are used in the reconstruction process. 
    Sometimes, the data quality is not poor due to mis-predictions or more
    generally anamolous data that should be ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    a = df['rec.energy.numu.lstmnu'] > 0
    # b = df['rec.sel.remid.pid'] > 0
    # c = df['rec.slc.nhit'] > 20
    # d = df['rec.slc.ncontplanes'] > 4
    # e = df['rec.trk.cosmic.ntracks'] > 0
    return a


def cut_numu_2020_pid(df: pd.DataFrame) -> pd.Series:
    """\
    Cut / NuMu / CVN PID Score (2020)

    A Convolutional Visual Network (CVN) Particle Identifier (PID) score is the
    probability that a given particle is of a certain class. These classes
    include: NuE, NuMu, NuTau, NC, and Cosmic. Other PID scores such at REMID
    are used as well to cut out non-NuMu events.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    a = df['rec.sel.remid.pid'] > 0.30
    b = df['rec.sel.cvnloosepreselptp.numuid'] > 0.80
    return a & b


def cut_numu_containment(df: pd.DataFrame) -> pd.Series:
    """\
    Cut / NuMu / Containment

    The containment cut (the cut we are interested in for this project) is 
    applied to ensure tha the particle tracks are well contained in the detector
    volume. This volume is called known as the 'fiducial volume' in the 
    publications.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    a = df['rec.sel.contain.kalfwdcell'] > 6
    b = df['rec.sel.contain.kalbakcell'] > 6
    c = df['rec.sel.contain.cosfwdcell'] > 5
    d = df['rec.sel.contain.cosbakcell'] > 7
    e = df['rec.slc.firstplane'] > 2
    f = (896 - df['rec.slc.lastplane']) > 3
    return  a & b & c & d & e & f


def cut_numu_cosmic_rej(df: pd.DataFrame) -> pd.Series:
    """\
    Cut / NuMu / Cosmic Rejection

    This cut is applied to isolate cosmic-like events from the data.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    return df['rec.sel.cosrej.numucontpid2020fhc'] > 0.45


def cut_numu_veto(df: pd.DataFrame) -> pd.Series:
    """\
    Cut / NuMu / Veto

    The three-flavour veto was introduced to discriminate between background 
    noise and signal-like data.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    return df['rec.sel.veto.keep'] > 0


# `Cuts` class
# ---------------------------------------------------------------------------- 


class Cuts:
    """\
    Cuts
    
    A class used to define and apply cuts to NOvA simulated data stored as a
    Pandas `DataFrame`.
    """

    verbose: bool = True

    def __init__(self) -> None:
        """\
        Initialises a `Cuts` object with no pre-defined cuts.
        """
        self._cuts: dict[str, CutFuncType] = dict()

    @staticmethod  # --> Should this be a cmethod?
    def init_nova_cuts() -> 'Cuts':
        """\
        Alternative constructor for the `Cuts` class. Returns a `Cuts` object 
        with the most recent NOvA cuts, which are: Detector Quality, Data 
        Quality, CVN PID Score, Containment, Cosmic Rej., Veto.
        
        Returns
        -------
        Cuts
            A `Cuts` object with pre-defined NOvA cuts.
        """
        cuts = Cuts()

        # Define the NOvA cuts.
        cuts.define_cut(
            name='Detector Quality',
            cut_func=cut_numu_detector_quality
        )
        cuts.define_cut(
            name='Data Quality',
            cut_func=cut_numu_data_quality
        )
        cuts.define_cut(
            name='CVN PID Score',
            cut_func=cut_numu_2020_pid
        )
        cuts.define_cut(
            name='Containment',
            cut_func=cut_numu_containment
        )
        cuts.define_cut(
            name='Cosmic Rej.',
            cut_func=cut_numu_cosmic_rej
        )
        cuts.define_cut(
            name='Veto',
            cut_func=cut_numu_veto
        )

        return cuts

    def reset(self) -> None:
        """\
        Rests the defined cuts.

        Notes
        -----
        Warning: This clears all the defined cuts! Use with caution!
        """
        self._cuts.clear()

    def define_cut(self, name: str, cut_func: CutFuncType) -> None:
        """\
        Define a new cut.

        Parameters
        ----------
        name : str
            The name of the cut.
        
        cut_func : CutFuncType
            The cut function.
        """
        self._cuts[name] = cut_func

    def get_cut(
            self,
            name: str,
            df: pd.DataFrame,
            passed: bool = True
        ) -> pd.Series:
        """
        """
        if passed:
            return self._cuts[name](df)
        
        return ~self._cuts[name](df)

    def apply_cut(
            self,
            name: str,
            df: pd.DataFrame,
            passed: bool = True
        ) -> pd.DataFrame:
        """\
        Apply a certain cut.
        """
        result = df[self.get_cut(name, df, passed)]

        if Cuts.verbose:
            print(
                f'Cuts     | Applied \'{name}\' cut ({len(df):_} -> '
                f'{len(result):_} events).'
            )

        return result

    def apply_cuts(
            self, 
            names: list[str], 
            df: pd.DataFrame, 
            passed: list[bool] | None = None
        ) -> pd.DataFrame:
        """\
        Apply a list of cuts.
        """
        if passed is None:
            passed = [True] * len(names)
        
        if len(passed) != len(names):
            # TODO: Write exception message
            raise ValueError('...')

        result = df

        for i, cut_name in enumerate(names):
            result = self.apply_cut(name=cut_name, df=result, passed=passed[i])

        return result

    def apply_all_cuts(
            self, 
            df: pd.DataFrame, 
            except_: list[str] | None = None, 
            passed: list[bool] | None = None
        ) -> pd.DataFrame:
        """\
        Apply all the defined cuts to a Pandas `DataFrame`.
        """
        result = df

        if except_ is None:
            except_ = list()

        if passed is None:
            passed = [True] * len(self._cuts)
        
        if len(passed) != len(self._cuts):
            # TODO: Write exception message
            raise ValueError('...')

        for i, cut_name in enumerate(self._cuts):
            if not (cut_name in except_):
                result = self.apply_cut(name=cut_name, df=result, passed=passed[i])

        return result

    def print_all_cuts(self) -> None:
        """
        """
        print('Defined Cuts\n------------')
        for cut in self._cuts:
            print('\t' + cut)

    def __str__(self) -> str:
        return f'Cuts(n_cuts={len(self._cuts)})'
    
    def __repr__(self) -> str:
        return str(self)
