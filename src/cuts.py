"""
src / cuts.py
--------------------------------------------------------------------------------

Aditya Marathe

Script contains the `Cuts` class used to define and apply cuts to NOvA simulated
data. Also, contains pre-defined cut functions that can be either loaded into a 
`Cuts` instance or used as is.
"""

# TODO: Add more detail to some of the docstrings here...

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


# Predefined cut functions (for the really mini dataset only!)
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
    return df['numuQuality']


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
    # Original code used numu.tracccE, but I think what I have done 
    # here is basically the same thing... (hopefully)
    # a = df['rec.energy.numu.lstmnu'] > 0
    # b = df['rec.sel.remid.pid'] > 0
    # c = df['rec.slc.nhit'] > 20
    # d = df['rec.slc.ncontplanes'] > 4
    # e = df['rec.trk.cosmic.ntracks'] > 0
    # return  a & b & c & d & e  # This gives me some problems...
    return df['numuBasicQuality']


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
    # Most of the keys from the original code are not in the "mini"
    # dataset! So I am going to cheat a little...
    a = df['numucontain']
    b = df['rec.sel.contain.cosbakcell'] > 7
    return  a & b


def cut_numu_cosmic_rej(df: pd.DataFrame) -> pd.Series:
    """\
    Cut / NuMu / Cosmic Rejection

    This cut is applied to isolate cosmic-like events from the data.

    Parameters
    ----------
    df : pd.DataFrame
        Target `DataFrame`.
    """
    # Again cheating a little bit because the keys are not avalible.
    return df['numucosrej']


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
    return df['3flavourveto']


# `Cuts` class
# ---------------------------------------------------------------------------- 


class Cuts:
    """\
    Cuts
    
    A class used to define and apply cuts to NOvA simulated data stored as a
    Pandas `DataFrame`.
    """
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

    def apply_cut(
            self, name: str, 
            df: pd.DataFrame, 
            passed: bool = True
        ) -> pd.DataFrame:
        """\
        Apply a certain cut.
        """
        cuts_list = df.attrs.get('_applied_cuts_list')

        if passed:
            if cuts_list:
                cuts_list.append(name)
            
            return df[self._cuts[name](df)]

        if cuts_list:
            cuts_list.append('Not ' + name)
        
        return df[~self._cuts[name](df)]

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
        print(''.join(self._cuts))
