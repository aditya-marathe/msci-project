from typing import Callable
from typing import TypeAlias

import pandas as pd

df_rename_map = {
    # Cuts (not used for training / testing)
    'numuBasicQuality': 'Cuts.DetectorQuality',
    'numuQuality': 'Cuts.DataQuality',
    'numucontain': 'Cuts.Containment',
    'numu2020pid': 'Cuts.PIDScore',
    'numucosrej': 'Cuts.CosmicRej',
    '3flavourveto': 'Cuts.Veto',
    'numufull': 'Cuts.Full',
    # MC Truth
    'trueEnu': 'MCTruth.ENu',
    # Derived Quant.
    'pmu': 'Derived.PMu',
    'pt': 'Derived.TransPMu',
    'ptp': 'Derived.TransPtoPRatio',
    'recoq2': 'Derived.QSquared',
    # Reconstructed Qant.
    'cosBeamFirst': 'Reconstructed.CosBeamFirst',
    'rec.energy.numu.hadclust.calE': 'Reconstructed.HadClust.CalE',
    'rec.energy.numu.hadclust.nhit': 'Reconstructed.HadClust.NHits',
    'rec.energy.numu.hadtrkE': 'Reconstructed.HadClust.TrackE',
    'rec.energy.numu.lstmmuon': 'Reconstructed.LSTM.EMu',
    'rec.energy.numu.lstmnu': 'Reconstructed.LSTM.ENu',
    # Pre-selection CVN
    'rec.sel.cvnloosepreselptp.cosmicid': 'Selection.PreSelCVN.CosmicID',
    'rec.sel.cvnloosepreselptp.ncid': 'Selection.PreSelCVN.NCID',
    'rec.sel.cvnloosepreselptp.nueid': 'Selection.PreSelCVN.NuEID',
    'rec.sel.cvnloosepreselptp.numuid': 'Selection.PreSelCVN.NuMuID',
    # Selection and PIDs
    'rec.sel.contain.cosbakcell': 'Selection.Cosmic.BackCell',
    'rec.sel.remid.pid': 'Selection.REMID.PID',
    'rec.sel.scann.scpid': 'Selection.SCPID',
    # Time Slice
    'rec.slc.ncontplanes': 'TimeSlice.NContPlanes',
    'rec.slc.nhit': 'TimeSlice.NHits',
    # Reconstructed Tracks
    'rec.trk.kalman.ntracks': 'Track.Kalman.NTracks'
}


def rename_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    return df.rename(columns=df_rename_map)


def numu_quality_cut(df: pd.DataFrame) -> pd.Series:
    return df['numuQuality']


def numu_data_quality_cut(df: pd.DataFrame) -> pd.Series:
    # Original code used numu.tracccE, but I think what I have done 
    # here is basically the same thing... (hopefully)
    a = df['rec.energy.numu.lstmnu'] > 0
    b = df['rec.sel.remid.pid'] > 0
    c = df['rec.slc.nhit'] > 20
    d = df['rec.slc.ncontplanes'] > 4
    e = df['rec.trk.cosmic.ntracks'] > 0
    # return  a & b & c & d & e  # This gives me some problems...
    return df['numuBasicQuality']


def numu_2020_pid_cut(df: pd.DataFrame) -> pd.Series:
    a = df['rec.sel.remid.pid'] > 0.30
    b = df['rec.sel.cvnloosepreselptp.numuid'] > 0.80
    return a & b


def numu_containment_cut(df: pd.DataFrame) -> pd.Series:
    # Most of the keys from the original code are not in the "mini"
    # dataset! So I am going to cheat a little...
    a = df['numucontain']
    b = df['rec.sel.contain.cosbakcell'] > 7
    return  a & b


def numu_cosmic_rejection_cut(df: pd.DataFrame) -> pd.Series:
    # Again cheating a little bit because the keys are not avalible.
    return df['numucosrej']


def numu_fd_veto_cut(df: pd.DataFrame) -> pd.Series:
    return df['3flavourveto']


CutFuncType: TypeAlias = Callable[..., pd.Series]


class Cuts:
    """\
    Stores and manages cuts made to a Pandas `DataFrame`.
    """
    def __init__(self) -> None:
        """\
        Initialises a `Cuts` object with no pre-defined cuts.
        """
        self._cuts: dict[str, CutFuncType] = dict()

    @staticmethod
    def init_nova_cuts() -> 'Cuts':
        """\
        Initialises a `Cuts` object with basic NOvA cuts.
        """
        my_df = Cuts()
        my_df.define_cut(name='Detector Quality', cut_func=numu_quality_cut)
        my_df.define_cut(name='Data Quality', cut_func=numu_data_quality_cut)
        my_df.define_cut(name='CVN PID Score', cut_func=numu_2020_pid_cut)
        my_df.define_cut(name='Containment', cut_func=numu_containment_cut)
        my_df.define_cut(name='Cosmic Rej.', cut_func=numu_cosmic_rejection_cut)
        my_df.define_cut(name='Veto', cut_func=numu_fd_veto_cut)
        return my_df

    def reset(self) -> None:
        """\
        Rests the defined cuts.
        """
        self._cuts.clear()

    def define_cut(self, name: str, cut_func: CutFuncType) -> None:
        """\
        Define a new cut.
        """
        self._cuts[name] = cut_func

    def apply_cut(self, name: str, df: pd.DataFrame, passed: bool = True) -> pd.DataFrame:
        """\
        Apply a certain cut.
        """
        if passed:
            return df[self._cuts[name](df)]
        
        return df[~self._cuts[name](df)]

    def apply_cuts(self, names: list[str], df: pd.DataFrame, passed: list[bool] | None = None) -> pd.DataFrame:
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

    def apply_all_cuts(self, df: pd.DataFrame, except_: list[str] | None = None, passed: list[bool] | None = None) -> pd.DataFrame:
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
