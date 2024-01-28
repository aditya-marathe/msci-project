"""
Functions used in the Initial Analysis notebook.
"""

# TODO: Delete this file...

from typing import Dict
from typing import List
from typing import Callable
from typing import Optional
from typing import Any
from typing import TypeAlias

import warnings

import pandas as pd


__all__ = [
    'get_event_from_pid_score',
    'print_event_info',
    'print_cut_flow',
    'draw_plot_info'
]


warnings.warn('This file is to be deleted soon...')


_CUT_FLOW = """\
== Cut Flow ===============================================
|
|   Cut         | True Sig | True NC  | Cosmic  | Total
|   ----------- | -------- | -------- | ------- | -------
|   Raw         | {0:8} | {5:8} | {10:6.1E} | {15:6.1E}
|   Quality     | {1:8} | {6:8} | {11:6.1E} | {16:6.1E}
|   Containment | {2:8} | {7:8} | {12:6.1E} | {17:6.1E}
|   Cosmic Rej. | {3:8} | {8:8} | {13:6.1E} | {18:6.1E}
|   NuMu CNV    | {4:8} | {9:8} | {14:6.1E} | {19:6.1E}
|
===========================================================
"""

_EVENT_INFO_STR = """\
== Event Info =============================================
|
|   Flavour                     {0}
|   Interaction                 {1}
|
|   Nu Energy                   {2:0.3f} GeV
|   Cos(Theta)                  {3:0.3f} deg
|
|   Reconstructed:
|
|       Nu Energy               {4:0.3f} GeV
|       Q-squared               {5:0.3f} GeV
|       Hadronic Inv. Mass      {6:0.3f} GeV
|       # of Hadronic Hits      {7}
|
|   LSTM Predictions:
|
|       Nu Energy               {8:0.3f} GeV
|       Muon Energy             {9:0.3f} GeV
|
|   Regression CVN Prediction:
|
|       Hadronic Energy         {10:0.3f} GeV
|
|   Cuts:
|
|       Quality                 {11}
|       Containment             {12}
|       Pre-selection CVN       ------>  Likely {13} ?
|       Cosmic Rejection        {14}
|       Hadronic (NuMu) CNV     {15}
| 
===========================================================
"""

# =============================================================================

# Misc. Funcs

def get_event_from_pid_score(data: pd.DataFrame) -> pd.Series:
    """\
    Gets the type of event by calculating the max PID score for an event.

    Args
    ----
    data: pd.DataFrame
        A sample of the "mini" dataset.
    """
    temp_df = pd.DataFrame(
        {
            'Cosmic': data['rec.sel.cvnloosepreselptp.cosmicid'],
            'NC'    : data['rec.sel.cvnloosepreselptp.ncid'],
            'NuE'   : data['rec.sel.cvnloosepreselptp.nueid'],
            'NuMu'  : data['rec.sel.cvnloosepreselptp.numuid'],
            'NuTau' : data['rec.sel.cvnloosepreselptp.nutauid']
        }
    )

    for event_name in temp_df.columns:
        # Corrects corrupted event data
        temp_df[event_name][(temp_df[event_name] <= 0.) & (temp_df[event_name] >= 1.)] = 0.

    return temp_df.idxmax(axis=1)


def print_event_info(event: pd.Series) -> None:
    """\
    Prints some of the important details of the event.

    Args:
    -----
    event: pd.Series
        Event data. This is simply a row/record of the loaded `pd.DataFrame`.
    """
    flavour = 'Unknown'
    current = ''
    
    if event['isCC']:
        current = 'CC'

    if event['isNC']:
        current = 'NC'
    
    anti_prefix = 'Anti-' if event['pdgAll'] < 0 else ''

    if event['isNotNu']:
        flavour = 'N/A'
    else:
        if abs(event['pdgAll']) == 12:
            flavour = 'NuE'
        elif abs(event['pdgAll']) == 14:
            flavour = 'NuMu'
        elif abs(event['pdgAll']) == 16:
            flavour = 'NuTau'

    mode_ref_table: Dict[int, str] = {
        -1 : 'Unknown',      
        0  : 'QE',
        1  : 'Res',
        2  : 'DIS',
        3  : 'Coh',
        4  : 'CohElastic',
        5  : 'ElectronScattering',
        6  : 'IMDAnnihilation',
        7  : 'InverseBetaDecay',
        8  : 'GlashowResonance',
        9  : 'AMNuGamma',
        10 : 'MEC',
        11 : 'Diffractive',
        12 : 'EM',
        13 : 'WeakMix'
    }

    quality_cuts = event['numuBasicQuality'] and event['numuQuality']

    print(
        _EVENT_INFO_STR.format(
            #
            anti_prefix + flavour,
            mode_ref_table[event['modeAll']] + ' ' + current,

            # True values
            event['trueEnu'],
            event['cosBeamFirst'],

            # Reconstructed
            event['rec.energy.numu.E'],
            event['recoq2'],
            event['recow'] if str(event['recow']) != 'nan' else 0.,
            event['rec.energy.numu.hadclust.nhit'],

            # Predictions
            event['rec.energy.numu.lstmnu'],
            event['rec.energy.numu.lstmmuon'],
            event['rec.energy.numu.regcvnhadE'],

            # Cuts
            'Passed' if quality_cuts else 'Failed',
            'Passed' if event['numucontain'] else 'Failed',
            event['Event'],
            'Passed' if event['numucosrej'] else 'Failed',
            'Passed' if event['numu2020pid'] else 'Failed'
        )
    )


# TODO: Improve this func...

def print_cut_flow(data: pd.DataFrame) -> None:
    """\
    Prints the cut flow for signal NuMu events.

    Args
    ----
    data: pd.DataFrame
        Full dataset or sample of the dataset.
    """
    quality, containment, cosmic_rej, numu_cvn = calc_cuts(data)

    print(
        _CUT_FLOW.format(
            len(data[data['isNuMu']]),
            len(quality[quality['isNuMu']]),
            len(containment[containment['isNuMu']]),
            len(cosmic_rej[cosmic_rej['isNuMu']]),
            len(numu_cvn[numu_cvn['isNuMu']]),

            len(data[data['isNC']]),
            len(quality[quality['isNC']]),
            len(containment[containment['isNC']]),
            len(cosmic_rej[cosmic_rej['isNC']]),
            len(numu_cvn[numu_cvn['isNC']]),

            len(data[
                data['rec.sel.cvnloosepreselptp.event'] == 'Cosmic'
            ]),
            len(quality[
                quality['rec.sel.cvnloosepreselptp.event'] == 'Cosmic'
            ]),
            len(containment[
                containment['rec.sel.cvnloosepreselptp.event'] == 'Cosmic'
            ]),
            len(cosmic_rej[
                cosmic_rej['rec.sel.cvnloosepreselptp.event'] == 'Cosmic'
            ]),
            len(numu_cvn[
                numu_cvn['rec.sel.cvnloosepreselptp.event'] == 'Cosmic'
            ]),

            len(data),
            len(quality),
            len(containment),
            len(cosmic_rej),
            len(numu_cvn)
        )
    )


def draw_plot_info(ax) -> Callable[[pd.Series, Any, str], None]:
    """\
    Draws simple stats beside the histogram axes - similar to ROOT.
    
    Args:
    -----
    ax: mpl.Axes
        Axes object of the plot (e.g. initialised via `plt.subplots`).

    Returns:
    --------
    Callable[[pd.Series, str], None]
        Function which updates the plot axes.
    """
    row = 0  # Updates the offset...
             # Plotting too many not reccomended!

    col = 0

    text_str = "{0}:\nEntries  {1}\nMean     {2:0.3f}\nStd Dev {3:0.3f}"

    def update(data: pd.Series, 
               bar_container, 
               label: str) -> None:
        """\
        Function updates the plot axes according to given data.

        Args:
        -----
        data: pd.Series
            Data plotted on the histogram. This is simply the column of the `pd.DataFrame`.

        bar_container: mpl.BarContainer
            The last thing that is retured by `plt.hist` or `ax.hist`.

        label: str
            Label of the histogram - will be drawn like a plot legend.
            
        """
        nonlocal row
        nonlocal col

        patch = bar_container.patches[0]

        bbox = dict(
            boxstyle='round', 
            facecolor=patch.get_facecolor(),
            alpha=patch.get_alpha()
        )

        ax.text(
            1.05 + 0.24 * col, 1.0 - 0.24 * row,
            text_str.format(
                label,
                len(data),
                data.mean(), 
                data.std()
            ),
            transform=ax.transAxes, 
            verticalalignment='top', 
            bbox=bbox
        )

        row += 1

        if row < 0:
            row = 0
            col += 1
    
    return update
