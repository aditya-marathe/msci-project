"""\
src / ana / plotting.py
--------------------------------------------------------------------------------

Aditya Marathe
"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement

from typing import Any
from typing import Literal

import matplotlib as mpl
import matplotlib.pyplot as plt

from ana.data import NOvAData
from ana.spectrum import Spectrum


_PLOT_FORMATTING: dict[str, Any] = {
    'MAJOR_TICKS_SIZE': 7.,
    'MINOR_TICKS_SIZE': 3.5,
    'TICK_FONT_SIZE': 12,
    'TICK_FONT_FAMILY': 'CMU Serif'
}


def init_global_plotting_style() -> None:
    """\
    
    """
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 14


def init_axes_formatting(ax: mpl.axes.Axes) -> None:
    """\
    Sets the formatting of the plot axes using the specifications in the
    `PLOT_FORMATTING` dictionary.
    
    Parameters
    ----------
    ax : axes.Axes
        Target Axes.
    """
    # Axis ticks
    ax.tick_params(
        which='major',
        right=True,
        left=True,
        top=True,
        bottom=True,
        direction='in',
        axis='both',
        size=7
    )
    ax.tick_params(
        which='minor',
        right=True,
        left=True,
        top=True,
        bottom=True,
        direction='in',
        axis='both',
        size=3.5
    )
    ax.minorticks_on()

    # Label formatting
    if ax.get_xlabel() is not None:
        ax.set_xlabel(ax.get_xlabel(), labelpad=15)

    if ax.get_ylabel() is not None:
        ax.set_ylabel(ax.get_ylabel(), labelpad=20)

    if ax.get_title() is not None:
        title_obj = ax.set_title(ax.get_title(), pad=10)
        title_obj.set_position([0.1, 1.05])

    ax.text(
        0.875,
        1.05,
        'NOvA Simulation',
        color='gray',
        fontsize=11,
        horizontalalignment='center',
        verticalalignment='center',
        transform = ax.transAxes
    )

    # Legend
    if ax.get_legend() is not None:
        legend = ax.legend(borderpad=1)
        legend.get_frame().set_linewidth(0)

    # Tight layout
    plt.tight_layout()


class PlotStyles:
    """\
    
    """
    STEP = 'step'
    HIST = 'hist'


def plot_spectrum(
        ax: mpl.axes.Axes,
        data: NOvAData,
        spectrum: Spectrum,
        label: str,
        style: Literal['steps'] = 'steps'
    ) -> None:
    """\
    Plot a spectrum.
    """
    (values, edges, centres), pot = spectrum(data=data)

    match style:
        case 'style':
            ax.step(edges[1:], values, label=label)
        case 'hist':
            ax.bar(
                centres,
                values,
                width=edges[:-1] - edges[1:],
                align='center',
                alpha=0.6,
                label=label
            )
            ax.step(edges[1:], values)
