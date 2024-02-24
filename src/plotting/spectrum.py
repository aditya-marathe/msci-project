"""\
src / plotting / spectrum.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function

__all__ = [
    'PLOT_DIR',
    'Style',
    'SpectrumSubplot'
]

from typing import TYPE_CHECKING
from typing import Literal
from typing import Sequence
from typing import Any

import enum

import pathlib

import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.ana.data import NOvAData
    from src.ana.spectrum import Spectrum


PLOT_DIR = pathlib.Path('./figures/')


def _init_axs_formatting(ax: Axes, fontsize: int) -> None:
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
        labelsize=fontsize,
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

    # ax.text(
    #     0.83,
    #     1.05,
    #     r'NO$\nu$A Simulation',
    #     color='gray',
    #     fontsize=15,
    #     horizontalalignment='center',
    #     verticalalignment='center',
    #     transform = ax.transAxes
    # )

    # Tight layout
    plt.tight_layout()


class Style(enum.Enum):
    """\
    Style
    -------------

    Avalible plotting styles for spectrum plots. 
    """
    HISTOGRAM = enum.auto()
    STEPS = enum.auto()
    L_POINTS = enum.auto()
    S_POINTS = enum.auto()


_Styles = Literal[
    Style.HISTOGRAM,
    Style.STEPS,
    Style.L_POINTS,
    Style.S_POINTS
]
Styles = _Styles | list[_Styles]

class SpectrumSubplot:
    """
    SpectrumSubplot
    ---------------

    Wrapper around Matplotlib `subplots`.
    """
    def __init__(
            self,
            nrows: int = 1,
            ncols: int = 1,
            *,
            fontsize: int = 12,
            sharex: bool | Literal['none', 'all', 'row', 'col'] = False,
            sharey: bool | Literal['none', 'all', 'row', 'col'] = False,
            squeeze: bool = True,
            width_ratios: Sequence[float] | None = None,
            height_ratios: Sequence[float] | None = None,
            subplot_kw: dict[str, Any] | None = None,
            gridspec_kw: dict[str, Any] | None = None,
            **fig_kw
        ):
        """\
        Initialises a `SpectrumSubplot`.

        Args
        ----
        fontsize: int
            Global (i.e., for the legend, labels, etc.) fontsize of the plot.
        *subplots_args, **subplots_kwargs
            Takes the same arguments as the `pyplot.subplots` function.
        """
        self.fontsize = fontsize

        self.fig, self.axs = plt.subplots(
            nrows, ncols,
            sharex=sharex, sharey=sharey,
            squeeze=squeeze,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            **fig_kw
        )

        if np.shape(self.axs):
            self.axs = self.axs.flatten()
        else:
            self.axs = np.asarray([self.axs])

        for ax in self.axs:
            _init_axs_formatting(ax=ax, fontsize=int(self.fontsize))

    def set_ax_labels(
            self,
            ax_idx: int,
            title: str | None = None,
            x_label: str | None = None,
            y_label: str | None = None,
        ) -> None:
        """\
        Set x and y labels for a certain axis.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        title: str | None
            Title of the selected `Axes`. Defaults to `None`.
        x_label: str | None
            The x-axis label. Defaults to `None`.
        y_label: str | None
            The y-axis label. Defaults to `None`.
        """
        self.axs[ax_idx].set_title(
            title,
            fontsize=self.fontsize * 1.5
        )
        self.axs[ax_idx].set_xlabel(
            x_label,
            fontsize=self.fontsize,
            labelpad=15
        )
        self.axs[ax_idx].set_ylabel(
            y_label,
            fontsize=self.fontsize,
            labelpad=15
        )

        plt.tight_layout()

    def set_energy_xy_labels(
            self,
            ax_idx: int,
            bin_width: int | float | None = None
        ) -> None:
        """\
        Sets the x and y axis labels to labels used for energy spectra plots.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        bin_width: int
            Histrogram bin width in GeV. Specify if using a constant bin width.
        """
        y_label = 'Events / Bin'

        if bin_width:
            y_label = f'Events / {bin_width} GeV'
        
        self.set_ax_labels(
            ax_idx=ax_idx,
            x_label=r'$\nu_\mu$ Energy (GeV)',
            y_label=y_label
        )

    def set_energy_residual_xy_labels(
            self,
            ax_idx: int
        ) -> None:
        """\
        Sets the x and y axis labels to labels used for energy residual plots.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        """
        self.set_ax_labels(
            ax_idx=ax_idx,
            x_label=r'$1 - E_{\nu, \text{obs.}} / E_{\nu, \text{exp.}}$',
            y_label='Events / Bin'
        )

    def plot_spectrum(
            self,
            ax_idx: int,
            data: 'NOvAData',
            spectrum: 'Spectrum',
            *,
            label: str | None = None,
            style: Styles = Style.HISTOGRAM,
            alpha: float = 0.8
        ) -> None:
        """\
        Plots the given spectrum.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        data: NOvAData
            The data from which the particular spectrum will be extracted.
        spectrum: Spectrum
            Initialised spectrum for a particular variable in the dataset.
        label: str | None
            Label displayed on the legend.
        style: Styles
            Chosen plotting style. Defaults to `PlottingStyle.HISTOGRAM`.
        alpha: float
            The transparency of the histogram (useful for plotting multiple 
            histograms).
        """
        spectrum_dict = spectrum(data)

        if not isinstance(style, list):
            style = [style]

        if Style.HISTOGRAM in style:
            self.axs[ax_idx].hist(
                spectrum_dict['BinEdges'][:-1],
                bins=spectrum_dict['BinEdges'],
                weights=spectrum_dict['YValues'],
                label=label,
                alpha=alpha
            )
        if Style.STEPS in style:
            self.axs[ax_idx].hist(
                spectrum_dict['BinEdges'][:-1],
                bins=spectrum_dict['BinEdges'],
                weights=spectrum_dict['YValues'],
                label=label,
                histtype='step',
                alpha=alpha
            )
        if Style.S_POINTS in style:
            self.axs[ax_idx].plot(
                spectrum_dict['XValues'],
                spectrum_dict['YValues'],
                '.',
                label=label,
                alpha=alpha
            )
        if Style.L_POINTS in style:
            self.axs[ax_idx].plot(
                spectrum_dict['XValues'],
                spectrum_dict['YValues'],
                'o',
                label=label,
                alpha=alpha
            )

        self.axs[ax_idx].set_xlim(
            np.min(spectrum.binning),
            np.max(spectrum.binning)
        )

    def plot_spectra(
            self,
            ax_idx: int,
            data: 'NOvAData',
            spectra: list['Spectrum'],
            *,
            labels: list[str | None] | None = None,
            style: list[Styles] | None = None,
            alpha: float = 0.8
        ) -> None:
        """\
        Plots the given spectra on the same `Axes`.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        data: NOvAData
            The data from which the particular spectrum will be extracted.
        spectra: list[Spectrum]
            List of initialised spectra.
        labels: list[str | None] | None
            List of labels to be displayed on the legend.
        style: list[Styles]
            Chosen plotting styles. Defaults to `None`.
        alpha: float
            The transparency of the histogram (useful for plotting multiple 
            histograms).
        """
        if labels is None:
            labels = [None] * len(spectra)  # type: ignore

        if style is None:
            style = [Style.HISTOGRAM] * len(spectra)  # type: ignore
        
        for i, spectrum in enumerate(spectra):
            self.plot_spectrum(
                ax_idx=ax_idx,
                data=data,
                spectrum=spectrum,
                label=labels[i],  # type: ignore
                style=style[i],  # type: ignore
                alpha=alpha
            )

    def plot_relative_residuals(
            self,
            ax_idx: int,
            data: "NOvAData",
            obs_var: str,
            exp_var: str,
            binning: npt.NDArray | None = None,
            *,
            label: str | None = None,
            alpha: float = 0.8
        ) -> None:
        """\
        Calculates the relative residuals and plots the histogram.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        data: NOvAData
            The data from which the particular spectrum will be extracted.
        obs_var: str
            The observed variable name (as is in the data table).
        exp_var: str
            The expected variable name (as is in the data table).
        binning: npt.NDArray
            Histogram binning for the residuals. 
        label: str | None
            Label displayed on the legend.
        alpha: float
            The transparency of the histogram (useful for plotting multiple 
            histograms).
        """
        observation = data.table[obs_var]
        expected = data.table[exp_var]
        residuals = 1 - (observation / expected)

        self.axs[ax_idx].hist(
            residuals,
            bins=binning,
            label=label,
            alpha=alpha
        )
        self.axs[ax_idx].set_xlim(-1, 1)

    def _add_legend(self, ax: Axes) -> None:
        """\
        Helper function for adding legends with custom formatting.
        """
        legend = ax.legend(
            borderpad=1,
            fontsize=int(self.fontsize * 1.2)
        )
        legend.get_frame().set_linewidth(0)

    def legend(self, ax_idx: int) -> None:
        """
        Add a legend on a selected axes.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        """
        self._add_legend(ax=self.axs[ax_idx])            

    @staticmethod
    def show() -> None:
        """\
        Uses the Tkinter backend to display the plot in a GUI.
        """
        plt.show()

    def save_plot(
            self,
            figname: str,
            force: bool = False,
            verbose: bool = True
        ) -> None:
        """\
        Saves the current figure as a high quality image in the '/figures' 
        folder.

        Args
        ----
        figname: str
            Name of the figure, and also the name of the file. Best practice: do
            not include spaces in the name use '_' instead.
        verbose: bool
            Prints out information if set to `True`. Defaults to `True`.
        """
        filedir = PLOT_DIR / (figname + '.png')

        response = 'y'

        if filedir.exists() and verbose and not force:
            response = input(
                f'SpectrumSubplot | A figure with name \'{figname}\' already '
                'exists. Do you want to overwrite this image (y / n)? '
            )
            if (response.lower() == 'n') or (response.lower() == 'no'):
                return
        
        if (response.lower() == 'y') or (response.lower() == 'yes'):
            self.fig.savefig(
                fname=filedir,
                transparent=False,
                # format='png',
                # metadata=   Would be pretty cool to have image metadata...
                dpi=200
            )

            if verbose:
                print('SpectrumSubplot | Figure saved!') 
