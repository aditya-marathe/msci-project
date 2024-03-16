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
    'Style',
    'Subplots'
]

from typing import TYPE_CHECKING
from typing import Literal
from typing import Sequence
from typing import Any

import enum

import pathlib

import numpy as np
import numpy.typing as npt

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from src.ana.data import NOvAData
    from src.ana.spectrum import Spectrum


def _init_axs_formatting(fig: Figure, ax: Axes, fontsize: int) -> None:
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
    fig.tight_layout()


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


def _plot_styled_hist(
        ax: Axes,
        y_values: npt.NDArray,
        bin_edges: npt.NDArray,
        x_values: npt.NDArray,
        label: str | None,
        style: Styles,
        alpha: float
    ) -> None:
    """\
    Helper function for plotting styled histograms.
    """
    if not isinstance(style, list):
            style = [style]

    if not (
        (Style.HISTOGRAM in style)
        or (Style.STEPS in style)
        or (Style.S_POINTS in style)
        or (Style.L_POINTS in style)
    ):
        raise ValueError(f'Unknown plotting style(s) `{style}`.')

    if Style.HISTOGRAM in style:
        ax.hist(
            bin_edges[:-1],
            bins=bin_edges,
            weights=y_values,
            label=label,
            alpha=alpha
        )
    if Style.STEPS in style:
        ax.hist(
            bin_edges[:-1],
            bins=bin_edges,
            weights=y_values,
            label=label,
            histtype='step',
            alpha=alpha
        )
    if Style.S_POINTS in style:
        ax.plot(
            x_values,
            y_values,
            '.',
            label=label,
            alpha=alpha
        )
    if Style.L_POINTS in style:
        ax.plot(
            x_values,
            y_values,
            'o',
            label=label,
            alpha=alpha
        )


class Subplots:
    """
    Subplots
    --------

    Wrapper around behaviour of Matplotlib `subplots` to make it easier to plot
    / analyse energy spectra and save high-quality images of the figure.
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
        Initialises `Subplots`.

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
            _init_axs_formatting(
                fig=self.fig,
                ax=ax,
                fontsize=int(self.fontsize)
            )

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
            fontsize=int(self.fontsize * 1.2)
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

        self.fig.tight_layout()

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
        ) -> dict[str, float]:
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
            Chosen plotting style. Defaults to `Style.HISTOGRAM`.
        alpha: float
            The transparency of the histogram (useful for plotting multiple 
            histograms).

        Returns
        -------
        dict[str, float]
            A dictionary containing the number of events, mean, and standard 
            deviation of the data whic are stored in keys 'Events', 'Mean', and
            'StdDev' respectively.
        """
        spectrum_dict = spectrum(data)

        _plot_styled_hist(
            self.axs[ax_idx],
            y_values=spectrum_dict['YValues'],
            bin_edges=spectrum_dict['BinEdges'],
            x_values=spectrum_dict['XValues'],
            label=label,
            style=style,
            alpha=alpha
        )

        self.axs[ax_idx].set_xlim(
            np.min(spectrum.binning),
            np.max(spectrum.binning)
        )

        return spectrum.get_stats(data=data)

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
            style: Styles = Style.HISTOGRAM,
            alpha: float = 0.8
        ) -> tuple[dict[str, float], dict[str, float]]:
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
        style: Styles
            Chosen plotting style. Defaults to `Style.HISTOGRAM`.
        alpha: float
            The transparency of the histogram (useful for plotting multiple 
            histograms).

        Returns
        -------
        tuple[dict[str, float], dict[str, float]]
            One dictionary containing the mean, standard deviation, and root
            mean square (aka energy resolution) of the residuals as keys 'Mean',
            'StdDev', and 'RMS' respectively. Another dictionary containing the
            y values (bar heights), bin edges, and x values (bin centres) of the
            residuals as keys 'YValues', 'BinEdges', and 'XValues' respectively.
        """
        observation = data.table[obs_var]
        expected = data.table[exp_var]
        residuals = 1 - (observation / expected)

        y_values, bin_edges = np.histogram(
            residuals, 
            bins=binning  # type: ignore
        )
        x_values = (bin_edges[:-1] + bin_edges[1:]) / 2

        _plot_styled_hist(
            ax=self.axs[ax_idx],
            y_values=y_values,
            bin_edges=bin_edges,
            x_values=x_values,
            label=label,
            style=style,
            alpha=alpha
        )

        self.axs[ax_idx].set_xlim(-1, 1)

        return ( # type: ignore ~ PyLance being stupid... I miss PyCharm :(
            {
                'Mean': np.mean(residuals),
                'StdDev': np.std(residuals),
                'RMS': np.sqrt(np.mean(residuals**2))
            },
            {
                'YValues': y_values,
                'BinEdges': bin_edges,
                'XValues': x_values
            }
        )

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
        """\
        Add a legend on a selected axes.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        """
        self._add_legend(ax=self.axs[ax_idx])            

    def fig_legend(
            self,
            ax_idx: int,
            *,
            loc: str | tuple[float, float] = 'lower center',
            anchor: tuple[float, float] = (0.5, -0.1),
            n_cols: int | None = None
        ) -> None:
        """\
        Adds a legend below the figure.

        Args
        ----
        ax_idx: int
            Axes index on the subplots.
        """
        handles, labels = self.axs[ax_idx].get_legend_handles_labels()

        if n_cols is None:
            n_cols = len(labels)
        
        self.fig.legend(
            handles=handles,
            labels=labels,
            ncol=n_cols,
            # Appearance
            fontsize=int(self.fontsize * 1.2),
            # Location
            loc=loc,
            bbox_to_anchor=anchor,
            bbox_transform=self.fig.transFigure
        )

        self.fig.tight_layout()

    @staticmethod
    def show() -> None:
        """\
        Uses the Tkinter backend to display the plot in a GUI.
        """
        plt.show()

    def save_plot(
            self,
            figname: str,
            plot_dir: str,
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
        plot_dir: str
            ...
        verbose: bool
            Prints out information if set to `True`. Defaults to `True`.
        """
        filedir = pathlib.Path(plot_dir) / (figname + '.png')

        response = 'y'

        if filedir.exists() and verbose and not force:
            response = input(
                f'Subplots | A figure with name \'{figname}\' already exists. '
                'Do you want to overwrite this image (y / n)? '
            )
            if (response.lower() == 'n') or (response.lower() == 'no'):
                return
        
        if (response.lower() == 'y') or (response.lower() == 'yes'):
            self.fig.savefig(
                fname=filedir,
                transparent=False,
                # format='png',
                # metadata=   Would be pretty cool to have image metadata...
                dpi=200,
                bbox_inches='tight'
            )

            if verbose:
                print('Subplots | Figure saved!')

    def __str__(self) -> str:
        return 'Subplots()'
    
    def __repr__(self) -> str:
        return str(self)
