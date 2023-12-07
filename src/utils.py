"""\
src / utils.py
--------------------------------------------------------------------------------

Aditya Marathe

Script contains the most basic tools required throughout the project.
"""
from __future__ import absolute_import
from __future__ import annotations
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import with_statement

__all__ = [
    'LocalDatasets',
    'load_nova_sample',
    'custom_subplots',
    'labelled_hist'
]

from typing import Any
from typing import Callable
from typing import Sequence
from typing import Literal
from typing import TypeAlias

import pathlib

import h5py
import pandas as pd

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.patches import Polygon

PLOT_FORMATTING: dict[str, Any] = {
    'MAJOR_TICKS_SIZE': 7.,
    'MINOR_TICKS_SIZE': 3.5,
    # TODO: Also add font size and other stuff here.
}


class LocalDatasets:
    # Note: Enum contains the locally saved datasets. Their directories are 
    #       stored in the environment variables. This enum stores the keys that
    #       are used to extract the local directory from the enviornment 
    #       variables.
    REALLY_MINI = 'MINI_DATA_DIR'


def _process_h5_file_path(
        file_path: str | pathlib.Path
    ) -> pathlib.Path:
    """\
    Processes the user specified file path and returns it as a Path object.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the file.

    Returns
    -------
    pathlib.Path
        Processed file path.
    """
    if isinstance(file_path, str):
        file_path = pathlib.Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            'The file stored at {} does not exist.'.format(file_path)
        )

    if not file_path.suffix == '.h5':
        raise ValueError(
            'The file stored at {} is not in HDF5 format.'.format(file_path)
        )

    return file_path


def _add_df_attrs(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """\
    Adds custom attributes to the DataFrame for QoL improvements. List of custom 
    attributes:

    - _applied_cuts_list : list[str]
    - get_applied_cuts () -> list[str]
    - has_applied_cuts () -> bool

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with custom attributes.
    """
    # setattr(df, '_applied_cuts_list', list())
    
    df.attrs['_applied_cuts_list'] = list()


    def _get_applied_cuts() -> list[str]:
        """\
        Gets a list of all the cuts applied to this DataFrame.

        Returns
        -------
        list[str]
            List of all the names of all the cuts applied to this DataFrame.
        """
        list_: list[str] = df.attrs['_applied_cuts_list']

        return list_.copy()


    # setattr(df, 'get_applied_cuts', _get_applied_cuts)

    df.attrs['get_applied_cuts'] = _get_applied_cuts
    

    def _has_applied_cuts() -> bool:
        """\
        Gets a list of all the cuts applied to this DataFrame.

        Returns
        -------
        bool
            List of all the names of all the cuts applied to this DataFrame.
        """
        list_: list[str] = df.attrs['_applied_cuts_list']

        return bool(list_)
    

    # setattr(df, 'has_applied_cuts', _has_applied_cuts)

    df.attrs['has_applied_cuts'] = _has_applied_cuts

    return df


def load_nova_sample(
        file_path: str | pathlib.Path,
        n_events: int = 1_000_000,
        shuffle: bool = False,
        random_state: int | None = None
    ) -> pd.DataFrame:
    """\
    Loads a sample of the first N events from simulated NOvA data stored in HDF5 
    format.

    Parameters
    ----------
    file_path : str | pathlib.Path
        Path to the HDF5 file.
    
    n_events : int
        Number of events in the sample. Defaults to 1,000,000 events.

    shuffle : bool
        Whether to shuffle the events in the HDF5 file. Defaults to False.

    random_state : int | None
        Random state used for shuffling the events. Defaults to None.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame object of the first N events from simulated NOvA data.

    Notes
    -----
    The returned DataFrame has the following custom attributes:
    - get_applied_cuts () -> list[str]
    - has_applied_cuts () -> bool

    These allow you to keep track of and check which cuts have been applied to 
    produce a certain DataFrame.
    """
    _process_h5_file_path(file_path)

    random = np.random.RandomState(random_state)

    with h5py.File(file_path, 'r') as file:
        file_keys = list(file.keys())
        
        # Note: For some reason my linter thinks that `__getitem__` is not 
        #       defined for the h5py.File object.

        file_n_events = len(file[file_keys[0]]) # type: ignore

        if shuffle:
            indices = np.sort(random.choice(
                file_n_events, 
                size=n_events, 
                replace=False
            ))
        else:
            indices = np.arange(min(n_events, file_n_events))

        df = pd.DataFrame(
            data={
                column : file[column][indices]  # type: ignore
                    for column in file_keys
            }
        )

    df = _add_df_attrs(df)

    return df


def _set_axes_formatting(ax: Axes) -> None:
    """\
    Sets the formatting of the plot axes using the specifications in the
    `PLOT_FORMATTING` dictionary.
    
    Parameters
    ----------
    ax : axes.Axes
        Target Axes.
    """
    ax.tick_params(which='both', axis='both', direction='in', right=True)
    ax.tick_params(
        which='major',
        axis='both',
        size=PLOT_FORMATTING['MAJOR_TICKS_SIZE']
    )
    ax.tick_params(
        which='minor',
        axis='both',
        size=PLOT_FORMATTING['MINOR_TICKS_SIZE']
    )


def _update_axes_labels(
        ax: Axes,
        data: npt.ArrayLike,
        bar_container: BarContainer,
        label: str
    ) -> None:
    """\
    Function updates the plot axes according to given data.

    Parameters
    ----------
    ax: axes.Axes
        Target Axes.

    data: npt.ArrayLike
        Data plotted on the histogram.

    bar_container: mpl.container.BarContainer
        Retured when `plt.hist` or `ax.hist` is called.

    label: str
        Label of the plotted histogram.
    """
    if not hasattr(ax, '_n_labelled_hist'):
        # Note: This error should not be raised... (hopefully)
        raise Exception('Unreachable: No `_n_labelled_hist` attribute!')

    patch = bar_container.patches[0]

    bbox = dict(
        boxstyle='round', 
        facecolor=patch.get_facecolor(),
        alpha=patch.get_alpha()
    )

    text = '{0}:\nEntries  {1}\nMean     {2:0.3f}\nStd Dev  {3:0.3f}'.format(
        label,
        np.shape(data)[0],
        np.mean(data),  # type: ignore
        np.std(data)    # type: ignore
    )

    ax.text(
        x=1.05, 
        y=1.0 - 0.24 * ax._n_labelled_hist,  # type: ignore
        s=text,
        transform=ax.transAxes, 
        verticalalignment='top', 
        bbox=bbox
    )

    ax._n_labelled_hist += 1  # type: ignore


_HIST_TYPES: TypeAlias = Literal['bar', 'barstacked', 'step', 'stepfilled']
_ALIGNMENTS: TypeAlias = Literal['left', 'mid', 'right']
_ORIENTATIONS: TypeAlias = Literal['vertical', 'horizontal']


def labelled_hist(
        ax: Axes,
        data: npt.ArrayLike,
        bins: int | Sequence[float] | str | None = None,
        range_: tuple[float, float] | None = None,
        density: bool = False,
        weights: npt.ArrayLike | None = None,
        cumulative: bool = False,
        bottom: npt.ArrayLike | None = None,
        histtype: _HIST_TYPES = 'bar',
        align: _ALIGNMENTS = 'mid',
        orientation: _ORIENTATIONS = 'vertical',
        rwidth: float | int | None = None,
        log: bool = False,
        color: str | None = None,
        label: str | None = None,
        stacked: bool = False,
        **kwargs
    ) -> tuple[npt.ArrayLike, 
                npt.ArrayLike, 
                BarContainer | Polygon | list[BarContainer | Polygon]]:
    
    return_args = ax.hist(
        data,
        bins=bins,
        range=range_,
        density=density,
        weights=weights,
        cumulative=cumulative,
        bottom=bottom,
        histtype=histtype,
        align=align,
        orientation=orientation,
        rwidth=rwidth,
        log=log,
        color=color,
        stacked=stacked,
        **kwargs
    )

    patches = return_args[-1]

    if not isinstance(patches, BarContainer):
        raise Exception('')

    # Adds the text label
    _update_axes_labels(
        ax=ax,
        data=data,
        # TODO: Fix this. Why do I even need bar container anyway?
        #       Who wrote this horrible function? (It was me :( ...)
        bar_container=return_args[-1],  # type: ignore
        label=label or 'Data'
    )

    return return_args


labelled_hist.__doc__ = """\
Parameters
----------
ax: matplotlib.axes.Axes
    The parent axes.

*args, **kwargs:
    Parameters for `matplotlib.pyplot.hist` (see below).

Notes
-----
Wraps the `matplotlib.pyplot.hist` function with the added functionality of 
also adding a `Text` widget on the Axes area which contains information about
the plotted distribution. This information includes: the label name, the number
of samples, the mean and the standard deviation.

This feature was inspired by plots produced by CERN's ROOT.

Original docstring
------------------
{}
""".format(plt.hist.__doc__)


def custom_subplots(
        nrows=1, 
        ncols=1, 
        *, 
        sharex=False,
        sharey=False,
        squeeze=True,
        width_ratios=None,
        height_ratios=None,
        subplot_kw=None,
        gridspec_kw=None,
        **fig_kw
    ) -> tuple[Figure, Axes]:
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
        **fig_kw
    )

    # Set my custom Axes formatting so plot style is consistent (and pretty).
    # Also, adds the custom function `labelled_hist` to the Axes object(s).
    if max(nrows, ncols) > 1:
        axs = axs.flatten()

        for ax in axs:
            _set_axes_formatting(ax)
            ax._n_labelled_hist: int = 0  # type: ignore

    else:
        _set_axes_formatting(axs)
        axs._n_labelled_hist: int = 0  # type: ignore

    return fig, axs


custom_subplots.__doc__ = """\
Notes
-----
Wraps `matplotlib.pyplot.subplots` with some extra functionality.

The Axes object(s) returned support the `labelled_hist` method which allows the 
user to add a nice little text label which displays the mean and the standard 
deviation of the distribution.

Also, the plot aesthetics are already formatted to reduce repeated code. :)

Original docstring
------------------
{}
""".format(plt.subplots.__doc__)