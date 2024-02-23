"""\
src / ana / spectrum.py
--------------------------------------------------------------------------------

Aditya Marathe
"""

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from pyosccalc.OscCalc import OscCalc
from pyosccalc.FluxTools import FluxTools

if TYPE_CHECKING:
    from ana.data import NOvAData


class Binning:
    """\
    Binning
    -------

    Standard binning used for NOvA data analysis.

    Notes
    -----
    For the full spectrum (from 0 to 10 GeV) use the `FULL_E` binning, and for
    the half spectrum (from 0 to 5 GeV) use the `HALF_E` binning.
    """
    FULL_E = np.linspace(0, 10, 100 + 1)
    HALF_E = np.linspace(0, 5, 50 + 1)


class Spectrum:
    """\
    Spectrum
    --------

    This class stores a histogram contaning the number of events per bin for a 
    given variable. 
    
    Notes
    -----
    In the future, also storing the exposure (protons on target).
    """
    def __init__(self, var: str, binning: npt.NDArray) -> None:
        """\
        Initialise a `Spectrum`.
        
        Args
        ----
        var: str
            Variable name (as given in the dataset).

        binning: npt.NDArray
            The bin configuration.

        Notes
        -----
        See `Binning` for the preset NOvA energy bins.
        """
        self._var = var
        self.binning = binning

    def get(self, data: "NOvAData") -> dict[str, npt.NDArray]:
        """\
        Get the spectrum for a dataset.

        Args
        ----
        data: NOvAData
            An initialised NOvA dataset.

        Returns
        -------
        dict[str, npt.NDArray]
            A dictionary containing the bin counts, bin edges and the bin 
            centres which are stored in keys 'YValues', 'BinEdges' and 'XValues'
            respectively.
        """
        values, bin_edges = np.histogram(
            data.table[self._var],
            bins=self.binning
        )
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'YValues': values,
            'BinEdges': bin_edges,
            'XValues': centres
        }

    def __call__(self, data: "NOvAData") -> dict[str, npt.NDArray]:
        """\
        Shortcut for the `Spectrum.get` method.
        """
        return self.get(data=data)


def get_flux_obj(
        energy: int | float | npt.NDArray, 
        counts: int | float | npt.NDArray
    ) -> object:
    """\
    Returns a `Flux` object, with a `flux` method.

    Notes
    -----
    This is used to mimic the flux class implemented in PyOscCalc, so that I can
    use it with my own experiment flux instead of some random lognormal flux.
    """
    class Flux:
        """\
        Flux
        ----
        
        Dummy flux class created for integration with the implementation in 
        PyOscCalc.        
        """
        def flux(
                self, 
                x: int | float | npt.NDArray
            ) -> int | float | npt.NDArray:
            """\
            Returns the weighting at the given x-value (which, in this case, is
            the muon neutrino energy).

            Notes
            -----
            NumPy's `intep` function is used to interpolate since we will not 
            have the same x-values in the data as the function input x-values.
            """
            return np.interp(
                x,
                energy,
                counts,
                left=0,  # No flux beyond measured points...
                right=0
            )
        
    return Flux()


# TODO: Documentation :( my worst enemy, and my best friend.


class OscillatableSpectrum:
    """\
    OscillatableSpectrum
    --------------------
    """
    def __init__(self, var: str, truth_var: str, binning: npt.NDArray) -> None:
        """\
        Initialises an `OscillatableSpectrum`.

        Args
        ----
        var: str
            ...
        truth_var: str
            ...
        binning: npt.NDarray
            ...
        """
        self._spectrum = Spectrum(var=var, binning=binning)
        self._truth_spectrum = Spectrum(var=truth_var, binning=binning)

    @property
    def binning(self) -> npt.NDArray:
        """\
        Getter for `binning`.
        """
        return self._spectrum.binning
    
    @binning.setter
    def binning(self, value: npt.NDArray) -> None:
        """\
        Setter for `binning` (updates binning for both spectra).
        """
        self._spectrum.binning = value
        self._truth_spectrum.binning = value
        self._fluxtools = FluxTools()

    def get(self, data: "NOvAData", oscalc: OscCalc) -> dict[str, npt.NDArray]:
        """\
        Get the oscillated spectrum for a dataset.

        Args
        ----
        data: NOvAData
            An initialised NOvA dataset.

        oscalc: OscCalc
            An initialised neutrino oscillation calculator (see `PyOscCalc`) 
            with certain oscillation parameter values.

        Returns
        -------
        dict[str, npt.NDArray]
            A dictionary containing the bin counts, bin edges and the bin 
            centres which are stored in keys 'YValues', 'BinEdges' and 'XValues'
            respectively.
        """
        spectrum = self._spectrum(data)

        flux_obj = get_flux_obj(
            energy=spectrum['XValues'],
            counts=spectrum['YValues']
        )

        spectrum_weights = np.asarray(
            self._fluxtools.getNuMuAsimov(flux=flux_obj, osccalc=oscalc)
        )
        bin_edges = np.asarray(self._fluxtools.binEdges)
        centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        return {
            'XValues': centres,
            'BinEdges': bin_edges,
            # Calculate the weighted flux...
            'YValues': spectrum_weights * spectrum['YValues']
        }
