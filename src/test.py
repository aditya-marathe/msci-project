import numpy as np

import ana
import plotting
import utils

ds = utils.Datasets()

data = ana.NOvAData.init_from_copymerge_h5(
    [
        ds.COPYMERGED_C9_DIR,  # type: ignore
        ds.COPYMERGED_C10_DIR  # type: ignore
    ]
)

cuts = ana.Cuts.init_nova_cuts()
data.table = cuts.apply_all_cuts(data.table)

spectrum = ana.Spectrum(
    var='rec.energy.numu.lstmnu',
    binning=ana.Binning.HALF_E
)
truth_spectrum = ana.Spectrum(
    var='rec.mc.nu.E',
    binning=ana.Binning.HALF_E
)

subplot = plotting.SpectrumSubplot(1, 2)
subplot.plot_spectra(
    ax_idx=0,
    data=data,
    spectra=[truth_spectrum, spectrum],
    labels=['truth', 'experiment'],
    style=[plotting.Style.HISTOGRAM, plotting.Style.S_POINTS],
    alpha=0.7
)
subplot.set_energy_xy_labels(ax_idx=0, bin_width=0.1)
subplot.legend(ax_idx=0)

subplot.plot_relative_residuals(
    ax_idx=1,
    data=data,
    obs_var=spectrum.var,
    exp_var=truth_spectrum.var,
    binning=np.linspace(-1, 1, 20)
)
subplot.set_energy_residual_xy_labels(ax_idx=1)

subplot.save_plot('testing123jeej', force=True)
subplot.show()
