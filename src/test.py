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

subplot = plotting.Subplots(1, 2)
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

stats, spec = subplot.plot_relative_residuals(
    ax_idx=1,
    data=data,
    obs_var=spectrum.var,
    exp_var=truth_spectrum.var,
    binning=np.linspace(-1, 1, 20)
)
x_arr = np.linspace(-1, 1, 100)
y_arr, params = plotting.fit_to_gaussian(
    x=x_arr,
    x_obs=spec['XValues'],  # type: ignore
    y_obs=spec['YValues'],  # type: ignore
    mean=stats['Mean'],
    std=stats['StdDev']
)
subplot.axs[1].plot(x_arr, y_arr)
subplot.set_energy_residual_xy_labels(ax_idx=1)

subplot.save_plot('testing123jeej', force=True)
subplot.show()
