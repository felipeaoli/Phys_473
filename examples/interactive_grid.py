"""
Interactive plot fitting and constraints
Zurich, 2024

Dr. Felipe Andrade-Oliveira
felipeaoli@gmail.com

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid

# Load chi2 grid from my previous constraints
# and define h, om sample arrays consistently
chi2_grid = np.loadtxt("../data/chi2sn.txt")
h_samples = np.loadtxt("../data/harr.txt")  # 50 values for h
om_samples = np.loadtxt("../data/omarr.txt")  # 70 values for om

# Load supernova data from 
# Union 2.1 catalog (Suzuki et al 2011)
zsn, musn, sigmu, probhost = np.genfromtxt("../data/SCPUnion2.1_mu_vs_z.txt", usecols=[1, 2, 3, 4]).T
zsort = np.sort(zsn)

ndata= len(zsn)
dof = ndata - 2 # 2 free-parameter

# Define function for luminosity distance prediction
# Flat Universe assumed for simplicity but you can extend it
def dl_interp(h, om):
    d_h = 2997.92458 / h  # Mpc
    zz = np.linspace(0, 1.5, 1000)
    ol = 1 - om
    Ez = np.sqrt(om * (1 + zz)**3 + ol + 5e-5 * (1 + zz)**4)
    return interp1d(zz, (1 + zz) * cumulative_trapezoid(1 / Ez, zz, initial=0) * d_h)

dl = lambda zarray, h, om: 5 * np.log10(dl_interp(h, om)(zarray)) + 25

# Set up figure and subplots with custom grid
fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], width_ratios=[1.3, 1.])
ax1 = fig.add_subplot(gs[0, 0])  # Left upper: Chi-squared contour plot
ax2 = fig.add_subplot(gs[0, 1])  # Right upper: Predictions plot
ax3 = fig.add_subplot(gs[1, 1])  # Right lower: Residuals plot
ax4 = fig.add_subplot(gs[1,0])   # Left bottom: Residuals histogram



# Left subplot: Credible level contours and subsampled points with color-coded levels
H, OM = np.meshgrid(h_samples, om_samples, indexing='ij')
confidence_levels = [0, 0.683, 0.95, 0.997][::-1]  # 68%, 95%, 99.7%
confidence_colors = ["lightblue", "lightgreen", "coral"]

# Draw contour lines only (no fill)
contour_lines = ax1.contourf(H, OM, np.exp(-(chi2_grid - chi2_grid.min()) / 2), 
                            levels=[1 - cl for cl in confidence_levels], cmap='coolwarm', alpha=0.7) #colors=confidence_colors, alpha=0.4)

ax1.contour(H, OM, np.exp(-(chi2_grid - chi2_grid.min()) / 2), 
                            levels=[1 - cl for cl in confidence_levels], color='k',lw=.3) #colors=confidence_colors, alpha=0.4)

ax1.set_xlabel("h")
ax1.set_ylabel("$\\Omega_m$")
ax1.set_title("Credible Levels")



# Right upper subplot: Predictions and measurements plot
line_measured = ax2.errorbar(zsn, musn, yerr=sigmu, capsize=3, capthick=0.5, c='dodgerblue', fmt='.')
line_pred, = ax2.plot([], [], '-', label="Predicted $\\mu(z | h, om)$", color="red", zorder=10)
ax2.set_title("Predictions vs Union 2.1 data")
ax2.set_xlabel("z")
ax2.set_ylabel("$\\mu$")
ax2.set_ylim(33, 47)
ax2.legend()



# Right lower subplot: Residuals plot
line_residual, = ax3.plot([], [], 'o', color="dodgerblue", markersize=3, label="Residuals")
ax3.axhline(0, color='gray', linestyle='--', linewidth=1)
ax3.set_title("Residuals: $(\\mu_{measured} - \\mu_{predicted})/\\sigma_\\mu$")
ax3.set_xlabel("z")
ax3.set_ylabel("Residual")
ax3.set_ylim(-8,8)
ax3.legend()


# Left lower subplot: Residuals histogram
hist_residuals = ax4.hist([], bins=15, color="purple", alpha=0.7)
_xx  =np.linspace(-8,8, 200)
ax4.plot(_xx, np.exp(-_xx**2/2)/np.sqrt(2*np.pi), c='tomato', zorder=0, label='Gaussian distribution')
ax4.set_xlim(-8,8)
ax4.set_ylim(0,.5)
ax4.set_title("Histogram of Residuals")
ax4.set_xlabel("Residual")
ax4.set_ylabel("Frequency")
ax4.legend(loc='upper left')


# Placeholder for chi2 display in text
chi2_text = fig.text(0.70, 0.6, "", fontsize=12, color="k")



# Event handler to update plots
def onclick(event):
    if event.inaxes != ax1:
        return

    # Get the (h, om) point where the user clicked
    h_click = event.xdata
    om_click = event.ydata

    # Find the nearest grid indices for h and om
    h_index = (np.abs(h_samples - h_click)).argmin()
    om_index = (np.abs(om_samples - om_click)).argmin()
    
    # Get the chi2 value for this point
    chi2_value = chi2_grid[h_index, om_index]
    
    # Calculate predicted dl values for this (h, om) on the zsn grid
    zsort = np.linspace(0.01, 1.05 * zsn.max(), 100)
    dl_pred = dl(zsort, h_samples[h_index], om_samples[om_index])
    
    # Update the right upper subplot with new predicted values
    line_pred.set_data(zsort, dl_pred)
    ax2.relim()
    ax2.autoscale_view()

    # Update residuals plot
    dl_obs = dl(zsn, h_samples[h_index], om_samples[om_index])
    residuals = (musn - dl_obs)/sigmu
    line_residual.set_data(zsn, residuals)
    ax3.relim()
    ax3.autoscale_view()

    ax4.cla()  # Clear previous histogram data
    ax4.hist(residuals, bins=16, histtype='step', alpha=0.7, density=True)
    ax4.plot(_xx, np.exp(-_xx**2/2)/np.sqrt(2*np.pi), c='tomato', zorder=0, label='Gaussian distribution')
    ax4.set_xlim(-8,8)
    ax4.set_ylim(0,.5)
    ax4.set_title("Histogram of Residuals")
    ax4.set_xlabel("Residual")
    ax4.set_ylabel("Frequency")
    ax4.legend(loc='upper left')
    # Update chi2 display for selected point
    chi2_text.set_text(f"$\\chi^2$: {chi2_value:.2f} (DoF: {dof})\n($h$, $\\Omega_m$) = ({h_samples[h_index]:.3f}, {om_samples[om_index]:.3f})")

    fig.canvas.draw_idle()


fig.canvas.mpl_connect('button_press_event', onclick)

plt.tight_layout()
plt.show()
