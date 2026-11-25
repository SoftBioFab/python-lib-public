"""2D random walk and mean squared displacement (MSD).

Each step: dx, dy drawn uniformly from [-1, 1].
Expected MSD for this walk: MSD(lag) = 2 * (1/3) * lag = (2/3) * lag
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(1)          # fixed seed for reproducibility

# Folder for saved figures (next to this script)
script_dir = Path(__file__).resolve().parent
fig_dir = script_dir / "figures"
fig_dir.mkdir(exist_ok=True)

# %% Parameters
n_traj = 100                            # number of trajectories
n_steps = 1000                          # number of points per trajectory
max_lag = 250                           # largest lag time for the MSD (in steps)
k = 2                                   # coverage factor for errors (~95%)

# %% Generate random walks (rows = time, columns = trajectories)
dx = rng.uniform(-1, 1, (n_steps - 1, n_traj))
dy = rng.uniform(-1, 1, (n_steps - 1, n_traj))
x = np.vstack([np.zeros((1, n_traj)), np.cumsum(dx, axis=0)])
y = np.vstack([np.zeros((1, n_traj)), np.cumsum(dy, axis=0)])

# %% Plot a few trajectories
n_show = 5
fig, ax = plt.subplots()
for i in range(n_show):
    ax.plot(x[:, i], y[:, i], lw=1, label=f"Traj {i + 1}")
ax.plot(0, 0, "k.", ms=10)              # common starting point
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"{n_show} example trajectories")
ax.legend(loc="best")
fig.savefig(fig_dir / "trajectories.png", dpi=300, bbox_inches="tight")

# %% Calculate MSD
# Time-averaged MSD for each trajectory, then mean and SEM over trajectories,
# which are independent.
lags = np.arange(1, max_lag + 1)
msd_traj = np.empty((max_lag, n_traj))  # rows = lag, columns = trajectory
for i, tau in enumerate(lags):
    ddx = x[tau:] - x[:-tau]
    ddy = y[tau:] - y[:-tau]
    msd_traj[i] = np.mean(ddx**2 + ddy**2, axis=0)
msd = msd_traj.mean(axis=1)
msd_sem = msd_traj.std(axis=1, ddof=1) / np.sqrt(n_traj)

# %% Weighted linear fit: MSD = slope*lag + offset, weights = 1/SEM^2
# MSD values at different lags are correlated, so a single fit to the mean
# curve underestimates the error. We fit each trajectory with the same
# weights and take the mean and SEM of the parameters over trajectories.
A = np.column_stack([lags, np.ones(max_lag)])
sw = 1 / msd_sem
P, *_ = np.linalg.lstsq(A * sw[:, None], msd_traj * sw[:, None], rcond=None)
p = P.mean(axis=1)                                   # [slope, offset]
p_err = k * P.std(axis=1, ddof=1) / np.sqrt(n_traj)  # expanded uncertainty


def fmt_err(val, err):
    """Round error to 1 significant digit and value to the same decimal."""
    n = -int(np.floor(np.log10(err)))
    err = round(err, n)
    n = -int(np.floor(np.log10(err)))   # recompute in case rounding gave e.g. 0.01
    nd = max(n, 0)
    return f"{round(val, n):.{nd}f} ± {err:.{nd}f}"


# %% Plot MSD
fig, ax = plt.subplots()
idx = slice(0, max_lag, 10)             # show every 10th lag to avoid clutter
ax.errorbar(lags[idx], msd[idx], k * msd_sem[idx], fmt="o", ms=4, capsize=2,
            label=f"Simulation (mean ± {k} SEM)")
ax.plot(lags, 2 / 3 * lags, "k--", lw=1.5, label=r"Theory: (2/3) $\tau$")
ax.plot(lags, A @ p, "r-", lw=1,
        label=f"Weighted fit: slope = {fmt_err(p[0], p_err[0])} (k={k})")
ax.set_xlabel("Lag (steps)")
ax.set_ylabel("MSD")
ax.set_title("Mean squared displacement")
ax.legend(loc="upper left")
fig.savefig(fig_dir / "msd.png", dpi=300, bbox_inches="tight")

# %% Results (diffusion coefficient in 2D: MSD = 4*D*lag)
D, D_err = p[0] / 4, p_err[0] / 4
print(f"Errors are expanded uncertainties, k = {k} (~95%)")
print(f"Fitted slope:  {fmt_err(p[0], p_err[0])} (theory: 0.667)")
print(f"Fitted offset: {fmt_err(p[1], p_err[1])} (theory: 0)")
print(f"Diffusion coefficient D: {fmt_err(D, D_err)} (theory: 0.167)")

plt.show()
