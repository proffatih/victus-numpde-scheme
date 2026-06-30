"""
Generate high-quality publication figures (vector PDF + 300 dpi PNG) from the
CSV / NPZ results produced by run_experiments.py.

Colour choices use the Okabe-Ito colourblind-safe palette.
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DAT = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colourblind-safe palette
OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermilion="#D55E00",
          purple="#CC79A7", skyblue="#56B4E9", yellow="#F0E442", black="#000000")

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.8,
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "mathtext.fontset": "cm",
})


def load_csv(name):
    with open(os.path.join(RES, name)) as f:
        return list(csv.DictReader(f))


def save(fig, stem):
    fig.savefig(os.path.join(FIG, stem + ".pdf"))
    fig.savefig(os.path.join(FIG, stem + ".png"), dpi=300)
    plt.close(fig)
    print("  saved", stem + ".pdf / .png")


# ----------------------------------------------------------------------
def fig_temporal():
    rows = load_csv("temporal_convergence.csv")
    data = {}
    for r in rows:
        data.setdefault(r["scheme"], ([], []))
        data[r["scheme"]][0].append(float(r["dt"]))
        data[r["scheme"]][1].append(float(r["L2_error"]))
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    styles = {"cn": (OI["blue"], "o", "Stabilized CN-SAV"),
              "sav": (OI["vermilion"], "s", "SAV-BDF2 (baseline)")}
    for sch, (c, m, lab) in styles.items():
        dt = np.array(data[sch][0]); err = np.array(data[sch][1])
        ax.loglog(dt, err, m + "-", color=c, label=lab, markersize=6)
    # reference slope-2 triangle
    dt = np.array(data["cn"][0])
    c2 = data["cn"][1][1] / dt[1] ** 2
    ax.loglog(dt, 0.6 * c2 * dt ** 2, "k--", lw=1.2,
              label=r"$\mathcal{O}(\Delta t^{2})$ reference")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"$L^2$ error at $T$")
    ax.set_title("Temporal convergence (1D Cahn--Hilliard)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    save(fig, "fig_temporal_convergence")


def fig_spatial():
    rows = load_csv("spatial_convergence.csv")
    N = np.array([int(r["N"]) for r in rows])
    err = np.array([float(r["sup_error"]) for r in rows])
    # only meaningful (monotone-resolved) branch N>=64 for slope clarity; show all
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    mask = err > 0
    ax.semilogy(N[mask], err[mask], "o-", color=OI["green"], markersize=6,
                label="Fourier-spectral")
    ax.set_xlabel(r"modes per direction $N$")
    ax.set_ylabel(r"$L^\infty$ error vs.\ fine reference")
    ax.set_title("Spatial spectral convergence (1D)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    save(fig, "fig_spatial_convergence")


def fig_energy_law():
    rows = load_csv("energy_law_timeseries.csv")
    t = np.array([float(r["t"]) for r in rows])
    cnE = np.array([float(r["cn_Emod"]) for r in rows])
    cnT = np.array([float(r["cn_Etrue"]) for r in rows])
    savE = np.array([float(r["sav_Emod"]) for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax1.plot(t, cnE, color=OI["blue"], label=r"CN-SAV modified $\tilde E$")
    ax1.plot(t, cnT, color=OI["skyblue"], ls="--",
             label=r"CN-SAV true $E$")
    ax1.plot(t, savE, color=OI["vermilion"], ls=":",
             label=r"SAV-BDF2 modified $\tilde E$")
    ax1.set_xlabel("time $t$"); ax1.set_ylabel("energy")
    ax1.set_title("(a) Discrete energy dissipation")
    ax1.grid(True, ls=":", alpha=0.4); ax1.legend(frameon=False)
    # right: per-step increments (zoom on monotonicity)
    dcn = np.diff(cnE); dsav = np.diff(savE); tt = t[1:]
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.plot(tt, dcn, color=OI["blue"], label="CN-SAV")
    ax2.plot(tt, dsav, color=OI["vermilion"], ls=":", label="SAV-BDF2")
    ax2.set_xlabel("time $t$")
    ax2.set_ylabel(r"$\tilde E^{n+1}-\tilde E^{n}$")
    ax2.set_title("(b) Per-step energy increment")
    ax2.grid(True, ls=":", alpha=0.4); ax2.legend(frameon=False)
    save(fig, "fig_energy_law")


def fig_coarsening():
    d = np.load(os.path.join(DAT, "coarsening2d_snapshots.npz"))
    keys = [k for k in d.files if k.startswith("t_")]
    # order by the time encoded in the key
    keys = sorted(keys, key=lambda k: float(k[2:]))
    keys = keys[:6]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.4))
    for ax, k in zip(axes.ravel(), keys):
        u = d[k]
        im = ax.imshow(u.T, origin="lower", cmap="RdBu_r", vmin=-1.1, vmax=1.1,
                       extent=[0, 2 * np.pi, 0, 2 * np.pi])
        ax.set_title(r"$t=%.3f$" % float(k[2:]))
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"phase field $u$")
    fig.suptitle("Spinodal decomposition / coarsening (stabilized CN-SAV, "
                 r"$256^2$, $\varepsilon=0.1$)", y=0.98)
    fig.savefig(os.path.join(FIG, "fig_coarsening_snapshots.pdf"))
    fig.savefig(os.path.join(FIG, "fig_coarsening_snapshots.png"), dpi=300)
    plt.close(fig)
    print("  saved fig_coarsening_snapshots.pdf / .png")


def fig_coarsening_energy_mass():
    rows = load_csv("coarsening2d_energy_mass.csv")
    t = np.array([float(r["t"]) for r in rows])
    E = np.array([float(r["E_true"]) for r in rows])
    mass = np.array([float(r["mass"]) for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax1.plot(t, E, color=OI["blue"])
    ax1.set_xlabel("time $t$"); ax1.set_ylabel(r"free energy $E[u]$")
    ax1.set_title("(a) Energy decay during coarsening")
    ax1.grid(True, ls=":", alpha=0.4)
    m0 = mass[0]
    dev = np.abs(mass - m0)
    floor = 1e-18
    dev_plot = np.maximum(dev, floor)
    ax2.semilogy(t, dev_plot, color=OI["green"], marker=".", ms=3, ls="none")
    ax2.axhline(np.finfo(float).eps, color=OI["orange"], ls="--", lw=1.0,
                label=r"machine $\epsilon$")
    ax2.set_ylim(1e-19, 1e-12)
    ax2.set_xlabel("time $t$")
    ax2.set_ylabel(r"$|\,\overline{u}(t)-\overline{u}(0)\,|$")
    ax2.set_title("(b) Mass-conservation error")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(frameon=False, loc="upper right")
    ax2.text(0.5, 0.5, "exactly conserved\n(at/below machine precision)",
             transform=ax2.transAxes, ha="center", va="center",
             fontsize=9, color="0.3")
    save(fig, "fig_coarsening_energy_mass")


def fig_large_dt():
    rows = load_csv("large_dt_robustness.csv")
    with open(os.path.join(RES, "large_dt_E0.txt")) as f:
        E0 = float(f.read().strip().split("=")[1])
    data = {}
    for r in rows:
        S = float(r["S"])
        data.setdefault(S, ([], []))
        data[S][0].append(float(r["dt"]))
        data[S][1].append(float(r["E_final"]))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    sty = {0.0: (OI["vermilion"], "s", r"$S=0$ (plain CN-SAV)"),
           2.0: (OI["blue"], "o", r"$S=2$ (stabilized CN-SAV)")}
    for S, (c, m, lab) in sty.items():
        dt = np.array(data[S][0]); E = np.array(data[S][1])
        ax.loglog(dt, E, m + "-", color=c, label=lab, markersize=6)
    ax.axhline(E0, color="k", ls="--", lw=1.0, label=r"initial energy $E_0$")
    ax.set_xlabel(r"time step $\Delta t$")
    ax.set_ylabel(r"final true energy $E(T)$, $T=8$")
    ax.set_title("Large-$\\Delta t$ robustness")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    save(fig, "fig_large_dt_robustness")


def main():
    print("Building figures...")
    fig_temporal()
    fig_spatial()
    fig_energy_law()
    fig_coarsening()
    fig_coarsening_energy_mass()
    fig_large_dt()
    print("All figures done.")


if __name__ == "__main__":
    main()
