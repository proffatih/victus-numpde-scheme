"""
Production experiment driver for the stabilized CN-SAV Cahn-Hilliard study.

Runs and records (real, computed at run time -- no fabricated numbers):

  E1  Temporal convergence  : CN-SAV vs SAV-BDF2 (1D), measured order.
  E2  Spatial convergence   : Fourier-spectral (1D), sup-norm vs fine reference.
  E3  Discrete energy law    : modified-energy dissipation, CN-SAV vs SAV-BDF2 (1D).
  E4  2D coarsening benchmark: stabilized CN-SAV spinodal decomposition,
                               energy decay + mass conservation + snapshots.
  E5  Large-dt robustness    : true-energy at increasing dt, S=0 vs S>0.
  E6  RSAV consistency probe : honest report of relaxation effect.

Outputs:
  results/*.csv   small tabular data (committed)
  data/*.npz      large fields  (gitignored)
"""
import os
import time
import json
import numpy as np

from sav_ch import SpectralGrid, solve_ch, energy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
DAT = os.path.join(ROOT, "data")
os.makedirs(RES, exist_ok=True)
os.makedirs(DAT, exist_ok=True)

rng = np.random.default_rng(2026)


def write_csv(name, header, rows):
    path = os.path.join(RES, name)
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{x:.10g}" if isinstance(x, float) else str(x)
                             for x in r) + "\n")
    print("  wrote", path)


# ----------------------------------------------------------------------
def exp_temporal_convergence():
    print("[E1] Temporal convergence (1D): CN-SAV vs SAV-BDF2")
    L = 2 * np.pi
    N = 256
    eps = 0.1
    M = 1.0
    C0 = 1.0
    T = 0.5
    g = SpectralGrid(N, L, dim=1)
    u0 = (0.05 * np.cos(g.x) + 0.03 * np.sin(2 * g.x)
          + 0.02 * np.cos(3 * g.x) + 0.01 * np.sin(5 * g.x))

    # recommended stabilized scheme (S>0); guarantees robust r>0 dynamics.
    Sstab = 1.0

    def run(dt, scheme):
        ns = int(round(T / dt))
        return solve_ch(g, u0, eps, M, dt, ns, C0, S=Sstab,
                        scheme=scheme, record_every=10 ** 9)["u_final"]

    dt_ref = T / 81920
    refs = {s: run(dt_ref, s) for s in ("cn", "sav")}
    dts = [T / 160, T / 320, T / 640, T / 1280, T / 2560]
    rows = []
    for scheme in ("cn", "sav"):
        prev = None
        for dt in dts:
            u = run(dt, scheme)
            err = float(np.sqrt(g.integral((u - refs[scheme]) ** 2)))
            rate = (float(np.log(prev[1] / err) / np.log(prev[0] / dt))
                    if prev else float("nan"))
            rows.append([scheme, dt, err, rate])
            prev = (dt, err)
            print(f"    {scheme:4s} dt={dt:.3e} L2err={err:.4e} rate={rate:.3f}")
    write_csv("temporal_convergence.csv",
              ["scheme", "dt", "L2_error", "rate"], rows)


# ----------------------------------------------------------------------
def exp_spatial_convergence():
    print("[E2] Spatial spectral convergence (1D)")
    L = 2 * np.pi
    eps = 0.08
    M = 1.0
    C0 = 1.0
    dt = 1e-4
    T = 0.1
    Nref = 1024
    gref = SpectralGrid(Nref, L, dim=1)

    def ic(g):
        return 0.6 * np.exp(-((g.x - np.pi) ** 2) / 0.3) - 0.3

    uref = solve_ch(gref, ic(gref), eps, M, dt, int(T / dt), C0, S=1.0,
                    scheme="cn", record_every=10 ** 9)["u_final"]
    rows = []
    for N in [16, 32, 64, 128, 256]:
        g = SpectralGrid(N, L, dim=1)
        u = solve_ch(g, ic(g), eps, M, dt, int(T / dt), C0, S=1.0,
                     scheme="cn", record_every=10 ** 9)["u_final"]
        step = Nref // N
        err = float(np.max(np.abs(u - uref[::step])))
        rows.append([N, err])
        print(f"    N={N:4d} sup-err={err:.4e}")
    write_csv("spatial_convergence.csv", ["N", "sup_error"], rows)


# ----------------------------------------------------------------------
def exp_energy_law():
    print("[E3] Discrete energy law (1D): CN-SAV vs SAV-BDF2")
    L = 2 * np.pi
    N = 256
    eps = 0.06
    M = 1.0
    C0 = 1.0
    dt = 5e-3
    T = 3.0
    g = SpectralGrid(N, L, dim=1)
    u0 = 0.1 * (2 * rng.random(N) - 1)
    series = {}
    summary = []
    for scheme in ("cn", "sav"):
        r = solve_ch(g, u0, eps, M, dt, int(T / dt), C0, S=1.0,
                     scheme=scheme, record_every=2)
        series[scheme] = r
        dE = np.diff(r["E_mod"])
        max_incr = float(dE.max())
        strict = bool(np.all(dE <= 1e-12))
        summary.append([scheme, max_incr, int(strict),
                        float(r["E_mod"][0]), float(r["E_mod"][-1])])
        print(f"    {scheme:4s} max modified-energy increment={max_incr:+.3e} "
              f"strictly_dissipative={strict}")
    # time series for plotting
    rows = []
    t = series["cn"]["t"]
    for i in range(len(t)):
        rows.append([float(t[i]),
                     float(series["cn"]["E_mod"][i]),
                     float(series["cn"]["E"][i]),
                     float(series["sav"]["E_mod"][i]),
                     float(series["sav"]["E"][i])])
    write_csv("energy_law_timeseries.csv",
              ["t", "cn_Emod", "cn_Etrue", "sav_Emod", "sav_Etrue"], rows)
    write_csv("energy_law_summary.csv",
              ["scheme", "max_increment", "strictly_dissipative",
               "Emod_initial", "Emod_final"], summary)


# ----------------------------------------------------------------------
def exp_coarsening_2d():
    print("[E4] 2D coarsening benchmark (stabilized CN-SAV)")
    L = 2 * np.pi
    N = 256
    eps = 0.1
    M = 1.0
    C0 = 1.0
    S = 2.0
    dt = 2e-4
    T = 2.0
    g = SpectralGrid(N, L, dim=2)
    u0 = 0.05 * (2 * rng.random((N, N)) - 1)
    snap_times = [0.0, 0.02, 0.1, 0.4, 1.0, 2.0]
    snap_steps = sorted(set(int(round(ts / dt)) for ts in snap_times))
    ns = int(T / dt)
    t0 = time.time()
    r = solve_ch(g, u0, eps, M, dt, ns, C0, scheme="cn", S=S,
                 record_every=25, return_fields=True)
    print(f"    {ns} steps, N={N}^2 in {time.time() - t0:.1f}s")
    # energy / mass time series
    rows = []
    for i in range(len(r["t"])):
        rows.append([float(r["t"][i]), float(r["E"][i]),
                     float(r["E_mod"][i]), float(r["mass"][i])])
    write_csv("coarsening2d_energy_mass.csv",
              ["t", "E_true", "E_mod", "mass"], rows)
    mass = r["mass"]
    print(f"    mass drift (max-min)={mass.max() - mass.min():.3e} "
          f"true-E {r['E'][0]:.3f}->{r['E'][-1]:.3f} "
          f"monotone={bool(np.all(np.diff(r['E']) <= 1e-5))}")
    # snapshots: pick nearest recorded fields
    fields = r["fields"]
    keys = np.array(sorted(fields.keys()))
    snaps = {}
    for st in snap_steps:
        k = keys[np.argmin(np.abs(keys - st))]
        snaps[f"t_{k * dt:.4f}"] = fields[k]
    np.savez_compressed(os.path.join(DAT, "coarsening2d_snapshots.npz"),
                        x=g.x, y=g.y, dt=dt, eps=eps, **snaps)
    print("    saved snapshots ->", os.path.join(DAT,
                                                  "coarsening2d_snapshots.npz"))


# ----------------------------------------------------------------------
def exp_large_dt():
    print("[E5] Large-dt robustness: true energy, S=0 vs S>0")
    L = 2 * np.pi
    N = 256
    eps = 0.06
    M = 1.0
    C0 = 1.0
    T = 8.0
    g = SpectralGrid(N, L, dim=1)
    u0 = 0.1 * (2 * rng.random(N) - 1)
    E0 = float(energy(g, u0, eps))
    dts = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    rows = []
    for S in (0.0, 2.0):
        for dt in dts:
            ns = max(2, int(T / dt))
            r = solve_ch(g, u0, eps, M, dt, ns, C0,
                         scheme="cn", S=S, record_every=10 ** 9)
            Ef = float(energy(g, r["u_final"], eps))
            finite = bool(np.isfinite(Ef))
            rows.append([S, dt, Ef, int(finite), int(Ef <= E0 + 1e-6)])
            print(f"    S={S} dt={dt:5.3f} E(T)={Ef:9.3f} finite={finite}")
    write_csv("large_dt_robustness.csv",
              ["S", "dt", "E_final", "finite", "leq_E0"], rows)
    with open(os.path.join(RES, "large_dt_E0.txt"), "w") as f:
        f.write(f"E0={E0:.10g}\n")


# ----------------------------------------------------------------------
def exp_rsav_probe():
    print("[E6] RSAV/RCN relaxation consistency probe (honest report)")
    L = 2 * np.pi
    N = 256
    eps = 0.08
    M = 1.0
    C0 = 1.0
    T = 2.0
    g = SpectralGrid(N, L, dim=1)
    u0 = 0.1 * (2 * rng.random(N) - 1)
    rows = []
    for dt in [1e-2, 5e-3, 1e-3, 5e-4]:
        ns = int(T / dt)
        out = {}
        for scheme in ("cn", "rcn"):
            r = solve_ch(g, u0, eps, M, dt, ns, C0,
                         scheme=scheme, record_every=4)
            gap = float(np.abs(r["r"] - r["r_exact"]).mean())
            out[scheme] = gap
        improvement = out["cn"] / out["rcn"] if out["rcn"] > 0 else float("nan")
        rows.append([dt, out["cn"], out["rcn"], improvement])
        print(f"    dt={dt:.0e} mean|r-r_exact| CN={out['cn']:.3e} "
              f"RCN={out['rcn']:.3e} ({improvement:.2f}x)")
    write_csv("rsav_consistency_probe.csv",
              ["dt", "cn_gap", "rcn_gap", "improvement_factor"], rows)


# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    exp_temporal_convergence()
    exp_spatial_convergence()
    exp_energy_law()
    exp_large_dt()
    exp_rsav_probe()
    exp_coarsening_2d()   # heaviest, last
    print(f"\nAll experiments done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
