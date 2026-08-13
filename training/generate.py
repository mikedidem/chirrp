# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import flopy
from scipy.interpolate import RegularGridInterpolator

# ==========================================================
# Paths
# ==========================================================
OUT_BASE = "modflow"
exe_path = "mf2005.exe"

print("current working directory:", os.getcwd())

# ==========================================================
# Model parameters  (Case 4 from paper)
# ==========================================================
Lx, Ly       = 1000.0, 1000.0
ncol, nrow   = 300, 300
nlay         = 1
delr         = Lx / ncol   # 10/3 m
delc         = Ly / nrow

top, botm    = 100.0, 0.0
K            = 33.33
Sy           = 0.10
Ss           = 1e-6
h0           = 90.0        # initial head — matches PINN (problem.py bc mode=0)

# Well at (700, 400) in model coords → (201.67, -98.33) in shifted PINN coords
iwell = int(400 / delc)  # row 120
jwell = int(700 / delr)  # col 210

# ==========================================================
# Time setup  (0 to 30 days)
# ==========================================================
times  = [0.25, 0.5, 0.75, 1.0] + list(range(2, 31))
perlen = [times[0]] + [times[i] - times[i - 1] for i in range(1, len(times))]
nper   = len(perlen)
nstp   = [5] + [3] * (nper - 1)
tsmult = [1.0] * nper
steady = [False] * nper

# ==========================================================
# Cell-centered output grid  (full 300x300)
# ==========================================================
xcent = np.linspace(delr / 2.0, Lx - delr / 2.0, ncol)
ycent = np.linspace(delc / 2.0, Ly - delc / 2.0, nrow)
Xg, Yg = np.meshgrid(xcent, ycent)
obs_points = np.column_stack([Xg.ravel(), Yg.ravel()])

# ==========================================================
# Q values to simulate (all 10 anchor rates reported in paper Table 4)
# ==========================================================
Q_list = [-50000.0, -45000.0, -44000.0, -42000.0, -40000.0,
          -38000.0, -36000.0, -35000.0, -30000.0, -25000.0]


def run_modflow(Qwell):
    OUT_DIR = os.path.join(OUT_BASE, f"Q{int(Qwell)}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Q = {Qwell:.0f} m3/day  ->  {OUT_DIR}")
    print(f"{'='*60}")

    modelname = "lrs_model"
    ml = flopy.modflow.Modflow(modelname, exe_name=exe_path, model_ws=OUT_DIR)

    flopy.modflow.ModflowDis(
        ml, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delr, delc=delc, top=top, botm=botm,
        nper=nper, perlen=perlen, nstp=nstp, tsmult=tsmult, steady=steady,
    )

    ibound = np.ones((nlay, nrow, ncol), dtype=int)
    flopy.modflow.ModflowBas(ml, ibound=ibound, strt=h0)
    flopy.modflow.ModflowLpf(ml, hk=K, vka=K, sy=Sy, ss=Ss, laytyp=1)

    # Point well
    flopy.modflow.ModflowWel(
        ml,
        stress_period_data={k: [[0, iwell, jwell, Qwell]] for k in range(nper)},
    )

    # CHD: north and south rows at h0
    chd_data = []
    for j in range(ncol):
        chd_data.append([0, 0,        j, h0, h0])
        chd_data.append([0, nrow - 1, j, h0, h0])
    flopy.modflow.ModflowChd(ml, stress_period_data={0: chd_data})

    flopy.modflow.ModflowPcg(
        ml, hclose=1e-5, rclose=5e-3, mxiter=400, iter1=150, relax=0.97, damp=0.7
    )
    flopy.modflow.ModflowOc(
        ml, stress_period_data={(k, 0): ['save head'] for k in range(nper)}
    )

    mg = ml.modelgrid
    xw = float(mg.xcellcenters[iwell, jwell])
    yw = float(mg.ycellcenters[iwell, jwell])
    print(f"  Well center (model coords): ({xw:.2f}, {yw:.2f})")
    print(f"  Well center (PINN shifted): ({xw-500:.2f}, {yw-500:.2f})")

    ml.write_input()
    _t0 = time.perf_counter()
    success, buff = ml.run_model(silent=True)
    _solver_time = time.perf_counter() - _t0
    if not success:
        print(f"  MODFLOW FAILED for Q={Qwell:.0f}!")
        return

    print(f"  MODFLOW completed successfully  |  solver wall clock: {_solver_time:.2f} s")

    hds     = flopy.utils.HeadFile(os.path.join(OUT_DIR, f"{modelname}.hds"))
    all_t   = np.array(hds.get_times())
    H_final = hds.get_data(totim=all_t[-1])[0]
    print(f"  Min head: {H_final.min():.2f}m  Drawdown: {h0 - H_final.min():.2f}m")

    for t in times:
        tsel = all_t[np.argmin(np.abs(all_t - t))]
        H    = hds.get_data(totim=tsel)[0]

        interp = RegularGridInterpolator(
            (ycent, xcent), H,
            bounds_error=False, fill_value=np.nan, method="linear"
        )
        heads = interp(obs_points[:, [1, 0]])
        out   = np.c_[obs_points, heads]

        t_str = str(int(t)) if float(t).is_integer() else str(t)
        np.savetxt(os.path.join(OUT_DIR, f"t{t_str}.txt"), out, fmt="%.4f %.4f %.4f")

    print(f"  Saved {len(times)} time steps")

    # Plot at t=1 and t=20
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, t_plot in zip(axes, [1.0, 20.0]):
        tsel = all_t[np.argmin(np.abs(all_t - t_plot))]
        H    = hds.get_data(totim=tsel)[0]
        cf   = ax.contourf(Xg, Yg, H, 20, cmap="viridis")
        ax.plot(xw, yw, 'r*', markersize=10)
        ax.set_title(f"Q={Qwell:.0f}  t={t_plot}d  min={H.min():.1f}m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        plt.colorbar(cf, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "head_evolution.png"), dpi=100)
    plt.close()


# ==========================================================
# Run all Q values
# Usage:
#   python generate.py           -> full run (all Q values)
#   python generate.py --bench   -> time one Q only, extrapolate
# ==========================================================
import sys
_bench_only = '--bench' in sys.argv

if _bench_only:
    # Single run for wall-clock estimate
    Qwell = Q_list[0]
    print(f"\nBench mode: running Q = {Qwell:.0f} only (will extrapolate to all {len(Q_list)} scenarios)")
    _t_start = time.perf_counter()
    run_modflow(Qwell)
    _one_time = time.perf_counter() - _t_start
    print(f"\n{'='*60}")
    print(f"  One MODFLOW scenario  : {_one_time:.2f} s")
    print(f"  Estimated total ({len(Q_list)} Q) : {_one_time * len(Q_list):.2f} s  (x{len(Q_list)})")
    print(f"{'='*60}")
else:
    _wall_times = {}
    for Qwell in Q_list:
        _t_start = time.perf_counter()
        run_modflow(Qwell)
        _wall_times[Qwell] = time.perf_counter() - _t_start

    print(f"\n{'='*60}")
    print("All Q simulations complete.")
    print(f"{'='*60}")
    print("\n  Wall-clock time per scenario (solver + I/O):")
    for Q, t in _wall_times.items():
        print(f"    Q = {Q:>10.0f} m3/day  ->  {t:.2f} s")
    print(f"\n  Mean per scenario : {sum(_wall_times.values())/len(_wall_times):.2f} s")
    print(f"  Total (6 Q runs)  : {sum(_wall_times.values()):.2f} s")
    print(f"{'='*60}")
