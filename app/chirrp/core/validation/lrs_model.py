"""
Live MODFLOW-2005 runner for the PINN's well problem (lrs_model).

Ported from the research data generator (param_pinn/generate.py). The output
convention matches the PINN exactly: head rows ascend with y, the well sits at
(201.67, -98.33) in shifted coordinates, and results are interpolated onto the
PINN's evaluation grid so the two engines can be differenced directly.
"""

import os
import platform
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

# Physical setup — must mirror the PINN training problem (see core.pinn.config)
LX, LY = 1000.0, 1000.0
NROW, NCOL = 300, 300
NLAY = 1
DELR = LX / NCOL
DELC = LY / NROW
TOP, BOTM = 100.0, 0.0
K = 33.33
SY = 0.10
SS = 1e-6
H0 = 90.0
IWELL = int(400 / DELC)   # row 120
JWELL = int(700 / DELR)   # col 210

# Output times (days) — same set the PINN engine uses by default
TIMES = [0.25, 0.5, 0.75, 1.0] + [float(t) for t in range(2, 31)]

_BIN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")


def resolve_mf2005_executable() -> str:
    """Pick the mf2005 binary for the current platform.

    Override with the MF2005_PATH environment variable. On Linux the binary is
    checked to be an ELF executable so a mistakenly-mounted macOS binary fails
    with a clear message instead of a cryptic exec error.
    """
    override = os.environ.get("MF2005_PATH")
    if override:
        path = override
    else:
        system = platform.system().lower()
        names = {
            "windows": "mf2005.exe",
            "linux": "mf2005.linux",
            "darwin": "mf2005.mac",
        }
        path = os.path.join(_BIN_DIR, names.get(system, "mf2005.linux"))

    exe = Path(path)
    if not exe.is_file():
        raise FileNotFoundError(
            f"MODFLOW executable not found: {path}. "
            "Set MF2005_PATH or place a binary under chirrp/core/validation/bin/."
        )

    if platform.system().lower() == "linux":
        with exe.open("rb") as f:
            magic = f.read(4)
        if not magic.startswith(b"\x7fELF"):
            raise ValueError(
                f"'{path}' is not a Linux ELF binary. Provide a Linux build "
                "of mf2005 (set MF2005_PATH or replace bin/mf2005.linux)."
            )
    return str(exe)


def run_lrs_model(q_rate: float,
                  resolution: int = 100,
                  times: Optional[Sequence[float]] = None,
                  workdir: Optional[str] = None,
                  keep_files: bool = False) -> Dict:
    """Build and run the lrs_model for one pumping rate.

    Returns heads interpolated onto the PINN evaluation grid:
      {"times", "x", "y" (PINN coords), "heads" (T, res, res) float32,
       "runtime_s" (solver+I/O wall clock), "head_min", "max_drawdown"}
    """
    import flopy
    from scipy.interpolate import RegularGridInterpolator

    exe_path = resolve_mf2005_executable()
    times = list(times) if times is not None else list(TIMES)

    perlen = [times[0]] + [times[i] - times[i - 1]
                           for i in range(1, len(times))]
    nper = len(perlen)
    nstp = [5] + [3] * (nper - 1)

    scratch = workdir or tempfile.mkdtemp(prefix="lrs_model_")
    os.makedirs(scratch, exist_ok=True)

    start = time.perf_counter()
    try:
        modelname = "lrs_model"
        ml = flopy.modflow.Modflow(modelname, exe_name=exe_path,
                                   model_ws=scratch)
        flopy.modflow.ModflowDis(
            ml, nlay=NLAY, nrow=NROW, ncol=NCOL,
            delr=DELR, delc=DELC, top=TOP, botm=BOTM,
            nper=nper, perlen=perlen, nstp=nstp,
            tsmult=[1.0] * nper, steady=[False] * nper,
        )
        ibound = np.ones((NLAY, NROW, NCOL), dtype=int)
        flopy.modflow.ModflowBas(ml, ibound=ibound, strt=H0)
        flopy.modflow.ModflowLpf(ml, hk=K, vka=K, sy=SY, ss=SS, laytyp=1)
        flopy.modflow.ModflowWel(
            ml,
            stress_period_data={k: [[0, IWELL, JWELL, q_rate]]
                                for k in range(nper)},
        )
        chd_data = []
        for j in range(NCOL):
            chd_data.append([0, 0, j, H0, H0])
            chd_data.append([0, NROW - 1, j, H0, H0])
        flopy.modflow.ModflowChd(ml, stress_period_data={0: chd_data})
        flopy.modflow.ModflowPcg(
            ml, hclose=1e-5, rclose=5e-3, mxiter=400, iter1=150,
            relax=0.97, damp=0.7,
        )
        flopy.modflow.ModflowOc(
            ml, stress_period_data={(k, 0): ["save head"]
                                    for k in range(nper)},
        )

        ml.write_input()
        success, _ = ml.run_model(silent=True)
        if not success:
            raise RuntimeError(
                f"MODFLOW (lrs_model) failed for Q={q_rate:,.0f} m³/day")

        hds = flopy.utils.HeadFile(
            os.path.join(scratch, f"{modelname}.hds"))
        all_t = np.array(hds.get_times())

        # Cell centers, ascending — head rows are treated as ascending y,
        # mirroring the y-axis exactly as the PINN training data did.
        xcent = np.linspace(DELR / 2.0, LX - DELR / 2.0, NCOL)
        ycent = np.linspace(DELC / 2.0, LY - DELC / 2.0, NROW)

        # PINN evaluation grid (shifted coords -> model coords)
        x_pinn = np.linspace(-500.0, 500.0, resolution)
        y_pinn = np.linspace(-500.0, 500.0, resolution)
        xq = np.clip(x_pinn + 500.0, xcent[0], xcent[-1])
        yq = np.clip(y_pinn + 500.0, ycent[0], ycent[-1])
        xx, yy = np.meshgrid(xq, yq)
        query = np.column_stack([yy.ravel(), xx.ravel()])  # (y, x) order

        heads = np.empty((len(times), resolution, resolution),
                         dtype=np.float32)
        for i, t in enumerate(times):
            tsel = all_t[np.argmin(np.abs(all_t - t))]
            h_grid = hds.get_data(totim=tsel)[0]
            interp = RegularGridInterpolator(
                (ycent, xcent), h_grid,
                bounds_error=False, fill_value=None, method="linear")
            heads[i] = interp(query).reshape(resolution, resolution)
        hds.close()
    finally:
        runtime_s = time.perf_counter() - start
        if not keep_files and workdir is None:
            shutil.rmtree(scratch, ignore_errors=True)

    return {
        "q_rate": q_rate,
        "times": times,
        "x": x_pinn.tolist(),
        "y": y_pinn.tolist(),
        "heads": heads,
        "runtime_s": runtime_s,
        "head_min": float(heads.min()),
        "head_max": float(heads.max()),
        "max_drawdown": float(H0 - heads.min()),
    }
