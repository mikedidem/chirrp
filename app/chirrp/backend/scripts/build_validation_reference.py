"""
One-time builder for the MODFLOW validation reference artifact.

Reads the research MODFLOW outputs (param_pinn/modflow/Q-*/t*.txt — one file
per time step, columns: x y head, 300x300 cell centers in model coordinates)
and compresses every complete Q set onto the PINN's 100x100 evaluation grid
in a single float32 npz that ships with the repository.

Usage:
    python chirrp/backend/scripts/build_validation_reference.py \
        --source "G:/My Drive/para_pinns/param_pinn/modflow" \
        [--out chirrp/core/validation/artifacts/reference_heads.npz]
"""

import argparse
import os
import re
import sys

import numpy as np
from scipy.interpolate import RegularGridInterpolator

TIMES = [0.25, 0.5, 0.75, 1.0] + [float(t) for t in range(2, 31)]
NROW = NCOL = 300
LX = LY = 1000.0
DELR, DELC = LX / NCOL, LY / NROW
RESOLUTION = 100

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUT = os.path.join(
    _REPO_ROOT, "chirrp", "core", "validation", "artifacts",
    "reference_heads.npz")


def t_filename(t: float) -> str:
    return f"t{int(t) if float(t).is_integer() else t}.txt"


def load_heads_file(path: str) -> np.ndarray:
    """Read one t*.txt (x y h rows) into a (300, 300) array, rows = y asc."""
    try:
        import pandas as pd
        h = pd.read_csv(path, sep=r"\s+", header=None,
                        usecols=[2], dtype=np.float64).values.ravel()
    except ImportError:
        h = np.loadtxt(path, usecols=2)
    if h.size != NROW * NCOL:
        raise ValueError(f"{path}: expected {NROW * NCOL} rows, got {h.size}")
    return h.reshape(NROW, NCOL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="Directory containing Q-*/t*.txt sets")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    q_dirs = []
    for name in sorted(os.listdir(args.source)):
        m = re.fullmatch(r"Q(-?\d+)", name)
        full = os.path.join(args.source, name)
        if m and os.path.isdir(full):
            missing = [t for t in TIMES
                       if not os.path.isfile(os.path.join(full, t_filename(t)))]
            if missing:
                print(f"  skipping {name}: {len(missing)} missing time files")
            else:
                q_dirs.append((float(m.group(1)), full))

    if not q_dirs:
        print("No complete Q directories found.")
        return 1

    q_dirs.sort()  # most negative (strongest pumping) first
    print(f"Found {len(q_dirs)} complete Q sets: "
          f"{[int(q) for q, _ in q_dirs]}")

    # Source cell centers (model coords) and PINN query grid
    xcent = np.linspace(DELR / 2.0, LX - DELR / 2.0, NCOL)
    ycent = np.linspace(DELC / 2.0, LY - DELC / 2.0, NROW)
    x_pinn = np.linspace(-500.0, 500.0, RESOLUTION)
    y_pinn = np.linspace(-500.0, 500.0, RESOLUTION)
    xq = np.clip(x_pinn + 500.0, xcent[0], xcent[-1])
    yq = np.clip(y_pinn + 500.0, ycent[0], ycent[-1])
    xx, yy = np.meshgrid(xq, yq)
    query = np.column_stack([yy.ravel(), xx.ravel()])

    heads = np.empty((len(q_dirs), len(TIMES), RESOLUTION, RESOLUTION),
                     dtype=np.float32)
    for qi, (q, qdir) in enumerate(q_dirs):
        print(f"  Q = {q:>10,.0f}: ", end="", flush=True)
        for ti, t in enumerate(TIMES):
            h300 = load_heads_file(os.path.join(qdir, t_filename(t)))
            interp = RegularGridInterpolator(
                (ycent, xcent), h300,
                bounds_error=False, fill_value=None, method="linear")
            heads[qi, ti] = interp(query).reshape(RESOLUTION, RESOLUTION)
        print(f"min head {heads[qi].min():.2f} m, "
              f"max drawdown {90.0 - heads[qi].min():.2f} m")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        q_values=np.array([q for q, _ in q_dirs], dtype=np.float64),
        times=np.array(TIMES, dtype=np.float64),
        x=x_pinn, y=y_pinn,
        heads=heads,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nWrote {args.out} ({size_mb:.1f} MB, "
          f"{len(q_dirs)} Q x {len(TIMES)} t x {RESOLUTION}^2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
