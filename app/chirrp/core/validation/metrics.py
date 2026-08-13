"""
Accuracy metrics between PINN and MODFLOW head fields.
"""

from typing import Dict

import numpy as np


def compare_heads(h_pinn: np.ndarray, h_modflow: np.ndarray) -> Dict:
    """Error metrics over matching head arrays (any shape, metres).

    Returns RMSE, MAE, max |error|, RRMSE (%), and R² computed over all
    space-time points, plus the per-cell error field at the final time when
    3-D (T, ny, nx) inputs are given.
    """
    if h_pinn.shape != h_modflow.shape:
        raise ValueError(
            f"Shape mismatch: PINN {h_pinn.shape} vs MODFLOW {h_modflow.shape}")

    err = h_pinn.astype(np.float64) - h_modflow.astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    max_err = float(np.max(np.abs(err)))

    h_true = h_modflow.astype(np.float64)
    span = float(h_true.max() - h_true.min())
    rrmse_pct = float(rmse / span * 100.0) if span > 0 else 0.0

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((h_true - h_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

    result = {
        "rmse_m": rmse,
        "mae_m": mae,
        "max_abs_error_m": max_err,
        "rrmse_pct": rrmse_pct,
        "r2": r2,
    }
    if err.ndim == 3:
        result["error_field_final"] = err[-1].astype(np.float32)
    return result
