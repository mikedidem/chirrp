"""
Precomputed MODFLOW reference heads at anchor pumping rates.

The artifact ``artifacts/reference_heads.npz`` is built offline by
``chirrp/backend/scripts/build_validation_reference.py`` from the research
MODFLOW runs. It lets the Validation Lab respond instantly at the anchor Q
values without invoking mf2005.
"""

import os
from typing import Dict, List, Optional

import numpy as np

_DEFAULT_NPZ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "artifacts", "reference_heads.npz")


class ReferenceHeads:
    """Lazy-loaded lookup of precomputed lrs_model head fields."""

    def __init__(self, npz_path: str = _DEFAULT_NPZ):
        self.npz_path = npz_path
        self._data = None

    @property
    def available(self) -> bool:
        return os.path.isfile(self.npz_path)

    def _load(self):
        if self._data is None:
            self._data = np.load(self.npz_path)
        return self._data

    @property
    def q_values(self) -> List[float]:
        return self._load()["q_values"].tolist() if self.available else []

    def find_anchor(self, q_rate: float, tol: float = 1.0) -> Optional[int]:
        """Index of the anchor matching q_rate within tol m³/day, else None."""
        if not self.available:
            return None
        q = self._load()["q_values"]
        idx = int(np.argmin(np.abs(q - q_rate)))
        return idx if abs(float(q[idx]) - q_rate) <= tol else None

    def get(self, q_rate: float) -> Optional[Dict]:
        """Reference result for an anchor Q, shaped like run_lrs_model()."""
        idx = self.find_anchor(q_rate)
        if idx is None:
            return None
        d = self._load()
        heads = d["heads"][idx]
        return {
            "q_rate": float(d["q_values"][idx]),
            "times": d["times"].tolist(),
            "x": d["x"].tolist(),
            "y": d["y"].tolist(),
            "heads": heads,
            "runtime_s": float(d["nominal_runtime_s"])
            if "nominal_runtime_s" in d else None,
            "head_min": float(heads.min()),
            "head_max": float(heads.max()),
            "max_drawdown": float(90.0 - heads.min()),
            "precomputed": True,
        }
