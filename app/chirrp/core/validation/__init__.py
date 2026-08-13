"""
MODFLOW-2005 validation engine for the PINN surrogate.

Runs the same well problem the PINN was trained on (lrs_model: 300x300 grid,
30 days, single well, CHD north/south at 90 m) so surrogate and simulator can
be compared apples-to-apples — accuracy metrics and wall-clock latency.

Two paths:
  * reference.py — instant lookups from precomputed heads at 6 anchor Q values
  * lrs_model.py — live mf2005 run for arbitrary Q (seconds to minutes)
"""

from .lrs_model import run_lrs_model, resolve_mf2005_executable
from .metrics import compare_heads
from .reference import ReferenceHeads

__all__ = [
    "run_lrs_model",
    "resolve_mf2005_executable",
    "compare_heads",
    "ReferenceHeads",
]
