"""
Physics-informed neural network (PINN) surrogate for unconfined groundwater flow.

This package provides millisecond-latency hydraulic head predictions as a
surrogate for the MODFLOW-2005 well problem (lrs_model). The trained weights
ship with the repository under ``artifacts/`` — no GPU or external storage
is required at runtime.

Public API:
    PinnConfig   — trained-model configuration (must match training)
    PinnEngine   — loads both stages once and serves predictions
    QOutOfRangeError — raised when a requested Q falls outside the trained range
"""

from .config import PinnConfig, QOutOfRangeError
from .engine import PinnEngine

__all__ = ["PinnConfig", "PinnEngine", "QOutOfRangeError"]
