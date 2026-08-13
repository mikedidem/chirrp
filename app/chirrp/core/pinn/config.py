"""
Trained-model configuration for the parameterized groundwater PINN.

Every value here must match the training run that produced the shipped
checkpoints (Stage{1,2}_HARD_5x50_tau1_sigma30). Do not change them without
retraining.
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


class QOutOfRangeError(ValueError):
    """Requested pumping rate falls outside the surrogate's trained range."""

    def __init__(self, q_rate: float, q_min: float, q_max: float):
        self.q_rate = q_rate
        self.q_min = q_min
        self.q_max = q_max
        super().__init__(
            f"Pumping rate {q_rate:,.0f} m³/day is outside the trained range "
            f"[{q_min:,.0f}, {q_max:,.0f}] m³/day. The surrogate is only valid "
            f"within its training envelope."
        )


@dataclass(frozen=True)
class PinnConfig:
    """Frozen description of the trained two-stage PINN."""

    # Network: [4 inputs (x, y, t, Q_norm)] -> 5 hidden x 50 -> [1 output (h)]
    hidden_layers: int = 5
    hidden_neurons: int = 50
    scale: float = 1.0                  # adaptive sin-activation scale
    constraint: str = "HARD"

    # Two-stage time split
    tau: float = 1.0                    # stage 1: t in [0, tau]; stage 2: [tau, 30]

    # Physical problem (well problem, PINN coordinates)
    domain: Tuple[float, float, float, float, float, float] = (
        -500.0, 500.0, -500.0, 500.0, 0.0, 30.0
    )                                   # (xmin, xmax, ymin, ymax, tmin, tmax)
    well_xy: Tuple[float, float] = (201.67, -98.33)
    sigma: float = 30.0                 # Gaussian well approximation width (m)
    hydraulic_conductivity: float = 33.33   # K (m/day)
    specific_yield: float = 0.1             # Sy / mu
    initial_head: float = 90.0              # h0 = Dirichlet BC value (m)

    # Parameterization (training envelope)
    q_min: float = -50000.0             # m³/day (most negative = strongest pumping)
    q_max: float = -25000.0
    q_baseline: float = -40000.0        # reference rate for percent-change scenarios

    # Checkpoints
    stage1_checkpoint: str = field(
        default=os.path.join(_ARTIFACTS_DIR, "stage1.pth.tar"))
    stage2_checkpoint: str = field(
        default=os.path.join(_ARTIFACTS_DIR, "stage2.pth.tar"))

    @property
    def layers(self) -> list:
        return [4] + self.hidden_layers * [self.hidden_neurons] + [1]

    # ----- Q handling -------------------------------------------------------

    def normalize_q(self, q_rate: float) -> float:
        return 2.0 * (q_rate - self.q_min) / (self.q_max - self.q_min) - 1.0

    def q_from_percent(self, percent_change: float) -> float:
        """Map a percent change (e.g. -15 ⇒ 15% less pumping) onto Q.

        Percentages are relative to the magnitude of the baseline rate, so
        +10% means 10% MORE pumping (more negative Q).
        """
        return self.q_baseline * (1.0 + percent_change / 100.0)

    def percent_from_q(self, q_rate: float) -> float:
        return (q_rate / self.q_baseline - 1.0) * 100.0

    def percent_bounds(self) -> Tuple[float, float]:
        """Valid percent-change interval implied by the trained Q range."""
        lo = self.percent_from_q(self.q_max)   # least pumping  (e.g. -37.5)
        hi = self.percent_from_q(self.q_min)   # most pumping   (e.g. +25.0)
        return (lo, hi)

    def check_q(self, q_rate: float) -> None:
        if not (self.q_min <= q_rate <= self.q_max):
            raise QOutOfRangeError(q_rate, self.q_min, self.q_max)
