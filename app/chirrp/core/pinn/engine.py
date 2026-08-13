"""
Inference engine for the parameterized groundwater PINN.

Loads both trained stages once and serves hydraulic-head predictions in
milliseconds on CPU. All public methods validate the pumping rate against the
trained envelope and report wall-clock latency, which the API surfaces to the
UI (the framework paper reports per-step latency transparently).
"""

import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .config import PinnConfig
from .model import Net, load_stage

# Evaluation times matching the MODFLOW reference outputs (days)
DEFAULT_TIMES: List[float] = [0.25, 0.5, 0.75] + [float(t) for t in range(1, 31)]


class PinnEngine:
    """Two-stage PINN surrogate. Instantiate once (e.g. FastAPI lifespan)."""

    def __init__(self, config: Optional[PinnConfig] = None):
        self.config = config or PinnConfig()
        self.stage1: Net = load_stage(self.config.stage1_checkpoint,
                                      self.config, stage=1)
        self.stage2: Net = load_stage(self.config.stage2_checkpoint,
                                      self.config, stage=2)

    # ------------------------------------------------------------------ #
    # Core prediction
    # ------------------------------------------------------------------ #

    def _heads_at(self, xy: np.ndarray, t: float, q_norm: float,
                  hstar: Optional[torch.Tensor] = None) -> np.ndarray:
        """Head at one time for a set of (x, y) points. Stage-aware."""
        n = xy.shape[0]
        xyt = torch.from_numpy(
            np.hstack([xy, np.full((n, 1), t)]).astype(np.float32))
        # no_grad here (not in __init__): torch grad mode is thread-local and
        # web frameworks dispatch requests to worker threads.
        with torch.no_grad():
            if t <= self.config.tau:
                h = self.stage1(xyt, q_norm)
            else:
                if hstar is None:
                    hstar = self._hstar(xy, q_norm)
                h = self.stage2(xyt, q_norm, hstar)
        return h.numpy().ravel()

    def _hstar(self, xy: np.ndarray, q_norm: float) -> torch.Tensor:
        """Stage-1 solution at t = tau for the given points (Stage-2 input)."""
        n = xy.shape[0]
        xytau = torch.from_numpy(
            np.hstack([xy, np.full((n, 1), self.config.tau)]).astype(np.float32))
        with torch.no_grad():
            return self.stage1(xytau, q_norm)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def predict_grid(self, q_rate: float,
                     times: Optional[Sequence[float]] = None,
                     resolution: int = 100) -> Dict:
        """Head field over a regular grid for each requested time.

        Returns a dict with grid axes, heads (T, ny, nx), drawdown, summary
        stats, the well drawdown series, and wall-clock latency in ms.
        Rows are y ascending (south to north), columns x ascending.
        """
        self.config.check_q(q_rate)
        times = list(times) if times is not None else list(DEFAULT_TIMES)
        q_norm = self.config.normalize_q(q_rate)

        xmin, xmax, ymin, ymax, _, _ = self.config.domain
        x_lin = np.linspace(xmin, xmax, resolution)
        y_lin = np.linspace(ymin, ymax, resolution)
        xx, yy = np.meshgrid(x_lin, y_lin)
        xy = np.column_stack([xx.ravel(), yy.ravel()])

        start = time.perf_counter()
        hstar = self._hstar(xy, q_norm)
        heads = np.empty((len(times), resolution, resolution), dtype=np.float32)
        for i, t in enumerate(times):
            heads[i] = self._heads_at(xy, t, q_norm, hstar=hstar) \
                .reshape(resolution, resolution)
        latency_ms = (time.perf_counter() - start) * 1000.0

        h0 = self.config.initial_head
        drawdown = h0 - heads
        well_series = self.well_drawdown_series(q_rate, times)

        return {
            "q_rate": q_rate,
            "percent_change": self.config.percent_from_q(q_rate),
            "times": times,
            "x": x_lin.tolist(),
            "y": y_lin.tolist(),
            "heads": heads,
            "drawdown": drawdown,
            "head_min": float(heads.min()),
            "head_max": float(heads.max()),
            "max_drawdown": float(drawdown.max()),
            "well_xy": list(self.config.well_xy),
            "well_drawdown_series": well_series["drawdown"],
            "latency_ms": latency_ms,
        }

    def predict_points(self, q_rate: float,
                       points: Sequence[Dict[str, float]]) -> Dict:
        """Mesh-free head/drawdown at arbitrary (x, y, t) query points.

        ``points`` is a sequence of {"x": ..., "y": ..., "t": ...} dicts.
        """
        self.config.check_q(q_rate)
        q_norm = self.config.normalize_q(q_rate)
        h0 = self.config.initial_head

        start = time.perf_counter()
        results = []
        for p in points:
            xy = np.array([[p["x"], p["y"]]], dtype=np.float32)
            h = float(self._heads_at(xy, float(p["t"]), q_norm)[0])
            results.append({
                "x": p["x"], "y": p["y"], "t": p["t"],
                "head": h, "drawdown": h0 - h,
            })
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {"q_rate": q_rate, "points": results, "latency_ms": latency_ms}

    def well_drawdown_series(self, q_rate: float,
                             times: Optional[Sequence[float]] = None) -> Dict:
        """Drawdown time series at the pumping well."""
        self.config.check_q(q_rate)
        times = list(times) if times is not None else list(DEFAULT_TIMES)
        q_norm = self.config.normalize_q(q_rate)
        h0 = self.config.initial_head

        xy = np.array([self.config.well_xy], dtype=np.float32)
        hstar = self._hstar(xy, q_norm)
        heads = [float(self._heads_at(xy, t, q_norm, hstar=hstar)[0])
                 for t in times]

        return {
            "times": times,
            "heads": heads,
            "drawdown": [h0 - h for h in heads],
        }

    def goal_seek_max_pumping(self, x: float, y: float,
                              max_drawdown_m: float,
                              t: Optional[float] = None,
                              n_scan: int = 500) -> Dict:
        """Largest pumping magnitude keeping drawdown at (x, y) under a limit.

        Scans the trained Q envelope densely on the surrogate (sub-second on
        CPU) and returns the strongest admissible rate plus the full
        drawdown-vs-Q curve for plotting.
        """
        t_eval = float(t) if t is not None else self.config.domain[5]
        q_values = np.linspace(self.config.q_max, self.config.q_min, n_scan)
        xy = np.array([[x, y]], dtype=np.float32)
        h0 = self.config.initial_head

        start = time.perf_counter()
        drawdowns = np.empty(n_scan, dtype=np.float64)
        for i, q in enumerate(q_values):
            q_norm = self.config.normalize_q(float(q))
            h = float(self._heads_at(xy, t_eval, q_norm)[0])
            drawdowns[i] = h0 - h
        latency_ms = (time.perf_counter() - start) * 1000.0

        admissible = drawdowns <= max_drawdown_m
        if admissible.any():
            # q_values runs from weakest to strongest pumping; take the
            # strongest rate that still satisfies the constraint.
            best_idx = int(np.where(admissible)[0][-1])
            best_q = float(q_values[best_idx])
            feasible = True
        else:
            best_idx = int(np.argmin(drawdowns))
            best_q = float(q_values[best_idx])
            feasible = False

        return {
            "feasible": feasible,
            "best_q_rate": best_q,
            "best_percent_change": self.config.percent_from_q(best_q),
            "predicted_drawdown": float(drawdowns[best_idx]),
            "constraint": {"x": x, "y": y, "max_drawdown_m": max_drawdown_m,
                           "t": t_eval},
            "curve": {
                "q_rates": q_values.tolist(),
                "percent_changes": [self.config.percent_from_q(float(q))
                                    for q in q_values],
                "drawdowns": drawdowns.tolist(),
            },
            "n_evaluations": n_scan,
            "latency_ms": latency_ms,
        }
