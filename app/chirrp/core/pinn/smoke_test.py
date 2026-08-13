"""
Smoke test for the PINN engine.

Run from the repo root:
    python -m chirrp.core.pinn.smoke_test

Checks (mirroring param_pinn/sensitivity_test.py):
  1. Checkpoints load with strict key matching.
  2. Baseline grid (Q = -40,000) is finite, near 90 m far from the well,
     with a clear drawdown cone at the well.
  3. Stage-1/Stage-2 continuity at t = tau.
  4. Monotonic response: stronger pumping => deeper drawdown.
  5. Q-range validation raises QOutOfRangeError.
  6. Grid latency on CPU is interactive (< 2000 ms for 33 x 100 x 100).
"""

import sys
import time

import numpy as np

from chirrp.core.pinn import PinnConfig, PinnEngine, QOutOfRangeError


def main() -> int:
    failures = []

    def check(name, cond, detail=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("Loading engine (both stages)...")
    t0 = time.perf_counter()
    engine = PinnEngine()
    print(f"  loaded in {(time.perf_counter() - t0) * 1000:.0f} ms\n")
    cfg = engine.config

    # --- Baseline grid ----------------------------------------------------
    print("1) Baseline grid, Q = -40,000 m³/day")
    res = engine.predict_grid(-40000.0, resolution=100)
    heads = res["heads"]
    check("finite heads", bool(np.isfinite(heads).all()))
    check("head_max ≈ 90 m (boundary)",
          abs(res["head_max"] - 90.0) < 1.0,
          f"head_max={res['head_max']:.3f}")
    check("drawdown cone exists (max_drawdown in (0.1, 30) m)",
          0.1 < res["max_drawdown"] < 30.0,
          f"max_drawdown={res['max_drawdown']:.3f} m")
    check("grid latency interactive", res["latency_ms"] < 2000.0,
          f"{res['latency_ms']:.0f} ms for {heads.shape}")

    # Deepest drawdown should be near the well
    t_final = heads[-1]
    iy, ix = np.unravel_index(np.argmin(t_final), t_final.shape)
    x_at = res["x"][ix]
    y_at = res["y"][iy]
    wx, wy = cfg.well_xy
    dist = ((x_at - wx) ** 2 + (y_at - wy) ** 2) ** 0.5
    check("deepest drawdown near well", dist < 60.0,
          f"min-head cell at ({x_at:.0f}, {y_at:.0f}), {dist:.0f} m from well")

    # --- Stage continuity at tau -------------------------------------------
    print("\n2) Stage-1/Stage-2 continuity at t = tau")
    q_norm = cfg.normalize_q(-40000.0)
    xy = np.array([[0.0, 0.0], [wx, wy], [-300.0, 200.0]], dtype=np.float32)
    h1 = engine._heads_at(xy, cfg.tau, q_norm)                 # stage 1
    hstar = engine._hstar(xy, q_norm)
    h2 = engine._heads_at(xy, cfg.tau + 1e-6, q_norm, hstar)   # stage 2
    max_jump = float(np.max(np.abs(h1 - h2)))
    check("continuity |h1(tau) - h2(tau+)| < 1e-3 m", max_jump < 1e-3,
          f"max jump {max_jump:.2e} m")

    # --- Monotonic sensitivity ---------------------------------------------
    print("\n3) Sensitivity: stronger pumping => deeper drawdown (t = 30 d)")
    dd = {}
    for q in (-25000.0, -40000.0, -50000.0):
        r = engine.well_drawdown_series(q, times=[30.0])
        dd[q] = r["drawdown"][0]
        print(f"     Q={q:>9,.0f}  drawdown at well = {dd[q]:6.3f} m")
    check("monotonic", dd[-25000.0] < dd[-40000.0] < dd[-50000.0])

    # --- Percent mapping ----------------------------------------------------
    print("\n4) Percent-change mapping")
    check("+25% -> Q_min", abs(cfg.q_from_percent(25.0) - cfg.q_min) < 1e-6)
    check("-37.5% -> Q_max", abs(cfg.q_from_percent(-37.5) - cfg.q_max) < 1e-6)
    lo, hi = cfg.percent_bounds()
    check("bounds = (-37.5, +25)", abs(lo + 37.5) < 1e-6 and abs(hi - 25.0) < 1e-6,
          f"({lo}, {hi})")

    # --- Range validation ----------------------------------------------------
    print("\n5) Q-range validation")
    try:
        engine.predict_grid(-60000.0)
        check("out-of-range raises", False)
    except QOutOfRangeError as e:
        check("out-of-range raises", True, str(e)[:60] + "...")

    # --- Goal seek -----------------------------------------------------------
    print("\n6) Goal seek (drawdown at well <= 2 m, t = 30 d)")
    gs = engine.goal_seek_max_pumping(wx, wy, max_drawdown_m=2.0)
    print(f"     feasible={gs['feasible']}  best_Q={gs['best_q_rate']:,.0f} "
          f"({gs['best_percent_change']:+.1f}%)  "
          f"drawdown={gs['predicted_drawdown']:.3f} m  "
          f"[{gs['n_evaluations']} evals in {gs['latency_ms']:.0f} ms]")
    check("goal-seek returns Q in range",
          cfg.q_min <= gs["best_q_rate"] <= cfg.q_max)
    check("goal-seek fast", gs["latency_ms"] < 5000.0,
          f"{gs['latency_ms']:.0f} ms")

    print(f"\n{'=' * 50}")
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
