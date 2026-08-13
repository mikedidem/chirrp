"""
PINN surrogate API — the physical layer of the Hydro-AI framework.

Endpoints expose the surrogate (millisecond inference), the chat-to-model
pipeline (LLM parse -> validate -> simulate -> summarize), MODFLOW
validation on the same well problem, and goal-seeking scenario search.
Every response carries wall-clock latency so the UI can report the
participatory-latency story transparently.
"""

import time
import uuid
from datetime import datetime
from typing import List, Literal, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from chirrp.core.pinn import PinnConfig, PinnEngine, QOutOfRangeError
from chirrp.core.validation import (
    ReferenceHeads,
    compare_heads,
    run_lrs_model,
)
from chirrp.core.llm_wrapper.graph.pinn_scenario_graph import (
    parse_pumping_instruction,
    summarize_scenario_result,
)
from chirrp.backend.dbo.database import get_session
from chirrp.backend.dbo.models.pinn_run import PinnRun
from chirrp.backend.dbo.services.pinn_run_services import (
    create_or_update_pinn_run,
    delete_pinn_run,
    get_all_pinn_runs,
    get_pinn_run_by_scenario,
)
from chirrp.backend.dbo.services.session_services import append_message

router = APIRouter(prefix="/pinn", tags=["PINN Surrogate"])

# Engine and reference are process-wide singletons (model loads in ~15 ms,
# but there is no reason to reload per request).
_engine: Optional[PinnEngine] = None
_reference = ReferenceHeads()


def get_engine() -> PinnEngine:
    global _engine
    if _engine is None:
        _engine = PinnEngine()
    return _engine


def _round_grid(arr: np.ndarray, decimals: int = 3) -> list:
    return np.round(arr.astype(np.float64), decimals).tolist()


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class SimulateRequest(BaseModel):
    scenario_name: Optional[str] = Field(
        default=None, description="If set, the run is saved under this name")
    percent_change: Optional[float] = Field(
        default=None,
        description="Pumping change in percent (positive = more pumping)")
    q_rate: Optional[float] = Field(
        default=None, description="Absolute pumping rate (m³/day, negative)")
    resolution: int = Field(default=100, ge=20, le=200)
    instruction: Optional[str] = Field(
        default=None, description="Original NL instruction (stored only)")


class ChatRequest(BaseModel):
    instruction: str = Field(..., min_length=1)
    scenario_name: Optional[str] = None
    resolution: int = Field(default=100, ge=20, le=200)
    session_id: Optional[uuid.UUID] = Field(
        default=None,
        description="If set, the user instruction and assistant reply are "
                    "persisted to this planning session.")


class ProbePoint(BaseModel):
    x: float = Field(..., ge=-500.0, le=500.0)
    y: float = Field(..., ge=-500.0, le=500.0)
    t: float = Field(..., ge=0.0, le=30.0)


class ProbeRequest(BaseModel):
    q_rate: Optional[float] = None
    percent_change: Optional[float] = None
    points: List[ProbePoint]


class ValidateRequest(BaseModel):
    q_rate: Optional[float] = None
    percent_change: Optional[float] = None
    live: bool = Field(
        default=False,
        description="Force a live mf2005 run even if a precomputed anchor "
                    "exists (live runs take ~30-60 s)")


class GoalSeekRequest(BaseModel):
    x: float = Field(default=201.67, ge=-500.0, le=500.0)
    y: float = Field(default=-98.33, ge=-500.0, le=500.0)
    max_drawdown_m: float = Field(..., gt=0.0)
    t: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    verify_with_modflow: bool = False


class CompareItem(BaseModel):
    """One scenario to compare — by percent change, absolute rate, or a
    previously saved scenario name."""
    label: Optional[str] = None
    percent_change: Optional[float] = None
    q_rate: Optional[float] = None
    scenario_name: Optional[str] = None


class CompareRequest(BaseModel):
    items: List[CompareItem] = Field(..., min_length=2, max_length=4)
    resolution: int = Field(default=100, ge=20, le=200)
    include_grids: bool = Field(
        default=False,
        description="Include each scenario's final-time head grid (heavier).")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _resolve_q(percent_change: Optional[float],
               q_rate: Optional[float],
               config: PinnConfig) -> float:
    if q_rate is not None:
        return float(q_rate)
    if percent_change is not None:
        return config.q_from_percent(float(percent_change))
    raise HTTPException(
        status_code=422,
        detail="Provide either percent_change or q_rate.")


def _simulation_payload(result: dict) -> dict:
    """JSON-safe simulation response (grids rounded to mm precision)."""
    return {
        "q_rate": result["q_rate"],
        "percent_change": result["percent_change"],
        "times": result["times"],
        "x": result["x"],
        "y": result["y"],
        "heads": _round_grid(result["heads"]),
        "head_min": result["head_min"],
        "head_max": result["head_max"],
        "max_drawdown": result["max_drawdown"],
        "well_xy": result["well_xy"],
        "well_drawdown_series": [round(v, 4)
                                 for v in result["well_drawdown_series"]],
        "latency_ms": result["latency_ms"],
        "initial_head": 90.0,
    }


async def _persist_run(db: AsyncSession, scenario_name: str,
                       instruction: Optional[str], result: dict,
                       latency_json: dict,
                       summary_text: Optional[str] = None) -> bool:
    """Save a run; returns False (instead of raising) if the DB is down so
    a broken database never blocks interactive simulation."""
    final_heads = result["heads"][-1]
    run = PinnRun(
        scenario_name=scenario_name,
        instruction=instruction,
        percent_change=result["percent_change"],
        q_rate=result["q_rate"],
        head_min=result["head_min"],
        head_max=result["head_max"],
        max_drawdown=result["max_drawdown"],
        well_drawdown_final=result["well_drawdown_series"][-1],
        head_grid=_round_grid(final_heads),
        grid_meta={
            "x": result["x"], "y": result["y"],
            "t": result["times"][-1],
            "resolution": len(result["x"]),
            "well_xy": result["well_xy"],
        },
        latency_json=latency_json,
        summary_text=summary_text,
    )
    try:
        await create_or_update_pinn_run(db, run)
        return True
    except Exception:
        return False


async def _record_session_exchange(
    db: AsyncSession,
    session_id: Optional[uuid.UUID],
    user_text: str,
    assistant_text: str,
    *,
    valid: bool,
    parse: Optional[dict] = None,
    latency: Optional[dict] = None,
    scenario_name: Optional[str] = None,
    sim_meta: Optional[dict] = None,
) -> None:
    """Persist a chat turn to a session; never blocks chat if the DB is down."""
    if session_id is None:
        return
    try:
        await append_message(db, session_id, "user", user_text)
        await append_message(
            db, session_id, "assistant", assistant_text,
            valid=valid, parse_json=parse, latency_json=latency,
            scenario_name=scenario_name, sim_meta=sim_meta,
        )
    except Exception:
        # Session persistence is best-effort; a broken DB must not break chat.
        pass


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@router.get("/meta")
def get_meta():
    """Static facts the UI needs: domain, ranges, anchors, well position."""
    config = get_engine().config
    lo, hi = config.percent_bounds()
    return {
        "domain": {
            "x": [config.domain[0], config.domain[1]],
            "y": [config.domain[2], config.domain[3]],
            "t": [config.domain[4], config.domain[5]],
        },
        "well_xy": list(config.well_xy),
        "initial_head": config.initial_head,
        "q_min": config.q_min,
        "q_max": config.q_max,
        "q_baseline": config.q_baseline,
        "percent_bounds": [lo, hi],
        "reference_anchors": _reference.q_values,
        "model": {
            "type": "Parameterized physics-informed neural network (two-stage)",
            "inputs": "(x, y, t, Q)",
            "hidden": "5 layers x 50 neurons, sin-residual",
            "constraint": "hard Dirichlet/IC enforcement",
        },
    }


@router.post("/simulate")
async def simulate(request: SimulateRequest,
                   db: AsyncSession = Depends(get_session)):
    engine = get_engine()
    q = _resolve_q(request.percent_change, request.q_rate, engine.config)
    try:
        result = await run_in_threadpool(
            engine.predict_grid, q, None, request.resolution)
    except QOutOfRangeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    payload = _simulation_payload(result)
    if request.scenario_name:
        saved = await _persist_run(
            db, request.scenario_name.strip(), request.instruction, result,
            {"pinn_ms": result["latency_ms"],
             "total_ms": result["latency_ms"]})
        payload["saved_as"] = request.scenario_name.strip() if saved else None
    return payload


@router.post("/chat")
async def chat(request: ChatRequest,
               db: AsyncSession = Depends(get_session)):
    """Chat-to-model pipeline: parse -> validate -> simulate -> summarize."""
    engine = get_engine()
    config = engine.config
    lo, hi = config.percent_bounds()
    t_start = time.perf_counter()

    parse = await run_in_threadpool(
        parse_pumping_instruction, request.instruction, lo, hi)

    if not parse["is_valid"]:
        latency = {
            "llm_parse_ms": parse["latency_ms"],
            "total_ms": (time.perf_counter() - t_start) * 1000.0,
        }
        await _record_session_exchange(
            db, request.session_id, request.instruction,
            parse["error"] or "The instruction could not be applied.",
            valid=False, parse=parse, latency=latency)
        return {
            "is_valid": False,
            "parse": parse,
            "error": parse["error"],
            "suggestion": parse["suggestion"],
            "latency": latency,
        }

    q = config.q_from_percent(parse["percent_change"])
    result = await run_in_threadpool(
        engine.predict_grid, q, None, request.resolution)

    summary = await run_in_threadpool(
        summarize_scenario_result,
        request.instruction,
        {
            "percent_change": result["percent_change"],
            "q_rate": result["q_rate"],
            "max_drawdown": result["max_drawdown"],
            "head_min": result["head_min"],
            "well_drawdown_final": result["well_drawdown_series"][-1],
            "latency_ms": result["latency_ms"],
        },
    )

    latency = {
        "llm_parse_ms": parse["latency_ms"],
        "pinn_ms": result["latency_ms"],
        "summary_ms": summary["latency_ms"],
        "total_ms": (time.perf_counter() - t_start) * 1000.0,
    }

    if request.scenario_name:
        saved_as = request.scenario_name.strip()
    else:
        saved_as = "chat-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    saved = await _persist_run(db, saved_as, request.instruction, result,
                               latency, summary_text=summary["summary"])
    if not saved:
        saved_as = None

    await _record_session_exchange(
        db, request.session_id, request.instruction, summary["summary"],
        valid=True, parse=parse, latency=latency, scenario_name=saved_as,
        sim_meta={
            "percent_change": result["percent_change"],
            "q_rate": result["q_rate"],
        })

    return {
        "is_valid": True,
        "parse": parse,
        "simulation": _simulation_payload(result),
        "summary": summary["summary"],
        "summary_source": summary["source"],
        "latency": latency,
        "saved_as": saved_as,
    }


@router.post("/probe")
async def probe(request: ProbeRequest):
    """Mesh-free head/drawdown at arbitrary (x, y, t) points."""
    engine = get_engine()
    q = _resolve_q(request.percent_change, request.q_rate, engine.config)
    try:
        return await run_in_threadpool(
            engine.predict_points, q,
            [p.model_dump() for p in request.points])
    except QOutOfRangeError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/validate")
async def validate(request: ValidateRequest):
    """Surrogate vs MODFLOW on the identical scenario.

    Uses the precomputed reference when the requested Q matches an anchor
    (instant); otherwise — or when live=true — runs mf2005 (~30-60 s).
    """
    engine = get_engine()
    q = _resolve_q(request.percent_change, request.q_rate, engine.config)
    try:
        engine.config.check_q(q)
    except QOutOfRangeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    modflow = None
    if not request.live:
        modflow = _reference.get(q)
    if modflow is None:
        modflow = await run_in_threadpool(run_lrs_model, q, 100)

    pinn = await run_in_threadpool(
        engine.predict_grid, q, modflow["times"], 100)

    metrics = compare_heads(pinn["heads"], modflow["heads"])
    err_t = pinn["heads"].astype(np.float64) - \
        modflow["heads"].astype(np.float64)
    rmse_per_time = np.sqrt((err_t ** 2).mean(axis=(1, 2)))

    # Drawdown at the well from both engines (nearest grid cell)
    x = np.asarray(modflow["x"])
    y = np.asarray(modflow["y"])
    wx, wy = engine.config.well_xy
    j = int(np.argmin(np.abs(x - wx)))
    i = int(np.argmin(np.abs(y - wy)))
    h0 = engine.config.initial_head

    return {
        "q_rate": q,
        "percent_change": engine.config.percent_from_q(q),
        "precomputed": bool(modflow.get("precomputed", False)),
        "times": modflow["times"],
        "x": modflow["x"],
        "y": modflow["y"],
        "pinn_heads_final": _round_grid(pinn["heads"][-1]),
        "modflow_heads_final": _round_grid(modflow["heads"][-1]),
        "error_field_final": _round_grid(metrics.pop("error_field_final")),
        "metrics": metrics,
        "rmse_per_time": np.round(rmse_per_time, 4).tolist(),
        "well_drawdown": {
            "pinn": [round(h0 - float(h), 4)
                     for h in pinn["heads"][:, i, j]],
            "modflow": [round(h0 - float(h), 4)
                        for h in modflow["heads"][:, i, j]],
        },
        "latency": {
            "pinn_ms": pinn["latency_ms"],
            "modflow_s": modflow.get("runtime_s"),
            "speedup": (modflow["runtime_s"] * 1000.0 / pinn["latency_ms"])
            if modflow.get("runtime_s") else None,
        },
    }


@router.post("/goal-seek")
async def goal_seek(request: GoalSeekRequest):
    """Strongest pumping that keeps drawdown at a point under a limit."""
    engine = get_engine()
    result = await run_in_threadpool(
        engine.goal_seek_max_pumping,
        request.x, request.y, request.max_drawdown_m, request.t)

    if request.verify_with_modflow and result["feasible"]:
        t_eval = result["constraint"]["t"]
        modflow = await run_in_threadpool(
            run_lrs_model, result["best_q_rate"], 100, [t_eval])
        x = np.asarray(modflow["x"])
        y = np.asarray(modflow["y"])
        j = int(np.argmin(np.abs(x - request.x)))
        i = int(np.argmin(np.abs(y - request.y)))
        mf_drawdown = 90.0 - float(modflow["heads"][-1, i, j])
        result["modflow_verification"] = {
            "drawdown_m": round(mf_drawdown, 4),
            "satisfies_constraint": mf_drawdown <= request.max_drawdown_m,
            "runtime_s": modflow["runtime_s"],
        }
    return result


@router.post("/compare")
async def compare(request: CompareRequest,
                  db: AsyncSession = Depends(get_session)):
    """Compare 2-4 scenarios side by side on the surrogate.

    Returns aligned drawdown-at-well series (for overlay) and summary stats per
    scenario, plus deltas relative to the first item (the baseline). Each run is
    a millisecond surrogate call, so comparison stays interactive.
    """
    engine = get_engine()
    config = engine.config

    async def _resolve_item_q(item: CompareItem) -> tuple[float, str]:
        if item.scenario_name:
            run = await get_pinn_run_by_scenario(db, item.scenario_name)
            if not run:
                raise HTTPException(
                    status_code=404,
                    detail=f"Scenario '{item.scenario_name}' not found")
            return float(run.q_rate), item.label or item.scenario_name
        q = _resolve_q(item.percent_change, item.q_rate, config)
        default_label = f"{config.percent_from_q(q):+.1f}%"
        return q, (item.label or default_label)

    scenarios = []
    for item in request.items:
        q, label = await _resolve_item_q(item)
        try:
            result = await run_in_threadpool(
                engine.predict_grid, q, None, request.resolution)
        except QOutOfRangeError as e:
            raise HTTPException(status_code=422, detail=str(e))

        entry = {
            "label": label,
            "percent_change": result["percent_change"],
            "q_rate": result["q_rate"],
            "times": result["times"],
            "well_drawdown_series": [round(v, 4)
                                     for v in result["well_drawdown_series"]],
            "max_drawdown": result["max_drawdown"],
            "head_min": result["head_min"],
            "well_drawdown_final": result["well_drawdown_series"][-1],
            "latency_ms": result["latency_ms"],
        }
        if request.include_grids:
            entry["x"] = result["x"]
            entry["y"] = result["y"]
            entry["head_grid_final"] = _round_grid(result["heads"][-1])
        scenarios.append(entry)

    base = scenarios[0]
    deltas = [
        {
            "label": s["label"],
            "d_max_drawdown": round(s["max_drawdown"] - base["max_drawdown"], 4),
            "d_head_min": round(s["head_min"] - base["head_min"], 4),
            "d_well_drawdown_final":
                round(s["well_drawdown_final"] - base["well_drawdown_final"], 4),
        }
        for s in scenarios
    ]

    return {"baseline_label": base["label"], "scenarios": scenarios,
            "deltas": deltas}


@router.get("/scenarios")
async def list_scenarios(db: AsyncSession = Depends(get_session)):
    runs = await get_all_pinn_runs(db)
    return [
        {
            "scenario_name": r.scenario_name,
            "instruction": r.instruction,
            "percent_change": r.percent_change,
            "q_rate": r.q_rate,
            "max_drawdown": r.max_drawdown,
            "well_drawdown_final": r.well_drawdown_final,
            "latency": r.latency_json,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in runs
    ]


@router.get("/scenarios/{scenario_name}")
async def get_scenario(scenario_name: str,
                       db: AsyncSession = Depends(get_session)):
    run = await get_pinn_run_by_scenario(db, scenario_name)
    if not run:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {
        "scenario_name": run.scenario_name,
        "instruction": run.instruction,
        "percent_change": run.percent_change,
        "q_rate": run.q_rate,
        "head_min": run.head_min,
        "head_max": run.head_max,
        "max_drawdown": run.max_drawdown,
        "well_drawdown_final": run.well_drawdown_final,
        "head_grid": run.head_grid,
        "grid_meta": run.grid_meta,
        "latency": run.latency_json,
        "summary_text": run.summary_text,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.delete("/scenarios/{scenario_name}")
async def delete_scenario(scenario_name: str,
                          db: AsyncSession = Depends(get_session)):
    deleted = await delete_pinn_run(db, scenario_name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Scenario not found")
    return {"deleted": scenario_name}
