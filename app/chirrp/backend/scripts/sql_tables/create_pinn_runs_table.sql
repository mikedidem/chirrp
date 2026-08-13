-- Saved PINN surrogate scenario runs (Hydro-AI framework).
-- One row per saved scenario: the parsed instruction, the pumping change,
-- summary statistics, the final-time head grid for re-display, and the
-- per-step latency breakdown reported to the UI.

CREATE TABLE IF NOT EXISTS pinn_runs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    scenario_name   TEXT UNIQUE NOT NULL,
    instruction     TEXT,
    percent_change  DOUBLE PRECISION NOT NULL,
    q_rate          DOUBLE PRECISION NOT NULL,
    head_min        DOUBLE PRECISION,
    head_max        DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    well_drawdown_final DOUBLE PRECISION,
    head_grid       JSONB,          -- final-time head field (resolution x resolution)
    grid_meta       JSONB,          -- {x: [...], y: [...], t: <days>, resolution: N}
    latency_json    JSONB,          -- {llm_parse_ms, pinn_ms, summary_ms, total_ms}
    engine_metrics  JSONB,          -- validation results vs MODFLOW, if computed
    summary_text    TEXT,           -- LLM plain-language summary
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_pinn_runs_scenario_name ON pinn_runs (scenario_name);
CREATE INDEX IF NOT EXISTS idx_pinn_runs_created_at ON pinn_runs (created_at DESC);

COMMENT ON TABLE pinn_runs IS 'Saved PINN surrogate scenario runs with latency and validation metadata';
