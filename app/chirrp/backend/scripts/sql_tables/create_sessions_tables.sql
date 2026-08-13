-- Stakeholder planning sessions (Hydro-AI framework).
-- A session is a persisted Studio chat so a stakeholder can leave and later
-- continue where they stopped, and read a plain-language recap of what was
-- achieved without re-sending the whole thread to the LLM.

CREATE TABLE IF NOT EXISTS sessions (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title               TEXT NOT NULL DEFAULT 'New session',
    summary_text        TEXT,            -- LLM "what was achieved" recap
    summary_updated_at  TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS session_messages (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    role            TEXT NOT NULL,       -- 'user' | 'assistant'
    content         TEXT NOT NULL,
    valid           BOOLEAN,             -- assistant: was the scenario valid
    scenario_name   TEXT,                -- linked pinn_runs scenario, if saved
    parse_json      JSONB,               -- parse result (percent_change, source)
    latency_json    JSONB,               -- {llm_parse_ms, pinn_ms, summary_ms, total_ms}
    sim_meta        JSONB,               -- {percent_change, q_rate} to restore charts
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_session_messages_session
    ON session_messages (session_id, created_at);

COMMENT ON TABLE sessions IS 'Persisted Studio planning chats with achievements summary';
COMMENT ON TABLE session_messages IS 'Ordered messages belonging to a planning session';
