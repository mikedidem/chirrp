-- Participatory planning: lightweight local participant profiles (no auth) so a
-- workshop on one machine can attribute sessions and decisions to named
-- stakeholders, and keep a shared decision log.

CREATE TABLE IF NOT EXISTS participants (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    role        TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The participatory decision log: notes and decisions on a session.
CREATE TABLE IF NOT EXISTS session_notes (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id      UUID NOT NULL REFERENCES sessions (id) ON DELETE CASCADE,
    participant_id  UUID REFERENCES participants (id) ON DELETE SET NULL,
    kind            TEXT NOT NULL DEFAULT 'note',   -- 'note' | 'decision'
    content         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Session ownership / sharing (additive; existing rows default to shared, no owner).
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS participant_id UUID
    REFERENCES participants (id) ON DELETE SET NULL;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS shared BOOLEAN NOT NULL DEFAULT true;

CREATE INDEX IF NOT EXISTS idx_session_notes_session
    ON session_notes (session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_participant
    ON sessions (participant_id);

COMMENT ON TABLE participants IS 'Local stakeholder profiles (no auth) for participatory sessions';
COMMENT ON TABLE session_notes IS 'Decision log: notes/decisions attributed to participants';
