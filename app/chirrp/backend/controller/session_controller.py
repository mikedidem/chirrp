"""
Planning-session API — persistence for the Studio chat.

Lets stakeholders leave and resume their participatory planning work, and read
a plain-language recap of what a session achieved (token-saving: the recap is
stored, so past work need not be re-sent to the LLM each visit).
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from chirrp.backend.dbo.database import get_session
from chirrp.backend.dbo.models.session import Session, SessionMessage
from chirrp.backend.dbo.services.session_services import (
    create_session,
    delete_session,
    get_session_with_messages,
    list_sessions,
    rename_session,
    set_summary,
)
from chirrp.backend.dbo.services.participant_services import add_note, list_notes
from chirrp.backend.dbo.models.participant import SessionNote
from chirrp.core.llm_wrapper.graph.pinn_scenario_graph import summarize_session

router = APIRouter(prefix="/sessions", tags=["Sessions"])


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: Optional[str] = None
    participant_id: Optional[uuid.UUID] = None
    shared: bool = True


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1)


class NoteRequest(BaseModel):
    content: str = Field(..., min_length=1)
    participant_id: Optional[uuid.UUID] = None
    kind: str = Field(default="note")  # 'note' | 'decision'


class NoteOut(BaseModel):
    id: uuid.UUID
    session_id: uuid.UUID
    participant_id: Optional[uuid.UUID] = None
    kind: str
    content: str
    created_at: Optional[str] = None


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    valid: Optional[bool] = None
    scenario_name: Optional[str] = None
    parse_json: Optional[dict] = None
    latency_json: Optional[dict] = None
    sim_meta: Optional[dict] = None
    created_at: Optional[str] = None


class SessionSummaryOut(BaseModel):
    id: uuid.UUID
    title: str
    participant_id: Optional[uuid.UUID] = None
    shared: bool = True
    has_summary: bool
    summary_text: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SessionDetailOut(SessionSummaryOut):
    messages: List[MessageOut] = []


def _iso(dt) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _message_out(m: SessionMessage) -> MessageOut:
    return MessageOut(
        id=m.id,
        role=m.role,
        content=m.content,
        valid=m.valid,
        scenario_name=m.scenario_name,
        parse_json=m.parse_json,
        latency_json=m.latency_json,
        sim_meta=m.sim_meta,
        created_at=_iso(m.created_at),
    )


def _summary_out(s: Session) -> SessionSummaryOut:
    return SessionSummaryOut(
        id=s.id,
        title=s.title,
        participant_id=s.participant_id,
        shared=bool(s.shared),
        has_summary=bool(s.summary_text),
        summary_text=s.summary_text,
        created_at=_iso(s.created_at),
        updated_at=_iso(s.updated_at),
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@router.post("", response_model=SessionSummaryOut)
async def post_session(request: CreateSessionRequest,
                       db: AsyncSession = Depends(get_session)):
    row = await create_session(db, request.title, request.participant_id,
                               request.shared)
    return _summary_out(row)


@router.get("", response_model=List[SessionSummaryOut])
async def get_sessions(participant_id: Optional[uuid.UUID] = None,
                       scope: str = "all",
                       db: AsyncSession = Depends(get_session)):
    rows = await list_sessions(db, participant_id=participant_id, scope=scope)
    return [_summary_out(r) for r in rows]


@router.get("/{session_id}", response_model=SessionDetailOut)
async def get_one_session(session_id: uuid.UUID,
                          db: AsyncSession = Depends(get_session)):
    row = await get_session_with_messages(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    out = SessionDetailOut(**_summary_out(row).model_dump())
    out.messages = [_message_out(m) for m in row.messages]
    return out


@router.patch("/{session_id}", response_model=SessionSummaryOut)
async def patch_session(session_id: uuid.UUID,
                        request: RenameSessionRequest,
                        db: AsyncSession = Depends(get_session)):
    row = await rename_session(db, session_id, request.title)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _summary_out(row)


@router.delete("/{session_id}")
async def remove_session(session_id: uuid.UUID,
                         db: AsyncSession = Depends(get_session)):
    ok = await delete_session(db, session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": str(session_id)}


@router.post("/{session_id}/summarize", response_model=SessionSummaryOut)
async def summarize_one_session(session_id: uuid.UUID,
                                db: AsyncSession = Depends(get_session)):
    """Generate + store an 'achievements' recap of the session."""
    row = await get_session_with_messages(db, session_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found")

    payload = [
        {"role": m.role, "content": m.content, "scenario_name": m.scenario_name}
        for m in row.messages
    ]
    result = await run_in_threadpool(summarize_session, payload)
    updated = await set_summary(db, session_id, result["summary"])
    return _summary_out(updated)


def _note_out(n: SessionNote) -> NoteOut:
    return NoteOut(
        id=n.id,
        session_id=n.session_id,
        participant_id=n.participant_id,
        kind=n.kind,
        content=n.content,
        created_at=_iso(n.created_at),
    )


@router.get("/{session_id}/notes", response_model=List[NoteOut])
async def get_notes(session_id: uuid.UUID,
                    db: AsyncSession = Depends(get_session)):
    """The session's participatory decision log (notes + decisions)."""
    notes = await list_notes(db, session_id)
    return [_note_out(n) for n in notes]


@router.post("/{session_id}/notes", response_model=NoteOut)
async def post_note(session_id: uuid.UUID,
                    request: NoteRequest,
                    db: AsyncSession = Depends(get_session)):
    note = await add_note(db, session_id, request.content,
                          participant_id=request.participant_id,
                          kind=request.kind)
    return _note_out(note)
