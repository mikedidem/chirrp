import logging
import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from chirrp.backend.dbo.models.participant import Participant, SessionNote

logger = logging.getLogger(__name__)


async def create_participant(session: AsyncSession, name: str,
                             role: Optional[str] = None) -> Participant:
    try:
        row = Participant(name=name.strip()[:80] or "Participant",
                          role=(role or None))
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error creating participant: {e}")
        raise


async def list_participants(session: AsyncSession) -> List[Participant]:
    try:
        result = await session.execute(
            select(Participant).order_by(Participant.created_at.asc())
        )
        return result.scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error listing participants: {e}")
        raise


async def add_note(session: AsyncSession,
                  session_id: uuid.UUID,
                  content: str,
                  *,
                  participant_id: Optional[uuid.UUID] = None,
                  kind: str = "note") -> SessionNote:
    try:
        note = SessionNote(
            session_id=session_id,
            participant_id=participant_id,
            kind=(kind if kind in ("note", "decision") else "note"),
            content=content,
        )
        session.add(note)
        await session.commit()
        await session.refresh(note)
        return note
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error adding session note: {e}")
        raise


async def list_notes(session: AsyncSession,
                     session_id: uuid.UUID) -> List[SessionNote]:
    try:
        result = await session.execute(
            select(SessionNote)
            .where(SessionNote.session_id == session_id)
            .order_by(SessionNote.created_at.asc())
        )
        return result.scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error listing session notes: {e}")
        raise
