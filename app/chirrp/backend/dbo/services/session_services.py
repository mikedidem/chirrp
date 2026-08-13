import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import selectinload

from chirrp.backend.dbo.models.session import Session, SessionMessage

logger = logging.getLogger(__name__)


async def create_session(session: AsyncSession,
                         title: Optional[str] = None,
                         participant_id: Optional[uuid.UUID] = None,
                         shared: bool = True) -> Session:
    """Create a new (empty) planning session, optionally owned by a participant."""
    try:
        row = Session(title=(title or "New session").strip()[:120]
                      or "New session",
                      participant_id=participant_id,
                      shared=shared)
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return row
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error creating session: {e}")
        raise


async def list_sessions(session: AsyncSession,
                        participant_id: Optional[uuid.UUID] = None,
                        scope: str = "all") -> List[Session]:
    """Sessions, most recently active first.

    scope='mine' (with participant_id) → only that participant's sessions;
    scope='all' → that participant's sessions plus every shared session.
    """
    try:
        stmt = select(Session).order_by(Session.updated_at.desc())
        if participant_id is not None:
            if scope == "mine":
                stmt = stmt.where(Session.participant_id == participant_id)
            else:
                stmt = stmt.where(
                    (Session.participant_id == participant_id)
                    | (Session.shared.is_(True))
                )
        result = await session.execute(stmt)
        return result.scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error listing sessions: {e}")
        raise


async def get_session_with_messages(session: AsyncSession,
                                    session_id: uuid.UUID) -> Optional[Session]:
    """Fetch a session and its ordered messages (resume-from-here)."""
    try:
        result = await session.execute(
            select(Session)
            .where(Session.id == session_id)
            .options(selectinload(Session.messages))
        )
        return result.scalars().first()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error fetching session: {e}")
        raise


async def append_message(session: AsyncSession,
                        session_id: uuid.UUID,
                        role: str,
                        content: str,
                        *,
                        valid: Optional[bool] = None,
                        scenario_name: Optional[str] = None,
                        parse_json: Optional[dict] = None,
                        latency_json: Optional[dict] = None,
                        sim_meta: Optional[dict] = None) -> SessionMessage:
    """Append a message and bump the parent session's updated_at."""
    try:
        msg = SessionMessage(
            session_id=session_id,
            role=role,
            content=content,
            valid=valid,
            scenario_name=scenario_name,
            parse_json=parse_json,
            latency_json=latency_json,
            sim_meta=sim_meta,
        )
        session.add(msg)
        parent = await session.get(Session, session_id)
        if parent is not None:
            parent.updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(msg)
        return msg
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error appending message: {e}")
        raise


async def rename_session(session: AsyncSession,
                        session_id: uuid.UUID,
                        title: str) -> Optional[Session]:
    try:
        row = await session.get(Session, session_id)
        if row is None:
            return None
        row.title = title.strip()[:120] or row.title
        await session.commit()
        await session.refresh(row)
        return row
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error renaming session: {e}")
        raise


async def set_summary(session: AsyncSession,
                     session_id: uuid.UUID,
                     summary_text: str) -> Optional[Session]:
    try:
        row = await session.get(Session, session_id)
        if row is None:
            return None
        row.summary_text = summary_text
        row.summary_updated_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(row)
        return row
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error saving session summary: {e}")
        raise


async def delete_session(session: AsyncSession,
                        session_id: uuid.UUID) -> bool:
    try:
        row = await session.get(Session, session_id)
        if row is None:
            return False
        await session.delete(row)
        await session.commit()
        return True
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error deleting session: {e}")
        raise
