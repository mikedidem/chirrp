from sqlalchemy import Column, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
import uuid

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Participant(Base):
    """A local stakeholder profile (no authentication) for participatory work."""

    __tablename__ = "participants"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    role = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self):
        return f"<Participant(id={self.id}, name={self.name!r})>"


class SessionNote(Base):
    """A note or decision in a session's participatory decision log."""

    __tablename__ = "session_notes"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # FK constraints are enforced in the DB (create_participants_tables.sql);
    # kept as plain columns here because `sessions` is mapped under a different
    # declarative Base, which the ORM can't resolve a ForeignKey across.
    session_id = Column(PG_UUID(as_uuid=True), nullable=False)
    participant_id = Column(PG_UUID(as_uuid=True), nullable=True)
    kind = Column(Text, nullable=False, default="note")   # 'note' | 'decision'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    def __repr__(self):
        return f"<SessionNote(session={self.session_id}, kind={self.kind})>"
