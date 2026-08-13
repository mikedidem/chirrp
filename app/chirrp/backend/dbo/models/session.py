from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
import uuid

Base = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Session(Base):
    """A persisted Studio planning chat a stakeholder can resume later."""

    __tablename__ = "sessions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=False, default="New session")
    participant_id = Column(PG_UUID(as_uuid=True), nullable=True)  # owner (local profile)
    shared = Column(Boolean, nullable=False, default=True)
    summary_text = Column(Text, nullable=True)
    summary_updated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow,
        nullable=False)

    messages = relationship(
        "SessionMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="SessionMessage.created_at",
    )

    def __repr__(self):
        return f"<Session(id={self.id}, title={self.title!r})>"


class SessionMessage(Base):
    """One message (user instruction or assistant reply) within a session."""

    __tablename__ = "session_messages"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        PG_UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(Text, nullable=False)            # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    valid = Column(Boolean, nullable=True)
    scenario_name = Column(Text, nullable=True)
    parse_json = Column(JSONB, nullable=True)
    latency_json = Column(JSONB, nullable=True)
    sim_meta = Column(JSONB, nullable=True)        # {percent_change, q_rate}
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    session = relationship("Session", back_populates="messages")

    def __repr__(self):
        return f"<SessionMessage(session={self.session_id}, role={self.role})>"
