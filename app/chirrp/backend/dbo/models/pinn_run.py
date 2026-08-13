from sqlalchemy import Column, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone
import uuid

Base = declarative_base()


class PinnRun(Base):
    """A saved PINN surrogate scenario run."""

    __tablename__ = "pinn_runs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    scenario_name = Column(Text, unique=True, nullable=False)
    instruction = Column(Text, nullable=True)
    percent_change = Column(Float, nullable=False)
    q_rate = Column(Float, nullable=False)
    head_min = Column(Float, nullable=True)
    head_max = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    well_drawdown_final = Column(Float, nullable=True)
    head_grid = Column(JSONB, nullable=True)
    grid_meta = Column(JSONB, nullable=True)
    latency_json = Column(JSONB, nullable=True)
    engine_metrics = Column(JSONB, nullable=True)
    summary_text = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self):
        return (f"<PinnRun(scenario_name={self.scenario_name}, "
                f"q_rate={self.q_rate})>")
