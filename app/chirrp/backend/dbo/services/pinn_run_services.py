import logging
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from chirrp.backend.dbo.models.pinn_run import PinnRun

logger = logging.getLogger(__name__)


async def get_pinn_run_by_scenario(session: AsyncSession,
                                   scenario_name: str) -> Optional[PinnRun]:
    """Fetch a saved PINN run by scenario name."""
    try:
        result = await session.execute(
            select(PinnRun).where(PinnRun.scenario_name == scenario_name)
        )
        return result.scalars().first()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error fetching pinn run: {e}")
        raise


async def get_all_pinn_runs(session: AsyncSession) -> List[PinnRun]:
    """All saved PINN runs, newest first."""
    try:
        result = await session.execute(
            select(PinnRun).order_by(PinnRun.created_at.desc())
        )
        return result.scalars().all()
    except SQLAlchemyError as e:
        logger.error(f"SQLAlchemy error fetching pinn runs: {e}")
        raise


async def create_or_update_pinn_run(session: AsyncSession,
                                    run: PinnRun) -> PinnRun:
    """Save a PINN run, replacing any existing run with the same name."""
    try:
        existing = await get_pinn_run_by_scenario(session, run.scenario_name)
        if existing:
            existing.instruction = run.instruction
            existing.percent_change = run.percent_change
            existing.q_rate = run.q_rate
            existing.head_min = run.head_min
            existing.head_max = run.head_max
            existing.max_drawdown = run.max_drawdown
            existing.well_drawdown_final = run.well_drawdown_final
            existing.head_grid = run.head_grid
            existing.grid_meta = run.grid_meta
            existing.latency_json = run.latency_json
            existing.engine_metrics = run.engine_metrics
            existing.summary_text = run.summary_text
            await session.commit()
            await session.refresh(existing)
            return existing

        session.add(run)
        await session.commit()
        await session.refresh(run)
        return run
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error saving pinn run: {e}")
        raise


async def delete_pinn_run(session: AsyncSession, scenario_name: str) -> bool:
    """Delete a saved PINN run by scenario name."""
    try:
        run = await get_pinn_run_by_scenario(session, scenario_name)
        if run:
            await session.delete(run)
            await session.commit()
            return True
        return False
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error(f"SQLAlchemy error deleting pinn run: {e}")
        raise
