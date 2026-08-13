"""
Participant API — lightweight local stakeholder profiles (no authentication).

Lets a workshop on one machine attribute planning sessions and decision-log
entries to named participants. This is a demo-grade identity layer: there is no
password/login, just selectable profiles.
"""

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from chirrp.backend.dbo.database import get_session
from chirrp.backend.dbo.models.participant import Participant
from chirrp.backend.dbo.services.participant_services import (
    create_participant,
    list_participants,
)

router = APIRouter(prefix="/participants", tags=["Participants"])


class CreateParticipantRequest(BaseModel):
    name: str = Field(..., min_length=1)
    role: Optional[str] = None


class ParticipantOut(BaseModel):
    id: uuid.UUID
    name: str
    role: Optional[str] = None
    created_at: Optional[str] = None


def _out(p: Participant) -> ParticipantOut:
    return ParticipantOut(
        id=p.id,
        name=p.name,
        role=p.role,
        created_at=p.created_at.isoformat() if p.created_at else None,
    )


@router.post("", response_model=ParticipantOut)
async def post_participant(request: CreateParticipantRequest,
                           db: AsyncSession = Depends(get_session)):
    row = await create_participant(db, request.name, request.role)
    return _out(row)


@router.get("", response_model=List[ParticipantOut])
async def get_participants(db: AsyncSession = Depends(get_session)):
    rows = await list_participants(db)
    return [_out(r) for r in rows]
