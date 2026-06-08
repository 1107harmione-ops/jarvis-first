"""Memory Pydantic schemas."""
from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MemoryCreate(BaseModel):
    fact: str = Field(..., min_length=1, max_length=5000)
    category: str = Field("general", max_length=50)
    importance: int = Field(3, ge=1, le=5)


class MemoryRead(BaseModel):
    id: int
    fact: str
    category: str
    importance: int
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class MemoryListResponse(BaseModel):
    entries: list[MemoryRead]
    total: int
