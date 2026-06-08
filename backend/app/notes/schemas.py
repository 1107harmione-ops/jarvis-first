"""Notes Pydantic schemas for API request/response."""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


CATEGORIES = ["learning", "project", "personal", "ideas", "research"]


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Note title")
    content: str = Field("", max_length=10000, description="Note content")
    category: str = Field("personal", pattern=r"^(learning|project|personal|ideas|research)$")
    tags: str = Field("", description="Comma-separated tags")
    priority: str = Field("medium", pattern=r"^(low|medium|high|urgent)$")


class NoteUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    content: Optional[str] = None
    category: Optional[str] = Field(None, pattern=r"^(learning|project|personal|ideas|research)$")
    tags: Optional[str] = None
    priority: Optional[str] = Field(None, pattern=r"^(low|medium|high|urgent)$")


class NoteRead(BaseModel):
    id: int
    title: str
    content: str
    category: str
    tags: str
    priority: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


class NoteListResponse(BaseModel):
    notes: list[NoteRead]
    total: int
