"""Task Pydantic schemas for API request/response."""

from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Task title")
    description: str = Field("", max_length=5000, description="Task description")
    priority: str = Field("medium", pattern=r"^(low|medium|high|urgent)$")
    due_date: Optional[datetime.datetime] = Field(None, description="ISO 8601 due date")
    tags: str = Field("", description="Comma-separated tags")
    category: str = Field("general", max_length=100)


class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern=r"^(pending|completed|cancelled)$")
    priority: Optional[str] = Field(None, pattern=r"^(low|medium|high|urgent)$")
    due_date: Optional[datetime.datetime] = None
    tags: Optional[str] = None
    category: Optional[str] = None


class TaskRead(BaseModel):
    id: int
    title: str
    description: str
    status: str
    priority: str
    due_date: Optional[datetime.datetime] = None
    tags: str
    category: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    model_config = {"from_attributes": True}


class TaskListResponse(BaseModel):
    tasks: list[TaskRead]
    total: int
