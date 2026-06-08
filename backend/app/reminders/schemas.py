"""Reminder Pydantic schemas."""
from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel, Field


REPEAT_TYPES = ["none", "daily", "weekly", "weekdays"]


class ReminderCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    reminder_time: datetime.datetime
    repeat_type: str = Field("none", pattern=r"^(none|daily|weekly|weekdays)$")


class ReminderUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=255)
    reminder_time: Optional[datetime.datetime] = None
    repeat_type: Optional[str] = Field(None, pattern=r"^(none|daily|weekly|weekdays)$")
    status: Optional[str] = None
    triggered: Optional[bool] = None


class ReminderRead(BaseModel):
    id: int
    title: str
    reminder_time: datetime.datetime
    repeat_type: str
    status: str
    triggered: bool
    created_at: datetime.datetime

    model_config = {"from_attributes": True}


class ReminderListResponse(BaseModel):
    reminders: list[ReminderRead]
    total: int
