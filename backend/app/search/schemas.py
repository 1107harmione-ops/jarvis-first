"""Search schemas."""
from __future__ import annotations

import datetime
from typing import Optional

from pydantic import BaseModel


class SearchResultItem(BaseModel):
    id: int
    type: str  # "task", "note", "memory"
    title: str
    snippet: str
    score: float
    created_at: str | None = None

    model_config = {"from_attributes": True}


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SearchResultItem]
