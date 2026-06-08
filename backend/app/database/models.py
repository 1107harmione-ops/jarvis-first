"""Database ORM models."""

from __future__ import annotations

import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, func

from app.database.connection import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, default="", server_default="")
    status = Column(String(20), default="pending", server_default="pending")
    priority = Column(String(20), default="medium", server_default="medium")
    due_date = Column(DateTime, nullable=True)
    tags = Column(String(500), default="", server_default="")
    category = Column(String(100), default="general", server_default="general")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<Task(id={self.id}, title='{self.title}', status='{self.status}')>"
