"""Database ORM models."""

from __future__ import annotations

import datetime

from sqlalchemy import Boolean, Column, Integer, String, Text, DateTime, func

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


class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, default="", server_default="")
    category = Column(String(50), default="personal", server_default="personal")
    tags = Column(String(500), default="", server_default="")
    priority = Column(String(20), default="medium", server_default="medium")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<Note(id={self.id}, title='{self.title}', category='{self.category}')>"


class Reminder(Base):
    __tablename__ = "reminders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    reminder_time = Column(DateTime, nullable=False)
    repeat_type = Column(String(20), default="none", server_default="none")
    status = Column(String(20), default="pending", server_default="pending")
    triggered = Column(Boolean, default=False, server_default="0")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<Reminder(id={self.id}, title='{self.title}', time='{self.reminder_time}')>"


class MemoryEntry(Base):
    __tablename__ = "memory_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fact = Column(Text, nullable=False)
    category = Column(String(50), default="general", server_default="general")
    importance = Column(Integer, default=3, server_default="3")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<MemoryEntry(id={self.id}, fact='{self.fact[:30]}...', importance={self.importance})>"
