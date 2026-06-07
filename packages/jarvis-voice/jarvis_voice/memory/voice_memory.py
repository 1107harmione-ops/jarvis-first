"""Voice memory: logs commands and sessions to a local JSON file store.

In production this would write to MongoDB via the `jarvis-memory` package.
The file-based store provides a zero-dependency fallback that works out of
the box for development and single-server deployments.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

from jarvis_voice.models import VoiceCommandLog, VoiceSessionLog

logger = logging.getLogger("jarvis_voice.memory")


class VoiceMemory:
    """Stores voice command history, session logs, and user metrics.

    Uses a local JSON file for persistence with thread-safe writes.
    Can be subclassed to use a real database adapter.
    """

    def __init__(self, storage_dir: str | Path = "voice_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._commands_path = self.storage_dir / "commands.jsonl"
        self._sessions_path = self.storage_dir / "sessions.jsonl"
        self._lock = Lock()

    # ── Command logging ────────────────────────────────────────────────

    async def log_command(
        self,
        user_id: str,
        text: str,
        language: str,
        duration_ms: int,
        success: bool,
    ) -> None:
        """Log a voice command to persistent storage."""
        entry = VoiceCommandLog(
            user_id=user_id,
            text=text,
            language=language,
            duration_ms=duration_ms,
            success=success,
        )
        await self._append_jsonl(self._commands_path, entry.model_dump(mode="json"))

    async def get_frequent_commands(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict]:
        """Get the most frequent commands for a user.

        Reads from the JSONL file and aggregates by text.
        """
        entries = await self._read_jsonl(self._commands_path)
        user_commands = [
            e for e in entries if e.get("user_id") == user_id
        ]
        # Count frequencies
        freq: dict[str, dict] = {}
        for e in user_commands:
            text = e.get("text", "").strip().lower()
            if not text:
                continue
            if text not in freq:
                freq[text] = {"text": e["text"], "language": e.get("language", "en"), "count": 0}
            freq[text]["count"] += 1

        sorted_commands = sorted(freq.values(), key=lambda x: x["count"], reverse=True)
        return sorted_commands[:limit]

    async def get_voice_history(
        self,
        user_id: str,
        days: int = 7,
    ) -> list[dict]:
        """Get recent voice commands within the given number of days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        entries = await self._read_jsonl(self._commands_path)
        return [
            e for e in entries
            if e.get("user_id") == user_id
            and _parse_timestamp(e.get("timestamp")) >= cutoff
        ]

    # ── Session logging ────────────────────────────────────────────────

    async def log_session(
        self,
        user_id: str,
        session_id: str,
        duration: float,
        commands: int,
    ) -> None:
        """Log a completed session."""
        entry = VoiceSessionLog(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow() - timedelta(seconds=duration),
            end_time=datetime.utcnow(),
            duration_seconds=duration,
            commands=commands,
        )
        await self._append_jsonl(self._sessions_path, entry.model_dump(mode="json"))

    async def get_voice_metrics(self, user_id: str) -> dict:
        """Get aggregated voice metrics for a user."""
        commands = await self._read_jsonl(self._commands_path)
        sessions = await self._read_jsonl(self._sessions_path)

        user_cmds = [c for c in commands if c.get("user_id") == user_id]
        user_sessions = [s for s in sessions if s.get("user_id") == user_id]

        total_commands = len(user_cmds)
        successful_commands = sum(1 for c in user_cmds if c.get("success", True))
        failed_commands = total_commands - successful_commands
        total_sessions = len(user_sessions)
        total_duration = sum(s.get("duration_seconds", 0) for s in user_sessions)

        # Language distribution
        lang_dist: dict[str, int] = {}
        for c in user_cmds:
            lang = c.get("language", "en")
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

        # Commands per language in the last 24h
        last_24h_cutoff = datetime.utcnow() - timedelta(hours=24)
        commands_last_24h = sum(
            1 for c in user_cmds
            if _parse_timestamp(c.get("timestamp")) >= last_24h_cutoff
        )

        return {
            "user_id": user_id,
            "total_commands": total_commands,
            "successful_commands": successful_commands,
            "failed_commands": failed_commands,
            "success_rate": successful_commands / max(total_commands, 1),
            "total_sessions": total_sessions,
            "total_duration_seconds": total_duration,
            "commands_last_24h": commands_last_24h,
            "language_distribution": lang_dist,
            "frequent_commands": await self.get_frequent_commands(user_id, limit=10),
        }

    # ── Internal helpers ───────────────────────────────────────────────

    async def _append_jsonl(self, path: Path, data: dict) -> None:
        """Append a JSON line to a file (thread-safe)."""
        def _write() -> None:
            with self._lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, default=str) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

        await _run_in_thread(_write)

    async def _read_jsonl(self, path: Path) -> list[dict]:
        """Read all JSON lines from a file."""
        if not path.exists():
            return []

        def _read() -> list[dict]:
            entries: list[dict] = []
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                entries.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
            except OSError:
                return []
            return entries

        return await _run_in_thread(_read)


def _parse_timestamp(ts) -> datetime | None:
    """Parse a timestamp string/datetime into a datetime object."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return None
    return None


async def _run_in_thread(func, *args, **kwargs):
    """Run a sync function in a thread."""
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args, **kwargs)
