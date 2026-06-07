"""
Voice Memory System — Stores voice interaction history, user preferences,
frequently used commands, and session metrics in MongoDB.

Features:
- Voice interaction history with transcripts, responses, confidence
- Frequently used commands tracking for faster routing
- User voice preferences (language, speed, pitch, wake word sensitivity)
- Session history with duration, interruptions, errors
- Audio metrics storage (STT latency, TTS latency, round-trip)
- Recent commands caching for quick recall
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from backend.database.mongodb import db
from backend.database.schemas import new_agent_log_doc, now_utc

logger = logging.getLogger("jarvis.voice_memory")


class VoiceMemoryService:
    """
    Service for storing and retrieving voice-related memory and preferences.

    Uses MongoDB collections:
    - voice_history: voice interaction records
    - voice_commands: frequently used command patterns
    - voice_preferences: per-user voice settings
    - voice_sessions: session metrics
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}
        self._cache_ttl: float = 300.0  # 5 minutes
        self._cache_timestamps: dict[str, float] = {}

    async def store_interaction(
        self,
        user_id: str,
        session_id: str,
        transcript: str,
        response: str,
        confidence: float,
        agent: str,
        language: str,
        metrics: dict[str, Any] | None = None,
        interrupted: bool = False,
        audio_duration_ms: float = 0.0,
    ) -> str:
        """Store a voice interaction history record."""
        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "type": "voice_interaction",
            "transcript": transcript,
            "response": response,
            "confidence": confidence,
            "agent": agent,
            "language": language,
            "interrupted": interrupted,
            "audio_duration_ms": audio_duration_ms,
            "metrics": metrics or {},
            "created_at": now_utc(),
        }

        result = await db.voice_history.insert_one(doc)
        log_doc = new_agent_log_doc(
            user_id=user_id,
            agent=agent,
            action="voice_interaction",
            input={"transcript": transcript, "language": language},
            output={"response_preview": response[:200]},
            tokens_used=0,
            duration_ms=metrics.get("total_pipeline_ms", 0) if metrics else 0,
        )
        await db.agent_logs.insert_one(log_doc)

        # Update command frequency
        await self._track_command(user_id, transcript, agent)

        logger.debug(
            "Stored voice interaction for user %s (session %s): %.0fms",
            user_id,
            session_id,
            audio_duration_ms,
        )

        return str(result.inserted_id)

    async def get_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get voice interaction history for a user."""
        query: dict[str, Any] = {
            "user_id": user_id,
            "type": "voice_interaction",
        }
        if language:
            query["language"] = language

        cursor = (
            db.voice_history.find(query)
            .sort("created_at", -1)
            .skip(offset)
            .limit(limit)
        )

        results = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            results.append(doc)

        return results

    async def get_history_count(
        self,
        user_id: str,
        language: str | None = None,
    ) -> int:
        """Get count of voice interactions for a user."""
        query: dict[str, Any] = {
            "user_id": user_id,
            "type": "voice_interaction",
        }
        if language:
            query["language"] = language
        return await db.voice_history.count_documents(query)

    async def _track_command(self, user_id: str, transcript: str, agent: str) -> None:
        """
        Track frequently used commands for faster routing.

        Increments a counter for normalized command patterns.
        """
        # Normalize: lowercase, strip punctuation, limit length
        normalized = transcript.lower().strip()[:200]
        if not normalized:
            return

        filter_query = {
            "user_id": user_id,
            "command": normalized,
            "type": "command_frequency",
        }
        update = {
            "$inc": {"count": 1, "total_count": 1},
            "$set": {
                "last_used": now_utc(),
                "agent": agent,
            },
            "$setOnInsert": {
                "created_at": now_utc(),
                "type": "command_frequency",
            },
        }

        await db.voice_commands.update_one(filter_query, update, upsert=True)

        # Update the aggregate count without touching the per-user record
        await db.voice_commands.update_one(
            {"command": normalized, "type": "command_global_frequency"},
            {"$inc": {"count": 1}, "$set": {"last_used": now_utc()}},
            upsert=True,
        )

    async def get_frequent_commands(
        self,
        user_id: str,
        limit: int = 20,
        min_count: int = 3,
    ) -> list[dict[str, Any]]:
        """Get the user's most frequently used voice commands."""
        cursor = (
            db.voice_commands.find({
                "user_id": user_id,
                "type": "command_frequency",
                "count": {"$gte": min_count},
            })
            .sort("count", -1)
            .limit(limit)
        )

        results = []
        async for doc in cursor:
            results.append({
                "command": doc["command"],
                "count": doc["count"],
                "agent": doc.get("agent", "router"),
                "last_used": doc.get("last_used", "").isoformat()
                if isinstance(doc.get("last_used"), datetime)
                else str(doc.get("last_used", "")),
            })

        return results

    async def store_preferences(
        self,
        user_id: str,
        preferences: dict[str, Any],
    ) -> None:
        """Store or update voice preferences for a user."""
        update = {"$set": {**preferences, "updated_at": now_utc()}}
        await db.voice_preferences.update_one(
            {"user_id": user_id},
            update,
            upsert=True,
        )
        # Invalidate cache
        self._cache.pop(f"prefs_{user_id}", None)
        logger.debug("Stored voice preferences for user %s", user_id)

    async def get_preferences(self, user_id: str) -> dict[str, Any]:
        """Get voice preferences for a user."""
        # Check cache
        cache_key = f"prefs_{user_id}"
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - self._cache_timestamps.get(cache_key, 0)) < self._cache_ttl:
            return cached

        doc = await db.voice_preferences.find_one({"user_id": user_id})
        if not doc:
            # Return defaults
            defaults = {
                "language": "en",
                "voice_speed": 1.0,
                "voice_pitch": 1.0,
                "wake_word_enabled": True,
                "wake_word_sensitivity": 0.5,
                "interrupt_enabled": True,
                "offline_mode": False,
                "bluetooth_sco_enabled": True,
            }
            return defaults

        if "_id" in doc:
            del doc["_id"]
        if "updated_at" in doc and isinstance(doc["updated_at"], datetime):
            doc["updated_at"] = doc["updated_at"].isoformat()

        # Update cache
        self._cache[cache_key] = doc
        self._cache_timestamps[cache_key] = time.monotonic()

        return doc

    async def store_session_metrics(
        self,
        user_id: str,
        session_id: str,
        metrics: dict[str, Any],
    ) -> None:
        """Store voice session metrics."""
        doc = {
            "user_id": user_id,
            "session_id": session_id,
            "type": "voice_session_metrics",
            **metrics,
            "created_at": now_utc(),
        }
        await db.voice_sessions.insert_one(doc)

    async def get_session_metrics(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent voice session metrics."""
        cursor = (
            db.voice_sessions.find({
                "user_id": user_id,
                "type": "voice_session_metrics",
            })
            .sort("created_at", -1)
            .limit(limit)
        )

        results = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            results.append(doc)

        return results

    async def get_voice_stats(self, user_id: str) -> dict[str, Any]:
        """Get aggregate voice usage statistics for a user."""
        # Total interactions
        total = await db.voice_history.count_documents({
            "user_id": user_id,
            "type": "voice_interaction",
        })

        # Total by language
        pipeline = [
            {"$match": {"user_id": user_id, "type": "voice_interaction"}},
            {"$group": {"_id": "$language", "count": {"$sum": 1}}},
        ]
        by_language = {}
        async for doc in db.voice_history.aggregate(pipeline):
            by_language[doc["_id"] or "unknown"] = doc["count"]

        # Average confidence
        avg_pipeline = [
            {"$match": {"user_id": user_id, "type": "voice_interaction"}},
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence"}}},
        ]
        avg_confidence = 0.0
        async for doc in db.voice_history.aggregate(avg_pipeline):
            avg_confidence = round(doc.get("avg_confidence", 0.0), 3)

        # Interruption rate
        interrupted = await db.voice_history.count_documents({
            "user_id": user_id,
            "type": "voice_interaction",
            "interrupted": True,
        })

        # Total commands tracked
        commands = await db.voice_commands.count_documents({
            "user_id": user_id,
            "type": "command_frequency",
        })

        return {
            "total_interactions": total,
            "by_language": by_language,
            "avg_confidence": avg_confidence,
            "interruptions": interrupted,
            "interruption_rate": round(interrupted / max(total, 1), 3),
            "unique_commands": commands,
        }

    async def search_history(
        self,
        user_id: str,
        query: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search voice history by transcript text."""
        if not query:
            return []

        cursor = (
            db.voice_history.find({
                "user_id": user_id,
                "type": "voice_interaction",
                "transcript": {
                    "$regex": query,
                    "$options": "i",
                },
            })
            .sort("created_at", -1)
            .limit(limit)
        )

        results = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            results.append(doc)

        return results


# Global singleton
voice_memory_service = VoiceMemoryService()
