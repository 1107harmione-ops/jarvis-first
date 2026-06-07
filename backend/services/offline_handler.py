"""
Offline Mode Handler — Manages offline state, command queuing, and connectivity monitoring.

When the backend loses connectivity to external services (LLM, STT, TTS),
or the Android client loses network connectivity, this handler provides:
- Graceful degradation
- Command queuing for later processing
- Basic offline responses
- Automatic reconnection and sync
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from backend.config.settings import settings
from backend.database.mongodb import mongodb
from backend.database.schemas import now_utc

logger = logging.getLogger("jarvis.offline_handler")


class ServiceStatus(str, Enum):
    """Status of external services."""

    ONLINE = "online"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class OfflineHandler:
    """
    Manages offline mode for the voice system.

    Tracks connectivity to external services and provides fallback behavior.
    """

    def __init__(self) -> None:
        self._status: dict[str, ServiceStatus] = {
            "llm": ServiceStatus.UNKNOWN,
            "stt": ServiceStatus.UNKNOWN,
            "tts": ServiceStatus.UNKNOWN,
            "database": ServiceStatus.UNKNOWN,
        }
        self._check_interval: float = 30.0  # check every 30s
        self._monitor_task: asyncio.Task[Any] | None = None
        self._running = False
        self._on_status_change: Callable[[str, ServiceStatus], Any] | None = None

        # Queued commands for offline → online sync
        self._processing_queue = False

    async def start_monitoring(
        self,
        on_status_change: Callable[[str, ServiceStatus], Any] | None = None,
    ) -> None:
        """Start background connectivity monitoring."""
        if self._running:
            return
        self._running = True
        self._on_status_change = on_status_change
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Offline handler monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background connectivity monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        logger.info("Offline handler monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Periodically check connectivity to external services."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in offline monitor loop: %s", e)
                await asyncio.sleep(self._check_interval * 2)  # Backoff on error

    async def _check_all_services(self) -> None:
        """Check connectivity to all external services."""
        # Check LLM (DeepSeek)
        await self._check_service(
            "llm",
            self._check_http_connectivity(settings.DEEPSEEK_BASE_URL, timeout=5.0),
        )

        # Check STT (Whisper API or DeepSeek audio)
        stt_url = settings.WHISPER_API_BASE_URL or settings.DEEPSEEK_BASE_URL
        await self._check_service(
            "stt",
            self._check_http_connectivity(stt_url, timeout=5.0),
        )

        # Check TTS (Piper local or API)
        if settings.PIPER_ENABLED:
            # Piper is local — always online unless binary is missing
            pass
        else:
            tts_url = settings.DEEPSEEK_BASE_URL
            await self._check_service(
                "tts",
                self._check_http_connectivity(tts_url, timeout=5.0),
            )

        # Check database
        await self._check_service(
            "database",
            self._check_database_connectivity(),
        )

    async def _check_service(
        self, service: str, check_coro: Any
    ) -> None:
        """Check a single service and update status."""
        try:
            result = await asyncio.wait_for(check_coro, timeout=10.0)
            new_status = ServiceStatus.ONLINE if result else ServiceStatus.OFFLINE
        except (asyncio.TimeoutError, Exception):
            new_status = ServiceStatus.OFFLINE

        old_status = self._status.get(service, ServiceStatus.UNKNOWN)
        if old_status != new_status:
            self._status[service] = new_status
            logger.info(
                "Service %s status changed: %s → %s",
                service,
                old_status.value,
                new_status.value,
            )
            if self._on_status_change:
                try:
                    self._on_status_change(service, new_status)
                except Exception as e:
                    logger.error("Status change callback error: %s", e)

    async def _check_http_connectivity(self, url: str, timeout: float = 5.0) -> bool:
        """Check if an HTTP endpoint is reachable."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url, follow_redirects=True)
                return response.status_code < 500
        except Exception:
            return False

    async def _check_database_connectivity(self) -> bool:
        """Check if MongoDB is reachable."""
        try:
            await mongodb.db.admin.command("ping")
            return True
        except Exception:
            return False

    @property
    def is_fully_online(self) -> bool:
        """All critical services are online."""
        critical = ["llm", "stt", "database"]
        return all(
            self._status.get(s) == ServiceStatus.ONLINE for s in critical
        )

    @property
    def is_online(self) -> bool:
        """At least one core service is online."""
        return self._status.get("database") == ServiceStatus.ONLINE

    @property
    def is_stt_available(self) -> bool:
        return self._status.get("stt") in (ServiceStatus.ONLINE, ServiceStatus.UNKNOWN)

    @property
    def is_tts_available(self) -> bool:
        return settings.PIPER_ENABLED or self._status.get("tts") in (
            ServiceStatus.ONLINE,
            ServiceStatus.UNKNOWN,
        )

    @property
    def status_summary(self) -> dict[str, str]:
        """Get summary of all service statuses."""
        return {k: v.value for k, v in self._status.items()}

    async def queue_command(
        self,
        user_id: str,
        transcript: str,
        language: str = "en",
        audio_data: bytes | None = None,
    ) -> str:
        """Queue a voice command for processing when back online."""
        doc = {
            "user_id": user_id,
            "transcript": transcript,
            "language": language,
            "audio_data": audio_data,
            "status": "queued",
            "created_at": now_utc(),
            "queued_while_offline": True,
        }

        result = await mongodb.offline_queue.insert_one(doc)
        logger.info(
            "Queued offline command for user %s: %s",
            user_id,
            transcript[:50],
        )
        return str(result.inserted_id)

    async def get_queued_commands(
        self,
        user_id: str,
        status: str = "queued",
    ) -> list[dict[str, Any]]:
        """Get queued commands for a user."""
        cursor = (
            mongodb.offline_queue.find({
                "user_id": user_id,
                "status": status,
            })
            .sort("created_at", 1)
        )

        results = []
        async for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            if isinstance(doc.get("created_at"), datetime):
                doc["created_at"] = doc["created_at"].isoformat()
            # Don't return binary audio data in list
            doc.pop("audio_data", None)
            results.append(doc)

        return results

    async def process_queued_commands(self, user_id: str) -> int:
        """
        Process all queued commands for a user.

        Called when connectivity is restored.
        Returns the number of commands processed.
        """
        if self._processing_queue:
            return 0

        self._processing_queue = True
        try:
            queued = await self.get_queued_commands(user_id, status="queued")
            processed = 0

            for cmd in queued:
                try:
                    # Import router_agent locally to avoid circular imports
                    from backend.agents.router_agent import RouterAgent

                    agent = RouterAgent()
                    result = await agent.process(
                        user_id=user_id,
                        message=cmd["transcript"],
                        language=cmd.get("language", "en"),
                    )

                    # Mark as processed
                    await mongodb.offline_queue.update_one(
                        {"_id": cmd["id"]},
                        {
                            "$set": {
                                "status": "processed",
                                "response": result.get("content", ""),
                                "processed_at": now_utc(),
                            }
                        },
                    )
                    processed += 1

                except Exception as e:
                    logger.error("Failed to process queued command %s: %s", cmd["id"], e)
                    await mongodb.offline_queue.update_one(
                        {"_id": cmd["id"]},
                        {"$set": {"status": "failed", "error": str(e)}},
                    )

            if processed > 0:
                logger.info("Processed %d queued commands for user %s", processed, user_id)

            return processed

        finally:
            self._processing_queue = False

    async def get_offline_response(self, transcript: str, language: str = "en") -> str:
        """Generate a basic response when offline."""
        transcript_lower = transcript.lower().strip()

        # Very basic offline command matching
        if any(greeting in transcript_lower for greeting in ["hello", "hi", "hey"]):
            return "Hello! I'm currently offline. I'll process your request when connected."

        if "stop" in transcript_lower or "cancel" in transcript_lower:
            return "Stopping. I'll be here when you need me."

        if "help" in transcript_lower:
            return (
                "I'm currently offline. I can still respond to basic commands. "
                "Say 'status' to check connectivity, or say 'queue' to save your request."
            )

        if "status" in transcript_lower or "online" in transcript_lower:
            return "I am currently offline. Trying to reconnect..."

        if "queue" in transcript_lower:
            return "I'll queue your request and process it when connectivity is restored."

        if "time" in transcript_lower or "date" in transcript_lower:
            from datetime import datetime
            now = datetime.now()
            return f"The current time is {now.strftime('%I:%M %p')}."

        # Generic offline response
        return (
            "I'm currently offline. I've saved your request and will process it "
            "when I reconnect to the server."
        )


# Global singleton
offline_handler = OfflineHandler()
