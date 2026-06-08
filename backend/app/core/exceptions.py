"""Application exception hierarchy."""

from __future__ import annotations


class JarvisError(Exception):
    """Base exception for all Jarvis errors."""
    pass


class NotFoundError(JarvisError):
    """Resource not found."""
    def __init__(self, resource: str, resource_id: int | str):
        self.resource = resource
        self.resource_id = resource_id
        super().__init__(f"{resource} with id {resource_id} not found")


class ValidationError(JarvisError):
    """Input validation failed."""
    pass


class VoiceProcessingError(JarvisError):
    """Voice processing (STT/TTS) failed."""
    pass


class IntentNotFoundError(JarvisError):
    """Could not determine intent from voice input."""
    def __init__(self, text: str):
        self.text = text
        super().__init__(f"Could not determine intent from: {text}")
