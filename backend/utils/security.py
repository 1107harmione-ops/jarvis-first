"""
Security utilities: JWT, password hashing, rate limiting, input sanitization.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
from jose import JWTError, jwt
from pydantic import BaseModel

from backend.config.settings import settings

# ── Password Hashing ─────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT Tokens ───────────────────────────────────────────────────


class TokenPayload(BaseModel):
    """Decoded JWT payload."""

    sub: str  # user_id
    exp: datetime
    iat: datetime
    role: str = "user"
    session_id: str | None = None


def create_access_token(
    user_id: str,
    role: str = "user",
    session_id: str | None = None,
    expires_delta: int | None = None,
) -> str:
    """Create a JWT access token."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(
        minutes=expires_delta or settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    )
    payload: dict[str, Any] = {
        "sub": user_id,
        "iat": now,
        "exp": expire,
        "role": role,
        "type": "access",
    }
    if session_id:
        payload["session_id"] = session_id
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Create a JWT refresh token."""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
    payload: dict[str, Any] = {
        "sub": user_id,
        "iat": now,
        "exp": expire,
        "type": "refresh",
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> TokenPayload | None:
    """Decode and validate a JWT token. Returns None on failure."""
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        return TokenPayload(
            sub=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
            role=payload.get("role", "user"),
            session_id=payload.get("session_id"),
        )
    except JWTError:
        return None


# ── API Key Management ───────────────────────────────────────────


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return f"jarvis_{secrets.token_hex(32)}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


# ── Input Sanitization ───────────────────────────────────────────

# Patterns that may indicate prompt injection
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+all\s+(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior)", re.IGNORECASE),
    re.compile(r"system\s+(prompt|message|instruction)", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now|not\s+required\s+to)", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    """Check if text contains prompt injection attempts."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input: strip control chars, enforce length."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned[:max_length]


# ── Rate Limiter (in-memory, replace with Redis in production) ───


class InMemoryRateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self) -> None:
        self._buckets: dict[str, list[datetime]] = {}

    def check(self, key: str, max_requests: int, window_seconds: int = 60) -> bool:
        """Check if request is allowed. Returns True if allowed."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=window_seconds)

        if key not in self._buckets:
            self._buckets[key] = []

        # Prune old entries
        self._buckets[key] = [t for t in self._buckets[key] if t > cutoff]

        if len(self._buckets[key]) >= max_requests:
            return False

        self._buckets[key].append(now)
        return True


rate_limiter = InMemoryRateLimiter()
