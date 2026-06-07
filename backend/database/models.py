"""
Pydantic models for the JARVIS backend.
All request/response schemas, database document models, and enums.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PREFERENCE = "preference"
    TASK = "task"
    KNOWLEDGE = "knowledge"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentType(str, Enum):
    ROUTER = "router"
    CODING = "coding"
    RESEARCH = "research"
    MEMORY = "memory"
    TASK = "task"
    VISION = "vision"
    PLANNER = "planner"
    MAIN = "main"


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    AGENT = "agent"


class VoiceState(str, Enum):
    IDLE = "idle"
    WAKE_PENDING = "wake_pending"
    LISTENING = "listening"
    PROCESSING_STT = "processing_stt"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    OFFLINE = "offline"
    ERROR = "error"


# ── Generic Response Models ─────────────────────────────────────


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    message: str = "OK"
    data: Any = None
    error: str | None = None


class PaginatedResponse(BaseModel):
    """Paginated list response."""

    items: list[Any]
    total: int
    page: int
    page_size: int
    total_pages: int


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = False
    error: str
    error_code: str | None = None
    details: dict[str, Any] | None = None


# ── Authentication ───────────────────────────────────────────────


class UserCreate(BaseModel):
    """User registration request."""

    email: str
    password: str = Field(min_length=8, max_length=128)
    name: str = Field(min_length=1, max_length=100)
    role: str = "user"

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Invalid email address")
        return v

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain an uppercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain a digit")
        return v


class UserLogin(BaseModel):
    """User login request."""

    email: str
    password: str


class UserResponse(BaseModel):
    """User data returned to clients."""

    id: str
    email: str
    name: str
    role: str
    created_at: str
    is_active: bool = True


class TokenResponse(BaseModel):
    """JWT token pair."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshRequest(BaseModel):
    """Refresh token request."""

    refresh_token: str


# ── Chat / Conversation ──────────────────────────────────────────


class ConversationCreate(BaseModel):
    """Start a new conversation."""

    title: str = "New Conversation"
    model: str = "deepseek-chat"


class ConversationResponse(BaseModel):
    """Conversation metadata."""

    id: str
    user_id: str
    title: str
    model: str
    message_count: int = 0
    created_at: str
    updated_at: str


class MessageCreate(BaseModel):
    """Send a message in a conversation."""

    content: str = Field(min_length=1, max_length=50000)
    role: MessageRole = MessageRole.USER
    attachments: list[str] = Field(default_factory=list, description="File URLs or image IDs")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageResponse(BaseModel):
    """A single message in a conversation."""

    id: str
    conversation_id: str
    role: str
    content: str
    agent: str | None = None
    attachments: list[str] = []
    tokens_used: int | None = None
    metadata: dict[str, Any] = {}
    created_at: str


class ChatRequest(BaseModel):
    """Request to the chat/agent endpoint."""

    message: str = Field(min_length=1, max_length=50000)
    conversation_id: str | None = None
    agent: AgentType = AgentType.ROUTER
    stream: bool = False
    attachments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Memory ───────────────────────────────────────────────────────


class MemoryCreate(BaseModel):
    """Store a new memory."""

    content: str = Field(min_length=1, max_length=10000)
    memory_type: MemoryType = MemoryType.SHORT_TERM
    tags: list[str] = Field(default_factory=list)
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryResponse(BaseModel):
    """Memory document returned to clients."""

    id: str
    user_id: str
    content: str
    memory_type: str
    importance_score: float
    tags: list[str] = []
    summary: str | None = None
    source: str | None = None
    consolidated: bool = False
    created_at: str
    updated_at: str


class MemorySearchRequest(BaseModel):
    """Search memories by query."""

    query: str = Field(min_length=1, max_length=1000)
    memory_type: MemoryType | None = None
    limit: int = Field(default=10, ge=1, le=50)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class MemorySearchResponse(BaseModel):
    """Search results with ranked memories."""

    results: list[MemoryResponse]
    total: int
    query_time_ms: float


# ── Tasks ────────────────────────────────────────────────────────


class TaskCreate(BaseModel):
    """Create a task (reminder, scheduled, recurring)."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=5000)
    priority: TaskPriority = TaskPriority.MEDIUM
    due_at: str | None = None
    scheduled_at: str | None = None
    recurring: str | None = Field(
        default=None,
        description="Cron expression for recurring tasks, e.g. '0 9 * * 1-5'",
    )
    tags: list[str] = Field(default_factory=list)


class TaskUpdate(BaseModel):
    """Update an existing task."""

    title: str | None = None
    description: str | None = None
    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    due_at: str | None = None
    tags: list[str] | None = None


class TaskResponse(BaseModel):
    """Task document returned to clients."""

    id: str
    user_id: str
    title: str
    description: str
    status: str
    priority: str
    due_at: str | None = None
    scheduled_at: str | None = None
    recurring: str | None = None
    tags: list[str] = []
    completed_at: str | None = None
    created_at: str
    updated_at: str


# ── Voice ────────────────────────────────────────────────────────


class VoiceSessionCreate(BaseModel):
    """Start a voice session."""

    language: str = "en"
    wake_word_enabled: bool = True
    client_info: dict[str, Any] = Field(default_factory=dict)


class VoiceSessionResponse(BaseModel):
    """Voice session info."""

    session_id: str
    state: str
    language: str
    started_at: str
    expires_at: str


class VoiceMessageResponse(BaseModel):
    """Voice interaction result."""

    session_id: str
    transcript: str
    response: str
    audio_url: str | None = None
    duration_ms: int
    interrupted: bool = False


# ── Vision ───────────────────────────────────────────────────────


class VisionRequest(BaseModel):
    """Analyze an image."""

    image_url: str = Field(min_length=1, description="URL or base64 data URI of the image")
    prompt: str = Field(default="Describe this image in detail", max_length=2000)
    detail_level: str = Field(default="auto", pattern=r"^(low|high|auto)$")


class VisionResponse(BaseModel):
    """Vision analysis result."""

    description: str
    objects: list[dict[str, Any]] = Field(default_factory=list)
    text_detected: str | None = None
    confidence: float = 0.0
    processing_time_ms: float


# ── Agent Logs ───────────────────────────────────────────────────


class AgentLog(BaseModel):
    """Structured agent execution log."""

    agent_name: str
    session_id: str
    user_id: str | None = None
    action: str
    input_summary: str = ""
    output_summary: str = ""
    tokens_used: int = 0
    duration_ms: float = 0.0
    status: Literal["success", "error", "timeout"] = "success"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


# ── WebSocket Messages ───────────────────────────────────────────


class WsClientMessage(BaseModel):
    """Message from client over WebSocket."""

    type: Literal["message", "typing", "ping", "interrupt", "config"] = "message"
    content: str = ""
    conversation_id: str | None = None
    attachments: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class WsServerMessage(BaseModel):
    """Message from server over WebSocket."""

    type: Literal[
        "token", "message", "error", "typing", "done", "pong", "state_change"
    ] = "message"
    content: str = ""
    conversation_id: str | None = None
    agent: str | None = None
    tokens: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Admin ────────────────────────────────────────────────────────


class SystemHealth(BaseModel):
    """System health check response."""

    status: str
    version: str
    environment: str
    mongodb: str
    uptime_seconds: float
    active_connections: int = 0
    memory_usage_mb: float = 0.0
