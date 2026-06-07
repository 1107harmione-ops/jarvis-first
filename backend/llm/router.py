"""
LLM Router — dynamically selects the best model for each request.
Routes to DeepSeek, Codex, Minimax, or Mimo based on task type,
cost optimization, and availability.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, AsyncGenerator

from backend.llm.codex import codex
from backend.llm.deepseek import deepseek
from backend.llm.minimax import minimax
from backend.llm.mimo import mimo
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class TaskCategory(str, Enum):
    """Task categories for model routing."""

    GENERAL_CHAT = "general_chat"
    CODING = "coding"
    CODE_REVIEW = "code_review"
    RESEARCH = "research"
    SUMMARIZATION = "summarization"
    VISION = "vision"
    OCR = "ocr"
    ROUTING = "routing"
    MEMORY = "memory"
    PLANNING = "planning"
    TASK_MANAGEMENT = "task_management"
    EXTRACTION = "extraction"


class LLMRouter:
    """Routes LLM requests to the optimal model provider.

    Routing logic:
    - Coding → Codex (GPT-4o)
    - Vision/OCR → Mimo (V2 Omni)
    - Research/Summarization → Minimax (M2.1)
    - Routing/Classification → DeepSeek (fast/cheap)
    - General chat → DeepSeek
    - Planning → DeepSeek (with structured output)
    """

    # Cost per 1K tokens (approximate, for optimization)
    MODEL_COSTS: dict[str, dict[str, float]] = {
        "deepseek": {"input": 0.0005, "output": 0.0015},
        "codex": {"input": 0.01, "output": 0.03},
        "minimax": {"input": 0.005, "output": 0.015},
        "mimo": {"input": 0.01, "output": 0.03},
    }

    def categorize_request(self, message: str, attachments: list[str] | None = None) -> TaskCategory:
        """Determine the task category from the user's message."""
        msg_lower = message.lower()

        # Image/Vision tasks
        if attachments and any(
            a.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"))
            or a.startswith("data:image")
            for a in attachments
        ):
            return TaskCategory.VISION

        # OCR indicators
        if any(word in msg_lower for word in ["extract text", "read text", "ocr", "what does this say"]):
            return TaskCategory.OCR

        # Coding tasks
        code_keywords = [
            "write code", "generate code", "implement", "function", "class ",
            "refactor", "debug", "fix bug", "code review", "pull request",
            "algorithm", "api endpoint", "database query", "migration",
        ]
        if any(kw in msg_lower for kw in code_keywords):
            return TaskCategory.CODING

        # Code review
        if any(kw in msg_lower for kw in ["review this code", "review code", "code quality"]):
            return TaskCategory.CODE_REVIEW

        # Research tasks
        research_keywords = [
            "research", "investigate", "find information", "search for",
            "what is", "who is", "explain", "analyze", "compare",
            "report on", "study", "tell me about",
        ]
        if any(kw in msg_lower for kw in research_keywords):
            # Short "what is" queries can stay on DeepSeek
            if len(message.split()) <= 8:
                return TaskCategory.GENERAL_CHAT
            return TaskCategory.RESEARCH

        # Summarization
        if any(kw in msg_lower for kw in ["summarize", "summary", "tl;dr", "tldr"]):
            return TaskCategory.SUMMARIZATION

        # Planning
        if any(kw in msg_lower for kw in ["plan", "strategy", "roadmap", "architecture", "design"]):
            return TaskCategory.PLANNING

        # Task management
        if any(kw in msg_lower for kw in ["remind me", "create task", "schedule", "to-do", "todo", "set reminder"]):
            return TaskCategory.TASK_MANAGEMENT

        # Memory operations
        if any(kw in msg_lower for kw in ["remember", "do you remember", "recall", "what did i say about"]):
            return TaskCategory.MEMORY

        return TaskCategory.GENERAL_CHAT

    def select_model(self, category: TaskCategory) -> str:
        """Select the model provider for a task category."""
        model_map: dict[TaskCategory, str] = {
            TaskCategory.GENERAL_CHAT: "deepseek",
            TaskCategory.CODING: "codex",
            TaskCategory.CODE_REVIEW: "codex",
            TaskCategory.RESEARCH: "minimax",
            TaskCategory.SUMMARIZATION: "minimax",
            TaskCategory.VISION: "mimo",
            TaskCategory.OCR: "mimo",
            TaskCategory.ROUTING: "deepseek",
            TaskCategory.MEMORY: "deepseek",
            TaskCategory.PLANNING: "deepseek",
            TaskCategory.TASK_MANAGEMENT: "deepseek",
            TaskCategory.EXTRACTION: "minimax",
        }
        return model_map.get(category, "deepseek")

    async def route(
        self,
        messages: list[dict[str, str]],
        category: TaskCategory | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any] | AsyncGenerator[str, None]:
        """Route a request to the appropriate model.

        Args:
            messages: Chat messages.
            category: Override auto-detection.
            stream: Enable streaming.
            **kwargs: Additional model parameters.

        Returns:
            API response dict or async generator for streaming.
        """
        # Auto-detect category from last user message
        if category is None:
            last_user = next(
                (m["content"] for m in reversed(messages) if m["role"] == "user"),
                "",
            )
            category = self.categorize_request(last_user)

        model = self.select_model(category)
        logger.info(
            "Routing request",
            extra={"category": category.value, "model": model, "stream": stream},
        )

        # Get model temperature based on category
        temps: dict[TaskCategory, float] = {
            TaskCategory.GENERAL_CHAT: 0.7,
            TaskCategory.CODING: 0.2,
            TaskCategory.CODE_REVIEW: 0.1,
            TaskCategory.RESEARCH: 0.3,
            TaskCategory.SUMMARIZATION: 0.3,
            TaskCategory.VISION: 0.3,
            TaskCategory.OCR: 0.1,
            TaskCategory.PLANNING: 0.5,
            TaskCategory.TASK_MANAGEMENT: 0.5,
        }
        temperature = temps.get(category, 0.7)

        if stream:
            return await self._route_stream(model, messages, temperature, **kwargs)

        return await self._route_chat(model, messages, temperature, **kwargs)

    async def _route_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route to a non-streaming chat completion."""
        if model == "codex":
            return await codex.chat(messages, temperature=temperature, **kwargs)
        elif model == "minimax":
            return await minimax.chat(messages, temperature=temperature, **kwargs)
        else:
            # Default: DeepSeek
            return await deepseek.chat(messages, temperature=temperature, **kwargs)

    async def _route_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Route to a streaming chat completion."""
        if model == "codex":
            async for token in codex.chat_stream(messages, temperature=temperature, **kwargs):
                yield token
        else:
            # Default: DeepSeek
            async for token in deepseek.chat_stream(messages, temperature=temperature, **kwargs):
                yield token

    def estimate_cost(
        self, input_tokens: int, output_tokens: int, model: str
    ) -> float:
        """Estimate API cost for a request."""
        costs = self.MODEL_COSTS.get(model, self.MODEL_COSTS["deepseek"])
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return round(input_cost + output_cost, 6)


# Global singleton
llm_router = LLMRouter()
