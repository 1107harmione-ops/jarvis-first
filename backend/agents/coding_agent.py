"""
Coding Agent — code generation, review, bug detection, and optimization.
Powered by Codex (GPT-4o) for superior code understanding.
"""

from __future__ import annotations

import time
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc
from backend.llm.codex import codex
from backend.llm.deepseek import deepseek
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class CodingAgent:
    """Specialized agent for all code-related tasks."""

    def __init__(self) -> None:
        self.name = "coding_agent"

    async def process(
        self,
        user_id: str,
        message: str,
        context: list[dict[str, str]] | None = None,
        attachments: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a coding request."""
        start = time.monotonic()
        session_id = session_id or f"coding_{int(start)}"

        # Detect specific coding task
        task_type = self._detect_task_type(message)
        logger.info(
            "Coding agent processing",
            extra={"task_type": task_type, "session_id": session_id, "user_id": user_id},
        )

        try:
            if task_type == "code_review":
                result = await self._code_review(message, context)
            elif task_type == "bug_detection":
                result = await self._bug_detection(message, context)
            elif task_type == "optimization":
                result = await self._optimization(message, context)
            else:
                result = await self._code_generation(message, context)

            # Log agent execution
            elapsed = (time.monotonic() - start) * 1000
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary=result.get("content", "")[:200],
                    duration_ms=elapsed,
                    tokens_used=result.get("tokens_used", 0),
                    status="success",
                )
            )

            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Coding agent failed", extra={"error": str(exc), "session_id": session_id})
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary="",
                    duration_ms=elapsed,
                    status="error",
                    error=str(exc),
                )
            )
            return {
                "content": f"I encountered an error while processing your coding request: {str(exc)}",
                "agent": self.name,
                "error": str(exc),
            }

    async def _code_generation(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Generate code from a description."""
        # Detect language from message
        language = self._detect_language(message)
        system = f"""You are an expert {language} developer. Generate clean, production-ready code.

Rules:
- Write complete, working code
- Include error handling
- Add comments for complex logic
- Follow {language} best practices and conventions
- Include type hints where applicable
- Output code in markdown code blocks with language tag
- If the request is unclear, ask clarifying questions before generating"""

        messages = context or []
        # Replace last message or add system
        messages = [msg for msg in messages if msg["role"] != "system"]
        messages.insert(0, {"role": "system", "content": system})
        messages.append({"role": "user", "content": message})

        response = await codex.chat(messages, temperature=0.2, max_tokens=8192)
        content = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {})

        return {
            "content": content,
            "agent": self.name,
            "tokens_used": tokens.get("total_tokens", 0),
            "metadata": {"language": language, "task": "code_generation"},
        }

    async def _code_review(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Review code for bugs, security, and improvements."""
        messages = context or []
        messages = [msg for msg in messages if msg["role"] != "system"]
        messages.insert(0, {
            "role": "system",
            "content": """You are a senior code reviewer. Review the provided code thoroughly.

Analyze for:
1. Bugs and logic errors (high priority)
2. Security vulnerabilities (high priority)
3. Performance issues (medium priority)
4. Code style and best practices (medium priority)
5. Test coverage suggestions (low priority)

Format your review with clear sections:
- Summary
- Critical Issues (if any)
- Suggested Improvements
- Security Notes
- Overall Assessment""",
        })
        # If message doesn't contain code, use context
        user_content = message
        if not any(marker in message for marker in ["```", "def ", "class ", "function ", "const ", "import "]):
            if context:
                last_user = next((m["content"] for m in reversed(context) if m["role"] == "user"), "")
                if last_user:
                    user_content = f"{message}\n\nCode to review:\n{last_user}"

        messages.append({"role": "user", "content": user_content})
        response = await codex.chat(messages, temperature=0.1, max_tokens=4096)
        content = response["choices"][0]["message"]["content"]

        return {
            "content": content,
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"task": "code_review"},
        }

    async def _bug_detection(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Detect bugs in provided code."""
        # Use Codex for structured bug detection
        code = message
        if context:
            last_user = next((m["content"] for m in reversed(context) if m["role"] == "user"), "")
            if last_user and "```" in last_user:
                code = last_user

        bugs = await codex.detect_bugs(code)
        if bugs:
            content = "## Bug Analysis Results\n\n"
            for bug in bugs:
                severity = bug.get("severity", "medium")
                emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                content += f"{emoji} **Line {bug.get('line', '?')}** ({severity}): {bug.get('description', '')}\n\n"
            content += "---\n### Recommended Fixes\nApply the fixes above and re-run tests."
        else:
            content = "No bugs detected in the reviewed code."

        return {
            "content": content,
            "agent": self.name,
            "tokens_used": 0,
            "metadata": {"task": "bug_detection", "bugs_found": len(bugs)},
        }

    async def _optimization(
        self, message: str, context: list[dict[str, str]] | None
    ) -> dict[str, Any]:
        """Suggest code optimizations."""
        messages = context or []
        messages = [msg for msg in messages if msg["role"] != "system"]
        messages.insert(0, {
            "role": "system",
            "content": """You are a performance optimization expert. Analyze the code and suggest optimizations.

Focus on:
1. Algorithmic improvements (time complexity)
2. Memory usage optimization
3. I/O and network efficiency
4. Caching opportunities
5. Parallelization potential

For each suggestion, provide:
- Current issue
- Optimized code
- Expected improvement""",
        })
        messages.append({"role": "user", "content": message})
        response = await codex.chat(messages, temperature=0.2, max_tokens=4096)

        return {
            "content": response["choices"][0]["message"]["content"],
            "agent": self.name,
            "tokens_used": response.get("usage", {}).get("total_tokens", 0),
            "metadata": {"task": "optimization"},
        }

    def _detect_task_type(self, message: str) -> str:
        """Detect the type of coding task."""
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["review", "code review", "review this"]):
            return "code_review"
        if any(kw in msg_lower for kw in ["bug", "debug", "fix", "error", "issue"]):
            return "bug_detection"
        if any(kw in msg_lower for kw in ["optimize", "optimization", "performance", "speed up", "faster"]):
            return "optimization"
        return "code_generation"

    def _detect_language(self, message: str) -> str:
        """Detect programming language from the message."""
        msg_lower = message.lower()
        language_map = {
            "python": ["python", "django", "flask", "fastapi", "pytorch", "tensorflow"],
            "javascript": ["javascript", "js", "node", "react", "vue", "angular", "typescript"],
            "typescript": ["typescript", "ts", "tsx", "angular", "nest"],
            "go": ["golang", "go "],
            "rust": ["rust", "cargo"],
            "java": ["java", "kotlin", "spring", "android"],
            "cpp": ["c++", "cpp", "c-plus-plus"],
            "c": ["c language", "c program"],
            "ruby": ["ruby", "rails"],
            "swift": ["swift", "ios"],
        }
        for lang, keywords in language_map.items():
            if any(kw in msg_lower for kw in keywords):
                return lang
        return "python"


# Global singleton
coding_agent = CodingAgent()
