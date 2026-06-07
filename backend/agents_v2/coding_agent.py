"""
Coding Agent
============
Handles code generation, review, debugging, optimization, and architecture
analysis using the Codex (GPT-4o) model. Integrates with the LangGraph
multi-agent system via BaseAgent.

Task types detected from the user message:
  - ``generate``: Write new code from a natural-language prompt.
  - ``review``:   Analyse existing code for bugs, security, style, performance.
  - ``debug``:    Locate bugs and produce a corrected version.
  - ``optimize``: Refactor code for performance, readability, or maintainability.
  - ``analyze``:  High-level architecture / design analysis (no code required).
"""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, List, Optional

from backend.agents_v2.state import AgentState, ExecutionStatus
from backend.agents_v2.base import BaseAgent
from backend.agents_v2.registry import get_agent_registry
from backend.agents_v2.tools import AgentTools
from backend.llm.codex import codex
from backend.llm.minimax import minimax
from backend.llm.mimo import mimo
from backend.llm.router import llm_router
from backend.services.search_service import search_service
from backend.database.mongodb import mongodb
from config.settings import settings


LANGUAGE_MAP: Dict[str, str] = {
    # ── Common explicit labels ────────────────────────────────
    "python": "python",
    "py": "python",
    "javascript": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    "java": "java",
    "kotlin": "kotlin",
    "kt": "kotlin",
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "rs": "rust",
    "c": "c",
    "c++": "cpp",
    "cpp": "cpp",
    "csharp": "csharp",
    "c#": "csharp",
    "objective-c": "objectivec",
    "swift": "swift",
    "ruby": "ruby",
    "rb": "ruby",
    "php": "php",
    "perl": "perl",
    "pl": "perl",
    "shell": "bash",
    "bash": "bash",
    "sh": "bash",
    "powershell": "powershell",
    "ps1": "powershell",
    "sql": "sql",
    "html": "html",
    "css": "css",
    "scss": "scss",
    "sass": "sass",
    "less": "less",
    "vue": "vue",
    "svelte": "svelte",
    "react": "jsx",
    "dart": "dart",
    "flutter": "dart",
    "lua": "lua",
    "r": "r",
    "scala": "scala",
    "haskell": "haskell",
    "hs": "haskell",
    "elixir": "elixir",
    "ex": "elixir",
    "clojure": "clojure",
    "clj": "clojure",
    "erlang": "erlang",
    "erl": "erlang",
    "groovy": "groovy",
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    "xml": "xml",
    "markdown": "markdown",
    "md": "markdown",
    "dockerfile": "dockerfile",
    "makefile": "makefile",
    "cmake": "cmake",
    "toml": "toml",
    "ini": "ini",
    "cfg": "ini",
    "conf": "ini",
    "protobuf": "protobuf",
    "proto": "protobuf",
    "graphql": "graphql",
    "gql": "graphql",
    "terraform": "hcl",
    "tf": "hcl",
    "solidity": "solidity",
    "sol": "solidity",
}

TASK_KEYWORDS: Dict[str, List[str]] = {
    "generate": [
        "write", "create", "generate", "implement", "build", "develop",
        "new function", "new class", "new file", "add feature",
        "boilerplate", "scaffold", "template", "from scratch",
    ],
    "review": [
        "review", "code review", "review this", "review code",
        "code quality", "audit", "inspect", "check my code",
        "feedback", "critique", "suggestions",
    ],
    "debug": [
        "debug", "fix bug", "bug fix", "issue", "error",
        "not working", "broken", "crash", "exception",
        "fails", "incorrect", "wrong output", "bug",
    ],
    "optimize": [
        "optimize", "optimization", "performance", "speed up",
        "refactor", "improve", "slow", "memory", "time complexity",
        "space complexity", "reduce", "clean up", "lint",
    ],
    "analyze": [
        "analyze", "architecture", "design", "explain", "diagram",
        "structure", "component", "dependency", "flow",
        "how does", "overview", "understand", "document",
    ],
}


class CodingAgent(BaseAgent):
    """Coding agent that generates, reviews, debugs, optimises and analyses code.

    Uses the **Codex** (GPT-4o) model directly for all code-related tasks.
    Detection of the task type and programming language is done via keyword
    analysis on the user message.
    """

    def __init__(self) -> None:
        super().__init__(
            name="coding",
            model_name="codex",
            system_prompt=(
                "You are JARVIS's Coding Agent, an expert software engineer. "
                "You generate clean, production-ready code; review code for bugs, "
                "security issues and style; debug and fix problems; optimise for "
                "performance and readability; and analyse software architecture.\n\n"
                "Rules:\n"
                "1. Always return complete, runnable code unless asked for a snippet.\n"
                "2. Include type hints and docstrings in supported languages.\n"
                "3. Handle errors gracefully and mention edge cases.\n"
                "4. Explain your reasoning concisely, then show the code.\n"
                "5. When reviewing, be constructive and specific.\n"
                "6. When debugging, explain root cause before showing the fix."
            ),
            description="Generates, reviews, debugs, and optimizes code using Codex",
        )
        # Auto-register so the graph and router can discover this agent.
        get_agent_registry().register(self)

    # ── Public API ──────────────────────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Execute the coding agent's logic on *state*.

        Steps:
        1. Read ``message`` and ``shared_context`` from the state.
        2. Load recent memory context for the user.
        3. Detect task type (generate / review / debug / optimize / analyse).
        4. Detect the target programming language.
        5. Call the appropriate Codex method.
        6. Store results in ``shared_context`` and ``final_response``.
        7. Log execution via ``_store_agent_log``.
        """
        message: str = state.get("message", "") or ""
        context: Dict[str, Any] = state.get("shared_context", {})
        memory_ctx = await self._load_memory_context(state)

        task_type = self._detect_task_type(message)
        language = self._detect_language(message)

        # Expose detection results in shared context for downstream agents.
        state.setdefault("shared_context", {})
        state["shared_context"]["coding_task_type"] = task_type
        state["shared_context"]["coding_language"] = language

        try:
            final: str = ""

            if task_type == "generate":
                final = await self._handle_generate(message, language, context, memory_ctx)

            elif task_type == "review":
                final = await self._handle_review(message, language)

            elif task_type == "debug":
                final = await self._handle_debug(message, language)

            elif task_type == "optimize":
                final = await self._handle_optimize(message, language, context)

            elif task_type == "analyze":
                final = await self._handle_analyze(message, context, memory_ctx)

            else:
                # Fallback: treat as generation.
                final = await codex.generate_code(message, language)

            state["final_response"] = final
            state["response_agent"] = self.name
            state["shared_context"]["coding_result"] = final

            await self._store_agent_log(
                state,
                action=f"code_{task_type}",
                input_summary=message,
                output_summary=final,
                status="success",
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            state.setdefault("errors", []).append({
                "step_id": f"{self.name}_{int(__import__('time').time() * 1000)}",
                "agent": self.name,
                "error": error_msg,
                "retry_count": state.get("retry_count", 0),
                "timestamp": __import__("time").time(),
            })
            state["final_response"] = (
                f"I encountered an error while processing your coding request "
                f"(task: {task_type}, language: {language}).\n\n{error_msg}"
            )
            state["response_agent"] = self.name

            await self._store_agent_log(
                state,
                action=f"code_{task_type}",
                input_summary=message,
                output_summary="",
                status="failed",
                error=error_msg,
            )

        return state

    # ── Task detection ──────────────────────────────────────────────────

    @staticmethod
    def _detect_task_type(message: str) -> str:
        """Return ``"generate"``, ``"review"``, ``"debug"``, ``"optimize"`` or ``"analyze"``.

        Scoring is used so that when multiple keyword groups match, the group
        with the most hits wins.
        """
        msg_lower = message.lower()
        scores: Dict[str, int] = {}
        for task, keywords in TASK_KEYWORDS.items():
            scores[task] = sum(1 for kw in keywords if kw in msg_lower)

        # If no clear signal, default to "generate".
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "generate"

    @staticmethod
    def _detect_language(message: str) -> str:
        """Return the programming language name inferred from *message*.

        Checks for explicit language names, common file extensions, fenced
        code block labels, and framework references.
        """
        msg_lower = message.lower()

        # 1. Explicit mentions preceded by keywords.
        for alias, lang in LANGUAGE_MAP.items():
            pattern = rf"(?:^|\s)(?:in\s+)?{re.escape(alias)}(?:\s|$|,|\.)"
            if re.search(pattern, msg_lower):
                return lang

        # 2. File-extension references (e.g. "my_app.py" or ".py file").
        ext_map = {v: k for k, v in LANGUAGE_MAP.items()}
        ext_match = re.findall(r"\.(\w+)\b", msg_lower)
        for ext in ext_match:
            for lang_alias, lang in LANGUAGE_MAP.items():
                if ext == lang_alias:
                    return lang

        # 3. Check for fenced code blocks in the message itself.
        block_lang_match = re.search(r"```(\w+)", message)
        if block_lang_match:
            raw = block_lang_match.group(1).lower()
            if raw in LANGUAGE_MAP:
                return LANGUAGE_MAP[raw]

        # 4. Framework / context hints.
        framework_langs: Dict[str, str] = {
            "react": "jsx",
            "vue": "vue",
            "angular": "typescript",
            "svelte": "svelte",
            "django": "python",
            "flask": "python",
            "fastapi": "python",
            "spring": "java",
            "rails": "ruby",
            "laravel": "php",
            "express": "javascript",
            "nextjs": "typescript",
            "tensorflow": "python",
            "pytorch": "python",
        }
        for framework, lang in framework_langs.items():
            if framework in msg_lower:
                return lang

        return "python"  # Sensible default.

    # ── Task handlers ───────────────────────────────────────────────────

    async def _handle_generate(
        self,
        message: str,
        language: str,
        context: Dict[str, Any],
        memory_ctx: List[Dict[str, Any]],
    ) -> str:
        """Generate code from a natural-language prompt."""
        context_str = self._build_context_str(context, memory_ctx)
        return await codex.generate_code(message, language, context=context_str or None)

    async def _handle_review(self, message: str, language: str) -> str:
        """Review code extracted from the message or passed inline."""
        code = self._extract_code_block(message) or message
        review_result = await codex.review_code(code, language)
        return self._format_review(review_result, code)

    async def _handle_debug(self, message: str, language: str) -> str:
        """Detect bugs in code and produce a corrected version."""
        code = self._extract_code_block(message) or message
        bugs = await codex.detect_bugs(code, language)
        # Ask codex to produce a fixed version.
        fix_prompt = (
            f"The following {language} code has bugs. Please provide the "
            f"complete corrected version with an explanation of each fix.\n\n"
            f"```{language}\n{code}\n```"
        )
        fixed_code = await codex.generate_code(fix_prompt, language)
        return self._format_debug(bugs, fixed_code)

    async def _handle_optimize(
        self,
        message: str,
        language: str,
        context: Dict[str, Any],
    ) -> str:
        """Optimise code for performance, readability, or maintainability."""
        code = self._extract_code_block(message) or message
        context_str = self._build_context_str(context, [])
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an expert {language} performance engineer. "
                    f"Analyse the following code and provide:\n"
                    f"1. A summary of performance/complexity issues found.\n"
                    f"2. Specific optimisation suggestions with before/after.\n"
                    f"3. The complete optimised version of the code.\n\n"
                    f"Be practical — suggest only changes that matter."
                ),
            },
            {"role": "user", "content": f"Context:\n{context_str}\n\nCode:\n```{language}\n{code}\n```"},
        ]
        if context_str:
            messages.insert(1, {"role": "system", "content": f"Additional context:\n{context_str}"})
        response = await codex.chat(messages, temperature=0.2)
        return response["choices"][0]["message"]["content"]

    async def _handle_analyze(
        self,
        message: str,
        context: Dict[str, Any],
        memory_ctx: List[Dict[str, Any]],
    ) -> str:
        """Analyse architecture, design, or code structure."""
        context_str = self._build_context_str(context, memory_ctx)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior software architect. Analyse the request "
                    "and provide:\n"
                    "1. High-level architecture overview.\n"
                    "2. Key components and their responsibilities.\n"
                    "3. Data flow and dependencies.\n"
                    "4. Design patterns identified.\n"
                    "5. Potential risks or improvements.\n\n"
                    "Be concrete and reference specific technologies where relevant."
                ),
            },
            {"role": "user", "content": message},
        ]
        if context_str:
            messages.insert(1, {"role": "system", "content": f"Context:\n{context_str}"})
        response = await codex.chat(messages, temperature=0.3)
        return response["choices"][0]["message"]["content"]

    # ── Formatting helpers ──────────────────────────────────────────────

    @staticmethod
    def _format_review(review: Dict[str, Any], original_code: str) -> str:
        """Format a review dict into a human-readable markdown report."""
        lines: List[str] = [
            "# Code Review Report\n",
        ]

        score = review.get("overall_score")
        if score is not None:
            lines.append(f"**Overall Score:** {score}/10\n")

        summary = review.get("summary")
        if summary:
            lines.append(f"**Summary:** {summary}\n")

        bugs = review.get("bugs", [])
        if bugs:
            lines.append("## Bugs & Logic Errors\n")
            for bug in bugs:
                sev = bug.get("severity", "unknown")
                desc = bug.get("description", "")
                line_num = bug.get("line", "")
                lines.append(f"- **Line {line_num}** (severity: {sev}): {desc}")

        vulnerabilities = review.get("vulnerabilities", [])
        if vulnerabilities:
            lines.append("\n## Security Vulnerabilities\n")
            for vuln in vulnerabilities:
                sev = vuln.get("severity", "unknown")
                desc = vuln.get("description", "")
                lines.append(f"- (severity: {sev}): {desc}")

        suggestions = review.get("suggestions", [])
        if suggestions:
            lines.append("\n## Suggestions\n")
            for s in suggestions:
                lines.append(f"- {s}")

        if not (bugs or vulnerabilities or suggestions):
            lines.append("\nNo issues found. The code looks clean!")

        return "\n".join(lines)

    @staticmethod
    def _format_debug(bugs: List[Dict[str, Any]], fixed_code: str) -> str:
        """Format bug detection results and the fixed code."""
        lines: List[str] = [
            "# Bug Analysis & Fix\n",
        ]

        if bugs:
            lines.append("## Detected Issues\n")
            for bug in bugs:
                line_num = bug.get("line", "?")
                severity = bug.get("severity", "unknown")
                desc = bug.get("description", bug.get("message", ""))
                lines.append(f"- **Line {line_num}** ({severity}): {desc}")
        else:
            lines.append("**No bugs detected by automated analysis.**\n")

        lines.append("\n## Corrected Code\n")
        lines.append(fixed_code)

        return "\n".join(lines)

    # ── Utility helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_code_block(text: str) -> Optional[str]:
        """Extract the first fenced code block from *text*, if any."""
        match = re.search(r"```(?:\w*)\n(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _build_context_str(
        context: Dict[str, Any],
        memory_ctx: List[Dict[str, Any]],
    ) -> str:
        """Build a plain-text context summary from shared context and memory."""
        parts: List[str] = []
        if context:
            # Exclude keys we know are ephemeral or circular.
            skip_keys = {"coding_result", "coding_task_type", "coding_language"}
            ctx_items = {k: v for k, v in context.items() if k not in skip_keys}
            if ctx_items:
                parts.append("Shared Context:")
                for k, v in ctx_items.items():
                    v_str = str(v)[:500]
                    parts.append(f"  {k}: {v_str}")
        if memory_ctx:
            parts.append("\nRecent Memory:")
            for m in memory_ctx[-5:]:
                content = m.get("content", str(m))[:200]
                parts.append(f"  - {content}")
        return "\n".join(parts)

    def __repr__(self) -> str:
        return f"<CodingAgent model={self.model_name}>"
