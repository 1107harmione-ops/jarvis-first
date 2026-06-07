"""
Utility Agent
=============
Handles quick, simple tasks efficiently using lightweight models.

Capabilities:
- Math calculations and formula evaluation
- JSON formatting, validation, and pretty-printing
- Code block extraction
- Unit conversions, text processing
- General Q&A and simple reasoning

Design principles:
- Pure math/format operations skip LLM calls entirely (fast path).
- Q&A and knowledge tasks use the base ``_call_llm()`` which routes
  through the LLM router for optimal model selection.
- Responses are kept concise and accurate.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.agents_v2.state import AgentState, ExecutionStatus
from backend.agents_v2.base import BaseAgent
from backend.agents_v2.registry import get_agent_registry
from backend.agents_v2.tools import AgentTools
from backend.llm.deepseek import deepseek
from backend.database.mongodb import mongodb
from memory.short_term import stm
from memory.long_term import ltm
from memory.vector_memory import vector_memory
from backend.services.memory_service import memory_service
from backend.services.task_service import task_service
from config.settings import settings


class UtilityAgent(BaseAgent):
    """
    Utility Agent — fast, lightweight operations.

    Operation detection (in priority order):

    1. **Math expression**  — ``AgentTools.calculate()``, no LLM.
    2. **JSON format**      — ``AgentTools.format_json()`` or parse-then-reformat.
    3. **Code extraction**  — ``AgentTools.extract_code_blocks()``.
    4. **General Q&A**      — routed through the base LLM call (temperature 0.1 for
       precision, 0.5 for creative/explanation tasks).

    The agent is designed so that simple operations complete in < 100 ms
    with zero external API calls.
    """

    # Regex to detect a pure math expression (only safe chars).
    _MATH_EXPR = re.compile(r"^[\d\s\+\-\*\/\(\)\.\,%\^]+$")

    # Patterns for JSON-related requests.
    _JSON_FORMAT = re.compile(
        r"(?:format\s*(?:as\s*)?json|pretty.?print|"
        r"beautify|validate\s*json|json\s*format)",
        re.IGNORECASE,
    )

    # Patterns for code extraction.
    _CODE_EXTRACT = re.compile(
        r"(?:extract\s*(?:code|block|snippet)|"
        r"find\s*(?:code|block)|show\s*code)",
        re.IGNORECASE,
    )

    # Patterns for unit conversion / translation / summarisation.
    _UNIT_CONVERT = re.compile(
        r"(?:convert|in\s+(?:cm|in|ft|m|km|lb|kg|gb|mb)|"
        r"how\s+many\s+(?:cm|in|ft|m|km|lb|kg|gb|mb))",
        re.IGNORECASE,
    )
    _SUMMARIZE = re.compile(
        r"(?:summarize|summary|tl;dr|brief|in\s*short|"
        r"give\s*me\s*(?:the\s*)?highlights)",
        re.IGNORECASE,
    )
    _TRANSLATE = re.compile(
        r"(?:translate|in\s+\w+(\s*\(.*\))?)\s*",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        super().__init__(
            name="utility",
            model_name="mimo",
            system_prompt="""You are JARVIS's Utility Agent. You handle quick, simple tasks efficiently.

You can:
- Math: calculations, conversions, formulas
- Formatting: format JSON, validate data, convert formats
- Quick answers: general knowledge, definitions, facts
- Simple reasoning: logic puzzles, sorting, filtering
- Text processing: summarization, extraction, translation

Keep responses concise and accurate.""",
            description="Handles quick tasks, math, formatting, and simple Q&A",
        )

    # ── Main entry point ──────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Detect the type of utility request and handle it."""
        message = state["message"]
        user_id = state["user_id"]

        try:
            result = await self._route_operation(message, user_id)
        except Exception as exc:
            result = {
                "response": f"I ran into an error while processing that: {exc}",
            }

        state["final_response"] = result.get("response", "")
        state.setdefault("shared_context", {})
        state["shared_context"]["utility_result"] = result

        await self._store_agent_log(
            state=state,
            action="utility_process",
            input_summary=message[:200],
            output_summary=result.get("response", "")[:200],
        )

        return state

    # ── Routing ───────────────────────────────────────────────

    async def _route_operation(
        self, message: str, user_id: str
    ) -> Dict[str, Any]:
        """
        Classify the request and dispatch to the appropriate handler.

        Priority:
        1. Pure math expression
        2. JSON formatting / validation
        3. Code extraction
        4. General Q&A (via LLM)
        """
        msg = message.strip()

        # ── 1. Pure math expression ────────────────────────────
        # If the message is entirely a mathematical expression, evaluate
        # directly without an LLM call.
        stripped = msg.replace(" ", "").replace(",", "")
        if self._MATH_EXPR.match(stripped) and re.search(r"[\d]", stripped):
            return self._do_math(msg)

        # ── 2. JSON formatting ─────────────────────────────────
        if self._JSON_FORMAT.search(msg):
            return await self._do_json_format(msg)

        # ── 3. Code extraction ─────────────────────────────────
        if self._CODE_EXTRACT.search(msg):
            return await self._do_code_extract(msg)

        # ── 4. General Q&A (LLM) ───────────────────────────────
        return await self._do_llm_response(msg, user_id)

    # ── Math (fast path, no LLM) ──────────────────────────────

    def _do_math(self, expression: str) -> Dict[str, Any]:
        """Evaluate a mathematical expression."""
        # Clean up the expression for the calculator
        cleaned = expression.replace("×", "*").replace("÷", "/").replace("^", "**")
        result = AgentTools.calculate(cleaned)

        return {"response": f"🧮 **{expression.strip()}** = {result}", "_type": "math"}

    # ── JSON formatting ───────────────────────────────────────

    async def _do_json_format(self, message: str) -> Dict[str, Any]:
        """
        Try to find JSON in the message and format it.
        Falls back to asking DeepSeek to generate sample JSON or explain format.
        """
        # Attempt to extract JSON from the message body
        json_match = re.search(
            r"```(?:json)?\s*\n?([\s\S]*?)```|(\{[\s\S]*\}|\[[\s\S]*\])",
            message,
        )
        if json_match:
            raw = json_match.group(1) or json_match.group(2) or ""
            try:
                parsed = json.loads(raw)
                formatted = AgentTools.format_json(parsed)
                return {
                    "response": f"✅ Formatted JSON:\n```json\n{formatted}\n```",
                    "_type": "json_format",
                }
            except json.JSONDecodeError as e:
                return {
                    "response": f"⚠️ Invalid JSON: {e}",
                    "_type": "json_format",
                }

        # No JSON found — ask the LLM to generate / handle it
        messages = self._build_system_messages(
            message,
            context="The user is asking about JSON formatting. Respond helpfully.",
        )
        text, tokens = await self._call_llm(messages, temperature=0.2)

        # Update token tracking
        return {"response": text, "_type": "json_format"}

    # ── Code extraction ───────────────────────────────────────

    async def _do_code_extract(self, message: str) -> Dict[str, Any]:
        """Extract code blocks from the provided text."""
        blocks = AgentTools.extract_code_blocks(message)

        if not blocks:
            # No code blocks found — maybe the user is asking how to extract
            messages = self._build_system_messages(
                message,
                context="The user wants code extracted. If there is code in their message, "
                "extract it. Otherwise explain how to extract code blocks.",
            )
            text, tokens = await self._call_llm(messages, temperature=0.2)
            return {"response": text, "_type": "code_extract"}

        lines = [f"Found {len(blocks)} code block(s):\n"]
        for i, block in enumerate(blocks, 1):
            lang = block.get("language", "unknown")
            code = block.get("code", "")
            lines.append(f"**Block {i}** ({lang}):")
            lines.append(f"```{lang}\n{code}\n```")
            lines.append("")

        return {"response": "\n".join(lines), "_type": "code_extract"}

    # ── General Q&A (LLM) ─────────────────────────────────────

    async def _do_llm_response(self, message: str, user_id: str) -> Dict[str, Any]:
        """
        Use the base LLM call for general questions.

        Chooses temperature based on the nature of the request:
        - 0.1 for factual / definition / precision questions
        - 0.5 for creative / explanatory / conversational
        """
        temperature = self._guess_temperature(message)

        messages = self._build_system_messages(message)
        text, tokens = await self._call_llm(messages, temperature=temperature)

        return {"response": text, "_type": "llm_qa"}

    # ── Temperature heuristics ────────────────────────────────

    def _guess_temperature(self, message: str) -> float:
        """
        Return a low temperature for precision tasks (math, facts,
        definitions) and a higher one for creative / open-ended queries.
        """
        precision_keywords = [
            "define", "definition", "what is", "calculate", "compute",
            "convert", "fact", "formula", "equation", "spell", "meaning",
            "synonym", "antonym", "translate", "capital", "population",
            "distance", "speed", "weight", "temperature",
        ]
        msg_lower = message.lower()
        for kw in precision_keywords:
            if kw in msg_lower:
                return 0.1
        return 0.5
