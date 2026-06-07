"""
Router Agent
============
Entry point for all user requests. Classifies intent using LLMRouter,
selects the appropriate agent(s), and populates the routing fields
in AgentState.

Flow:
1. Quick classification via ``llm_router.categorize_request()``
2. For complex/vague inputs, refine with DeepSeek
3. Map category → agent(s) via routing table
4. Detect multi-agent planning requirements
5. Update state with routing decisions

The populated state fields (``category``, ``confidence``, ``detected_intent``,
``selected_agents``) drive downstream graph routing decisions.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from backend.agents_v2.state import AgentState
from backend.agents_v2.base import BaseAgent
from backend.llm.router import llm_router, TaskCategory
from backend.llm.deepseek import deepseek


class RouterAgent(BaseAgent):
    """Routes user requests to the appropriate agent(s) based on intent classification.

    This agent is the first node in the LangGraph execution flow. It:
    - Classifies the user's intent using keyword-based analysis (``LLMRouter``)
    - For complex, vague, or multi-intent inputs, uses DeepSeek for deeper analysis
    - Selects the best agent(s) for the task via a configurable category mapping
    - Detects multi-agent or planning requirements
    - Routes image attachments directly to the vision agent
    """

    def __init__(self) -> None:
        super().__init__(
            name="router",
            model_name="deepseek",
            system_prompt=(
                "You are JARVIS's Router Agent. Your job is to:\n"
                "1. Understand the user's intent\n"
                "2. Determine if this is a single-agent or multi-agent task\n"
                "3. Select the best agent(s) for the task\n\n"
                "Available agents:\n"
                "- coding: Code generation, review, debug, architecture\n"
                "- research: Web search, deep research, summarization\n"
                "- vision: Image analysis, OCR, screenshot analysis\n"
                "- memory: Store, recall, search memories\n"
                "- task: Task management, reminders, scheduling\n"
                "- planner: Complex multi-step tasks, project planning\n"
                "- utility: Quick answers, math, formatting, simple questions\n\n"
                "Respond with a clear intent classification."
            ),
            description="Routes user requests to the appropriate agent(s)",
        )

        # ── Category-to-agent mapping ──────────────────────────────────────
        # Maps each TaskCategory to the agent(s) that should handle it.
        self._agent_map: Dict[TaskCategory, List[str]] = {
            TaskCategory.GENERAL_CHAT: ["utility"],
            TaskCategory.CODING: ["coding"],
            TaskCategory.CODE_REVIEW: ["coding"],
            TaskCategory.RESEARCH: ["research"],
            TaskCategory.SUMMARIZATION: ["research"],
            TaskCategory.VISION: ["vision"],
            TaskCategory.OCR: ["vision"],
            TaskCategory.ROUTING: ["planner"],
            TaskCategory.MEMORY: ["memory"],
            TaskCategory.PLANNING: ["planner"],
            TaskCategory.TASK_MANAGEMENT: ["task"],
            TaskCategory.EXTRACTION: ["utility"],
        }

        # ── Complex task indicators ────────────────────────────────────────
        # Keywords that suggest the request needs planning or multi-agent
        # orchestration rather than a single-agent response.
        self._complex_keywords: List[str] = [
            "multi-step", "multi step", "complex", "plan", "strategy",
            "compare and contrast", "build a", "create a project",
            "architecture", "design a", "research then", "first",
            "step by step", "comprehensive", "project", "system design",
        ]

    async def process(self, state: AgentState) -> AgentState:
        """Classify the request and select target agent(s).

        Performs classification in stages:
        1. Extract attachment metadata for routing hints
        2. Run keyword-based classification via ``llm_router.categorize_request()``
        3. For low-confidence or complex inputs, invoke DeepSeek for deep analysis
        4. Map the final category to agent(s) and update the state

        Args:
            state: The current ``AgentState`` with at minimum ``message`` populated.

        Returns:
            Updated ``AgentState`` with routing fields set:
            ``category``, ``confidence``, ``detected_intent``, ``selected_agents``.
        """
        message = state.get("message", "")
        attachments = state.get("attachments", [])
        start_time = time.monotonic()

        # ── Step 1: Extract attachment paths ──────────────────────────────
        attachment_paths = self._extract_attachment_paths(attachments)

        # ── Step 2: Keyword-based classification ──────────────────────────
        category: TaskCategory = llm_router.categorize_request(message, attachment_paths)
        confidence: float = self._compute_confidence(category, message)
        detected_intent: str = self._describe_intent(category, message)

        # ── Step 3: DeepSeek refinement for complex/vague inputs ──────────
        if self._needs_deep_classification(message, category):
            try:
                deep_category, deep_confidence, deep_intent = await self._deep_classify(
                    state, message, attachment_paths,
                )
                if deep_confidence > confidence:
                    category = deep_category
                    confidence = deep_confidence
                    detected_intent = deep_intent
            except Exception:
                # If DeepSeek fails, keep the initial classification
                pass

        # ── Step 4: Select agents ─────────────────────────────────────────
        selected_agents = self._get_agent_for_category(category, message, attachments)

        # ── Step 5: Persist to state ──────────────────────────────────────
        state["category"] = category.value
        state["confidence"] = round(confidence, 4)
        state["detected_intent"] = detected_intent
        state["selected_agents"] = selected_agents
        state["start_time"] = start_time
        state.setdefault("graph_execution_path", []).append(self.name)

        return state

    # ── Private helpers ──────────────────────────────────────────────────

    def _extract_attachment_paths(self, attachments: Any) -> Optional[List[str]]:
        """Extract file path/URL strings from attachment data.

        Supports both ``list[dict]`` (standard ``AgentState`` format) and
        ``list[str]`` (simplified format accepted by ``LLMRouter``).

        Returns ``None`` when no valid paths are found so the router can
        distinguish "no attachments" from "empty list".
        """
        if not attachments:
            return None

        paths: List[str] = []
        for att in attachments:
            if isinstance(att, dict):
                path = att.get("path") or att.get("url") or att.get("file_path")
                if path:
                    paths.append(str(path))
            elif isinstance(att, str):
                paths.append(att)

        return paths or None

    def _compute_confidence(self, category: TaskCategory, message: str) -> float:
        """Compute classification confidence (0.0 – 1.0) based on match strength.

        Confidence is highest when there are unambiguous indicators (image
        attachments for VISION, multiple code keywords for CODING, etc.).
        """
        if category == TaskCategory.VISION:
            return 0.95  # Attachment-triggered routing is highly reliable

        if category == TaskCategory.CODING:
            code_keywords = [
                "write code", "implement", "function", "class ",
                "refactor", "debug", "api endpoint", "def ",
            ]
            matches = sum(1 for kw in code_keywords if kw in message.lower())
            return min(0.95, 0.5 + matches * 0.12)

        if category in (TaskCategory.GENERAL_CHAT, TaskCategory.ROUTING):
            return 0.50  # Ambiguous — may need further refinement

        if category in (TaskCategory.MEMORY, TaskCategory.TASK_MANAGEMENT):
            memory_kw = ["remember", "recall", "remind", "task", "todo", "schedule"]
            matches = sum(1 for kw in memory_kw if kw in message.lower())
            return min(0.90, 0.5 + matches * 0.1)

        return 0.85  # Most other categories have reasonable keyword signal

    def _describe_intent(self, category: TaskCategory, message: str) -> str:
        """Produce a short human-readable intent description."""
        descriptions = {
            TaskCategory.GENERAL_CHAT: "General conversation or simple question",
            TaskCategory.CODING: "Code generation or programming task",
            TaskCategory.CODE_REVIEW: "Code review request",
            TaskCategory.RESEARCH: "Research or information gathering",
            TaskCategory.SUMMARIZATION: "Content summarization",
            TaskCategory.VISION: "Image or visual content analysis",
            TaskCategory.OCR: "Text extraction from image",
            TaskCategory.ROUTING: "Routing decision needed",
            TaskCategory.MEMORY: "Memory storage or recall operation",
            TaskCategory.PLANNING: "Complex multi-step planning required",
            TaskCategory.TASK_MANAGEMENT: "Task or reminder management",
            TaskCategory.EXTRACTION: "Data or content extraction",
        }
        base = descriptions.get(category, f"Categorized as {category.value}")
        return f"{base}: {message[:120].strip()}"

    def _needs_deep_classification(self, message: str, category: TaskCategory) -> bool:
        """Determine whether the request warrants DeepSeek-based analysis.

        Returns ``True`` when:
        - The category is low-confidence (GENERAL_CHAT, ROUTING)
        - The message contains planning/complex keywords
        - The message is very short (≤ 3 words) and thus ambiguous
        """
        if category in (TaskCategory.GENERAL_CHAT, TaskCategory.ROUTING):
            return True

        if any(kw in message.lower() for kw in self._complex_keywords):
            return True

        if len(message.strip().split()) <= 3:
            return True

        return False

    async def _deep_classify(
        self,
        state: AgentState,
        message: str,
        attachment_paths: Optional[List[str]],
    ) -> Tuple[TaskCategory, float, str]:
        """Use DeepSeek to deeply analyse the user's intent.

        Builds a context-rich prompt including recent memory context and
        attachment information, then calls ``deepseek.extract_json()`` to
        obtain a structured classification.

        Returns:
            A tuple of ``(category, confidence, intent_description)``.
        """
        # Build contextual hints
        parts = [f"User message: {message}"]

        if attachment_paths:
            parts.append(f"Attachments: {', '.join(attachment_paths)}")

        memory_context = state.get("memory_context", [])
        if memory_context:
            recent = memory_context[-5:]
            lines = "\n".join(
                f"- {item.get('content', str(item))[:200]}" for item in recent
            )
            parts.append(f"Recent context:\n{lines}")

        user_content = "\n\n".join(parts)

        result = await deepseek.extract_json(
            system_prompt=(
                "You classify user requests for the JARVIS AI assistant. "
                "Analyse the input and respond with valid JSON only."
            ),
            user_message=(
                f"{user_content}\n\n"
                "Respond with this exact JSON structure:\n"
                "{\n"
                '  "category": "general_chat|coding|code_review|research|'
                "summarization|vision|ocr|memory|planning|task_management|extraction\",\n"
                '  "confidence": 0.0-1.0,\n'
                '  "intent": "short description of the user goal",\n'
                '  "agents_required": ["agent_name", ...],\n'
                '  "requires_planning": true/false\n'
                "}"
            ),
        )

        # Parse category string back to enum
        category_str = result.get("category", "general_chat")
        try:
            category = TaskCategory(category_str)
        except ValueError:
            category = TaskCategory.GENERAL_CHAT

        confidence = min(1.0, max(0.0, float(result.get("confidence", 0.7))))
        intent = result.get("intent", category.value)

        return category, confidence, intent

    def _get_agent_for_category(
        self,
        category: TaskCategory,
        message: str,
        attachments: List[Dict[str, Any]],
    ) -> List[str]:
        """Map a classified category to the agent(s) that should handle the request.

        Routing rules applied in priority order:
        1. Image attachments → vision agent (overrides category)
        2. PLANNING category → planner agent
        3. Multi-agent tasks (research + implement) → planner agent
        4. Single-agent lookup via ``_agent_map``
        """
        # Image attachments force vision routing
        if attachments and category in (TaskCategory.VISION, TaskCategory.OCR):
            return ["vision"]

        # Planning always goes to planner for decomposition
        if category == TaskCategory.PLANNING:
            return ["planner"]

        # Multi-agent tasks go through the planner
        if self._is_multi_agent_task(category, message):
            return ["planner"]

        # Default: single agent from mapping
        return list(self._agent_map.get(category, ["utility"]))

    def _is_multi_agent_task(self, category: TaskCategory, message: str) -> bool:
        """Detect requests that would benefit from multi-agent orchestration.

        Looks for patterns that combine research and implementation, or
        explicit planning keywords.
        """
        msg_lower = message.lower()

        if any(kw in msg_lower for kw in self._complex_keywords):
            return True

        # Research-then-implement pattern (e.g. "research X and implement Y")
        has_research = any(
            kw in msg_lower
            for kw in ["research", "find", "search", "look up", "investigate"]
        )
        has_implement = any(
            kw in msg_lower
            for kw in ["implement", "code", "build", "create", "write", "develop"]
        )
        if has_research and has_implement:
            return True

        return False
