"""Intent router — keyword → regex → fuzzy matching pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from thefuzz import fuzz


class IntentType(str, Enum):
    TASK_CREATE = "TASK_CREATE"
    TASK_LIST = "TASK_LIST"
    TASK_COMPLETE = "TASK_COMPLETE"
    TASK_DELETE = "TASK_DELETE"
    TASK_SEARCH = "TASK_SEARCH"

    NOTE_CREATE = "NOTE_CREATE"
    NOTE_SEARCH = "NOTE_SEARCH"

    REMINDER_CREATE = "REMINDER_CREATE"

    MEMORY_SAVE = "MEMORY_SAVE"
    MEMORY_RECALL = "MEMORY_RECALL"

    UNKNOWN = "UNKNOWN"


@dataclass
class IntentResult:
    type: IntentType
    confidence: float  # 0.0 to 1.0
    entities: dict[str, str] = field(default_factory=dict)
    raw_text: str = ""

    def is_known(self) -> bool:
        return self.type != IntentType.UNKNOWN


# ── Pattern Definitions ────────────────────────────────────────

EXACT_PATTERNS: dict[str, IntentType] = {
    "show my tasks": IntentType.TASK_LIST,
    "show my pending tasks": IntentType.TASK_LIST,
    "show my completed tasks": IntentType.TASK_LIST,
    "list my tasks": IntentType.TASK_LIST,
    "what are my tasks": IntentType.TASK_LIST,
}

REGEX_PATTERNS: list[tuple[re.Pattern, IntentType, list[str]]] = [
    # TASK_CREATE
    (re.compile(
        r"(?:create|add|make|new)\s+(?:a\s+)?(?:task|todo)\s+(?:to\s+)?(?:called\s+)?(?:titled\s+)?['\"]?(.+?)['\"]?(?:\s+(?:with|by|due|priority|on|for))?(?:\s+(?:high|medium|low|urgent))?(?:\s+priority)?(?:\s+due\s+(.+?))?$",
        re.IGNORECASE,
    ), IntentType.TASK_CREATE, ["title", "due_date"]),

    # TASK_LIST
    (re.compile(
        r"(?:show|list|get|display|what(?:'s| is| are))\s+(?:my\s+)?(?:pending\s+)?(?:completed\s+)?(?:tasks|task|todos|todo)",
        re.IGNORECASE,
    ), IntentType.TASK_LIST, []),

    # TASK_COMPLETE
    (re.compile(
        r"(?:complete|mark\s+(?:as\s+)?done|finish|done)\s+(?:my\s+)?(?:task\s+)?(?:called\s+)?(?:titled\s+)?['\"]?(.+?)['\"]?$",
        re.IGNORECASE,
    ), IntentType.TASK_COMPLETE, ["title"]),

    # TASK_DELETE
    (re.compile(
        r"(?:delete|remove|erase)\s+(?:my\s+)?(?:task\s+)?(?:called\s+)?(?:titled\s+)?['\"]?(.+?)['\"]?$",
        re.IGNORECASE,
    ), IntentType.TASK_DELETE, ["title"]),

    # TASK_SEARCH
    (re.compile(
        r"(?:search|find|look\s+(?:for|up))\s+(?:my\s+)?(?:tasks?\s+(?:about|for|with|containing)\s+)?['\"]?(.+?)['\"]?$",
        re.IGNORECASE,
    ), IntentType.TASK_SEARCH, ["query"]),
]

FUZZY_THRESHOLD = 75

FUZZY_PATTERNS: list[tuple[str, IntentType]] = [
    ("create task", IntentType.TASK_CREATE),
    ("add task", IntentType.TASK_CREATE),
    ("make task", IntentType.TASK_CREATE),
    ("new task", IntentType.TASK_CREATE),
    ("show tasks", IntentType.TASK_LIST),
    ("list tasks", IntentType.TASK_LIST),
    ("my tasks", IntentType.TASK_LIST),
    ("complete task", IntentType.TASK_COMPLETE),
    ("mark done", IntentType.TASK_COMPLETE),
    ("finish task", IntentType.TASK_COMPLETE),
    ("delete task", IntentType.TASK_DELETE),
    ("remove task", IntentType.TASK_DELETE),
    ("search task", IntentType.TASK_SEARCH),
    ("find task", IntentType.TASK_SEARCH),
    ("create note", IntentType.NOTE_CREATE),
    ("find note", IntentType.NOTE_SEARCH),
    ("remind me", IntentType.REMINDER_CREATE),
    ("remember", IntentType.MEMORY_SAVE),
    ("recall", IntentType.MEMORY_RECALL),
    ("what do you know", IntentType.MEMORY_RECALL),
]


class IntentRouter:
    """Routes natural language to structured intents."""

    def route(self, text: str) -> IntentResult:
        """Route text through the matching pipeline."""
        text = text.strip().lower()
        result = IntentResult(raw_text=text, type=IntentType.UNKNOWN, confidence=0.0)

        # 1. Exact match
        if text in EXACT_PATTERNS:
            result.type = EXACT_PATTERNS[text]
            result.confidence = 1.0
            return result

        # 2. Regex match
        for pattern, intent_type, entity_keys in REGEX_PATTERNS:
            match = pattern.search(text)
            if match:
                result.type = intent_type
                result.confidence = 0.95
                for i, key in enumerate(entity_keys):
                    if i < len(match.groups()) and match.group(i + 1):
                        result.entities[key] = match.group(i + 1).strip()
                return result

        # 3. Fuzzy match
        best_score = 0
        best_intent = IntentType.UNKNOWN
        for pattern_text, intent_type in FUZZY_PATTERNS:
            score = fuzz.ratio(text, pattern_text)
            if score > best_score:
                best_score = score
                best_intent = intent_type

        if best_score >= FUZZY_THRESHOLD:
            result.type = best_intent
            result.confidence = best_score / 100.0
            return result

        return result


intent_router = IntentRouter()
