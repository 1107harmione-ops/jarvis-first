from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from thefuzz import fuzz


class IntentType(str, Enum):
    TASK_CREATE = "task_create"
    TASK_LIST = "task_list"
    TASK_COMPLETE = "task_complete"
    TASK_DELETE = "task_delete"
    TASK_SEARCH = "task_search"
    NOTE_CREATE = "note_create"
    NOTE_SEARCH = "note_search"
    REMINDER_CREATE = "reminder_create"
    MEMORY_SAVE = "memory_save"
    MEMORY_RECALL = "memory_recall"
    MEMORY_FORGET = "memory_forget"
    SEARCH = "search"
    HELP = "help"
    STOP = "stop"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class IntentResult:
    intent: IntentType
    confidence: float
    params: dict = field(default_factory=dict)
    raw_text: str = ""


class IntentRouter:
    def __init__(self, fuzzy_threshold: int = 80):
        self.fuzzy_threshold = fuzzy_threshold
        self._exact_patterns: list[tuple[re.Pattern, IntentType, dict]] = []
        self._exact_phrases: list[tuple[str, IntentType, dict]] = []
        self._setup_patterns()

    def _setup_patterns(self):
        self._exact_phrases = [
            ("show my tasks", IntentType.TASK_LIST, {"status": None}),
            ("show tasks", IntentType.TASK_LIST, {"status": None}),
            ("list tasks", IntentType.TASK_LIST, {"status": None}),
            ("list my tasks", IntentType.TASK_LIST, {"status": None}),
            ("my tasks", IntentType.TASK_LIST, {"status": None}),
            ("pending tasks", IntentType.TASK_LIST, {"status": "pending"}),
            ("completed tasks", IntentType.TASK_LIST, {"status": "completed"}),
            ("help", IntentType.HELP, {}),
            ("stop", IntentType.STOP, {}),
            ("exit", IntentType.STOP, {}),
            ("quiet", IntentType.STOP, {}),
            ("hello", IntentType.GREETING, {}),
            ("hi", IntentType.GREETING, {}),
            ("hey", IntentType.GREETING, {}),
            ("good morning", IntentType.GREETING, {}),
            ("good evening", IntentType.GREETING, {}),
        ]

        self._exact_patterns = [
            (re.compile(r"create\s+(?:a\s+)?task\s+(?:to\s+)?(?:called\s+)?(?:for\s+)?(.+)", re.IGNORECASE),
             IntentType.TASK_CREATE, {"title_extract": True}),

            (re.compile(r"add\s+(?:a\s+)?task\s+(?:to\s+)?(?:called\s+)?(.+)", re.IGNORECASE),
             IntentType.TASK_CREATE, {"title_extract": True}),

            (re.compile(r"new\s+task\s+(?:called\s+)?(.+)", re.IGNORECASE),
             IntentType.TASK_CREATE, {"title_extract": True}),

            (re.compile(r"complete\s+(?:the\s+)?(?:my\s+)?task\s+(\d+)", re.IGNORECASE),
             IntentType.TASK_COMPLETE, {}),

            (re.compile(r"mark\s+(?:the\s+)?(?:my\s+)?task\s+(\d+)\s+(?:as\s+)?(?:done|complete|completed)", re.IGNORECASE),
             IntentType.TASK_COMPLETE, {}),

            (re.compile(r"finish\s+(?:the\s+)?task\s+(\d+)", re.IGNORECASE),
             IntentType.TASK_COMPLETE, {}),

            (re.compile(r"delete\s+(?:the\s+)?(?:my\s+)?task\s+(\d+)", re.IGNORECASE),
             IntentType.TASK_DELETE, {}),

            (re.compile(r"remove\s+(?:the\s+)?task\s+(\d+)", re.IGNORECASE),
             IntentType.TASK_DELETE, {}),

            (re.compile(r"search\s+(?:for\s+)?(?:tasks?\s+)?(?:about\s+)?(.+)", re.IGNORECASE),
             IntentType.TASK_SEARCH, {}),

            (re.compile(r"find\s+(?:tasks?\s+)?(?:about\s+)?(.+)", re.IGNORECASE),
             IntentType.TASK_SEARCH, {}),

            (re.compile(r"create\s+(?:a\s+)?note\s+(?:about\s+|called\s+)?(.+)", re.IGNORECASE),
             IntentType.NOTE_CREATE, {}),

            (re.compile(r"save\s+(?:a\s+)?note\s+(?:about\s+|called\s+)?(.+)", re.IGNORECASE),
             IntentType.NOTE_CREATE, {}),

            (re.compile(r"find\s+(?:a\s+)?note\s+(?:about\s+)?(.+)", re.IGNORECASE),
             IntentType.NOTE_SEARCH, {}),

            (re.compile(r"search\s+(?:for\s+)?note\s+(?:about\s+)?(.+)", re.IGNORECASE),
             IntentType.NOTE_SEARCH, {}),

            (re.compile(r"remind\s+(?:me\s+)?(.+)", re.IGNORECASE),
             IntentType.REMINDER_CREATE, {}),

            (re.compile(r"set\s+(?:a\s+)?reminder\s+(?:for\s+)?(.+)", re.IGNORECASE),
             IntentType.REMINDER_CREATE, {}),

            (re.compile(r"remember\s+(?:that\s+)?(.+)", re.IGNORECASE),
             IntentType.MEMORY_SAVE, {}),

            (re.compile(r"save\s+(?:that\s+)?(.+)", re.IGNORECASE),
             IntentType.MEMORY_SAVE, {}),

            (re.compile(r"what\s+do\s+you\s+know\s+(?:about\s+)?(.+)", re.IGNORECASE),
             IntentType.MEMORY_RECALL, {}),

            (re.compile(r"recall\s+(?:about\s+)?(.+)", re.IGNORECASE),
             IntentType.MEMORY_RECALL, {}),

            (re.compile(r"forget\s+(?:that\s+)?(.+)", re.IGNORECASE),
             IntentType.MEMORY_FORGET, {}),

            (re.compile(r"search\s+(?:for\s+)?(.+)", re.IGNORECASE),
             IntentType.SEARCH, {}),
        ]

    def _extract_task_id(self, text: str, pattern: re.Pattern) -> Optional[int]:
        match = pattern.search(text)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return None
        return None

    def route(self, text: str) -> IntentResult:
        cleaned = text.strip().lower()
        result = IntentResult(intent=IntentType.UNKNOWN, confidence=0.0, raw_text=text)

        for phrase, intent, params in self._exact_phrases:
            if cleaned == phrase or cleaned.startswith(phrase):
                return IntentResult(
                    intent=intent,
                    confidence=1.0,
                    params=params,
                    raw_text=text,
                )

        for pattern, intent, params in self._exact_patterns:
            match = pattern.search(text)
            if match:
                extracted_params = dict(params)
                if params.get("title_extract"):
                    extracted_params["title"] = match.group(1).strip()
                else:
                    groups = match.groups()
                    if groups:
                        extracted_params["value"] = groups[0].strip()
                return IntentResult(
                    intent=intent,
                    confidence=0.95,
                    params=extracted_params,
                    raw_text=text,
                )

        best_intent = IntentType.UNKNOWN
        best_score = 0
        best_params: dict = {}

        for phrase, intent, params in self._exact_phrases:
            score = fuzz.partial_ratio(cleaned, phrase)
            if score > best_score:
                best_score = score
                best_intent = intent
                best_params = dict(params)

        if best_score >= self.fuzzy_threshold:
            return IntentResult(
                intent=best_intent,
                confidence=best_score / 100.0,
                params=best_params,
                raw_text=text,
            )

        return result


router = IntentRouter()
