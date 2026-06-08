"""Intent router unit tests."""

from __future__ import annotations

from app.voice.router import IntentType, intent_router


class TestIntentRouter:
    """Test suite for the intent router."""

    def test_exact_match(self):
        result = intent_router.route("show my tasks")
        assert result.type == IntentType.TASK_LIST
        assert result.confidence == 1.0

    def test_exact_match_pending(self):
        result = intent_router.route("show my pending tasks")
        assert result.type == IntentType.TASK_LIST

    def test_regex_create_task(self):
        result = intent_router.route("create a task to learn FastAPI")
        assert result.type == IntentType.TASK_CREATE
        assert result.confidence == 0.95
        assert "title" in result.entities

    def test_regex_create_simple(self):
        result = intent_router.route("add task buy groceries")
        assert result.type == IntentType.TASK_CREATE
        assert "groceries" in result.entities.get("title", "")

    def test_regex_complete_task(self):
        result = intent_router.route("complete my task learn FastAPI")
        assert result.type == IntentType.TASK_COMPLETE
        assert result.confidence == 0.95

    def test_regex_delete_task(self):
        result = intent_router.route("delete my task test note")
        assert result.type == IntentType.TASK_DELETE

    def test_regex_search(self):
        result = intent_router.route("search my tasks about FastAPI")
        assert result.type == IntentType.TASK_SEARCH

    def test_unknown_intent(self):
        result = intent_router.route("what is the weather today")
        assert result.type == IntentType.UNKNOWN
        assert not result.is_known()

    def test_case_insensitive(self):
        result = intent_router.route("SHOW MY TASKS")
        assert result.type == IntentType.TASK_LIST

    def test_empty_string(self):
        result = intent_router.route("")
        assert result.type == IntentType.UNKNOWN
