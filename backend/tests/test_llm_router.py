"""
Tests for the LLM router module — category classification, model selection, cost estimation.
"""

from __future__ import annotations

from backend.llm.router import LLMRouter, TaskCategory


class TestLLMRouter:
    """Tests for LLM router."""

    def setup_method(self) -> None:
        self.router = LLMRouter()

    def test_categorize_code_request(self) -> None:
        category = self.router.categorize_request("Write a Python function to sort a list")
        assert category == TaskCategory.CODE

    def test_categorize_research_request(self) -> None:
        category = self.router.categorize_request("Research the latest AI trends in healthcare")
        assert category == TaskCategory.RESEARCH

    def test_categorize_vision_request_with_attachment(self) -> None:
        category = self.router.categorize_request(
            "What's in this image?",
            attachments=["http://example.com/photo.jpg"],
        )
        assert category == TaskCategory.VISION

    def test_categorize_memory_request(self) -> None:
        category = self.router.categorize_request("Remember that I like coffee")
        assert category == TaskCategory.MEMORY

    def test_categorize_task_request(self) -> None:
        category = self.router.categorize_request("Create a task to buy groceries tomorrow")
        assert category == TaskCategory.TASK

    def test_categorize_plan_request(self) -> None:
        category = self.router.categorize_request("Plan the architecture for a microservice app")
        assert category == TaskCategory.PLAN

    def test_categorize_general_chat(self) -> None:
        category = self.router.categorize_request("Hello, how are you?")
        assert category == TaskCategory.CHAT

    def test_select_model_code(self) -> None:
        model = self.router.select_model(TaskCategory.CODE)
        assert "codex" in model or "gpt-4" in model

    def test_select_model_research(self) -> None:
        model = self.router.select_model(TaskCategory.RESEARCH)
        assert "minimax" in model

    def test_select_model_vision(self) -> None:
        model = self.router.select_model(TaskCategory.VISION)
        assert "mimo" in model

    def test_select_model_chat(self) -> None:
        model = self.router.select_model(TaskCategory.CHAT)
        assert "deepseek" in model

    def test_select_model_default(self) -> None:
        model = self.router.select_model(TaskCategory.SUMMARIZE)
        assert "deepseek" in model

    def test_estimate_cost_code(self) -> None:
        cost = self.router.estimate_cost(TaskCategory.CODE, input_tokens=500, output_tokens=200)
        assert cost["model"].startswith("codex")
        assert cost["input_cost"] > 0
        assert cost["output_cost"] > 0
        assert cost["total_cost"] == cost["input_cost"] + cost["output_cost"]

    def test_route_to_deepseek(self) -> None:
        """route() should return a provider name for chat category."""
        provider = self.router.route("Tell me about AI", TaskCategory.CHAT)
        assert provider in ("deepseek", "codex", "minimax", "mimo")

    def test_route_to_codex(self) -> None:
        provider = self.router.route("Write code", TaskCategory.CODE)
        assert provider == "codex"

    def test_categorize_keyword_priority(self) -> None:
        """Memory keyword should win over general in mixed message."""
        category = self.router.categorize_request("I remember you said we should plan a trip")
        assert category == TaskCategory.MEMORY

    def test_categorize_image_keyword_triggers_vision(self) -> None:
        """Vision keywords without attachments may still classify as vision."""
        category = self.router.categorize_request("Analyze this image I'm describing")
        assert category in (TaskCategory.VISION, TaskCategory.CHAT)
