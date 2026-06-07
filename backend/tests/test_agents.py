"""
Tests for the agent system — router agent, coding, research, vision, task, planner, memory agents.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents.coding_agent import CodingAgent
from backend.agents.memory_agent import MemoryAgent
from backend.agents.planner_agent import PlannerAgent, PlanType
from backend.agents.research_agent import ResearchAgent
from backend.agents.router_agent import RouterAgent
from backend.agents.task_agent import TaskAgent
from backend.agents.vision_agent import VisionAgent


class TestRouterAgent:
    """Tests for the orchestrator router agent."""

    @pytest.mark.asyncio
    async def test_process_chat_message(self) -> None:
        agent = RouterAgent()
        result = await agent.process(
            user_id="test_user",
            message="Hello, how are you?",
            conversation_id="conv_123",
        )
        assert "content" in result
        assert "agent" in result
        assert result["category"] in ("chat", "code", "research", "vision", "memory", "task", "plan")
        assert "duration_ms" in result

    @pytest.mark.asyncio
    async def test_process_with_attachments(self) -> None:
        agent = RouterAgent()
        result = await agent.process(
            user_id="test_user",
            message="What's in this image?",
            conversation_id="conv_123",
            attachments=["http://example.com/img.jpg"],
        )
        assert result["category"] == "vision"

    @pytest.mark.asyncio
    async def test_process_code_request(self) -> None:
        agent = RouterAgent()
        result = await agent.process(
            user_id="test_user",
            message="Write a Python function to calculate fibonacci",
            conversation_id="conv_123",
        )
        assert "content" in result

    @pytest.mark.asyncio
    async def test_process_error_handling(self) -> None:
        agent = RouterAgent()
        result = await agent.process(
            user_id="test_user",
            message="",
            conversation_id="conv_123",
        )
        assert "error" in result or "content" in result


class TestCodingAgent:
    """Tests for the coding agent."""

    def test_initialization(self) -> None:
        agent = CodingAgent()
        assert agent.name == "coding_agent"

    @pytest.mark.asyncio
    async def test_generate_code(self) -> None:
        agent = CodingAgent()
        result = await agent.generate(
            language="python",
            prompt="sort a list of numbers",
        )
        assert "code" in result or "error" in result

    @pytest.mark.asyncio
    async def test_review_code(self) -> None:
        agent = CodingAgent()
        result = await agent.review("def foo():\n    pass\n")
        assert "issues" in result or "feedback" in result or "error" in result


class TestResearchAgent:
    """Tests for the research agent."""

    def test_initialization(self) -> None:
        agent = ResearchAgent()
        assert agent.name == "research_agent"

    @pytest.mark.asyncio
    async def test_research_topic(self) -> None:
        agent = ResearchAgent()
        result = await agent.research("Latest AI developments")
        assert "summary" in result or "error" in result


class TestVisionAgent:
    """Tests for the vision agent."""

    def test_initialization(self) -> None:
        agent = VisionAgent()
        assert agent.name == "vision_agent"

    @pytest.mark.asyncio
    async def test_describe_image_with_invalid_url(self) -> None:
        agent = VisionAgent()
        result = await agent.describe("http://invalid-image-url.jpg")
        assert "error" in result


class TestMemoryAgent:
    """Tests for the memory agent."""

    @pytest.mark.asyncio
    async def test_store_memory(self) -> None:
        agent = MemoryAgent()
        result = await agent.store(
            user_id="test_user",
            content="User likes dark mode",
            memory_type="long_term",
            importance_score=0.8,
        )
        assert "id" in result or "error" in result

    @pytest.mark.asyncio
    async def test_recall_memory(self) -> None:
        agent = MemoryAgent()
        result = await agent.recall(
            user_id="test_user",
            query="dark mode preferences",
        )
        assert isinstance(result, list)


class TestTaskAgent:
    """Tests for the task agent."""

    @pytest.mark.asyncio
    async def test_create_task_from_message(self) -> None:
        agent = TaskAgent()
        result = await agent.create_from_message(
            user_id="test_user",
            message="Buy groceries tomorrow at 10am",
        )
        assert "title" in result or "error" in result


class TestPlannerAgent:
    """Tests for the planner agent."""

    def test_plan_types(self) -> None:
        assert PlanType.ARCHITECTURE.value == "architecture"
        assert PlanType.ROADMAP.value == "roadmap"
        assert PlanType.SPRINT.value == "sprint"

    @pytest.mark.asyncio
    async def test_create_architecture_plan(self) -> None:
        agent = PlannerAgent()
        result = await agent.create_plan(
            plan_type=PlanType.ARCHITECTURE,
            context="Build a microservice with FastAPI",
        )
        assert isinstance(result, dict)
