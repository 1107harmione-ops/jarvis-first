"""
Tests for the v2 Multi-Agent System (LangGraph-based).

Covers:
- State creation
- Agent registry
- Base agent lifecycle
- Individual agent implementations
- LangGraph workflow
- Error recovery
- Monitoring
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.agents_v2.state import (
    AgentState,
    ExecutionStatus,
    WorkflowType,
    create_initial_state,
)
from backend.agents_v2.registry import AgentRegistry, get_agent_registry
from backend.agents_v2.base import BaseAgent
from backend.agents_v2.tools import AgentTools
from backend.agents_v2.monitor import AgentMonitor, get_agent_monitor


# ══════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════


@pytest.fixture
def sample_state() -> AgentState:
    """Create a standard test state."""
    return create_initial_state(
        user_id="test_user",
        message="What is quantum computing?",
        conversation_id="conv_123",
    )


@pytest.fixture
def clean_registry():
    """Get a clean registry for testing."""
    registry = get_agent_registry()
    registry.reset()
    yield registry
    registry.reset()


@pytest.fixture
def clean_monitor():
    """Get a clean monitor for testing."""
    monitor = get_agent_monitor()
    monitor.clear()
    yield monitor
    monitor.clear()


# ══════════════════════════════════════════════
# State Tests
# ══════════════════════════════════════════════


class TestAgentState:
    """AgentState creation and defaults."""

    def test_create_initial_state_defaults(self):
        """Verify all default fields are set correctly."""
        state = create_initial_state(user_id="u1", message="hello")

        assert state["user_id"] == "u1"
        assert state["message"] == "hello"
        assert state["conversation_id"] is None
        assert state["attachments"] == []
        assert state["metadata"] == {}
        assert state["category"] is None
        assert state["confidence"] == 0.0
        assert state["detected_intent"] is None
        assert state["selected_agents"] == []
        assert state["plan"] is None
        assert state["current_step_index"] == 0
        assert state["execution_order"] == []
        assert state["results"] == {}
        assert state["step_results"] == {}
        assert state["shared_context"] == {}
        assert state["memory_context"] == []
        assert state["errors"] == []
        assert state["retry_count"] == 0
        assert state["max_retries"] == 3
        assert state["fallback_activated"] is False
        assert state["final_response"] is None
        assert state["response_agent"] is None
        assert state["session_id"] is not None
        assert state["start_time"] == 0.0
        assert state["end_time"] is None
        assert state["total_tokens"] == 0
        assert state["total_latency_ms"] == 0.0
        assert state["graph_execution_path"] == []

    def test_create_state_with_optional_params(self):
        """Verify optional parameters are passed through."""
        state = create_initial_state(
            user_id="u1",
            message="hello",
            conversation_id="conv_1",
            attachments=[{"type": "image", "url": "https://example.com/img.jpg"}],
            metadata={"source": "test"},
            max_retries=5,
        )

        assert state["conversation_id"] == "conv_1"
        assert len(state["attachments"]) == 1
        assert state["attachments"][0]["type"] == "image"
        assert state["metadata"]["source"] == "test"
        assert state["max_retries"] == 5

    def test_state_is_mutable_dict(self):
        """Verify AgentState behaves as a mutable dictionary."""
        state = create_initial_state(user_id="u1", message="hi")
        state["category"] = "RESEARCH"
        state["confidence"] = 0.95
        state["results"]["research"] = {
            "agent_name": "research",
            "status": ExecutionStatus.SUCCESS,
            "output": "Research results",
            "error": None,
            "tokens_used": 100,
            "latency_ms": 500.0,
            "metadata": None,
        }
        state["final_response"] = "Here is your answer"

        assert state["category"] == "RESEARCH"
        assert state["results"]["research"]["status"] == ExecutionStatus.SUCCESS
        assert state["final_response"] == "Here is your answer"


# ══════════════════════════════════════════════
# Registry Tests
# ══════════════════════════════════════════════


class TestAgentRegistry:
    """Agent registry operations."""

    def test_registry_singleton(self):
        """Verify AgentRegistry is a singleton."""
        r1 = AgentRegistry()
        r2 = AgentRegistry()
        assert r1 is r2

    def test_register_and_get(self, clean_registry):
        """Verify agent registration and retrieval."""
        agent = MagicMock(spec=BaseAgent)
        agent.name = "test_agent"
        agent.model_name = "deepseek"
        agent.description = "Test agent"

        clean_registry.register(agent)
        assert clean_registry.is_registered("test_agent")
        assert clean_registry.get("test_agent") is agent
        assert clean_registry.count == 1

    def test_get_nonexistent(self, clean_registry):
        """Verify get returns None for unregistered agents."""
        assert clean_registry.get("nonexistent") is None

    def test_get_all(self, clean_registry):
        """Verify get_all returns all agents."""
        agents = []
        for i in range(3):
            agent = MagicMock(spec=BaseAgent)
            agent.name = f"agent_{i}"
            agents.append(agent)
            clean_registry.register(agent)

        all_agents = clean_registry.get_all()
        assert len(all_agents) == 3
        for agent in agents:
            assert agent.name in all_agents

    def test_get_names(self, clean_registry):
        """Verify get_names returns correct list."""
        names = ["alpha", "beta", "gamma"]
        for name in names:
            agent = MagicMock(spec=BaseAgent)
            agent.name = name
            clean_registry.register(agent)

        assert sorted(clean_registry.get_names()) == sorted(names)

    def test_get_by_capability(self, clean_registry):
        """Verify capability-based agent lookup."""
        descriptions = {
            "coder": "Generates and reviews code using AI",
            "researcher": "Searches the web for information",
            "vision": "Analyzes images and performs OCR",
        }
        for name, desc in descriptions.items():
            agent = MagicMock(spec=BaseAgent)
            agent.name = name
            agent.description = desc
            clean_registry.register(agent)

        code_agents = clean_registry.get_by_capability("code")
        assert len(code_agents) == 1
        assert code_agents[0].name == "coder"

        # Case insensitive
        web_agents = clean_registry.get_by_capability("WEB")
        assert len(web_agents) == 1
        assert web_agents[0].name == "researcher"

    def test_reset(self, clean_registry):
        """Verify reset clears all agents."""
        agent = MagicMock(spec=BaseAgent)
        agent.name = "test"
        clean_registry.register(agent)
        assert clean_registry.count == 1

        clean_registry.reset()
        assert clean_registry.count == 0


# ══════════════════════════════════════════════
# Base Agent Tests
# ══════════════════════════════════════════════


class TestBaseAgent:
    """BaseAgent lifecycle and utilities."""

    @pytest.mark.asyncio
    async def test_safe_process_success(self):
        """Verify safe_process records success correctly."""
        class TestAgent(BaseAgent):
            async def process(self, state):
                state["final_response"] = "Test response"
                return state

        agent = TestAgent(
            name="test",
            model_name="deepseek",
            system_prompt="You are a test agent.",
            description="Test agent for unit tests",
        )

        state = create_initial_state(user_id="u1", message="test")
        result = await agent.safe_process(state)

        assert result["results"]["test"]["status"] == ExecutionStatus.SUCCESS
        assert result["results"]["test"]["output"] == "Test response"
        assert result["results"]["test"]["latency_ms"] > 0
        assert result["total_latency_ms"] > 0
        assert "test" in result["graph_execution_path"]

    @pytest.mark.asyncio
    async def test_safe_process_failure(self):
        """Verify safe_process records failure correctly."""
        class FailingAgent(BaseAgent):
            async def process(self, state):
                raise ValueError("Something went wrong")

        agent = FailingAgent(
            name="failing",
            model_name="deepseek",
            system_prompt="",
            description="Agent that fails",
        )

        state = create_initial_state(user_id="u1", message="test")
        result = await agent.safe_process(state)

        assert result["results"]["failing"]["status"] == ExecutionStatus.FAILED
        assert result["results"]["failing"]["output"] is None
        assert "Something went wrong" in result["results"]["failing"]["error"]
        assert len(result["errors"]) == 1
        assert result["errors"][0]["agent"] == "failing"

    def test_build_system_messages(self):
        """Verify message construction works."""
        class ChatAgent(BaseAgent):
            async def process(self, state):
                return state

        agent = ChatAgent(
            name="chat",
            model_name="deepseek",
            system_prompt="System prompt",
            description="Chat agent",
        )

        messages = agent._build_system_messages("User message", context="Memory context")
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "system"
        assert "Memory context" in messages[1]["content"]
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "User message"

    def test_build_system_messages_no_context(self):
        """Verify messages work without context."""
        class ChatAgent(BaseAgent):
            async def process(self, state):
                return state

        agent = ChatAgent(
            name="chat",
            model_name="deepseek",
            system_prompt="System prompt",
            description="Chat agent",
        )

        messages = agent._build_system_messages("User message")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


# ══════════════════════════════════════════════
# Tools Tests
# ══════════════════════════════════════════════


class TestAgentTools:
    """AgentTools utility functions."""

    def test_calculate_basic(self):
        assert AgentTools.calculate("2 + 2") == "Result: 4"

    def test_calculate_complex(self):
        result = AgentTools.calculate("(3 + 5) * 2")
        assert result == "Result: 16"

    def test_calculate_division_by_zero(self):
        assert "division by zero" in AgentTools.calculate("1 / 0")

    def test_calculate_invalid_input(self):
        assert "disallowed characters" in AgentTools.calculate("import os")

    def test_extract_code_blocks(self):
        text = 'Some text\n```python\nprint("hello")\n```\nmore text'
        blocks = AgentTools.extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"
        assert 'print("hello")' in blocks[0]["code"]

    def test_extract_code_blocks_no_language(self):
        text = '```\nplain code\n```'
        blocks = AgentTools.extract_code_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["language"] == "unknown"

    def test_extract_urls(self):
        text = "Visit https://example.com and http://test.org/path"
        urls = AgentTools.extract_urls(text)
        assert len(urls) == 2
        assert "https://example.com" in urls

    def test_truncate_short(self):
        text = "Short text"
        assert AgentTools.truncate(text, max_length=100) == text

    def test_truncate_long(self):
        text = "A" * 1000
        result = AgentTools.truncate(text, max_length=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_count_tokens(self):
        text = "Hello world"  # 11 chars
        assert AgentTools.count_tokens(text) == 2  # 11 // 4 = 2

    def test_split_markdown_sections(self):
        text = "# Intro\nHello\n# Details\nMore info"
        sections = AgentTools.split_markdown_sections(text)
        assert "Intro" in sections
        assert sections["Intro"] == "Hello"
        assert "Details" in sections
        assert sections["Details"] == "More info"


# ══════════════════════════════════════════════
# Monitor Tests
# ══════════════════════════════════════════════


class TestAgentMonitor:
    """AgentMonitor metrics collection."""

    def test_monitor_singleton(self, clean_monitor):
        """Verify monitor is singleton."""
        m1 = get_agent_monitor()
        m2 = get_agent_monitor()
        assert m1 is m2

    def test_start_and_end_session(self, clean_monitor):
        """Verify session lifecycle tracking."""
        state = create_initial_state(user_id="u1", message="test")
        state["category"] = "RESEARCH"

        clean_monitor.start_session(state)
        state["end_time"] = time.time() + 0.1
        state["results"]["research"] = {
            "agent_name": "research",
            "status": ExecutionStatus.SUCCESS,
            "output": "done",
            "error": None,
            "tokens_used": 50,
            "latency_ms": 100.0,
            "metadata": None,
        }

        metrics = clean_monitor.end_session(state)
        assert metrics.session_id == state["session_id"]
        assert metrics.category == "RESEARCH"
        assert metrics.success is True
        assert metrics.agent_count == 1
        assert metrics.error_count == 0

    def test_session_with_errors(self, clean_monitor):
        """Verify session tracking with errors."""
        state = create_initial_state(user_id="u1", message="test")
        state["errors"].append({
            "step_id": "step_1",
            "agent": "coding",
            "error": "Syntax error",
            "retry_count": 1,
            "timestamp": time.time(),
        })

        clean_monitor.start_session(state)
        state["end_time"] = time.time()
        metrics = clean_monitor.end_session(state)

        assert metrics.success is False
        assert metrics.error_count == 1

    def test_agent_summary(self, clean_monitor):
        """Verify per-agent aggregate metrics."""
        # Simulate multiple sessions for different agents
        for agent_name in ["coding", "research", "coding"]:
            state = create_initial_state(user_id="u1", message="test")
            state["category"] = agent_name.upper()

            success = agent_name != "research"  # Make one fail
            state["results"][agent_name] = {
                "agent_name": agent_name,
                "status": ExecutionStatus.SUCCESS if success else ExecutionStatus.FAILED,
                "output": "done" if success else None,
                "error": None if success else "Failure",
                "tokens_used": 100,
                "latency_ms": 200.0,
                "metadata": None,
            }
            if not success:
                state["errors"].append({"agent": agent_name, "error": "Failure"})

            clean_monitor.start_session(state)
            state["end_time"] = time.time()
            clean_monitor.end_session(state)

        summary = clean_monitor.get_agent_summary()

        assert "coding" in summary
        assert summary["coding"].total_calls == 2
        assert summary["coding"].successful_calls == 2
        assert summary["coding"].failed_calls == 0
        assert summary["coding"].success_rate == 100.0

        assert "research" in summary
        assert summary["research"].total_calls == 1
        assert summary["research"].failed_calls == 1
        assert summary["research"].success_rate == 0.0

    def test_global_stats(self, clean_monitor):
        """Verify global statistics computation."""
        # No sessions
        stats = clean_monitor.get_global_stats()
        assert stats["total_sessions"] == 0

        # Add a session
        state = create_initial_state(user_id="u1", message="test")
        state["category"] = "TEST"
        clean_monitor.start_session(state)
        state["end_time"] = time.time() + 0.5
        clean_monitor.end_session(state)

        stats = clean_monitor.get_global_stats()
        assert stats["total_sessions"] == 1
        assert stats["successful_sessions"] == 1
        assert stats["average_duration_seconds"] > 0

    def test_get_session_metrics(self, clean_monitor):
        """Verify individual session lookup."""
        state = create_initial_state(user_id="u1", message="test")
        clean_monitor.start_session(state)
        state["end_time"] = time.time()
        clean_monitor.end_session(state)

        metrics = clean_monitor.get_session_metrics(state["session_id"])
        assert metrics is not None
        assert metrics.session_id == state["session_id"]

        # Non-existent session
        assert clean_monitor.get_session_metrics("nonexistent") is None

    def test_category_breakdown(self, clean_monitor):
        """Verify category counting."""
        categories = ["CODING", "RESEARCH", "CODING", "VISION"]
        for i, cat in enumerate(categories):
            state = create_initial_state(user_id="u1", message=f"test {i}")
            state["category"] = cat
            clean_monitor.start_session(state)
            state["end_time"] = time.time()
            clean_monitor.end_session(state)

        breakdown = clean_monitor.get_category_breakdown()
        assert breakdown.get("CODING") == 2
        assert breakdown.get("RESEARCH") == 1
        assert breakdown.get("VISION") == 1


# ══════════════════════════════════════════════
# Graph Tests (mock-based to avoid LangGraph dep)
# ══════════════════════════════════════════════


class TestGraphFallbackExecutor:
    """Tests for the fallback sequential executor."""

    @pytest.mark.asyncio
    async def test_fallback_executor_single_agent(self, clean_registry):
        """Verify fallback executor routes to a single agent correctly."""
        from backend.agents_v2.graph import AgentGraph

        # Register a mock agent
        agent = MagicMock(spec=BaseAgent)
        agent.name = "research"
        agent.safe_process = AsyncMock(return_value=None)
        clean_registry.register(agent)

        graph = AgentGraph()
        state = create_initial_state(user_id="u1", message="test")
        state["selected_agents"] = ["research"]
        state["category"] = "RESEARCH"

        # Patch safe_process to return updated state
        async def mock_process(s):
            s["final_response"] = "Research done"
            return s

        agent.safe_process = mock_process

        result = await graph._fallback_execute(state)
        assert result["final_response"] is not None

    @pytest.mark.asyncio
    async def test_fallback_executor_with_plan(self, clean_registry):
        """Verify fallback executor handles multi-step plans."""
        from backend.agents_v2.graph import AgentGraph
        from backend.agents_v2.state import AgentPlanStep

        # Register mock agents
        for name in ["research", "coding"]:
            agent = MagicMock(spec=BaseAgent)
            agent.name = name

            async def make_mock(n):
                async def mock_process(s):
                    s["final_response"] = f"{n} done"
                    return s
                return mock_process

            agent.safe_process = await make_mock(name)
            clean_registry.register(agent)

        graph = AgentGraph()
        state = create_initial_state(user_id="u1", message="test")
        state["selected_agents"] = ["research", "coding"]
        state["category"] = "COMPLEX"
        state["plan"] = {
            "goal": "Build a weather app",
            "workflow_type": "sequential",
            "steps": [
                {
                    "step_id": "step_1",
                    "agent": "research",
                    "input": "Research weather API",
                    "depends_on": [],
                    "expected_output": "API docs",
                    "max_retries": 2,
                    "timeout_seconds": 30,
                    "parallel_group": None,
                },
                {
                    "step_id": "step_2",
                    "agent": "coding",
                    "input": "Write weather app code",
                    "depends_on": ["step_1"],
                    "expected_output": "Working code",
                    "max_retries": 2,
                    "timeout_seconds": 60,
                    "parallel_group": None,
                },
            ],
            "parallel_groups": {},
        }

        result = await graph._fallback_execute(state)
        assert result["current_step_index"] == 2


# ══════════════════════════════════════════════
# Integration sanity
# ══════════════════════════════════════════════


class TestImports:
    """Verify all modules can be imported."""

    def test_import_state(self):
        from backend.agents_v2.state import AgentState, ExecutionStatus, WorkflowType, create_initial_state
        assert AgentState is not None

    def test_import_base(self):
        from backend.agents_v2.base import BaseAgent
        assert BaseAgent is not None

    def test_import_registry(self):
        from backend.agents_v2.registry import AgentRegistry, get_agent_registry
        assert AgentRegistry is not None

    def test_import_tools(self):
        from backend.agents_v2.tools import AgentTools
        assert AgentTools is not None

    def test_import_graph(self):
        from backend.agents_v2.graph import AgentGraph, create_agent_graph
        assert AgentGraph is not None

    def test_import_monitor(self):
        from backend.agents_v2.monitor import AgentMonitor, get_agent_monitor
        assert AgentMonitor is not None

    def test_import_init(self):
        from backend.agents_v2.init import initialize_agent_system, shutdown_agent_system
        assert initialize_agent_system is not None
