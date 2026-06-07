"""tests/agents_v2/test_research_agent.py"""
import pytest
from backend.agents_v2.state import create_initial_state
from backend.agents_v2.research_agent import ResearchAgent

pytestmark = pytest.mark.asyncio


class TestResearchAgent:
    @pytest.fixture
    def agent(self):
        return ResearchAgent()

    @pytest.fixture
    def base_state(self):
        return create_initial_state(
            user_id="test_user",
            message="What is quantum computing?",
        )

    async def test_agent_name_and_model(self, agent):
        assert agent.name == "research"
        assert agent.model_name == "minimax"

    async def test_detect_research_type_quick(self, agent):
        rtype = await agent.detect_research_type("What is the capital of France?")
        assert rtype == "quick"

    async def test_detect_research_type_deep(self, agent):
        rtype = await agent.detect_research_type(
            "I need a comprehensive deep research report"
        )
        assert rtype == "deep"

    async def test_detect_research_type_comparative(self, agent):
        rtype = await agent.detect_research_type(
            "Compare Python vs JavaScript"
        )
        assert rtype == "comparative"

    async def test_detect_research_type_defaults_to_quick(self, agent):
        rtype = await agent.detect_research_type("Hello there")
        assert rtype == "quick"

    async def test_determine_depth_quick_for_quick_type(self, agent):
        depth = agent._determine_depth("What is X?", "quick")
        assert depth == "quick"

    async def test_determine_depth_deep_for_long_messages(self, agent):
        depth = agent._determine_depth(
            "This is a very long message that has many words so it should be "
            "classified as deep research because the user clearly wants a "
            "comprehensive analysis of whatever topic they are asking about",
            "deep",
        )
        assert depth == "deep"

    async def test_determine_depth_moderate_for_medium_messages(self, agent):
        depth = agent._determine_depth(
            "This is a moderately long message with some detail",
            "deep",
        )
        assert depth == "moderate"

    async def test_extract_title_short_message(self, agent):
        title = agent._extract_title("What is Python?", "response text")
        assert title == "What is Python?"

    async def test_extract_title_long_message(self, agent):
        long_msg = "x" * 100
        title = agent._extract_title(long_msg, "response")
        assert len(title) <= 80
        assert title.endswith("...")

    async def test_process_returns_state(self, agent, base_state):
        """process() should always return the state with final_response set."""
        result = await agent.process(base_state)
        assert "final_response" in result
        assert result["response_agent"] == "research"

    async def test_format_deep_report(self, agent):
        result = {
            "title": "Test Report",
            "executive_summary": "Summary",
            "key_findings": ["Finding 1", "Finding 2"],
            "detailed_analysis": "Analysis text",
            "pros": ["Pro 1"],
            "cons": ["Con 1"],
            "recommendations": ["Rec 1"],
            "conclusions": "Conclusion",
            "fact_check": {
                "verified_claims": ["Claim 1"],
                "contradicted_claims": [],
                "unverifiable_claims": [],
                "overall_confidence": 0.9,
            },
            "sources": [
                {"title": "Source 1", "url": "https://example.com", "overall_score": 0.85},
            ],
        }
        report = agent._format_deep_report(result)
        assert "# Test Report" in report
        assert "Summary" in report
        assert "Finding 1" in report
        assert "Pro 1" in report
        assert "Con 1" in report
        assert "Rec 1" in report
        assert "Claim 1" in report
        assert "Source 1" in report
