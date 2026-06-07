"""
Minimax M2.1 integration — research agent model.
Specialized for long-context analysis, web search, and report generation.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MinimaxLLM:
    """Minimax API client for research tasks.

    M2.1 model excels at:
    - Long-context understanding (up to 1M tokens)
    - Web search integration
    - Structured report generation
    - Multi-source synthesis
    """

    def __init__(self) -> None:
        self.api_key = settings.MINIMAX_API_KEY
        self.base_url = settings.MINIMAX_BASE_URL.rstrip("/")
        self.model = settings.MINIMAX_MODEL
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Send a research-oriented chat request."""
        client = await self._client()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()

    async def research(
        self,
        topic: str,
        context: str | None = None,
        depth: str = "moderate",
    ) -> dict[str, Any]:
        """Perform deep research on a topic.

        Returns a structured research report.
        """
        system = f"""You are a research analyst. Research the given topic thoroughly.
Depth level: {depth}

Return a structured report in JSON:
{{
    "title": "...",
    "executive_summary": "...",
    "key_findings": ["...", "..."],
    "detailed_analysis": "...",
    "sources": [{{"title": "...", "url": "...", "relevance": "..."}}],
    "conclusions": "...",
    "further_reading": ["..."]
}}"""

        user_message = f"Research topic: {topic}"
        if context:
            user_message += f"\n\nContext:\n{context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        return await self._extract_json(messages)

    async def summarize(
        self,
        text: str,
        max_length: int = 500,
        format: str = "paragraph",
    ) -> str:
        """Summarize a long text."""
        messages = [
            {
                "role": "system",
                "content": f"Summarize the following text in {format} format, max {max_length} words. Be concise and capture key points.",
            },
            {"role": "user", "content": text},
        ]
        response = await self.chat(messages, temperature=0.3, max_tokens=2000)
        return response["choices"][0]["message"]["content"]

    async def extract_knowledge(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """Extract structured knowledge from text."""
        messages = [
            {
                "role": "system",
                "content": """Extract knowledge from the text as JSON array:
[{"concept": "...", "description": "...", "category": "...", "related_concepts": ["..."]}]""",
            },
            {"role": "user", "content": text},
        ]
        result = await self._extract_json(messages)
        if isinstance(result, list):
            return result
        return result.get("concepts", [])

    async def web_search_query(
        self,
        query: str,
    ) -> str:
        """Generate optimized web search queries from a natural language question."""
        messages = [
            {
                "role": "system",
                "content": "Generate 3 optimized web search queries for the given question. Return as JSON array of strings.",
            },
            {"role": "user", "content": query},
        ]
        result = await self._extract_json(messages)
        if isinstance(result, list):
            return "\n".join(result)
        return query

    async def _extract_json(self, messages: list[dict[str, str]]) -> Any:
        """Extract JSON from response."""
        response = await self.chat(messages, temperature=0.1)
        content = response["choices"][0]["message"]["content"]
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as exc:
            logger.error("Minimax JSON extraction failed", extra={"error": str(exc)})
            return {"error": "Failed to parse JSON", "raw": content[:500]}

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None


minimax = MinimaxLLM()
