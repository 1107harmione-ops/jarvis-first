"""
Codex (OpenAI GPT-4o) integration — coding agent model.
Specialized for code generation, review, and analysis.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class CodexLLM:
    """OpenAI-compatible coding model client."""

    def __init__(self) -> None:
        self.api_key = settings.CODEX_API_KEY
        self.base_url = settings.CODEX_BASE_URL.rstrip("/")
        self.model = settings.CODEX_MODEL
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
        temperature: float = 0.2,
        max_tokens: int = 8192,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Send a chat request with low temperature for code."""
        client = await self._client()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> AsyncGenerator[str, None]:
        """Stream code generation tokens."""
        client = await self._client()
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with client.stream("POST", "/chat/completions", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if data_str:
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        context: str | None = None,
    ) -> str:
        """Generate code from a natural language prompt."""
        system = f"""You are an expert {language} developer. Generate clean, production-ready code.
Follow best practices, add comments, handle errors.
Return ONLY the code, no explanations unless asked."""
        if context:
            system += f"\n\nContext:\n{context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        response = await self.chat(messages)
        return response["choices"][0]["message"]["content"]

    async def review_code(
        self,
        code: str,
        language: str = "python",
    ) -> dict[str, Any]:
        """Review code for bugs, security issues, and improvements."""
        messages = [
            {
                "role": "system",
                "content": f"""Review this {language} code for:
1. Bugs and logic errors
2. Security vulnerabilities
3. Performance issues
4. Code style and best practices
5. Specific suggestions for improvement

Respond in JSON format:
{{
    "bugs": [{{"line": int, "severity": "high|medium|low", "description": "..."}}],
    "vulnerabilities": [{{"description": "...", "severity": "..."}}],
    "suggestions": ["..."],
    "overall_score": 0-10,
    "summary": "..."
}}""",
            },
            {"role": "user", "content": code},
        ]
        return await self._extract_json(messages)

    async def detect_bugs(self, code: str, language: str = "python") -> list[dict[str, Any]]:
        """Detect potential bugs in code."""
        messages = [
            {
                "role": "system",
                "content": f"Find bugs in this {language} code. Return JSON array of bugs with line, description, severity.",
            },
            {"role": "user", "content": code},
        ]
        result = await self._extract_json(messages)
        if isinstance(result, dict) and "error" not in result:
            return result.get("bugs", [])
        return []

    async def _extract_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Helper to extract JSON from response."""
        response = await self.chat(messages, temperature=0.1)
        content = response["choices"][0]["message"]["content"]
        try:
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            result = json.loads(json_str)
            return result if isinstance(result, dict) else {"data": result}
        except (json.JSONDecodeError, IndexError) as exc:
            logger.error("Codex JSON extraction failed", extra={"error": str(exc)})
            return {"error": "Failed to parse JSON", "raw": content[:500]}

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None


codex = CodexLLM()
