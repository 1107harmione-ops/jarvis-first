"""
DeepSeek LLM integration — main assistant model.
OpenAI-compatible API, supports streaming, function calling.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekLLM:
    """DeepSeek Chat API client.

    Used as the primary assistant and router agent model.
    """

    def __init__(self) -> None:
        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL.rstrip("/")
        self.model = settings.DEEPSEEK_MODEL
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=60.0,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._http

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature (0-2).
            max_tokens: Maximum tokens in response.
            stream: Enable streaming response.
            tools: Optional function calling tools.

        Returns:
            Full API response dict.
        """
        client = await self._client()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()
        return response.json()

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion tokens.

        Yields:
            Content delta strings as they arrive.
        """
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

    async def extract_json(
        self,
        system_prompt: str,
        user_message: str,
    ) -> dict[str, Any]:
        """Extract structured JSON from a response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        response = await self.chat(messages, temperature=0.1, max_tokens=2000)
        content = response["choices"][0]["message"]["content"]

        # Try to parse JSON from the response
        try:
            # Find JSON block if wrapped in ```json ... ```
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError, KeyError) as exc:
            logger.error("JSON extraction failed", extra={"error": str(exc), "content": content[:200]})
            return {"error": "Failed to parse structured output", "raw": content}

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None


deepseek = DeepSeekLLM()
