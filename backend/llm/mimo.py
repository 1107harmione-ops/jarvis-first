"""
Mimo V2 Omni integration — vision agent model.
Multi-modal: image understanding, OCR, visual QA, detailed description.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class MimoLLM:
    """Mimo V2 Omni API client for vision tasks.

    Capabilities:
    - Image description and captioning
    - OCR (optical character recognition)
    - Visual question answering
    - Object detection (via description)
    - Screenshot analysis
    """

    def __init__(self) -> None:
        self.api_key = settings.MIMO_API_KEY
        self.base_url = settings.MIMO_BASE_URL.rstrip("/")
        self.model = settings.MIMO_MODEL
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

    async def analyze_image(
        self,
        image_url: str,
        prompt: str = "Describe this image in detail.",
        detail_level: str = "auto",
    ) -> dict[str, Any]:
        """Analyze an image with a text prompt.

        Args:
            image_url: URL or base64 data URI of the image.
            prompt: Instruction for what to analyze.
            detail_level: "low", "high", or "auto".

        Returns:
            Analysis result with description and metadata.
        """
        client = await self._client()
        start = time.monotonic()

        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt},
        ]

        # Handle both URL and base64 inputs
        if image_url.startswith("data:") or image_url.startswith("http"):
            content.append({"type": "image_url", "image_url": {"url": image_url, "detail": detail_level}})
        else:
            # Assume base64 string
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_url}", "detail": detail_level}})

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4096,
            "temperature": 0.3,
        }

        response = await client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        elapsed = (time.monotonic() - start) * 1000
        description = data["choices"][0]["message"]["content"]

        return {
            "description": description,
            "model": self.model,
            "processing_time_ms": round(elapsed, 1),
        }

    async def describe(self, image_url: str) -> str:
        """Get a detailed description of an image."""
        result = await self.analyze_image(
            image_url,
            prompt="Describe this image in great detail. Include colors, objects, people, text, setting, and any notable features.",
        )
        return result["description"]

    async def ocr(self, image_url: str) -> str:
        """Extract text from an image (OCR)."""
        result = await self.analyze_image(
            image_url,
            prompt="Extract all text visible in this image. Return only the extracted text, preserving formatting where possible.",
        )
        return result["description"]

    async def ask_question(self, image_url: str, question: str) -> str:
        """Ask a question about an image."""
        result = await self.analyze_image(image_url, prompt=question)
        return result["description"]

    async def detect_objects(self, image_url: str) -> list[dict[str, Any]]:
        """Detect objects in an image and return structured data."""
        result = await self.analyze_image(
            image_url,
            prompt="""List all objects you can see in this image. For each object, provide:
- name
- approximate position (e.g., "top-left", "center", "bottom-right")
- color
- estimated quantity if multiple

Return as JSON array of objects.""",
        )
        # Try to extract structured data from description
        description = result["description"]
        try:
            if "```json" in description:
                json_str = description.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in description:
                json_str = description.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            pass
        # Fallback: return description as a single item
        return [{"name": description, "position": "unknown", "confidence": "high"}]

    async def analyze_screenshot(self, image_url: str) -> dict[str, Any]:
        """Analyze a screenshot (UI/UX analysis)."""
        result = await self.analyze_image(
            image_url,
            prompt="""Analyze this screenshot in detail:
1. What application or website is shown?
2. What are the key UI elements? (buttons, menus, text fields)
3. What is the main content/action?
4. Any errors or notable states?
5. Layout structure description

Return as JSON with keys: application, ui_elements, main_content, notes, layout""",
        )
        description = result["description"]
        try:
            if "```json" in description:
                json_str = description.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_str)
                parsed["processing_time_ms"] = result["processing_time_ms"]
                return parsed
        except (json.JSONDecodeError, IndexError):
            pass
        return {
            "description": description,
            "processing_time_ms": result["processing_time_ms"],
        }

    async def close(self) -> None:
        if self._http:
            await self._http.aclose()
            self._http = None


mimo = MimoLLM()
