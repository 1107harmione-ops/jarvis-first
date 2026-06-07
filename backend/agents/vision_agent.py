"""
Vision Agent — image analysis, OCR, visual QA, and screenshot analysis.
Powered by Mimo V2 Omni for multi-modal understanding.
"""

from __future__ import annotations

import time
from typing import Any

from backend.database.mongodb import mongodb
from backend.database.schemas import new_agent_log_doc
from backend.llm.mimo import mimo
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class VisionAgent:
    """Specialized agent for all vision-related tasks."""

    def __init__(self) -> None:
        self.name = "vision_agent"

    async def process(
        self,
        user_id: str,
        message: str,
        context: list[dict[str, str]] | None = None,
        attachments: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a vision/image request."""
        start = time.monotonic()
        session_id = session_id or f"vision_{int(start)}"

        # Extract image URL from attachments or message
        image_url = self._extract_image_url(message, attachments)
        if not image_url:
            return {
                "content": "I need an image to analyze. Please attach an image or provide an image URL.",
                "agent": self.name,
            }

        task_type = self._detect_task_type(message)
        logger.info(
            "Vision agent processing",
            extra={"task_type": task_type, "session_id": session_id},
        )

        try:
            if task_type == "ocr":
                result = await self._ocr(image_url, message)
            elif task_type == "describe":
                result = await self._describe(image_url, message)
            elif task_type == "qa":
                result = await self._qa(image_url, message)
            elif task_type == "screenshot":
                result = await self._analyze_screenshot(image_url, message)
            elif task_type == "objects":
                result = await self._detect_objects(image_url, message)
            else:
                result = await self._analyze(image_url, message)

            elapsed = (time.monotonic() - start) * 1000
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary=result.get("content", "")[:200],
                    duration_ms=elapsed,
                    status="success",
                )
            )
            return result

        except Exception as exc:
            elapsed = (time.monotonic() - start) * 1000
            logger.error("Vision agent failed", extra={"error": str(exc), "session_id": session_id})
            await mongodb.agent_logs.insert_one(
                new_agent_log_doc(
                    agent_name=self.name,
                    session_id=session_id,
                    user_id=user_id,
                    action=task_type,
                    input_summary=message[:200],
                    output_summary="",
                    duration_ms=elapsed,
                    status="error",
                    error=str(exc),
                )
            )
            return {"content": f"Image analysis failed: {str(exc)}", "agent": self.name, "error": str(exc)}

    async def _describe(self, image_url: str, prompt: str) -> dict[str, Any]:
        """Detailed image description."""
        description = await mimo.describe(image_url)
        return {
            "content": description,
            "agent": self.name,
            "metadata": {"task": "describe"},
        }

    async def _ocr(self, image_url: str, prompt: str) -> dict[str, Any]:
        """Extract text from image."""
        text = await mimo.ocr(image_url)
        if text.strip():
            content = f"**Extracted Text:**\n\n```\n{text}\n```"
        else:
            content = "No text was detected in this image."
        return {
            "content": content,
            "agent": self.name,
            "metadata": {"task": "ocr", "text_found": bool(text.strip())},
        }

    async def _qa(self, image_url: str, prompt: str) -> dict[str, Any]:
        """Answer a question about an image."""
        # Extract the question from the prompt
        question = prompt
        for prefix in ["answer", "question:", "?"]:
            if prefix in prompt.lower():
                break

        answer = await mimo.ask_question(image_url, question)
        return {
            "content": answer,
            "agent": self.name,
            "metadata": {"task": "visual_qa", "question": question},
        }

    async def _analyze_screenshot(self, image_url: str, prompt: str) -> dict[str, Any]:
        """Analyze a screenshot."""
        analysis = await mimo.analyze_screenshot(image_url)
        content = f"""## Screenshot Analysis

**Application:** {analysis.get('application', 'Unknown')}

**UI Elements:** {analysis.get('ui_elements', 'N/A')}

**Main Content:** {analysis.get('main_content', 'N/A')}

**Layout:** {analysis.get('layout', 'N/A')}

**Notes:** {analysis.get('notes', 'N/A')}"""
        return {
            "content": content,
            "agent": self.name,
            "metadata": {"task": "screenshot_analysis"},
        }

    async def _detect_objects(self, image_url: str, prompt: str) -> dict[str, Any]:
        """Detect objects in an image."""
        objects = await mimo.detect_objects(image_url)
        content = "## Detected Objects\n\n"
        for obj in objects:
            name = obj.get("name", "Unknown")
            position = obj.get("position", "unknown")
            color = obj.get("color", "unknown")
            content += f"- **{name}** — Position: {position}, Color: {color}\n"

        if not objects:
            content += "No objects were detected."
        return {
            "content": content,
            "agent": self.name,
            "metadata": {"task": "object_detection", "objects_count": len(objects)},
        }

    async def _analyze(self, image_url: str, prompt: str) -> dict[str, Any]:
        """General image analysis."""
        result = await mimo.analyze_image(image_url, prompt=prompt)
        return {
            "content": result.get("description", ""),
            "agent": self.name,
            "metadata": {"task": "general_analysis", "processing_time_ms": result.get("processing_time_ms")},
        }

    def _extract_image_url(
        self, message: str, attachments: list[str] | None
    ) -> str | None:
        """Extract image URL from attachments or message content."""
        # Check attachments first
        if attachments:
            for att in attachments:
                if any(ext in att.lower() for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
                    return att
                if att.startswith("data:image"):
                    return att

        # Check message for URLs
        import re
        urls = re.findall(r"https?://[^\s]+(?:png|jpg|jpeg|gif|webp)", message.lower())
        if urls:
            return urls[0]

        # Check for base64 in message
        if "base64," in message:
            import re as re2
            match = re2.search(r"data:image/[^;]+;base64,[^\s]+", message)
            if match:
                return match.group(0)

        return None

    def _detect_task_type(self, message: str) -> str:
        """Detect the vision task type."""
        msg_lower = message.lower()
        if any(kw in msg_lower for kw in ["ocr", "extract text", "read text", "what does this say"]):
            return "ocr"
        if any(kw in msg_lower for kw in ["describe", "what do you see", "tell me about", "caption"]):
            return "describe"
        if any(kw in msg_lower for kw in ["question", "answer", "?"]):
            return "qa"
        if any(kw in msg_lower for kw in ["screenshot", "screen shot", "ui", "interface"]):
            return "screenshot"
        if any(kw in msg_lower for kw in ["objects", "what objects", "detect", "find"]):
            return "objects"
        return "analyze"


# Global singleton
vision_agent = VisionAgent()
