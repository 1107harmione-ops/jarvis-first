"""
Vision Agent
============
Analyses images, performs OCR, answers visual questions, analyses
screenshots, and detects objects using the Mimo V2 Omni multi-modal model.

The agent sources images from:
  1. ``state["attachments"]`` — a list of dicts with a ``url`` or ``path`` key.
  2. URLs embedded in the message text.
  3. Base64 data URIs embedded in the message text.

Task types:
  - **describe**:   Free-form image description with rich detail.
  - **ocr**:        Extract visible text (optical character recognition).
  - **qa**:         Answer a user question about the image.
  - **screenshot**: UI/UX analysis of a screenshot.
  - **objects**:    Structured object detection.
  - **analyze**:    General-purpose analysis with a custom prompt deduced from
                    the message.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional

from agents_v2.state import AgentState, ExecutionStatus
from agents_v2.base import BaseAgent
from agents_v2.registry import get_agent_registry
from agents_v2.tools import AgentTools
from llm.codex import codex
from llm.minimax import minimax
from llm.mimo import mimo
from llm.router import llm_router
from services.search_service import search_service
from database.mongodb import mongodb
from config.settings import settings


VISION_KEYWORDS: Dict[str, List[str]] = {
    "describe": [
        "describe", "describe this image", "what's in this image",
        "what do you see", "what you see", "caption",
        "explain this image", "tell me about this image",
        "what is shown",
    ],
    "ocr": [
        "ocr", "extract text", "read text", "what does this say",
        "text from image", "read this", "transcribe",
    ],
    "qa": [
        "what is", "who is", "how many", "where is", "can you see",
        "is there", "are there", "do you see", "question",
    ],
    "screenshot": [
        "screenshot", "screen capture", "ui", "ux", "interface",
        "app screen", "website screenshot", "dashboard",
    ],
    "objects": [
        "object", "detect", "what objects", "list everything",
        "find objects", "identify objects",
    ],
}


class VisionAgent(BaseAgent):
    """Agent that processes images via Mimo V2 Omni.

    Supports describing images, extracting text (OCR), answering questions
    about images, analysing screenshots, and detecting objects.

    Automatically registers itself with the global ``AgentRegistry`` so it
    can be discovered by the LangGraph router and planner.
    """

    def __init__(self) -> None:
        super().__init__(
            name="vision",
            model_name="mimo",
            system_prompt=(
                "You are JARVIS's Vision Agent. You analyse images, extract "
                "text, detect objects, and answer visual questions using the "
                "Mimo V2 Omni multi-modal model.\n\n"
                "Rules:\n"
                "1. Be precise and descriptive when analysing images.\n"
                "2. For OCR, preserve the original text formatting.\n"
                "3. For screenshots, identify UI elements and layout.\n"
                "4. If the image quality is poor, note limitations.\n"
                "5. Store analysis results for other agents to reuse."
            ),
            description=(
                "Analyzes images, performs OCR, and answers visual questions "
                "using Mimo"
            ),
        )
        get_agent_registry().register(self)

    # ── Public API ──────────────────────────────────────────────────────

    async def process(self, state: AgentState) -> AgentState:
        """Execute the vision agent's logic on *state*.

        Steps:
        1. Read ``message``, ``attachments``, and ``shared_context``.
        2. Extract an image URL from attachments or message text.
        3. Detect the vision task type.
        4. Call the appropriate Mimo method.
        5. Update ``final_response`` and ``shared_context``.
        6. Log execution.
        """
        message: str = state.get("message", "") or ""
        attachments: List[Dict[str, Any]] = state.get("attachments", []) or []
        context: Dict[str, Any] = state.get("shared_context", {})
        await self._load_memory_context(state)

        # Extract image URL — fail early if none found.
        image_url = self._extract_image_url(state)
        if not image_url:
            state["final_response"] = (
                "I need an image to work with. Please attach an image or "
                "provide a URL to one."
            )
            state["response_agent"] = self.name
            return state

        vision_task = self._detect_vision_task(message)
        state.setdefault("shared_context", {})
        state["shared_context"]["vision_task_type"] = vision_task
        state["shared_context"]["vision_image_url"] = image_url

        try:
            final: str = ""

            if vision_task == "describe":
                result = await mimo.describe(image_url)
                final = f"# Image Description\n\n{result}"

            elif vision_task == "ocr":
                text = await mimo.ocr(image_url)
                extracted = text.strip() or "No text could be extracted from this image."
                final = f"# OCR Result\n\n{extracted}"

            elif vision_task == "qa":
                question = self._extract_question(message)
                answer = await mimo.ask_question(image_url, question)
                final = f"# Visual Q&A\n\n**Question:** {question}\n\n**Answer:** {answer}"

            elif vision_task == "screenshot":
                analysis = await mimo.analyze_screenshot(image_url)
                final = self._format_screenshot_analysis(analysis)

            elif vision_task == "objects":
                objects = await mimo.detect_objects(image_url)
                final = self._format_objects(objects)

            elif vision_task == "analyze":
                prompt = self._build_analysis_prompt(message)
                result = await mimo.analyze_image(image_url, prompt=prompt)
                final = f"# Image Analysis\n\n{result['description']}"

            else:
                # Fallback: describe the image.
                result = await mimo.describe(image_url)
                final = f"# Image Description\n\n{result}"

            state["final_response"] = final
            state["response_agent"] = self.name
            state["shared_context"]["vision_result"] = final

            await self._store_agent_log(
                state,
                action=f"vision_{vision_task}",
                input_summary=message,
                output_summary=final,
                status="success",
            )

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            state.setdefault("errors", []).append({
                "step_id": f"{self.name}_{int(time.time() * 1000)}",
                "agent": self.name,
                "error": error_msg,
                "retry_count": state.get("retry_count", 0),
                "timestamp": time.time(),
            })
            state["final_response"] = (
                f"I encountered an error while processing the image "
                f"(task: {vision_task}).\n\n{error_msg}"
            )
            state["response_agent"] = self.name

            await self._store_agent_log(
                state,
                action=f"vision_{vision_task}",
                input_summary=message,
                output_summary="",
                status="failed",
                error=error_msg,
            )

        return state

    # ── Image URL extraction ────────────────────────────────────────────

    @staticmethod
    def _extract_image_url(state: AgentState) -> Optional[str]:
        """Extract an image URL from the state's attachments or message text.

        Priority:
        1. ``attachments`` list — first item with a ``url``, ``path``,
           ``data``, ``image_url``, or ``file_url`` key.
        2. URLs in the message text ending with image extensions or starting
           with ``data:image``.
        3. Base64 data URIs embedded in the message.
        """
        attachments: List[Any] = state.get("attachments", []) or []

        # 1. Check structured attachments.
        for att in attachments:
            if isinstance(att, dict):
                for key in ("url", "path", "data", "image_url", "file_url"):
                    value = att.get(key)
                    if value and isinstance(value, str) and len(value) > 10:
                        return value
            elif isinstance(att, str) and len(att) > 10:
                # Attachment is a raw URL string.
                if att.startswith(("http://", "https://", "data:")):
                    return att

        message: str = state.get("message", "") or ""

        # 2. URLs in the message text.
        urls = AgentTools.extract_urls(message)
        for url in urls:
            if any(
                url.lower().endswith(ext)
                for ext in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg")
            ):
                return url

        # 3. Data URIs embedded in the message.
        data_uri_match = re.search(r"data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]{50,}", message)
        if data_uri_match:
            return data_uri_match.group(0)

        # 4. URLs in markdown image syntax: ![alt](url)
        md_img_match = re.search(r"!\[.*?\]\((https?://[^\s)]+)\)", message)
        if md_img_match:
            return md_img_match.group(1)

        return None

    # ── Vision task detection ───────────────────────────────────────────

    @staticmethod
    def _detect_vision_task(message: str) -> str:
        """Return ``"describe"``, ``"ocr"``, ``"qa"``, ``"screenshot"``,
        ``"objects"``, or ``"analyze"``.

        Uses keyword scoring; the group with the most keyword hits wins.
        Tie-breaking rules:
        - ``describe`` wins over ``qa`` when both have equal scores (description
          keywords are more specific).
        - A standalone ``?`` with no keyword match triggers ``qa`` only if the
          question is not a generic description request.
        """
        msg_lower = message.lower()
        scores: Dict[str, int] = {}
        for task, keywords in VISION_KEYWORDS.items():
            scores[task] = sum(1 for kw in keywords if kw in msg_lower)

        best = max(scores, key=scores.get)  # type: ignore[arg-type]

        # Tie-breaking: if describe and qa are tied, describe wins.
        if scores.get("describe", 0) > 0 and scores["describe"] >= scores.get("qa", 0):
            return "describe"

        # A question mark with no keyword match: check context.
        if scores[best] == 0 and "?" in message:
            # Clear description questions → describe.
            q_desc = [
                "what do you see", "what's in this image", "what is shown",
                "describe",
            ]
            if any(p in msg_lower for p in q_desc):
                return "describe"
            # Generic open-ended questions (no specific Q&A intent) → analyze.
            open_ended = [
                "what can you tell", "tell me about", "anything about",
                "what about", "what do you think",
            ]
            if any(p in msg_lower for p in open_ended):
                return "analyze"
            return "qa"

        # Catch-all.
        if scores[best] == 0:
            return "analyze"

        return best

    # ── Task handlers are thin wrappers around Mimo calls; the actual
    #    logic lives inside the MimoLLM integration.  We only format the
    #    results into user-facing responses.
    # ── Formatting helpers ──────────────────────────────────────────────

    @staticmethod
    def _format_screenshot_analysis(analysis: Any) -> str:
        """Convert the screenshot analysis result into a markdown report."""
        if isinstance(analysis, dict):
            lines: List[str] = ["# Screenshot Analysis\n"]
            app = analysis.get("application")
            if app:
                lines.append(f"**Application / Website:** {app}\n")
            main_content = analysis.get("main_content")
            if main_content:
                lines.append(f"**Main Content / Action:** {main_content}\n")
            ui_elements = analysis.get("ui_elements")
            if ui_elements:
                lines.append("## UI Elements\n")
                if isinstance(ui_elements, list):
                    for el in ui_elements:
                        lines.append(f"- {el}")
                else:
                    lines.append(str(ui_elements))
            notes = analysis.get("notes")
            if notes:
                lines.append(f"\n**Notes:** {notes}")
            layout = analysis.get("layout")
            if layout:
                lines.append(f"\n**Layout:** {layout}")
            processing_ms = analysis.get("processing_time_ms")
            if processing_ms:
                lines.append(f"\n---\n*Processed in {processing_ms} ms*")
            return "\n".join(lines)

        # Fallback: analysis is a plain string.
        return f"# Screenshot Analysis\n\n{analysis}"

    @staticmethod
    def _format_objects(objects: List[Dict[str, Any]]) -> str:
        """Format detected objects into a readable list."""
        if not objects:
            return "# Objects Detected\n\nNo objects were detected in this image."

        lines: List[str] = ["# Detected Objects\n"]
        for i, obj in enumerate(objects, 1):
            name = obj.get("name", obj.get("object", f"Object {i}"))
            position = obj.get("position", obj.get("approximate position", "unknown"))
            color = obj.get("color", "")
            quantity = obj.get("estimated quantity", obj.get("quantity", ""))
            details = f"{name}"
            if color:
                details += f" ({color})"
            if position and position != "unknown":
                details += f" — {position}"
            if quantity:
                details += f" ×{quantity}"
            lines.append(f"{i}. {details}")

        return "\n".join(lines)

    # ── Utility helpers ─────────────────────────────────────────────────

    @staticmethod
    def _extract_question(message: str) -> str:
        """Extract the actual question from the message.

        Strips leading task-labelling phrases like "answer this about the
        image" so that only the question itself is sent to Mimo.
        """
        # Remove common prefixes.
        cleaned = re.sub(
            r"^(?:answer|tell me|look at|check|examine|analyze)\s*(?:this|the)\s*(?:image|picture|photo)?[:\s,]*",
            "",
            message,
            flags=re.IGNORECASE,
        )
        return cleaned.strip() or message

    @staticmethod
    def _build_analysis_prompt(message: str) -> str:
        """Derive a detailed analysis prompt from the user message.

        For the ``"analyze"`` task type we this reuse the original message
        as the prompt so the user gets a custom analysis.
        """
        # If the message is very generic, provide a rich default.
        generic_patterns = [
            r"^analyze\s*(?:this|the)?\s*(?:image|picture|photo)?\s*$",
            r"^what\s+(?:can|do)\s+you\s+(?:see|tell)\s*(?:me)?\s*(?:about)?\s*(?:this)?\s*(?:image)?\s*\??$",
        ]
        for pattern in generic_patterns:
            if re.match(pattern, message.strip(), re.IGNORECASE):
                return (
                    "Analyse this image in detail. Include:\n"
                    "1. Overall scene and setting.\n"
                    "2. Key objects, people, and their arrangement.\n"
                    "3. Colours, lighting, and visual style.\n"
                    "4. Any text visible.\n"
                    "5. The mood or purpose of the image."
                )
        return message

    def __repr__(self) -> str:
        return f"<VisionAgent model={self.model_name}>"
