"""
Router Agent — the central orchestrator for the JARVIS multi-agent system.

Responsibilities:
1. Receive all user requests
2. Classify intent (coding, research, vision, memory, task, general)
3. Route to the correct specialized agent
4. Aggregate responses from sub-agents
5. Manage conversation context and memory injection
6. Handle errors and fallbacks gracefully
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from backend.agents.coding_agent import coding_agent
from backend.agents.memory_agent import memory_agent
from backend.agents.planner_agent import planner_agent
from backend.agents.research_agent import research_agent
from backend.agents.task_agent import task_agent
from backend.agents.vision_agent import vision_agent
from backend.database.mongodb import mongodb
from backend.database.schemas import new_message_doc, serialize_doc
from backend.llm.router import LLMRouter, TaskCategory, llm_router
from backend.memory.long_term import ltm
from backend.memory.short_term import stm
from backend.memory.vector_memory import vector_memory
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class RouterAgent:
    """Central agent orchestrator.

    Routes requests to specialized agents based on intent classification.
    Manages conversation state, memory injection, and response aggregation.
    """

    def __init__(self) -> None:
        self.router = llm_router
        self.name = "router"

    async def process(
        self,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        stream: bool = False,
        attachments: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a user message through the agent pipeline.

        Args:
            user_id: Authenticated user ID.
            message: User's text message.
            conversation_id: Existing conversation ID or None for new.
            stream: Enable streaming response.
            attachments: File URLs or image IDs.
            metadata: Additional context.

        Returns:
            Response dict with content, agent used, and metadata.
        """
        start_time = time.monotonic()
        session_id = str(uuid.uuid4())
        metadata = metadata or {}

        # 1. Classify intent
        category = self.router.categorize_request(message, attachments)
        logger.info(
            "Request classified",
            extra={"category": category.value, "user_id": user_id, "session_id": session_id},
        )

        # 2. Build enriched context with memory
        context_messages = await self._build_context(user_id, message, category)

        # 3. Route to specialized agent or handle directly
        try:
            response = await self._route_to_agent(
                category=category,
                user_id=user_id,
                message=message,
                context_messages=context_messages,
                attachments=attachments,
                session_id=session_id,
                stream=stream,
            )

            # 4. Store in conversation
            conv_id = await self._store_messages(
                user_id=user_id,
                conversation_id=conversation_id,
                user_message=message,
                assistant_response=response.get("content", ""),
                agent=response.get("agent", self.name),
                attachments=attachments,
                metadata=metadata,
            )

            # 5. Store in short-term memory
            await self._store_memory(user_id, message, response)

            elapsed = (time.monotonic() - start_time) * 1000
            logger.info(
                "Request processed",
                extra={
                    "agent": response.get("agent"),
                    "duration_ms": f"{elapsed:.1f}",
                    "session_id": session_id,
                    "conversation_id": conv_id,
                },
            )

            return {
                "content": response.get("content", ""),
                "agent": response.get("agent", self.name),
                "conversation_id": conv_id,
                "category": category.value,
                "duration_ms": round(elapsed, 1),
                "tokens_used": response.get("tokens_used"),
                "metadata": response.get("metadata", {}),
            }

        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            logger.error(
                "Agent processing failed",
                extra={"error": str(exc), "session_id": session_id, "category": category.value},
            )
            # Graceful fallback
            return {
                "content": f"I encountered an error processing your request. Please try again or rephrase. Error: {str(exc)}",
                "agent": "fallback",
                "conversation_id": conversation_id,
                "category": category.value,
                "duration_ms": round(elapsed, 1),
                "error": str(exc),
            }

    async def _route_to_agent(
        self,
        category: TaskCategory,
        user_id: str,
        message: str,
        context_messages: list[dict[str, str]],
        attachments: list[str] | None,
        session_id: str,
        stream: bool,
    ) -> dict[str, Any]:
        """Route to the appropriate specialized agent."""
        agent_map = {
            TaskCategory.CODING: ("coding_agent", coding_agent.process),
            TaskCategory.CODE_REVIEW: ("coding_agent", coding_agent.process),
            TaskCategory.RESEARCH: ("research_agent", research_agent.process),
            TaskCategory.SUMMARIZATION: ("research_agent", research_agent.process),
            TaskCategory.VISION: ("vision_agent", lambda **kw: vision_agent.process(**kw)),
            TaskCategory.OCR: ("vision_agent", lambda **kw: vision_agent.process(**kw)),
            TaskCategory.PLANNING: ("planner_agent", planner_agent.process),
            TaskCategory.TASK_MANAGEMENT: ("task_agent", task_agent.process),
            TaskCategory.MEMORY: ("memory_agent", memory_agent.process),
        }

        if category in agent_map:
            agent_name, agent_func = agent_map[category]
            logger.debug("Routing to agent", extra={"agent": agent_name, "session_id": session_id})
            result = await agent_func(
                user_id=user_id,
                message=message,
                context=context_messages,
                attachments=attachments,
                session_id=session_id,
            )
            if isinstance(result, dict):
                result["agent"] = agent_name
                return result
            return {"content": str(result), "agent": agent_name}

        # General chat — use router directly
        return await self._direct_chat(context_messages, stream)

    async def _direct_chat(
        self,
        messages: list[dict[str, str]],
        stream: bool,
    ) -> dict[str, Any]:
        """Handle general chat directly via DeepSeek."""
        response = await self.router.route(
            messages, category=TaskCategory.GENERAL_CHAT, stream=False
        )
        content = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {})
        return {
            "content": content,
            "agent": "router",
            "tokens_used": tokens.get("total_tokens", 0),
        }

    async def _build_context(
        self,
        user_id: str,
        message: str,
        category: TaskCategory,
    ) -> list[dict[str, str]]:
        """Build enriched context with memory retrieval."""
        system_prompt = """You are JARVIS, an advanced AI assistant. You are helpful, concise, and capable.

Capabilities:
- Answer questions and have natural conversations
- Write, review, and debug code (Python, JavaScript, TypeScript, Go, Rust, and more)
- Research topics and provide detailed reports
- Analyze images and extract text (OCR)
- Manage tasks, reminders, and schedules
- Remember context from previous conversations
- Help plan projects and architectures

Guidelines:
- Be concise but thorough
- Admit when you don't know something
- Ask clarifying questions when needed
- Use code blocks with language tags for code
- For technical topics, provide examples"""
        messages = [{"role": "system", "content": system_prompt}]

        # Inject relevant memories
        try:
            recent_stm = await stm.get_context_window(user_id, max_items=5)
            if recent_stm:
                memory_text = "\n".join(
                    f"- [{m['memory_type']}] {m['content'][:200]}"
                    for m in recent_stm
                )
                messages.append({
                    "role": "system",
                    "content": f"Recent context from memory:\n{memory_text}",
                })
        except Exception as exc:
            logger.warning("Memory context build failed", extra={"error": str(exc)})

        messages.append({"role": "user", "content": message})
        return messages

    async def _store_messages(
        self,
        user_id: str,
        conversation_id: str | None,
        user_message: str,
        assistant_response: str,
        agent: str,
        attachments: list[str] | None,
        metadata: dict[str, Any],
    ) -> str:
        """Store user message and assistant response in conversation."""
        # Create conversation if new
        if not conversation_id:
            from backend.database.schemas import new_conversation_doc
            title = user_message[:80] + ("..." if len(user_message) > 80 else "")
            conv_doc = new_conversation_doc(user_id, title=title)
            result = await mongodb.conversations.insert_one(conv_doc)
            conversation_id = str(result.inserted_id)
        else:
            # Update conversation timestamp
            await mongodb.conversations.update_one(
                {"_id": conversation_id},
                {"$set": {"updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                 "$inc": {"message_count": 1}},
            )

        # Store user message
        user_doc = new_message_doc(
            conversation_id=conversation_id,
            role="user",
            content=user_message,
            attachments=attachments,
        )
        await mongodb.messages.insert_one(user_doc)

        # Store assistant message
        assistant_doc = new_message_doc(
            conversation_id=conversation_id,
            role="assistant",
            content=assistant_response,
            agent=agent,
            metadata=metadata,
        )
        await mongodb.messages.insert_one(assistant_doc)

        return conversation_id

    async def _store_memory(
        self, user_id: str, user_message: str, response: dict[str, Any]
    ) -> None:
        """Store the interaction in short-term memory."""
        try:
            content = f"User: {user_message[:300]}\nAssistant: {response.get('content', '')[:300]}"
            await stm.store(
                user_id=user_id,
                content=content,
                tags=["conversation", response.get("agent", "router")],
                importance_score=0.3,
            )
        except Exception as exc:
            logger.warning("Memory storage failed", extra={"error": str(exc)})


# Global singleton
router_agent = RouterAgent()
