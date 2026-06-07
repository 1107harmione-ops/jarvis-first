#!/usr/bin/env python3
"""Create all BSON indexes for jarvis-memory collections.

Usage::

    python scripts/create_indexes.py --uri "mongodb://localhost:27017" --db jarvis

If ``--uri`` is omitted the script reads ``MONGODB_URI`` from the environment
(or falls back to ``mongodb://localhost:27017``).
"""

import argparse
import asyncio
import logging
import sys

from motor.motor_asyncio import AsyncIOMotorClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("create_indexes")


async def create_indexes(uri: str, db_name: str) -> None:
    """Create all BSON indexes for the jarvis-memory database.

    Args:
        uri: MongoDB connection string.
        db_name: Database name.
    """
    client = AsyncIOMotorClient(uri, tz_aware=True)
    db = client[db_name]

    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    users = db["users"]
    await users.create_index("user_id", unique=True, name="idx_users_user_id_unique")
    await users.create_index("email", unique=True, name="idx_users_email_unique")
    await users.create_index("username", name="idx_users_username")
    await users.create_index("tags", name="idx_users_tags")
    await users.create_index(
        [("is_active", 1), ("created_at", -1)],
        name="idx_users_active_created",
    )
    logger.info("Users indexes created")

    # ------------------------------------------------------------------
    # conversations
    # ------------------------------------------------------------------
    conversations = db["conversations"]
    await conversations.create_index(
        "conversation_id", unique=True, name="idx_conv_conversation_id_unique"
    )
    await conversations.create_index(
        [("user_id", 1), ("updated_at", -1)],
        name="idx_conv_user_updated",
    )
    await conversations.create_index(
        [("user_id", 1), ("created_at", -1)],
        name="idx_conv_user_created",
    )
    await conversations.create_index(
        [("user_id", 1), ("tags", 1)],
        name="idx_conv_user_tags",
    )
    await conversations.create_index(
        [("session_id", 1), ("updated_at", -1)],
        name="idx_conv_session_updated",
    )
    logger.info("Conversations indexes created")

    # ------------------------------------------------------------------
    # messages
    # ------------------------------------------------------------------
    messages = db["messages"]
    await messages.create_index(
        "message_id", unique=True, name="idx_msg_message_id_unique"
    )
    await messages.create_index(
        [("conversation_id", 1), ("timestamp", 1)],
        name="idx_msg_conversation_timestamp",
    )
    await messages.create_index(
        [("user_id", 1), ("timestamp", -1)],
        name="idx_msg_user_timestamp",
    )
    await messages.create_index(
        [("intent", 1), ("timestamp", -1)],
        name="idx_msg_intent_timestamp",
    )
    await messages.create_index(
        [("user_id", 1), ("importance_score", -1)],
        name="idx_msg_user_importance",
    )
    logger.info("Messages indexes created")

    # ------------------------------------------------------------------
    # memories
    # ------------------------------------------------------------------
    memories = db["memories"]
    await memories.create_index(
        "memory_id", unique=True, name="idx_mem_memory_id_unique"
    )
    await memories.create_index(
        [("user_id", 1), ("memory_type", 1), ("importance_score", -1)],
        name="idx_mem_user_type_importance",
    )
    await memories.create_index(
        [("user_id", 1), ("last_accessed", -1)],
        name="idx_mem_user_last_accessed",
    )
    await memories.create_index(
        [("user_id", 1), ("memory_type", 1), ("created_at", -1)],
        name="idx_mem_user_type_created",
    )
    await memories.create_index(
        [("user_id", 1), ("access_count", -1)],
        name="idx_mem_user_access_count",
    )
    await memories.create_index(
        [("user_id", 1), ("consolidated", 1), ("memory_type", 1)],
        name="idx_mem_user_consolidated_type",
    )
    await memories.create_index("tags", name="idx_mem_tags")
    await memories.create_index(
        "expires_at", expireAfterSeconds=0, name="idx_mem_expires_at_ttl"
    )
    logger.info("Memories indexes created")

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------
    tasks = db["tasks"]
    await tasks.create_index(
        "task_id", unique=True, name="idx_task_task_id_unique"
    )
    await tasks.create_index(
        [("user_id", 1), ("status", 1), ("due_at", 1)],
        name="idx_task_user_status_due",
    )
    await tasks.create_index(
        [("user_id", 1), ("status", 1), ("priority", -1)],
        name="idx_task_user_status_priority",
    )
    await tasks.create_index(
        [("user_id", 1), ("task_type", 1), ("status", 1)],
        name="idx_task_user_type_status",
    )
    await tasks.create_index(
        [("user_id", 1), ("completed_at", -1)],
        name="idx_task_user_completed",
    )
    logger.info("Tasks indexes created")

    # ------------------------------------------------------------------
    # knowledge
    # ------------------------------------------------------------------
    knowledge = db["knowledge"]
    await knowledge.create_index(
        "knowledge_id", unique=True, name="idx_knowledge_knowledge_id_unique"
    )
    await knowledge.create_index(
        [("user_id", 1), ("source_type", 1), ("created_at", -1)],
        name="idx_knowledge_user_source_created",
    )
    await knowledge.create_index("tags", name="idx_knowledge_tags")
    await knowledge.create_index(
        [("title", "text"), ("content", "text")],
        name="idx_knowledge_text",
    )
    await knowledge.create_index(
        [("user_id", 1), ("chunk_index", 1)],
        name="idx_knowledge_user_chunk",
    )
    logger.info("Knowledge indexes created")

    # ------------------------------------------------------------------
    # agent_logs
    # ------------------------------------------------------------------
    agent_logs = db["agent_logs"]
    await agent_logs.create_index(
        "timestamp", name="idx_log_timestamp"
    )
    await agent_logs.create_index(
        [("user_id", 1), ("timestamp", -1)],
        name="idx_log_user_timestamp",
    )
    await agent_logs.create_index(
        [("level", 1), ("timestamp", -1)],
        name="idx_log_level_timestamp",
    )
    await agent_logs.create_index(
        [("action", 1), ("timestamp", -1)],
        name="idx_log_action_timestamp",
    )
    await agent_logs.create_index(
        "timestamp", expireAfterSeconds=2592000, name="idx_log_timestamp_ttl"
    )
    logger.info("Agent_logs indexes created")

    # ------------------------------------------------------------------
    # analytics
    # ------------------------------------------------------------------
    analytics = db["analytics"]
    await analytics.create_index(
        [("user_id", 1), ("period", 1), ("date", 1)],
        unique=True,
        name="idx_analytics_user_period_date_unique",
    )
    await analytics.create_index(
        [("event_type", 1), ("date", 1)],
        name="idx_analytics_event_date",
    )
    logger.info("Analytics indexes created")

    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------
    settings_coll = db["settings"]
    await settings_coll.create_index(
        [("scope", 1), ("scope_id", 1), ("key", 1)],
        unique=True,
        name="idx_settings_scope_scopeid_key_unique",
    )
    logger.info("Settings indexes created")

    client.close()
    logger.info("All indexes created successfully for database '%s'", db_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create BSON indexes for jarvis-memory"
    )
    parser.add_argument(
        "--uri",
        default=None,
        help="MongoDB connection string (default: MONGODB_URI env or "
        "mongodb://localhost:27017)",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Database name (default: MONGODB_DB_NAME env or 'jarvis')",
    )
    args = parser.parse_args()

    import os

    uri = args.uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    db_name = args.db or os.getenv("MONGODB_DB_NAME", "jarvis")

    asyncio.run(create_indexes(uri, db_name))


if __name__ == "__main__":
    main()
