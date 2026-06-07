"""
JARVIS Backend — Conftest with test fixtures.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from motor.motor_asyncio import AsyncIOMotorClient

from backend.config.settings import settings
from backend.database.mongodb import db
from backend.database.schemas import new_user_doc
from backend.utils.security import create_access_token, hash_password

# Test database
TEST_MONGODB_URI = "mongodb://localhost:27017"
TEST_DB_NAME = "jarvis_test"

TEST_USER_EMAIL = "test@jarvis.ai"
TEST_USER_PASSWORD = "TestPass123"
TEST_ADMIN_EMAIL = "admin@jarvis.ai"
TEST_ADMIN_PASSWORD = "AdminPass123"


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def mongo_client() -> AsyncGenerator[AsyncIOMotorClient[Any], None]:
    """Create MongoDB client for testing."""
    client: AsyncIOMotorClient[Any] = AsyncIOMotorClient(TEST_MONGODB_URI)
    yield client
    client.close()


@pytest_asyncio.fixture(autouse=True)
async def clean_db(mongo_client: AsyncIOMotorClient[Any]) -> AsyncGenerator[None, None]:
    """Clean test database before each test."""
    db_instance = mongo_client[TEST_DB_NAME]
    collections = await db_instance.list_collection_names()
    for collection in collections:
        await db_instance[collection].delete_many({})
    yield


@pytest_asyncio.fixture
async def test_user(mongo_client: AsyncIOMotorClient[Any]) -> dict[str, Any]:
    """Create a test user in the database and return the doc."""
    db_instance = mongo_client[TEST_DB_NAME]
    doc = new_user_doc(
        email=TEST_USER_EMAIL,
        password_hash=hash_password(TEST_USER_PASSWORD),
        name="Test User",
        role="user",
    )
    result = await db_instance.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


@pytest_asyncio.fixture
async def test_admin(mongo_client: AsyncIOMotorClient[Any]) -> dict[str, Any]:
    """Create a test admin in the database and return the doc."""
    db_instance = mongo_client[TEST_DB_NAME]
    doc = new_user_doc(
        email=TEST_ADMIN_EMAIL,
        password_hash=hash_password(TEST_ADMIN_PASSWORD),
        name="Admin User",
        role="admin",
    )
    result = await db_instance.users.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


@pytest_asyncio.fixture
async def user_token(test_user: dict[str, Any]) -> str:
    """Generate a JWT token for the test user."""
    token = create_access_token(
        data={"sub": str(test_user["_id"]), "email": test_user["email"], "role": "user"},
    )
    return token


@pytest_asyncio.fixture
async def admin_token(test_admin: dict[str, Any]) -> str:
    """Generate a JWT token for the test admin."""
    token = create_access_token(
        data={"sub": str(test_admin["_id"]), "email": test_admin["email"], "role": "admin"},
    )
    return token


@pytest_asyncio.fixture
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create HTTP client for API testing."""
    async with httpx.AsyncClient(base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def async_client(http_client: httpx.AsyncClient) -> httpx.AsyncClient:
    """Alias for http_client for compatibility."""
    return http_client
