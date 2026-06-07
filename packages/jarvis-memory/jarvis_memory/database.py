"""Motor async MongoDB client singleton."""

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

logger = logging.getLogger(__name__)


class Database:
    """Async MongoDB database wrapper using Motor.

    Usage::

        from jarvis_memory.database import db

        await db.connect("mongodb://localhost:27017", "jarvis")
        col = db.get_collection("memories")
    """

    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None

    async def connect(self, uri: str, db_name: str) -> None:
        """Create the Motor client and select the database.

        Args:
            uri: MongoDB connection string.
            db_name: Database name.
        """
        self.client = AsyncIOMotorClient(uri, tz_aware=True)
        self.db = self.client[db_name]
        logger.info("Connected to MongoDB: %s / %s", uri.rsplit("@", 1)[-1], db_name)

    async def disconnect(self) -> None:
        """Close the Motor client and release resources."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None
            logger.info("Disconnected from MongoDB")

    def get_collection(self, name: str) -> AsyncIOMotorCollection:
        """Return a reference to the named collection.

        Args:
            name: Collection name.

        Returns:
            AsyncIOMotorCollection instance.
        """
        if self.db is None:
            raise RuntimeError("Database not connected. Call `connect()` first.")
        return self.db[name]


db = Database()
