"""
Vector memory — embedding generation and similarity search abstraction.
Supports multiple embedding providers with unified interface.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np

from backend.config.settings import settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding generators."""

    async def embed(self, text: str) -> list[float]:
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedding:
    """Local embedding via sentence-transformers (default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._dimension = settings.MEMORY_VECTOR_DIMENSION

    async def _load_model(self) -> None:
        if self._model is None:
            # Lazy import to avoid heavy dependency at startup
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded", extra={"model": self._model_name})

    async def embed(self, text: str) -> list[float]:
        await self._load_model()
        assert self._model is not None
        emb = self._model.encode(text, normalize_embeddings=True)
        return emb.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        await self._load_model()
        assert self._model is not None
        embs = self._model.encode(texts, normalize_embeddings=True)
        return [e.tolist() for e in embs]


class OpenAIEmbedding:
    """Remote embedding via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = settings.MEMORY_VECTOR_DIMENSION
        self._http: Any = None

    async def _ensure_client(self) -> None:
        if self._http is None:
            import httpx
            self._http = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=30.0,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )

    async def embed(self, text: str) -> list[float]:
        await self._ensure_client()
        assert self._http is not None
        response = await self._http.post(
            "/embeddings",
            json={"input": text, "model": self._model},
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        await self._ensure_client()
        assert self._http is not None
        response = await self._http.post(
            "/embeddings",
            json={"input": texts, "model": self._model},
        )
        response.raise_for_status()
        data = response.json()
        return [item["embedding"] for item in data["data"]]


class VectorMemory:
    """High-level vector memory service.

    Combines embedding generation with storage/retrieval.
    Auto-selects embedding provider based on configuration.
    """

    def __init__(self) -> None:
        self._provider: EmbeddingProvider | None = None

    async def get_provider(self) -> EmbeddingProvider:
        """Get or initialize the embedding provider."""
        if self._provider is None:
            if settings.DEEPSEEK_API_KEY:
                self._provider = OpenAIEmbedding(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url=settings.DEEPSEEK_BASE_URL,
                    model="text-embedding-ada-002",
                )
            else:
                self._provider = SentenceTransformerEmbedding()
            logger.info(
                "Vector memory provider initialized",
                extra={"provider": type(self._provider).__name__},
            )
        return self._provider

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        provider = await self.get_provider()
        start = time.monotonic()
        vector = await provider.embed(text)
        elapsed = (time.monotonic() - start) * 1000
        logger.debug("Embedding generated", extra={"duration_ms": f"{elapsed:.1f}", "dim": len(vector)})
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        provider = await self.get_provider()
        return await provider.embed_batch(texts)

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a, dtype=np.float32)
        b_np = np.array(b, dtype=np.float32)
        if np.linalg.norm(a_np) == 0 or np.linalg.norm(b_np) == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np)))

    def rank_by_similarity(
        self, query: list[float], candidates: list[tuple[str, list[float]]]
    ) -> list[tuple[str, float]]:
        """Rank candidates by cosine similarity to query."""
        scored: list[tuple[str, float]] = []
        for content, emb in candidates:
            sim = self.cosine_similarity(query, emb)
            scored.append((content, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored


# Global singleton
vector_memory = VectorMemory()
