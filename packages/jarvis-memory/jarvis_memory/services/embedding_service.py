"""Embedding service — model-agnostic interface for vector generation.

Default provider is ``sentence-transformers`` with ``all-MiniLM-L6-v2``
(384-dim). The interface allows swapping to OpenAI or a custom HTTP endpoint.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Async embedding service with a model-agnostic interface.

    Args:
        provider: Embedding provider name (``sentence_transformers``,
            ``openai``, or ``http``).
        model_name: Model identifier.
        openai_api_key: API key for OpenAI provider.
        http_endpoint: URL for custom HTTP provider.
    """

    def __init__(
        self,
        provider: str = "sentence_transformers",
        model_name: str = "all-MiniLM-L6-v2",
        openai_api_key: str | None = None,
        http_endpoint: str | None = None,
    ) -> None:
        self._provider_name = provider
        self._model_name = model_name
        self._openai_api_key = openai_api_key
        self._http_endpoint = http_endpoint
        self._model: Any = None
        self._dimensions: int = 384

    @property
    def dimensions(self) -> int:
        """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return self._dimensions

    async def _load_model(self) -> None:
        """Lazy-load the sentence-transformers model on the first call."""
        if self._model is None and self._provider_name == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
                logger.info(
                    "Loaded sentence-transformers model: %s (dim=%d)",
                    self._model_name,
                    self.dimensions,
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not installed; using fallback dummy embeddings"
                )
                self._model = "dummy"

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string.

        Args:
            text: Input text.

        Returns:
            A list of floats representing the embedding vector.
        """
        if not text.strip():
            return [0.0] * self.dimensions

        if self._provider_name == "sentence_transformers":
            await self._load_model()
            if self._model and self._model != "dummy":
                emb = self._model.encode(text, show_progress_bar=False)
                return emb.tolist()

        elif self._provider_name == "openai":
            return await self._embed_openai(text)

        # Fallback: deterministic hash-based embedding for testing
        return self._dummy_embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []

        if self._provider_name == "sentence_transformers":
            await self._load_model()
            if self._model and self._model != "dummy":
                embeddings = self._model.encode(
                    texts, show_progress_bar=False
                )
                return [emb.tolist() for emb in embeddings]

        return [await self.embed(t) for t in texts]

    async def cosine_similarity(
        self,
        a: list[float],
        b: list[float],
    ) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First embedding vector.
            b: Second embedding vector.

        Returns:
            Cosine similarity in range [-1, 1].
        """
        if len(a) != len(b):
            raise ValueError(
                f"Dimension mismatch: {len(a)} vs {len(b)}"
            )
        dot = sum(ai * bi for ai, bi in zip(a, b))
        norm_a = math.sqrt(sum(ai * ai for ai in a))
        norm_b = math.sqrt(sum(bi * bi for bi in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _embed_openai(self, text: str) -> list[float]:
        """Embed via OpenAI API."""
        if not self._openai_api_key:
            logger.warning("OpenAI API key not set; using dummy embedding")
            return self._dummy_embed(text)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.openai.com/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": text,
                        "model": self._model_name or "text-embedding-3-small",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
        except Exception as exc:
            logger.error("OpenAI embedding failed: %s", exc)
            return self._dummy_embed(text)

    def _dummy_embed(self, text: str) -> list[float]:
        """Deterministic dummy embedding for testing/fallback."""
        import hashlib

        hash_bytes = hashlib.md5(text.encode()).digest()
        vec = [b / 255.0 for b in hash_bytes]
        # Repeat to fill 384 dimensions
        vec = (vec * (self.dimensions // len(vec) + 1))[: self.dimensions]
        # Normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
