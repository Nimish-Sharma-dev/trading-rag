"""
app/embeddings/embedder.py
──────────────────────────
Embedding pipeline that converts text → dense vectors.

Supports two providers:
  • openai   — text-embedding-3-small / text-embedding-3-large (API call)
  • local    — sentence-transformers all-MiniLM-L6-v2 (no API key needed)

The Embedder is a singleton; call get_embedder() to obtain the instance.
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import List

from app.core.config import get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbedder:
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @staticmethod
    def _fingerprint(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI embeddings
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """Uses OpenAI text-embedding-3-* models."""

    _DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model
        self._dim = self._DIMS.get(model, 1536)
        logger.info("OpenAIEmbedder initialised (model=%s, dim=%d)", model, self._dim)

    def embed(self, text: str) -> List[float]:
        text = text.replace("\n", " ").strip()
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        texts = [t.replace("\n", " ").strip() for t in texts]
        response = self._client.embeddings.create(input=texts, model=self._model)
        # API returns items in order
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def dimension(self) -> int:
        return self._dim


# ─────────────────────────────────────────────────────────────────────────────
# Local embeddings (sentence-transformers, zero API cost)
# ─────────────────────────────────────────────────────────────────────────────

class LocalEmbedder(BaseEmbedder):
    """
    Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim).
    Falls back gracefully if sentence-transformers is not installed.
    """

    _MODEL_NAME = "all-MiniLM-L6-v2"
    _DIM = 384

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._MODEL_NAME)
            logger.info("LocalEmbedder initialised (model=%s, dim=%d)", self._MODEL_NAME, self._DIM)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Run: pip install sentence-transformers"
            )

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    @property
    def dimension(self) -> int:
        return self._DIM


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    settings = get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set for embedding_provider=openai")
        return OpenAIEmbedder(
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
        )

    if provider == "local":
        return LocalEmbedder()

    raise ValueError(f"Unknown embedding provider: {provider!r}. Choose 'openai' or 'local'.")
