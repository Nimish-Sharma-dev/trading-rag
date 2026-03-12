"""
app/retrieval/vector_store.py
──────────────────────────────
ChromaDB vector store wrapper.

Responsibilities:
  • Add / update / delete prompt embeddings
  • Run cosine-similarity nearest-neighbour search
  • Return (prompt_id, score) pairs to the retriever

ChromaDB stores vectors on disk at CHROMA_PERSIST_DIR.
A single collection named CHROMA_COLLECTION_NAME holds all trading prompts.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.embeddings.embedder import get_embedder

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(self):
        cfg = get_settings()
        self._embedder = get_embedder()

        self._client = chromadb.PersistentClient(
            path=cfg.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=cfg.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "VectorStore ready — collection=%r, count=%d",
            cfg.chroma_collection_name,
            self._collection.count(),
        )

    # ─── Write ────────────────────────────────────────────────────────────

    def upsert(self, prompt_id: str, text: str, metadata: Dict) -> None:
        """
        Insert or update a single prompt vector.
        metadata dict should include: title, category, tags, is_active.
        """
        embedding = self._embedder.embed(text)
        self._collection.upsert(
            ids=[prompt_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
        )
        logger.debug("Upserted vector for prompt %s", prompt_id)

    def upsert_batch(self, records: List[Dict]) -> None:
        """
        Bulk upsert.  Each record: {id, text, metadata}.
        Embeddings are generated in one batched API call.
        """
        ids = [r["id"] for r in records]
        texts = [r["text"] for r in records]
        metadatas = [r["metadata"] for r in records]

        embeddings = self._embedder.embed_batch(texts)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Batch upserted %d vectors", len(records))

    def delete(self, prompt_id: str) -> None:
        self._collection.delete(ids=[prompt_id])
        logger.debug("Deleted vector for prompt %s", prompt_id)

    # ─── Read ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> List[Tuple[str, float, str]]:
        """
        Semantic nearest-neighbour search.

        Returns list of (prompt_id, relevance_score, document_text)
        sorted by descending relevance.  Relevance is 1 − cosine_distance.
        """
        query_embedding = self._embedder.embed(query)

        where_filter = {"is_active": True}
        if category_filter:
            where_filter["category"] = category_filter

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            where=where_filter,
            include=["distances", "documents", "metadatas"],
        )

        hits = []
        for pid, dist, doc in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
        ):
            # ChromaDB cosine space: distance = 1 − similarity
            score = round(1.0 - dist, 4)
            hits.append((pid, score, doc))

        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def count(self) -> int:
        return self._collection.count()


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore()
