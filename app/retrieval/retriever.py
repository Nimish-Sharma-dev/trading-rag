"""
app/retrieval/retriever.py
──────────────────────────
Retrieval engine — sits between the vector store and the pipeline.

Steps
─────
1. Vector search  → top-K candidates by cosine similarity
2. Metadata fetch → load full prompt records from SQLite
3. Re-ranking     → apply lightweight heuristics on top of cosine score
4. Threshold gate → drop prompts below MIN_RELEVANCE_SCORE
5. Return         → ordered list of RetrievedPrompt dataclasses
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import TradingPrompt
from app.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPrompt:
    id: str
    title: str
    content: str
    category: str
    tags: List[str]
    cosine_score: float
    final_score: float      # after re-ranking


# ─── Re-ranking helpers ───────────────────────────────────────────────────────

_ANALYSIS_TYPE_TO_CATEGORY = {
    "technical":    "technical",
    "fundamental":  "fundamental",
    "sentiment":    "sentiment",
    "macro":        "macro",
    "risk":         "risk",
    "options":      "options",
}


def _category_bonus(prompt_category: str, requested_type: Optional[str]) -> float:
    """
    Small additive boost when the prompt category exactly matches
    the analysis type the user requested.
    """
    if not requested_type:
        return 0.0
    mapped = _ANALYSIS_TYPE_TO_CATEGORY.get(requested_type.lower(), "")
    return 0.08 if prompt_category.lower() == mapped else 0.0


def _usage_bonus(usage_count: int) -> float:
    """
    Tiny popularity boost — proven prompts float slightly higher.
    Capped at 0.05 to avoid swamping semantic relevance.
    """
    return min(usage_count / 1000.0, 0.05)


def _tag_bonus(tags: List[str], query: str) -> float:
    """
    Keyword overlap bonus — boost prompts whose tags appear in the query.
    """
    query_lower = query.lower()
    matches = sum(1 for t in tags if t.lower() in query_lower)
    return min(matches * 0.02, 0.06)


# ─── Retriever ────────────────────────────────────────────────────────────────

class Retriever:
    def __init__(self):
        self._vector_store = get_vector_store()
        self._settings = get_settings()

    def retrieve(
        self,
        query: str,
        db: Session,
        top_k: int | None = None,
        analysis_type: str | None = None,
        category_filter: str | None = None,
    ) -> List[RetrievedPrompt]:
        """
        Full retrieval pipeline.

        Parameters
        ──────────
        query           The raw user query text.
        db              SQLAlchemy session for metadata lookups.
        top_k           Number of prompts to return (default from settings).
        analysis_type   Optional hint (technical/fundamental/…) used in re-ranking.
        category_filter Hard filter to restrict search to one category.

        Returns
        ───────
        List of RetrievedPrompt sorted by final_score descending.
        """
        k = top_k or self._settings.default_top_k
        threshold = self._settings.min_relevance_score

        # ── Step 1: vector search ──────────────────────────────────────────
        # Fetch slightly more candidates than needed so re-ranking has room
        candidates = self._vector_store.search(
            query=query,
            top_k=k * 3,
            category_filter=category_filter,
        )
        logger.debug("Vector search returned %d candidates", len(candidates))

        if not candidates:
            logger.warning("No vectors found in store — run ingestion first.")
            return []

        # ── Step 2: metadata hydration ────────────────────────────────────
        prompt_ids = [pid for pid, _, _ in candidates]
        db_prompts: dict[str, TradingPrompt] = {
            p.id: p
            for p in db.query(TradingPrompt)
                       .filter(TradingPrompt.id.in_(prompt_ids))
                       .filter(TradingPrompt.is_active == True)  # noqa
                       .all()
        }

        # ── Step 3: re-ranking ────────────────────────────────────────────
        ranked: List[RetrievedPrompt] = []
        for pid, cosine_score, _ in candidates:
            if cosine_score < threshold:
                continue
            prompt = db_prompts.get(pid)
            if not prompt:
                continue

            tags = json.loads(prompt.tags or "[]")
            final_score = (
                cosine_score
                + _category_bonus(prompt.category, analysis_type)
                + _usage_bonus(prompt.usage_count)
                + _tag_bonus(tags, query)
            )

            ranked.append(RetrievedPrompt(
                id=prompt.id,
                title=prompt.title,
                content=prompt.content,
                category=prompt.category,
                tags=tags,
                cosine_score=cosine_score,
                final_score=round(final_score, 4),
            ))

        ranked.sort(key=lambda x: x.final_score, reverse=True)
        top = ranked[:k]

        logger.info(
            "Retrieval done — query=%r, returned=%d/%d candidates above threshold",
            query[:60], len(top), len(candidates),
        )
        return top

    def update_usage_stats(
        self, prompt_ids: List[str], scores: List[float], db: Session
    ) -> None:
        """
        Increment usage counts and update running avg_relevance for
        each retrieved prompt (called after a successful LLM response).
        """
        for pid, score in zip(prompt_ids, scores):
            prompt = db.get(TradingPrompt, pid)
            if not prompt:
                continue
            n = prompt.usage_count
            prompt.avg_relevance = round(
                (prompt.avg_relevance * n + score) / (n + 1), 4
            )
            prompt.usage_count = n + 1

        db.commit()
