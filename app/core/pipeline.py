"""
app/core/pipeline.py
────────────────────
Main RAG orchestration pipeline.

Wires together:
  Embedder → VectorStore → Retriever → PromptInjector → LLMClient

Entry point: RAGPipeline.run(query, ...)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import QueryLog
from app.llm.client import get_llm_client
from app.prompts.injector import PromptInjector
from app.retrieval.retriever import Retriever, RetrievedPrompt

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    query: str
    ticker: Optional[str]
    analysis_type: Optional[str]
    answer: str
    retrieved_prompts: List[RetrievedPrompt]
    llm_provider: str
    llm_model: str
    latency_ms: int
    injection_metadata: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Orchestrates the full RAG flow:

    1. Retrieve relevant expert prompts for the query
    2. Inject prompts into LLM context
    3. Call the LLM
    4. Log the query + results
    5. Update prompt usage stats
    """

    def __init__(self):
        self._retriever = Retriever()
        self._injector  = PromptInjector()
        self._llm       = get_llm_client()

    def run(
        self,
        query: str,
        db: Session,
        ticker: str | None = None,
        analysis_type: str | None = None,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> PipelineResult:
        t_start = time.monotonic()

        # ── Step 1: Retrieve ──────────────────────────────────────────────
        logger.info("RAG pipeline — query=%r, ticker=%s, type=%s", query[:80], ticker, analysis_type)
        retrieved = self._retriever.retrieve(
            query=query,
            db=db,
            top_k=top_k,
            analysis_type=analysis_type,
            category_filter=category_filter,
        )
        logger.info("Retrieved %d prompts", len(retrieved))

        # ── Step 2: Inject prompts into context ───────────────────────────
        context = self._injector.build_context(
            query=query,
            retrieved_prompts=retrieved,
            ticker=ticker,
            analysis_type=analysis_type,
        )

        # ── Step 3: Call LLM ──────────────────────────────────────────────
        llm_response = self._llm.complete(
            system_prompt=context["system_prompt"],
            user_message=context["user_message"],
        )
        logger.info(
            "LLM responded — provider=%s, model=%s, tokens_in=%d, tokens_out=%d, ms=%d",
            llm_response.provider, llm_response.model,
            llm_response.input_tokens, llm_response.output_tokens,
            llm_response.latency_ms,
        )

        total_ms = int((time.monotonic() - t_start) * 1000)

        # ── Step 4: Log to DB ─────────────────────────────────────────────
        self._log_query(
            query=query,
            ticker=ticker,
            analysis_type=analysis_type,
            retrieved_ids=[p.id for p in retrieved],
            provider=llm_response.provider,
            model=llm_response.model,
            latency_ms=total_ms,
            db=db,
        )

        # ── Step 5: Update usage stats ────────────────────────────────────
        if retrieved:
            self._retriever.update_usage_stats(
                prompt_ids=[p.id for p in retrieved],
                scores=[p.final_score for p in retrieved],
                db=db,
            )

        return PipelineResult(
            query=query,
            ticker=ticker,
            analysis_type=analysis_type,
            answer=llm_response.text,
            retrieved_prompts=retrieved,
            llm_provider=llm_response.provider,
            llm_model=llm_response.model,
            latency_ms=total_ms,
            injection_metadata=context["metadata"],
        )

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _log_query(
        self,
        query, ticker, analysis_type,
        retrieved_ids, provider, model, latency_ms, db,
    ):
        try:
            log = QueryLog(
                query=query,
                ticker=ticker,
                analysis_type=analysis_type,
                retrieved_ids=json.dumps(retrieved_ids),
                llm_provider=provider,
                llm_model=model,
                response_ms=latency_ms,
            )
            db.add(log)
            db.commit()
        except Exception as exc:
            logger.warning("Failed to write query log: %s", exc)
            db.rollback()
