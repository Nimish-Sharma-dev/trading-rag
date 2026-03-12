"""
app/api/routes.py
─────────────────
FastAPI route handlers.

Endpoints
─────────
POST /api/v1/analyze          — main RAG analysis endpoint
POST /api/v1/prompts          — ingest a new expert prompt
GET  /api/v1/prompts          — list all prompts (paginated)
GET  /api/v1/prompts/{id}     — get a single prompt
DELETE /api/v1/prompts/{id}   — soft-delete a prompt
GET  /api/v1/status           — health check + stats
"""
from __future__ import annotations

import json
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.schemas import (
    AnalysisRequest, AnalysisResponse, RetrievedPromptSummary,
    PromptCreate, PromptResponse, StatusResponse,
)
from app.core.config import get_settings
from app.core.pipeline import RAGPipeline
from app.db.database import get_db
from app.db.models import TradingPrompt
from app.retrieval.vector_store import get_vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

# Singleton pipeline (initialised once at import time)
_pipeline = RAGPipeline()


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
def analyze(request: AnalysisRequest, db: Session = Depends(get_db)):
    """
    **Main RAG endpoint.**

    1. Retrieves the most relevant expert trading prompts for the query.
    2. Injects them into the LLM context as analysis frameworks.
    3. Returns a structured trading analysis.
    """
    try:
        result = _pipeline.run(
            query=request.query,
            db=db,
            ticker=request.ticker,
            analysis_type=request.analysis_type,
            top_k=request.top_k,
            category_filter=request.category_filter,
        )
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    return AnalysisResponse(
        query=request.query,
        ticker=request.ticker,
        analysis_type=request.analysis_type,
        answer=result.answer,
        retrieved_prompts=[
            RetrievedPromptSummary(
                id=p.id,
                title=p.title,
                category=p.category,
                cosine_score=p.cosine_score,
                final_score=p.final_score,
            )
            for p in result.retrieved_prompts
        ],
        llm_provider=result.llm_provider,
        llm_model=result.llm_model,
        latency_ms=result.latency_ms,
        num_frameworks=len(result.retrieved_prompts),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt management
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/prompts", response_model=PromptResponse, status_code=201, tags=["Prompts"])
def create_prompt(body: PromptCreate, db: Session = Depends(get_db)):
    """Ingest a new expert prompt into the knowledge base."""
    prompt = TradingPrompt(
        title=body.title,
        content=body.content,
        category=body.category,
        tags=json.dumps(body.tags),
        source=body.source,
    )
    db.add(prompt)
    db.flush()   # get the generated ID

    # Index in vector store
    get_vector_store().upsert(
        prompt_id=prompt.id,
        text=f"{body.title}\n{body.content}",
        metadata={
            "title":     body.title,
            "category":  body.category,
            "tags":      " ".join(body.tags),
            "is_active": True,
        },
    )

    db.commit()
    db.refresh(prompt)
    return _prompt_to_schema(prompt)


@router.get("/prompts", response_model=List[PromptResponse], tags=["Prompts"])
def list_prompts(
    category: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    db: Session = Depends(get_db),
):
    """List prompts with optional category filter and pagination."""
    q = db.query(TradingPrompt)
    if active_only:
        q = q.filter(TradingPrompt.is_active == True)  # noqa
    if category:
        q = q.filter(TradingPrompt.category == category.lower())
    prompts = q.order_by(TradingPrompt.created_at.desc()).offset(offset).limit(limit).all()
    return [_prompt_to_schema(p) for p in prompts]


@router.get("/prompts/{prompt_id}", response_model=PromptResponse, tags=["Prompts"])
def get_prompt(prompt_id: str, db: Session = Depends(get_db)):
    """Fetch a single prompt by ID."""
    prompt = db.get(TradingPrompt, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return _prompt_to_schema(prompt)


@router.delete("/prompts/{prompt_id}", status_code=204, tags=["Prompts"])
def delete_prompt(prompt_id: str, db: Session = Depends(get_db)):
    """Soft-delete a prompt (sets is_active=False, removes from vector store)."""
    prompt = db.get(TradingPrompt, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")

    prompt.is_active = False
    get_vector_store().delete(prompt_id)
    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse, tags=["System"])
def status(db: Session = Depends(get_db)):
    """Health check and system stats."""
    settings = get_settings()
    prompt_count = db.query(TradingPrompt).filter(TradingPrompt.is_active == True).count()  # noqa
    vector_count = get_vector_store().count()

    return StatusResponse(
        status="ok",
        prompt_count=prompt_count,
        vector_count=vector_count,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
    )


# ─── Helper ───────────────────────────────────────────────────────────────────

def _prompt_to_schema(p: TradingPrompt) -> PromptResponse:
    return PromptResponse(
        id=p.id,
        title=p.title,
        content=p.content,
        category=p.category,
        tags=json.loads(p.tags or "[]"),
        source=p.source,
        usage_count=p.usage_count,
        avg_relevance=p.avg_relevance,
        is_active=p.is_active,
        created_at=p.created_at,
    )
