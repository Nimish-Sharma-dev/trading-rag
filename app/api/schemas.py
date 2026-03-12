"""
app/api/schemas.py
──────────────────
Pydantic v2 request / response models for the Trading RAG API.
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ─── Prompt management ────────────────────────────────────────────────────────

class PromptCreate(BaseModel):
    title:    str = Field(..., min_length=3, max_length=200)
    content:  str = Field(..., min_length=20)
    category: str = Field(..., description="technical|fundamental|sentiment|macro|risk|options")
    tags:     List[str] = Field(default_factory=list)
    source:   str = "internal"

    @field_validator("category")
    @classmethod
    def validate_category(cls, v):
        allowed = {"technical", "fundamental", "sentiment", "macro", "risk", "options"}
        if v.lower() not in allowed:
            raise ValueError(f"category must be one of {allowed}")
        return v.lower()


class PromptResponse(BaseModel):
    id:            str
    title:         str
    content:       str
    category:      str
    tags:          List[str]
    source:        str
    usage_count:   int
    avg_relevance: float
    is_active:     bool
    created_at:    datetime

    model_config = {"from_attributes": True}


# ─── Analysis query ───────────────────────────────────────────────────────────

class AnalysisRequest(BaseModel):
    query:         str  = Field(..., min_length=5, description="Trading or stock analysis question")
    ticker:        Optional[str] = Field(None, description="Optional stock ticker, e.g. AAPL")
    analysis_type: Optional[str] = Field(None, description="technical|fundamental|sentiment|macro|risk|options")
    top_k:         int  = Field(default=3, ge=1, le=10, description="Number of expert prompts to retrieve")
    category_filter: Optional[str] = Field(None, description="Hard-filter retrieval to one category")

    @field_validator("ticker")
    @classmethod
    def uppercase_ticker(cls, v):
        return v.upper() if v else v


class RetrievedPromptSummary(BaseModel):
    id:           str
    title:        str
    category:     str
    cosine_score: float
    final_score:  float


class AnalysisResponse(BaseModel):
    query:          str
    ticker:         Optional[str]
    analysis_type:  Optional[str]
    answer:         str
    retrieved_prompts: List[RetrievedPromptSummary]
    llm_provider:   str
    llm_model:      str
    latency_ms:     int
    num_frameworks: int


# ─── Health / status ──────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    status:          str
    prompt_count:    int
    vector_count:    int
    llm_provider:    str
    llm_model:       str
    embedding_provider: str
