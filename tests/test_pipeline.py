"""
tests/test_pipeline.py
──────────────────────
Unit tests for the RAG pipeline components.

Run with:  pytest tests/ -v
"""
import json
import pytest
from unittest.mock import MagicMock, patch


# ─── Prompt Injector ──────────────────────────────────────────────────────────

class TestPromptInjector:
    def setup_method(self):
        from app.prompts.injector import PromptInjector
        from app.retrieval.retriever import RetrievedPrompt

        self.injector = PromptInjector()
        self.mock_prompt = RetrievedPrompt(
            id="test-001",
            title="RSI Framework",
            content="When RSI > 70 consider overbought...",
            category="technical",
            tags=["rsi", "momentum"],
            cosine_score=0.87,
            final_score=0.92,
        )

    def test_build_context_with_prompts(self):
        ctx = self.injector.build_context(
            query="Analyse NVDA momentum",
            retrieved_prompts=[self.mock_prompt],
            ticker="NVDA",
            analysis_type="technical",
        )
        assert "system_prompt" in ctx
        assert "user_message" in ctx
        assert "metadata" in ctx

        # System prompt should contain the framework
        assert "RSI Framework" in ctx["system_prompt"]
        assert "EXPERT ANALYSIS FRAMEWORKS" in ctx["system_prompt"]

        # User message should mention ticker
        assert "NVDA" in ctx["user_message"]
        assert "technical" in ctx["user_message"]

    def test_build_context_no_prompts(self):
        ctx = self.injector.build_context(
            query="Generic market question",
            retrieved_prompts=[],
        )
        assert "No specific frameworks retrieved" in ctx["system_prompt"]
        assert ctx["metadata"]["num_frameworks_injected"] == 0

    def test_metadata_structure(self):
        ctx = self.injector.build_context(
            query="test",
            retrieved_prompts=[self.mock_prompt],
        )
        meta = ctx["metadata"]
        assert meta["retrieved_prompt_ids"] == ["test-001"]
        assert meta["num_frameworks_injected"] == 1
        assert meta["cosine_scores"] == [0.87]


# ─── Re-ranking helpers ───────────────────────────────────────────────────────

class TestReranking:
    def test_category_bonus_match(self):
        from app.retrieval.retriever import _category_bonus
        assert _category_bonus("technical", "technical") == 0.08

    def test_category_bonus_no_match(self):
        from app.retrieval.retriever import _category_bonus
        assert _category_bonus("fundamental", "technical") == 0.0

    def test_category_bonus_no_type(self):
        from app.retrieval.retriever import _category_bonus
        assert _category_bonus("technical", None) == 0.0

    def test_usage_bonus_cap(self):
        from app.retrieval.retriever import _usage_bonus
        # Should cap at 0.05
        assert _usage_bonus(10_000) == 0.05
        assert _usage_bonus(0) == 0.0

    def test_tag_bonus(self):
        from app.retrieval.retriever import _tag_bonus
        score = _tag_bonus(["rsi", "momentum", "macd"], "analyse RSI momentum for NVDA")
        assert score > 0.0
        assert score <= 0.06

    def test_tag_bonus_no_match(self):
        from app.retrieval.retriever import _tag_bonus
        score = _tag_bonus(["inflation", "macro"], "NVDA RSI analysis")
        assert score == 0.0


# ─── API schemas ──────────────────────────────────────────────────────────────

class TestAPISchemas:
    def test_analysis_request_ticker_uppercased(self):
        from app.api.schemas import AnalysisRequest
        req = AnalysisRequest(query="analyse this stock", ticker="nvda")
        assert req.ticker == "NVDA"

    def test_analysis_request_invalid_top_k(self):
        from app.api.schemas import AnalysisRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnalysisRequest(query="test", top_k=50)  # exceeds max of 10

    def test_prompt_create_invalid_category(self):
        from app.api.schemas import PromptCreate
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PromptCreate(
                title="Test",
                content="Some content here that is long enough",
                category="INVALID_CATEGORY",
            )

    def test_prompt_create_valid_categories(self):
        from app.api.schemas import PromptCreate
        for cat in ["technical", "fundamental", "sentiment", "macro", "risk", "options"]:
            p = PromptCreate(
                title="Test Framework",
                content="Some content here that is long enough to pass validation",
                category=cat,
            )
            assert p.category == cat


# ─── Config ───────────────────────────────────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        import os
        # Patch env to avoid requiring a .env file
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-4o",
            "EMBEDDING_PROVIDER": "local",
        }):
            from app.core.config import Settings
            s = Settings()
            assert s.llm_provider == "openai"
            assert s.default_top_k == 3
            assert s.min_relevance_score == 0.30
