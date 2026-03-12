# API Request & Response Examples

## 1. Main Analysis Endpoint

### Request
```bash
POST /api/v1/analyze
Content-Type: application/json

{
  "query": "Analyze NVDA momentum. Is the uptrend still intact? Identify key levels and assess if this is a good entry point.",
  "ticker": "NVDA",
  "analysis_type": "technical",
  "top_k": 3
}
```

### Response
```json
{
  "query": "Analyze NVDA momentum. Is the uptrend still intact? Identify key levels and assess if this is a good entry point.",
  "ticker": "NVDA",
  "analysis_type": "technical",
  "answer": "## NVDA Technical Analysis\n\n### Multi-Timeframe Trend Assessment\n\n**Weekly Chart (Primary Trend):** NVDA remains in a primary uptrend, trading above both the 50-week and 200-week moving averages. The weekly structure shows a series of higher highs and higher lows since the October 2022 low at $108. Key weekly resistance sits at $950 (all-time high zone) with major support at $750 (previous breakout level).\n\n**Daily Chart (Intermediate Trend):** The intermediate trend is bullish but showing signs of consolidation. Price is trading between the 20-day EMA ($820) and the recent high of $875. The 50-day EMA ($780) remains a critical support level — a daily close below would signal intermediate trend deterioration.\n\n**Momentum Assessment (RSI + MACD):**\n- RSI(14): 58 — neutral-bullish territory, no overbought warning\n- MACD: Histogram contracting slightly, signal line convergence forming — watching for potential bearish crossover\n- Confluence Score: +2 (bullish bias, but not strongly so)\n\n**Volume Analysis:**\n- Recent pullback from highs occurred on declining volume — healthy consolidation\n- VWAP on daily: $834 — price is above VWAP, institutional buyers are net positive\n\n### Key Levels\n\n**Support:**\n1. $820 — 20-day EMA (immediate)\n2. $780 — 50-day EMA + prior breakout level (key)\n3. $750 — major structural support (stop-loss anchor)\n\n**Resistance:**\n1. $875 — recent swing high\n2. $910 — measured move target\n3. $950 — all-time high zone\n\n### Entry Assessment\n\nThe current level ($845 assumed) offers a moderate-quality entry:\n- **Risk/Reward:** Entry at $845, stop at $778 (below $780 support), target $910–$950\n- **R/R Ratio:** 1:1.9 to 1:3.2 depending on target — acceptable\n- **Recommendation:** Wait for a pullback into the $820–$830 zone for a better risk/reward entry. Current level is mid-range between support and resistance.\n\n### Risk Flags\n- Broader market risk-off could drag NVDA to 50-day EMA regardless of fundamentals\n- AI spend narrative is fully priced — any guidance revision = sharp pullback\n- Options market showing elevated put activity above $900 — dealer hedging may cap upside short-term\n\n### Summary\nNVDA's uptrend remains structurally intact. The stock is in a healthy consolidation phase. Ideal entry is a retest of $820–$830 with a stop below $778, targeting $910+. Current levels are acceptable but not optimal. Reduce size by 30% versus normal if entering here.",
  "retrieved_prompts": [
    {
      "id": "a1b2c3d4-...",
      "title": "Multi-Timeframe Technical Analysis Framework",
      "category": "technical",
      "cosine_score": 0.8932,
      "final_score": 0.9732
    },
    {
      "id": "e5f6g7h8-...",
      "title": "RSI + MACD Momentum Analysis Framework",
      "category": "technical",
      "cosine_score": 0.8715,
      "final_score": 0.9515
    },
    {
      "id": "i9j0k1l2-...",
      "title": "Volume Profile and Market Structure Analysis",
      "category": "technical",
      "cosine_score": 0.7841,
      "final_score": 0.8441
    }
  ],
  "llm_provider": "anthropic",
  "llm_model": "claude-sonnet-4-6",
  "latency_ms": 3247,
  "num_frameworks": 3
}
```

---

## 2. Ingest a New Expert Prompt

### Request
```bash
POST /api/v1/prompts
Content-Type: application/json

{
  "title": "Elliott Wave Count Framework",
  "content": "Elliott Wave analysis identifies repetitive wave patterns driven by market psychology...\n\n1. IMPULSIVE WAVES (5-wave structure)...\n2. CORRECTIVE WAVES (3-wave structure, A-B-C)...",
  "category": "technical",
  "tags": ["elliott-wave", "wave-count", "pattern", "fibonacci"],
  "source": "internal"
}
```

### Response (201 Created)
```json
{
  "id": "m3n4o5p6-...",
  "title": "Elliott Wave Count Framework",
  "content": "Elliott Wave analysis identifies...",
  "category": "technical",
  "tags": ["elliott-wave", "wave-count", "pattern", "fibonacci"],
  "source": "internal",
  "usage_count": 0,
  "avg_relevance": 0.0,
  "is_active": true,
  "created_at": "2026-03-12T09:30:00"
}
```

---

## 3. Status Check

### Request
```bash
GET /api/v1/status
```

### Response
```json
{
  "status": "ok",
  "prompt_count": 8,
  "vector_count": 8,
  "llm_provider": "anthropic",
  "llm_model": "claude-sonnet-4-6",
  "embedding_provider": "openai"
}
```

---

## Pipeline Flow Diagram

```
POST /api/v1/analyze
{query: "Analyze NVDA momentum", ticker: "NVDA", analysis_type: "technical"}
         │
         ▼
  [1] Embedder.embed(query)
      → vector: [0.023, -0.187, 0.441, ...]  (1536-dim)
         │
         ▼
  [2] VectorStore.search(query_vector, top_k=9)
      → [(prompt_id_1, score=0.89), (prompt_id_2, score=0.87), ...]
         │
         ▼
  [3] Metadata fetch from SQLite
      → TradingPrompt objects with full content
         │
         ▼
  [4] Re-ranking
      final_score = cosine_score
                  + category_bonus  (+0.08 if category matches analysis_type)
                  + usage_bonus     (+0.00–0.05 based on past usage)
                  + tag_bonus       (+0.02/match, capped at 0.06)
         │
         ▼
  [5] PromptInjector.build_context(query, top_3_prompts)
      → system_prompt: "You are an expert analyst. Follow frameworks: [Framework 1]..."
      → user_message:  "Ticker: NVDA\nAnalysis: technical\n\n{original query}"
         │
         ▼
  [6] LLMClient.complete(system_prompt, user_message)
      → Anthropic/OpenAI/Google API call
      → structured analysis response text
         │
         ▼
  [7] Log to QueryLog table + update prompt usage stats
         │
         ▼
  Return AnalysisResponse JSON
```

---

## Switching LLM Providers

Edit `.env`:

```bash
# Use OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o

# Use Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6

# Use Google
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-pro
```

No code changes required — the `get_llm_client()` factory reads from config.