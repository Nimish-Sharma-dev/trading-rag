# Trading RAG System

A Retrieval-Augmented Generation (RAG) system that enhances LLM responses for trading and stock analysis by injecting expert prompts from a structured knowledge base.

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Gateway                       │
│                  POST /api/v1/analyze                    │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌─────────────┐  ┌──────────┐  ┌──────────┐
   │  Embedding  │  │  SQLite  │  │  Chroma  │
   │  Pipeline   │  │ Metadata │  │  Vector  │
   │(text→vector)│  │   DB     │  │   DB     │
   └──────┬──────┘  └──────────┘  └────┬─────┘
          │                            │
          └────────────┬───────────────┘
                       ▼
              ┌─────────────────┐
              │ Retrieval Engine│
              │ (semantic search│
              │  + re-ranking)  │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Prompt Injection│
              │ Layer (context  │
              │  assembly)      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │   LLM Client    │
              │ OpenAI/Anthropic│
              │    /Google      │
              └────────┬────────┘
                       ▼
              Structured Analysis Response
```

## Folder Structure

```
trading-rag/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # FastAPI route handlers
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # App configuration & env vars
│   │   └── pipeline.py        # Main RAG orchestration pipeline
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   └── database.py        # DB connection & session management
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedder.py        # Embedding generation (OpenAI/local)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py    # ChromaDB vector operations
│   │   └── retriever.py       # Retrieval + re-ranking logic
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py          # Multi-provider LLM client
│   └── prompts/
│       ├── __init__.py
│       └── injector.py        # Prompt formatting & injection
├── data/
│   └── seed_prompts/
│       └── trading_prompts.json  # Initial expert prompts
├── scripts/
│   └── ingest_prompts.py      # CLI tool to load prompts into DB
├── config/
│   └── .env.example           # Environment variable template
├── tests/
│   ├── test_pipeline.py
│   └── test_retrieval.py
├── main.py                    # Application entry point
└── requirements.txt
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp config/.env.example .env
# Edit .env with your API keys

# 3. Ingest seed prompts
python scripts/ingest_prompts.py

# 4. Start server
uvicorn main:app --reload --port 8000
```

## API Usage

### Analyze a stock query

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Analyze NVDA momentum and identify key support/resistance levels",
    "ticker": "NVDA",
    "analysis_type": "technical",
    "top_k": 3
  }'
```

### Ingest a new prompt

```bash
curl -X POST http://localhost:8000/api/v1/prompts \
  -H "Content-Type: application/json" \
  -d '{
    "title": "RSI Divergence Framework",
    "content": "When analyzing RSI divergence...",
    "category": "technical",
    "tags": ["rsi", "divergence", "momentum"]
  }'
```

## Supported LLM Providers

| Provider  | Models                          | Env Key              |
|-----------|---------------------------------|----------------------|
| OpenAI    | gpt-4o, gpt-4-turbo, gpt-3.5   | OPENAI_API_KEY       |
| Anthropic | claude-opus-4-6, sonnet, haiku  | ANTHROPIC_API_KEY    |
| Google    | gemini-1.5-pro, gemini-flash    | GOOGLE_API_KEY       |
