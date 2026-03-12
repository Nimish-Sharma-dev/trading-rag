"""
app/db/models.py
────────────────
SQLAlchemy ORM models for the Trading RAG metadata store.

Schema
──────
trading_prompts          — master table for expert prompts
  id             TEXT PK  UUID
  title          TEXT     human-readable name
  content        TEXT     full prompt text
  category       TEXT     technical | fundamental | sentiment | macro | risk
  tags           TEXT     JSON-encoded list of keyword tags
  source         TEXT     origin (e.g. "internal", author name)
  usage_count    INT      how many times this prompt was retrieved
  avg_relevance  REAL     running average of cosine similarity scores
  is_active      BOOL     soft-delete / enable flag
  created_at     DATETIME
  updated_at     DATETIME

query_logs               — audit table for every RAG pipeline run
  id             TEXT PK  UUID
  query          TEXT     original user query
  ticker         TEXT     optional ticker symbol
  analysis_type  TEXT     optional requested analysis type
  retrieved_ids  TEXT     JSON list of prompt IDs used
  llm_provider   TEXT
  llm_model      TEXT
  response_ms    INT      total pipeline latency
  created_at     DATETIME
"""
import uuid
from datetime import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
)
from app.db.database import Base


def _now():
    return datetime.utcnow()


def _uuid():
    return str(uuid.uuid4())


class TradingPrompt(Base):
    __tablename__ = "trading_prompts"

    id            = Column(String, primary_key=True, default=_uuid)
    title         = Column(String(200), nullable=False)
    content       = Column(Text, nullable=False)
    category      = Column(String(50), nullable=False, index=True)
    tags          = Column(Text, default="[]")          # JSON list
    source        = Column(String(100), default="internal")
    usage_count   = Column(Integer, default=0)
    avg_relevance = Column(Float, default=0.0)
    is_active     = Column(Boolean, default=True, index=True)
    created_at    = Column(DateTime, default=_now)
    updated_at    = Column(DateTime, default=_now, onupdate=_now)

    def __repr__(self):
        return f"<TradingPrompt id={self.id} title={self.title!r}>"


class QueryLog(Base):
    __tablename__ = "query_logs"

    id            = Column(String, primary_key=True, default=_uuid)
    query         = Column(Text, nullable=False)
    ticker        = Column(String(20))
    analysis_type = Column(String(50))
    retrieved_ids = Column(Text, default="[]")          # JSON list
    llm_provider  = Column(String(50))
    llm_model     = Column(String(100))
    response_ms   = Column(Integer)
    created_at    = Column(DateTime, default=_now)

    def __repr__(self):
        return f"<QueryLog id={self.id} query={self.query[:40]!r}>"
