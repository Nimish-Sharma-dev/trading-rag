"""
app/db/database.py
──────────────────
SQLAlchemy engine, session factory, and Base declarative class.
"""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _get_engine():
    settings = get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    engine = create_engine(
        settings.database_url,
        connect_args=connect_args,
        echo=(settings.app_env == "development"),
    )

    # Enable WAL mode for SQLite — improves concurrent read performance
    if settings.database_url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_wal(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA journal_mode=WAL")

    return engine


engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables (idempotent)."""
    from app.db import models  # noqa: F401 — registers models with Base
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
