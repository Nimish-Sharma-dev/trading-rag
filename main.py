"""
main.py
───────
Trading RAG — FastAPI application entry point.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.db.database import init_db
from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    settings = get_settings()
    logger.info("Starting Trading RAG (env=%s)", settings.app_env)

    # Create DB tables on startup (idempotent)
    init_db()
    logger.info("Database tables ready")

    yield

    logger.info("Trading RAG shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Trading RAG API",
        description=(
            "Retrieval-Augmented Generation system for trading and stock analysis. "
            "Retrieves expert prompts from a knowledge base and injects them into "
            "an LLM context to produce structured, framework-driven analysis."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", tags=["Root"])
    def root():
        return {
            "service": "Trading RAG API",
            "docs": "/docs",
            "status": "/api/v1/status",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
