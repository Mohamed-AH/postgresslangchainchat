"""FastAPI application factory.

Uses the modern ``lifespan`` context manager (not the deprecated ``on_event`` hooks) to
build the :class:`~ragchat.service.RAGService` once at startup and dispose the database
engine at shutdown.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ragchat import __version__
from ragchat.api.routes import router
from ragchat.config import get_settings
from ragchat.logging_config import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources on startup; tear them down on shutdown."""
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger(__name__)

    # Import here so the module imports cleanly even when providers aren't configured
    # (e.g. when tests import create_app and override dependencies).
    from ragchat.db.engine import get_engine, init_db

    logger.info("Starting ragchat API v%s", __version__)
    # Ensure the relational schema exists up front. Per-session services (and their LLM
    # providers) are built lazily on first request, so startup stays fast and the
    # readiness probe never depends on the model layer.
    init_db()
    try:
        yield
    finally:
        logger.info("Shutting down ragchat API")
        get_engine().dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="ragchat",
        version=__version__,
        summary="Retrieval-augmented Q&A over a PostgreSQL + pgvector knowledge base.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


# ASGI entrypoint for `uvicorn ragchat.api.app:app`.
app = create_app()
