"""SQLAlchemy engine and session factory.

The engine is created lazily from :class:`~ragchat.config.Settings` and cached for the
process lifetime, so connection pooling is shared across requests.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ragchat.config import Settings, get_settings
from ragchat.db.models import Base


@lru_cache
def get_engine() -> Engine:
    """Return a cached SQLAlchemy engine using the psycopg3 driver."""
    settings: Settings = get_settings()
    return create_engine(
        settings.sqlalchemy_url,
        pool_pre_ping=True,  # transparently recover from dropped connections
        future=True,
    )


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Return a cached session factory bound to the engine."""
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_db() -> None:
    """Create the application's relational tables if they don't already exist.

    Idempotent, so it is safe to call on every startup / ingest. The pgvector-managed
    tables are created by the vector store itself; this covers the ``knowledge_base``
    source-of-truth table. (A real deployment would graduate this to Alembic migrations.)
    """
    Base.metadata.create_all(get_engine())
