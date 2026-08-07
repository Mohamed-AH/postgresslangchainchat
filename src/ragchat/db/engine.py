"""SQLAlchemy engine and session factory.

The engine is created lazily from :class:`~ragchat.config.Settings` and cached for the
process lifetime, so connection pooling is shared across requests.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from ragchat.config import Settings, get_settings


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
