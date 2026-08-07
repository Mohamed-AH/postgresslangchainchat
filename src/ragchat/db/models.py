"""SQLAlchemy models.

``KnowledgeBase`` is the *relational source of truth* for the corpus. Embeddings are
derived from these rows and stored in pgvector-managed tables; keeping the canonical
text here means the index can always be rebuilt from the database alone.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class KnowledgeBase(Base):
    """A single titled section of source content."""

    __tablename__ = "knowledge_base"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"KnowledgeBase(id={self.id!r}, title={self.title!r})"
