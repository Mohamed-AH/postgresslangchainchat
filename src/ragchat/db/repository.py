"""Data-access helpers for the ``knowledge_base`` table.

Keeps SQLAlchemy usage in one place so the service layer works with plain domain
objects (``Section``) rather than ORM/session details.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ragchat.db.models import KnowledgeBase
from ragchat.ingestion.parser import Section


def replace_all_sections(session: Session, sections: Sequence[Section]) -> int:
    """Replace the entire corpus with ``sections`` within the caller's transaction.

    The delete + insert run in a single transaction (the caller controls commit/rollback),
    so a failure mid-ingest never leaves the table half-populated.
    """
    session.query(KnowledgeBase).delete()
    session.add_all([KnowledgeBase(title=s.title, content=s.content) for s in sections])
    session.flush()
    return len(sections)


def count_sections(session: Session) -> int:
    """Return the number of rows in ``knowledge_base``."""
    return session.execute(select(func.count()).select_from(KnowledgeBase)).scalar_one()


def get_all_sections(session: Session) -> list[Section]:
    """Return every stored section as domain objects."""
    rows = session.execute(select(KnowledgeBase).order_by(KnowledgeBase.id)).scalars().all()
    return [Section(title=row.title, content=row.content) for row in rows]
