"""Application service layer.

``RAGService`` is the single orchestration point the API and CLI both call, and it is
**session-scoped**: each instance is bound to one tenant's ``session_id`` and only ever
reads or writes that tenant's relational rows and pgvector collection. It owns three
flows:

* **ingest** — parse content, write the relational source of truth *and* the per-session
  vector index in one consistent operation;
* **ask** — run the RAG chain and return an answer with its supporting sources;
* **purge** — delete all of a session's data (used by TTL cleanup).

All external dependencies (session factory, vector store, chain) are injected, so tests
construct a service backed by fakes and never touch a real database or LLM. Use
:func:`build_session_service` for the fully wired production instance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from sqlalchemy import text
from sqlalchemy.orm import Session

from ragchat.config import Settings, get_settings
from ragchat.db import repository
from ragchat.ingestion.parser import Section, parse_markdown_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """A retrieved document backing an answer."""

    content: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """The result of a question: the generated answer plus its sources."""

    answer: str
    sources: list[SourceDocument]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Summary of an ingestion run."""

    sections_written: int


class RAGService:
    """Coordinates ingestion and question answering for a single session."""

    def __init__(
        self,
        *,
        session_id: str,
        session_factory: Callable[[], Session],
        vector_store: Any,  # PGVector in production; a test double in tests
        chain: Runnable[str, dict[str, Any]],
        ttl_hours: int = 24,
    ) -> None:
        self._session_id = session_id
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._chain = chain
        self._ttl_hours = ttl_hours

    @property
    def session_id(self) -> str:
        return self._session_id

    # -- Ingestion ---------------------------------------------------------
    def ingest_sections(self, sections: Sequence[Section]) -> IngestResult:
        """Atomically replace *this session's* corpus with ``sections``.

        Ordering guarantees consistency without a distributed transaction: relational
        rows are written but *not committed* until the vector index has been rebuilt
        successfully. If embedding/indexing fails, the relational write is rolled back
        and any partially written vectors are cleared, so the two stores never drift.
        """
        documents = [
            Document(
                page_content=s.as_document_text(),
                metadata={"title": s.title, "session_id": self._session_id},
            )
            for s in sections
        ]

        db = self._session_factory()
        try:
            expires_at = datetime.now(UTC) + timedelta(hours=self._ttl_hours)
            repository.upsert_session(db, self._session_id, expires_at)
            count = repository.replace_all_sections(db, self._session_id, sections)
            # Rebuild this session's vector index from scratch to stay in sync with the
            # table: drop the collection, recreate it, then load the fresh embeddings.
            # (langchain-postgres does not auto-create a collection on add.)
            self._vector_store.delete_collection()
            self._vector_store.create_collection()
            if documents:
                self._vector_store.add_documents(documents)
            db.commit()
            logger.info("Ingested %d sections for session %s", count, self._session_id)
            return IngestResult(sections_written=count)
        except Exception:
            db.rollback()
            # Best-effort cleanup of any vectors written before the failure.
            try:
                self._vector_store.delete_collection()
            except Exception:  # pragma: no cover - cleanup is best-effort
                logger.exception("Failed to clean up vector collection after ingest error")
            logger.exception("Ingestion failed for session %s; rolled back", self._session_id)
            raise
        finally:
            db.close()

    def ingest_markdown_file(self, path: str | Path) -> IngestResult:
        """Parse a markdown file and ingest its sections."""
        sections = parse_markdown_file(path)
        logger.info("Parsed %d sections from %s", len(sections), path)
        return self.ingest_sections(sections)

    # -- Querying ----------------------------------------------------------
    def ask(self, question: str) -> AnswerResult:
        """Answer ``question`` using retrieval-augmented generation over this session."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        result = self._chain.invoke(question)
        documents: list[Document] = result.get("documents", [])
        sources = [
            SourceDocument(content=doc.page_content, metadata=dict(doc.metadata))
            for doc in documents
        ]
        return AnswerResult(answer=result["answer"], sources=sources)

    # -- Lifecycle ---------------------------------------------------------
    def purge(self) -> None:
        """Delete all of this session's data (relational rows + vector collection)."""
        db = self._session_factory()
        try:
            repository.delete_session(db, self._session_id)
            db.commit()
        finally:
            db.close()
        try:
            self._vector_store.delete_collection()
        except Exception:  # pragma: no cover - collection may not exist
            logger.exception("Failed to drop vector collection for session %s", self._session_id)

    def section_count(self) -> int:
        """Return how many sections this session currently has stored."""
        db = self._session_factory()
        try:
            return repository.count_sections(db, self._session_id)
        finally:
            db.close()

    # -- Health ------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the database answers a trivial query, else raise."""
        db = self._session_factory()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()


@dataclass(frozen=True, slots=True)
class _Providers:
    """Process-shared, stateless building blocks reused across all sessions."""

    settings: Settings
    embeddings: Any
    llm: Any


@lru_cache(maxsize=1)
def _get_providers() -> _Providers:
    """Build and cache the embedding model and LLM once per process.

    These are keyed only on configuration and carry no per-session state, so they are
    safely shared; only the vector collection and relational scoping vary by session.
    """
    from ragchat.rag.embeddings import build_embeddings

    settings = get_settings()
    embeddings = build_embeddings(settings)

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
    )
    return _Providers(settings=settings, embeddings=embeddings, llm=llm)


def build_session_service(session_id: str) -> RAGService:
    """Wire a fully configured, session-scoped :class:`RAGService`."""
    from ragchat.db.engine import get_session_factory, init_db
    from ragchat.rag.pipeline import build_rag_chain
    from ragchat.rag.vectorstore import build_vector_store, session_collection_name

    init_db()  # ensure the relational schema exists (idempotent)
    providers = _get_providers()
    vector_store = build_vector_store(
        providers.embeddings,
        providers.settings,
        collection_name=session_collection_name(session_id),
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": providers.settings.retriever_k})
    chain = build_rag_chain(retriever, providers.llm)

    return RAGService(
        session_id=session_id,
        session_factory=get_session_factory(),
        vector_store=vector_store,
        chain=chain,
        ttl_hours=providers.settings.session_ttl_hours,
    )
