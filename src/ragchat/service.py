"""Application service layer.

``RAGService`` is the single orchestration point the API and CLI both call. It owns two
flows:

* **ingest** — parse content, write the relational source of truth *and* the vector
  index in one consistent operation;
* **ask** — run the RAG chain and return an answer with its supporting sources.

All external dependencies (session factory, vector store, chain) are injected, so tests
construct a service backed by fakes and never touch a real database or LLM. Use
:func:`build_service` for the fully wired production instance.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from sqlalchemy import text
from sqlalchemy.orm import Session

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
    """Coordinates ingestion and question answering."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        vector_store: Any,  # PGVector in production; a test double in tests
        chain: Runnable[str, dict[str, Any]],
    ) -> None:
        self._session_factory = session_factory
        self._vector_store = vector_store
        self._chain = chain

    # -- Ingestion ---------------------------------------------------------
    def ingest_sections(self, sections: Sequence[Section]) -> IngestResult:
        """Atomically replace the corpus with ``sections`` (relational + vectors).

        Ordering guarantees consistency without a distributed transaction: relational
        rows are written but *not committed* until the vector index has been rebuilt
        successfully. If embedding/indexing fails, the relational write is rolled back
        and any partially written vectors are cleared, so the two stores never drift.
        """
        documents = [
            Document(page_content=s.as_document_text(), metadata={"title": s.title})
            for s in sections
        ]

        session = self._session_factory()
        try:
            count = repository.replace_all_sections(session, sections)
            # Rebuild the vector index from scratch to stay in sync with the table.
            self._vector_store.delete_collection()
            if documents:
                self._vector_store.add_documents(documents)
            session.commit()
            logger.info("Ingested %d sections into relational store and vector index", count)
            return IngestResult(sections_written=count)
        except Exception:
            session.rollback()
            # Best-effort cleanup of any vectors written before the failure.
            try:
                self._vector_store.delete_collection()
            except Exception:  # pragma: no cover - cleanup is best-effort
                logger.exception("Failed to clean up vector collection after ingest error")
            logger.exception("Ingestion failed; rolled back relational and vector writes")
            raise
        finally:
            session.close()

    def ingest_markdown_file(self, path: str | Path) -> IngestResult:
        """Parse a markdown file and ingest its sections."""
        sections = parse_markdown_file(path)
        logger.info("Parsed %d sections from %s", len(sections), path)
        return self.ingest_sections(sections)

    # -- Querying ----------------------------------------------------------
    def ask(self, question: str) -> AnswerResult:
        """Answer ``question`` using retrieval-augmented generation."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")

        result = self._chain.invoke(question)
        documents: list[Document] = result.get("documents", [])
        sources = [
            SourceDocument(content=doc.page_content, metadata=dict(doc.metadata))
            for doc in documents
        ]
        return AnswerResult(answer=result["answer"], sources=sources)

    # -- Health ------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the database answers a trivial query, else raise."""
        session = self._session_factory()
        try:
            session.execute(text("SELECT 1"))
            return True
        finally:
            session.close()


def build_service() -> RAGService:
    """Wire a fully configured :class:`RAGService` from application settings."""
    from ragchat.config import get_settings
    from ragchat.db.engine import get_session_factory
    from ragchat.rag.embeddings import build_embeddings
    from ragchat.rag.pipeline import build_rag_chain
    from ragchat.rag.vectorstore import build_vector_store

    settings = get_settings()
    embeddings = build_embeddings(settings)
    vector_store = build_vector_store(embeddings, settings)
    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
    )
    chain = build_rag_chain(retriever, llm)

    return RAGService(
        session_factory=get_session_factory(),
        vector_store=vector_store,
        chain=chain,
    )
