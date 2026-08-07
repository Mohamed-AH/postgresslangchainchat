"""Tests for the RAG service orchestration (no keys, no live DB)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sqlalchemy.orm import Session

from ragchat.db import repository
from ragchat.ingestion.parser import Section
from ragchat.service import RAGService

SECTIONS = [
    Section(title="VPC", content="A private, isolated cloud network."),
    Section(title="Subnets", content="Ranges within a VPC."),
]


def test_ask_returns_answer_and_sources(rag_service: RAGService) -> None:
    result = rag_service.ask("What is a VPC?")
    assert "VPC" in result.answer
    assert len(result.sources) == 2
    assert result.sources[0].metadata["title"] == "VPC"


def test_ask_rejects_empty_question(rag_service: RAGService) -> None:
    with pytest.raises(ValueError):
        rag_service.ask("   ")


def test_ingest_writes_relational_and_vectors(
    rag_service: RAGService,
    fake_vector_store,
    session_factory: Callable[[], Session],
) -> None:
    result = rag_service.ingest_sections(SECTIONS)

    assert result.sections_written == 2
    # Relational store persisted and committed.
    with session_factory() as session:
        assert repository.count_sections(session) == 2
    # Vector index rebuilt: cleared once, then documents added.
    assert fake_vector_store.delete_collection_calls == 1
    assert len(fake_vector_store.documents) == 2


def test_ingest_is_atomic_on_vector_failure(
    rag_service: RAGService,
    fake_vector_store,
    session_factory: Callable[[], Session],
) -> None:
    fake_vector_store.fail_on_add = True

    with pytest.raises(RuntimeError):
        rag_service.ingest_sections(SECTIONS)

    # Relational write rolled back — no orphan rows without matching embeddings.
    with session_factory() as session:
        assert repository.count_sections(session) == 0


def test_health_check_passes_against_live_session(rag_service: RAGService) -> None:
    assert rag_service.health_check() is True
