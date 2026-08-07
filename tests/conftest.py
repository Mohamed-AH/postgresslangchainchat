"""Shared test fixtures and fakes.

The whole suite runs with **no API keys and no live database**:

* the relational layer is exercised against real SQLite in-memory (so models and the
  repository are genuinely tested), and
* embeddings / LLM / pgvector are replaced with deterministic fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ragchat.db.models import Base
from ragchat.rag.pipeline import build_rag_chain
from ragchat.service import RAGService

# --- Fakes ----------------------------------------------------------------


class FakeRetriever(BaseRetriever):
    """Retriever that always returns a fixed set of documents."""

    documents: list[Document]

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.documents


class FakeVectorStore:
    """Minimal stand-in for ``PGVector`` that records ingestion calls."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.delete_collection_calls = 0
        self.create_collection_calls = 0
        self.fail_on_add = False

    def delete_collection(self) -> None:
        self.delete_collection_calls += 1
        self.documents = []

    def create_collection(self) -> None:
        self.create_collection_calls += 1

    def add_documents(self, documents: list[Document]) -> None:
        if self.fail_on_add:
            raise RuntimeError("simulated embedding failure")
        self.documents.extend(documents)


# --- Fixtures -------------------------------------------------------------


@pytest.fixture
def session_factory() -> Callable[[], Session]:
    """A real SQLite in-memory session factory (shared connection)."""
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="A VPC is a private, isolated cloud network.",
            metadata={"title": "VPC"},
        ),
        Document(
            page_content="Subnets divide a VPC into smaller ranges.",
            metadata={"title": "Subnets"},
        ),
    ]


@pytest.fixture
def fake_vector_store() -> FakeVectorStore:
    return FakeVectorStore()


@pytest.fixture
def rag_service(
    session_factory: Callable[[], Session],
    fake_vector_store: FakeVectorStore,
    sample_documents: list[Document],
) -> RAGService:
    """A service wired with SQLite + fake vector store + a real LCEL chain over fakes."""
    retriever = FakeRetriever(documents=sample_documents)
    llm = FakeListChatModel(responses=["A VPC is a private, isolated network in the cloud."])
    chain = build_rag_chain(retriever, llm)
    return RAGService(
        session_factory=session_factory,
        vector_store=fake_vector_store,
        chain=chain,
    )


@pytest.fixture
def api_client(rag_service: RAGService) -> Iterator[TestClient]:
    """A FastAPI TestClient with the service dependency overridden by ``rag_service``."""
    from ragchat.api.app import create_app
    from ragchat.api.routes import get_service

    app = create_app()
    app.dependency_overrides[get_service] = lambda: rag_service
    # Construct the client WITHOUT the `with` block: entering it would run the real
    # lifespan (building production services that need API keys). The overridden
    # dependency supplies the fake service, so startup isn't needed.
    yield TestClient(app)
