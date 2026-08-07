"""HTTP routes and the service dependency provider."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ragchat.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceSchema,
)
from ragchat.service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter()


def get_service(request: Request) -> RAGService:
    """Provide the shared :class:`RAGService` created during app startup.

    Overridden in tests via ``app.dependency_overrides`` to inject a fake service.
    """
    service: RAGService | None = getattr(request.app.state, "service", None)
    if service is None:  # pragma: no cover - guards against misconfiguration
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not initialised",
        )
    return service


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(service: RAGService = Depends(get_service)) -> HealthResponse:
    """Readiness probe: verifies the database is reachable with ``SELECT 1``.

    Returns 503 (not 200) while the database is unavailable, so orchestrators can gate
    traffic until the app is genuinely ready.
    """
    try:
        service.health_check()
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not reachable",
        ) from exc
    return HealthResponse(status="ok")


@router.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(payload: AskRequest, service: RAGService = Depends(get_service)) -> AskResponse:
    """Answer a question using retrieval-augmented generation."""
    try:
        result = service.ask(payload.question)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return AskResponse(
        answer=result.answer,
        sources=[SourceSchema(content=s.content, metadata=s.metadata) for s in result.sources],
    )


@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
def ingest(payload: IngestRequest, service: RAGService = Depends(get_service)) -> IngestResponse:
    """Re-ingest a markdown file available on the server into the knowledge base."""
    try:
        result = service.ingest_markdown_file(payload.path)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"File not found: {payload.path}"
        ) from exc
    return IngestResponse(sections_written=result.sections_written)
