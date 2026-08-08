"""HTTP routes, session resolution, and dependency providers."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, Response, UploadFile, status
from sqlalchemy import text
from sqlalchemy.orm import Session as DbSession

from ragchat.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestResponse,
    SourceSchema,
)
from ragchat.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    TooManySectionsError,
    UnsupportedFileTypeError,
)
from ragchat.service import RAGService, build_session_service

logger = logging.getLogger(__name__)

router = APIRouter()

SESSION_COOKIE = "sid"
_SESSION_TTL_SECONDS = 24 * 60 * 60
# Session ids become part of a pgvector collection name, so only accept the exact
# shape we mint (a uuid4 hex). Anything else is replaced with a fresh id.
_VALID_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


def get_session_id(request: Request, response: Response) -> str:
    """Resolve the caller's session id from a cookie, minting one if absent/invalid."""
    sid = request.cookies.get(SESSION_COOKIE)
    if sid is None or not _VALID_SESSION_ID.match(sid):
        sid = uuid4().hex
        response.set_cookie(
            SESSION_COOKIE,
            sid,
            max_age=_SESSION_TTL_SECONDS,
            httponly=True,
            samesite="lax",
        )
    return sid


def get_service(session_id: str = Depends(get_session_id)) -> RAGService:
    """Build a session-scoped service. Overridden in tests to inject a fake."""
    return build_session_service(session_id)


def get_db_session_factory() -> Callable[[], DbSession]:
    """Provide the DB session factory for the readiness probe (no LLM providers).

    Kept separate from :func:`get_service` so readiness never depends on the model layer.
    Overridden in tests.
    """
    from ragchat.db.engine import get_session_factory

    return get_session_factory()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(
    session_factory: Callable[[], DbSession] = Depends(get_db_session_factory),
) -> HealthResponse:
    """Readiness probe: verifies the database is reachable with ``SELECT 1``.

    Returns 503 (not 200) while the database is unavailable, so orchestrators can gate
    traffic until the app is genuinely ready.
    """
    try:
        db = session_factory()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as exc:
        logger.warning("Health check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database is not reachable",
        ) from exc
    return HealthResponse(status="ok")


@router.post("/ask", response_model=AskResponse, tags=["qa"])
def ask(payload: AskRequest, service: RAGService = Depends(get_service)) -> AskResponse:
    """Answer a question using retrieval-augmented generation over the caller's session."""
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


@router.post("/ingest/file", response_model=IngestResponse, tags=["ingest"])
async def ingest_file(
    file: UploadFile = File(...),
    service: RAGService = Depends(get_service),
) -> IngestResponse:
    """Upload a file (.md/.txt/.pdf/.docx) and (re)build the caller's knowledge base.

    The read is bounded by the configured size cap, and all caps are enforced before any
    embedding work happens, so an oversized or unsupported upload never incurs cost.
    """
    limit = service.max_upload_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {limit}-byte limit.",
        )
    try:
        result = service.ingest_upload(file.filename or "upload", data)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)
        ) from exc
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)
        ) from exc
    except (TooManySectionsError, EmptyDocumentError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    return IngestResponse(sections_written=result.sections_written)
