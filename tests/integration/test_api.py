"""API tests driving the real FastAPI app with the service dependency overridden.

These are integration-style (full request/response through routing, validation, and
serialization) but self-contained: the injected service uses SQLite + fakes, so no keys
or live database are required.
"""

from __future__ import annotations


def test_ask_returns_answer_and_sources(api_client) -> None:
    response = api_client.post("/ask", json={"question": "What is a VPC?"})
    assert response.status_code == 200
    body = response.json()
    assert "VPC" in body["answer"]
    assert len(body["sources"]) == 2
    assert body["sources"][0]["metadata"]["title"] == "VPC"


def test_ask_rejects_blank_question_with_422(api_client) -> None:
    response = api_client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_requires_question_field(api_client) -> None:
    response = api_client.post("/ask", json={})
    assert response.status_code == 422


def test_health_reports_ok_when_db_reachable(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_db_unreachable(api_client) -> None:
    from ragchat.api.routes import get_db_session_factory

    def _broken_factory():
        def _session():
            raise RuntimeError("db down")

        return _session

    # Point the readiness probe at a factory whose sessions fail.
    api_client.app.dependency_overrides[get_db_session_factory] = _broken_factory
    response = api_client.get("/health")
    assert response.status_code == 503


def test_ingest_file_upload_succeeds(api_client) -> None:
    response = api_client.post(
        "/ingest/file",
        files={"file": ("notes.txt", b"Networking notes about VPCs and subnets.", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["sections_written"] >= 1


def test_ingest_file_rejects_unsupported_type(api_client) -> None:
    response = api_client.post(
        "/ingest/file",
        files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
    )
    assert response.status_code == 415


def test_index_serves_web_ui(api_client) -> None:
    response = api_client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ragchat" in response.text


def test_openapi_schema_is_served(api_client) -> None:
    response = api_client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/ask" in paths
    assert "/ingest/file" in paths
