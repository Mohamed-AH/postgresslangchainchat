"""Embedding-model factory.

Isolated behind a single function so the concrete provider (Cohere today) is an
implementation detail. Tests substitute a deterministic fake via the service layer's
dependency injection, so no network or API key is needed to exercise retrieval logic.
"""

from __future__ import annotations

from typing import cast

from langchain_core.embeddings import Embeddings

from ragchat.config import Settings, get_settings


def build_embeddings(settings: Settings | None = None) -> Embeddings:
    """Construct the Cohere embedding model from settings."""
    settings = settings or get_settings()
    # Imported lazily so importing this module (e.g. in unit tests) never requires the
    # optional provider SDK to be installed or configured.
    from langchain_cohere import CohereEmbeddings

    return cast(
        Embeddings,
        CohereEmbeddings(
            cohere_api_key=settings.cohere_api_key.get_secret_value(),
            model=settings.embedding_model,
        ),
    )
