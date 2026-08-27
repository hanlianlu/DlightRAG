# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Inline Retrieval use case and caller contracts."""

from .service import (
    CorpusUnavailableError,
    ProjectedRetrieval,
    QueryImagePreparer,
    RetrievalService,
    RetrievalSettings,
    RetrievalTimeoutError,
    RetrieveProjection,
    RetrieveRequest,
    RetrieveResponse,
    SchemaLookup,
)

__all__ = [
    "CorpusUnavailableError",
    "ProjectedRetrieval",
    "QueryImagePreparer",
    "RetrievalService",
    "RetrievalSettings",
    "RetrievalTimeoutError",
    "RetrieveProjection",
    "RetrieveRequest",
    "RetrieveResponse",
    "SchemaLookup",
]
