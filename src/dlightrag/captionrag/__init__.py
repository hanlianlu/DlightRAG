# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Caption-based RAG mode — retrieval, ingestion, and document processing."""

from dlightrag.captionrag.pipeline import (
    IngestionCancelledError,
    IngestionPipeline,
    IngestionResult,
)
from dlightrag.captionrag.retrieval import RetrievalEngine

__all__ = [
    "IngestionCancelledError",
    "IngestionPipeline",
    "IngestionResult",
    "RetrievalEngine",
]
