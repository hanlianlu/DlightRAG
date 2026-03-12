# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared retrieval types and utilities."""

from dlightrag.core.retrieval.path_resolver import PathResolver
from dlightrag.core.retrieval.protocols import (
    ChunkContext,
    EntityContext,
    RelationshipContext,
    RetrievalBackend,
    RetrievalContexts,
    RetrievalResult,
)

__all__ = [
    "ChunkContext",
    "EntityContext",
    "PathResolver",
    "RelationshipContext",
    "RetrievalBackend",
    "RetrievalContexts",
    "RetrievalResult",
]
