# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG: PostgreSQL-backed multimodal RAG built on LightRAG main.

Exposable as both a REST API (bulk ingestion) and MCP server (agent integration).
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag")
except Exception:
    __version__ = "0.0.0"
__maintainer__ = "HanlianLyu"
__credits__ = ["hllyu"]

from dlightrag.config import DlightragConfig

__all__ = [
    "DlightragConfig",
    "__version__",
]


def _lazy_imports():
    """Lazy imports for heavy modules — only loaded when accessed."""
    from dlightrag.core.retrieval.protocols import RetrievalResult
    from dlightrag.core.service import RAGService
    from dlightrag.core.servicemanager import RAGServiceManager

    return RAGService, RAGServiceManager, RetrievalResult


# Re-export for convenience (lazy to avoid heavy import on package load)
def __getattr__(name: str):
    if name in ("RAGService", "RAGServiceManager", "RetrievalResult"):
        RAGService, RAGServiceManager, RetrievalResult = _lazy_imports()
        return {
            "RAGService": RAGService,
            "RAGServiceManager": RAGServiceManager,
            "RetrievalResult": RetrievalResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
