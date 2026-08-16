# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""DlightRAG: PostgreSQL-backed multimodal RAG built on LightRAG main.

Exposable as both a REST API (bulk ingestion) and MCP server (agent integration).
"""

from typing import TYPE_CHECKING

try:
    from importlib.metadata import version as _version

    __version__ = _version("dlightrag")
except Exception:
    __version__ = "0.0.0"
__maintainer__ = "HanlianLyu"
__credits__ = ["hllyu"]

from dlightrag.config import DlightragConfig
from dlightrag.core.client_contracts import IngestSpec, QueryImage

if TYPE_CHECKING:
    from dlightrag.core.answer.errors import (
        ANSWER_IMAGE_CAPABILITY_UNKNOWN,
        ANSWER_STREAM_FAILED,
        CURRENT_IMAGE_LIMIT_EXCEEDED,
        CURRENT_IMAGES_UNSUPPORTED,
        AnswerImageError,
        classify_answer_error,
    )
    from dlightrag.core.servicemanager import RAGServiceManager

_ANSWER_EXPORTS = frozenset(
    {
        "ANSWER_IMAGE_CAPABILITY_UNKNOWN",
        "ANSWER_STREAM_FAILED",
        "CURRENT_IMAGE_LIMIT_EXCEEDED",
        "CURRENT_IMAGES_UNSUPPORTED",
        "AnswerImageError",
        "classify_answer_error",
    }
)

__all__ = [
    "DlightragConfig",
    "IngestSpec",
    "QueryImage",
    "RAGServiceManager",
    "AnswerImageError",
    "ANSWER_IMAGE_CAPABILITY_UNKNOWN",
    "ANSWER_STREAM_FAILED",
    "CURRENT_IMAGES_UNSUPPORTED",
    "CURRENT_IMAGE_LIMIT_EXCEEDED",
    "classify_answer_error",
    "__version__",
]


def _lazy_imports():
    """Lazy imports for heavy modules — only loaded when accessed."""
    from dlightrag.core.servicemanager import RAGServiceManager

    return RAGServiceManager


def _lazy_answer_exports():
    from dlightrag.core.answer import errors

    return {name: getattr(errors, name) for name in _ANSWER_EXPORTS}


# Re-export for convenience (lazy to avoid heavy import on package load)
def __getattr__(name: str):
    if name in _ANSWER_EXPORTS:
        return _lazy_answer_exports()[name]
    if name == "RAGServiceManager":
        return _lazy_imports()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
