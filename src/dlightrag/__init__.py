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

from dlightrag.access_control import AccessAction, AccessControl, access_control_from_config
from dlightrag.config import AccessControlConfig, AccessControlRuleConfig, DlightragConfig
from dlightrag.core.client_contracts import IngestDocument, IngestSpec

if TYPE_CHECKING:
    from dlightrag.core.retrieval.protocols import RetrievalResult
    from dlightrag.core.servicemanager import RAGServiceManager

__all__ = [
    "DlightragConfig",
    "IngestDocument",
    "IngestSpec",
    "AccessAction",
    "AccessControl",
    "AccessControlConfig",
    "AccessControlRuleConfig",
    "RAGServiceManager",
    "RetrievalResult",
    "__version__",
    "access_control_from_config",
]


def _lazy_imports():
    """Lazy imports for heavy modules — only loaded when accessed."""
    from dlightrag.core.retrieval.protocols import RetrievalResult
    from dlightrag.core.servicemanager import RAGServiceManager

    return RAGServiceManager, RetrievalResult


# Re-export for convenience (lazy to avoid heavy import on package load)
def __getattr__(name: str):
    if name in ("RAGServiceManager", "RetrievalResult"):
        RAGServiceManager, RetrievalResult = _lazy_imports()
        return {
            "RAGServiceManager": RAGServiceManager,
            "RetrievalResult": RetrievalResult,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
