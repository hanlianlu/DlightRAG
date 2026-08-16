# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Provider-neutral checks for the LightRAG runtime surface RAG consumes."""

from typing import Any, ClassVar


class LightRAGContractGuard:
    """Fail fast when the public LightRAG runtime surface has drifted."""

    _REQUIRED_CALLABLES: ClassVar[tuple[str, ...]] = (
        "initialize_storages",
        "finalize_storages",
        "aquery_data",
        "apipeline_enqueue_documents",
        "apipeline_process_enqueue_documents",
    )
    _REQUIRED_ATTRIBUTES: ClassVar[tuple[str, ...]] = (
        "chunks_vdb",
        "text_chunks",
        "full_docs",
        "doc_status",
    )

    def __init__(self, lightrag: Any) -> None:
        self._lightrag = lightrag

    def verify(self) -> None:
        """Collect runtime-surface drift into one actionable failure."""
        errors = [
            f"LightRAG missing callable {name!r}"
            for name in self._REQUIRED_CALLABLES
            if not callable(getattr(self._lightrag, name, None))
        ]
        errors.extend(
            f"LightRAG missing attribute {name!r}"
            for name in self._REQUIRED_ATTRIBUTES
            if not hasattr(self._lightrag, name)
        )
        if errors:
            raise RuntimeError(
                f"LightRAG runtime contract check failed ({len(errors)} issue(s)):\n"
                + "\n".join(f"  - {error}" for error in errors)
            )


__all__ = ["LightRAGContractGuard"]
