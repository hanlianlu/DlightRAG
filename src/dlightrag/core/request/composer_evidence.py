# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Composer-only evidence selection for temporary document attachments.

This module never imports workspace retrieval or storage. It selects evidence
only from current-turn and planner-selected historical Web attachments; the
LightRAG lane remains independently retrieved and ranked.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from dlightrag.core.request.attachments import ATTACHMENT_CONTEXT_TOKEN_LIMIT
from dlightrag.core.request.composer_lexical import rank_composer_bm25
from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.core.retrieval.protocols import ContextRow
from dlightrag.core.retrieval.rerank import rerank_with_fallback
from dlightrag.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

COMPOSER_CANDIDATE_LIMIT = 30

_STRUCTURE_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s+|\[(?:table|image|equation) name\]|\[table\])")
_STRUCTURAL_TYPES = frozenset({"table", "drawing", "image", "figure", "equation"})


class ComposerEvidenceSelector:
    """Select temporary attachment evidence without touching RAG results."""

    def __init__(
        self,
        *,
        attachment_token_limit: int = ATTACHMENT_CONTEXT_TOKEN_LIMIT,
        candidate_limit: int = COMPOSER_CANDIDATE_LIMIT,
    ) -> None:
        self._attachment_token_limit = max(0, attachment_token_limit)
        self._candidate_limit = max(1, candidate_limit)

    async def select(
        self,
        *,
        query: str,
        current_rows: list[ContextRow],
        history_rows: list[ContextRow],
        dense_rankings: list[ContextRow],
        retrieval_attachment_ids: set[str],
        rerank_func: Any | None,
    ) -> tuple[list[ContextRow], dict[str, Any]]:
        if not current_rows and not history_rows:
            return [], self._trace("empty", [], [])

        all_rows = [*current_rows, *history_rows]
        retrieval_rows = [
            row for row in all_rows if _row_attachment_id(row) in retrieval_attachment_ids
        ]
        full_pass_rows = [
            dict(row) for row in all_rows if _row_attachment_id(row) not in retrieval_attachment_ids
        ]
        if not retrieval_rows:
            full_pass_ids = {str(row.get("chunk_id") or "") for row in full_pass_rows}
            selected_current = [
                dict(row) for row in current_rows if str(row.get("chunk_id") or "") in full_pass_ids
            ]
            selected_history = [
                dict(row) for row in history_rows if str(row.get("chunk_id") or "") in full_pass_ids
            ]
            return [*selected_current, *selected_history], self._trace(
                "full",
                selected_current,
                selected_history,
            )

        lexical_error: str | None = None
        try:
            lexical = await asyncio.to_thread(
                rank_composer_bm25,
                query,
                retrieval_rows,
                limit=self._candidate_limit,
            )
        except Exception as exc:
            lexical = []
            lexical_error = type(exc).__name__
            logger.warning("Composer BM25 ranking failed", exc_info=True)
        structural = [dict(row) for row in retrieval_rows if _is_structural(row)]
        coverage = _coverage_ranking(retrieval_rows)
        dense = [
            dict(row)
            for row in dense_rankings
            if _row_attachment_id(row) in retrieval_attachment_ids
        ]
        candidates = rrf_fuse([lexical, structural, coverage, dense])[: self._candidate_limit]
        outcome = await rerank_with_fallback(
            query=query,
            chunks=candidates,
            top_k=len(candidates),
            rerank_func=rerank_func,
        )
        selected_retrieval = _pack_with_document_guarantees(
            outcome.chunks,
            original_rows=retrieval_rows,
            token_budget=self._attachment_token_limit,
        )

        selected_ids = {
            str(row.get("chunk_id") or "") for row in [*full_pass_rows, *selected_retrieval]
        }
        selected_current = [
            dict(row) for row in current_rows if str(row.get("chunk_id") or "") in selected_ids
        ]
        selected_history = [
            dict(row) for row in history_rows if str(row.get("chunk_id") or "") in selected_ids
        ]
        trace = self._trace(
            "retrieved",
            selected_current,
            selected_history,
        )
        trace.update(
            {
                "composer_evidence_candidates": len(candidates),
                "composer_evidence_reranked": outcome.reranked,
            }
        )
        if outcome.error_type:
            trace["composer_evidence_rerank_error"] = outcome.error_type
        if lexical_error:
            trace["composer_evidence_lexical_error"] = lexical_error
        return [*selected_current, *selected_history], trace

    @staticmethod
    def _trace(
        strategy: str,
        current_rows: list[ContextRow],
        history_rows: list[ContextRow],
    ) -> dict[str, Any]:
        return {
            "composer_evidence_strategy": strategy,
            "composer_evidence_current_chunks": len(current_rows),
            "composer_evidence_history_chunks": len(history_rows),
            "composer_evidence_tokens": _rows_tokens([*current_rows, *history_rows]),
            "composer_evidence_candidates": 0,
            "composer_evidence_reranked": False,
        }


def _row_tokens(row: ContextRow) -> int:
    return estimate_tokens(str(row.get("content") or ""))


def _rows_tokens(rows: list[ContextRow]) -> int:
    return sum(_row_tokens(row) for row in rows)


def _row_attachment_id(row: ContextRow) -> str:
    return str(row.get("full_doc_id") or row.get("reference_id") or "")


def _is_structural(row: ContextRow) -> bool:
    metadata = row.get("metadata") or {}
    sidecar_type = str(metadata.get("sidecar_type") or row.get("sidecar_type") or "")
    if sidecar_type.casefold() in _STRUCTURAL_TYPES:
        return True
    if any(
        metadata.get(key)
        for key in ("heading", "title", "caption", "parent_headings", "block_type")
    ):
        return True
    return bool(_STRUCTURE_RE.search(str(row.get("content") or "")))


def _coverage_ranking(rows: list[ContextRow]) -> list[ContextRow]:
    groups = _group_by_document(rows)

    coverage: list[ContextRow] = []
    for group in groups.values():
        positions = sorted({0, len(group) // 2, len(group) - 1})
        coverage.extend(dict(group[position]) for position in positions)
    return coverage


def _group_by_document(rows: list[ContextRow]) -> dict[str, list[ContextRow]]:
    groups: dict[str, list[ContextRow]] = {}
    for row in rows:
        groups.setdefault(str(row.get("reference_id") or ""), []).append(row)
    return groups


def _pack_with_document_guarantees(
    ranked_rows: list[ContextRow],
    *,
    original_rows: list[ContextRow],
    token_budget: int,
) -> list[ContextRow]:
    ranked_by_doc: dict[str, list[ContextRow]] = {}
    for row in ranked_rows:
        ranked_by_doc.setdefault(str(row.get("reference_id") or ""), []).append(row)

    doc_order: list[str] = []
    original_by_doc: dict[str, list[ContextRow]] = {}
    input_order: dict[str, int] = {}
    for index, row in enumerate(original_rows):
        chunk_id = str(row.get("chunk_id") or "")
        input_order[chunk_id] = index
        reference_id = str(row.get("reference_id") or "")
        if reference_id not in original_by_doc:
            original_by_doc[reference_id] = []
            doc_order.append(reference_id)
        original_by_doc[reference_id].append(row)

    selected_ids: set[str] = set()
    used_by_reference: dict[str, int] = {}

    def add(row: ContextRow) -> bool:
        chunk_id = str(row.get("chunk_id") or "")
        reference_id = str(row.get("reference_id") or "")
        tokens = _row_tokens(row)
        used = used_by_reference.get(reference_id, 0)
        if not chunk_id or chunk_id in selected_ids or used + tokens > token_budget:
            return False
        selected_ids.add(chunk_id)
        used_by_reference[reference_id] = used + tokens
        return True

    # Every referenced attachment gets its best-ranked chunk and, when room
    # permits, a separate structure/coverage representative.
    coverage_ids = {str(row.get("chunk_id") or "") for row in _coverage_ranking(original_rows)}
    for reference_id in doc_order:
        doc_ranked = ranked_by_doc.get(reference_id, [])
        if not any(add(row) for row in doc_ranked):
            next((row for row in original_by_doc[reference_id] if add(row)), None)
        for row in [*doc_ranked, *original_by_doc[reference_id]]:
            chunk_id = str(row.get("chunk_id") or "")
            if chunk_id not in selected_ids and (chunk_id in coverage_ids or _is_structural(row)):
                if add(row):
                    break

    for row in ranked_rows:
        add(row)

    return [
        dict(row)
        for row in sorted(
            (row for row in original_rows if str(row.get("chunk_id") or "") in selected_ids),
            key=lambda row: input_order[str(row.get("chunk_id") or "")],
        )
    ]


__all__ = [
    "ComposerEvidenceSelector",
]
