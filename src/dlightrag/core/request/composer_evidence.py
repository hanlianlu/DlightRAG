# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Composer-only evidence selection for temporary document attachments.

This module never imports workspace retrieval or storage. It selects evidence
only from current-turn and planner-selected historical Web attachments; the
LightRAG lane remains independently retrieved and ranked.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
from collections.abc import Awaitable
from typing import Any, cast

from dlightrag.core.request.composer_lexical import rank_composer_bm25
from dlightrag.core.retrieval.fusion import rrf_fuse
from dlightrag.core.retrieval.protocols import ContextRow
from dlightrag.utils.tokens import estimate_tokens

logger = logging.getLogger(__name__)

COMPOSER_FULL_PASS_TOKENS = 24_576
COMPOSER_CURRENT_TARGET_TOKENS = 32_768
COMPOSER_HISTORY_TARGET_TOKENS = 12_288
COMPOSER_TOTAL_TOKENS = 45_056
COMPOSER_CANDIDATE_LIMIT = 30

_STRUCTURE_RE = re.compile(r"(?im)^\s*(?:#{1,6}\s+|\[(?:table|image|equation) name\]|\[table\])")
_STRUCTURAL_TYPES = frozenset({"table", "drawing", "image", "figure", "equation"})


class ComposerEvidenceSelector:
    """Select temporary attachment evidence without touching RAG results."""

    def __init__(
        self,
        *,
        embedding_func: Any | None = None,
        embedding_owner: Any | None = None,
        rerank_func: Any | None = None,
        full_pass_tokens: int = COMPOSER_FULL_PASS_TOKENS,
        current_target_tokens: int = COMPOSER_CURRENT_TARGET_TOKENS,
        history_target_tokens: int = COMPOSER_HISTORY_TARGET_TOKENS,
        total_tokens: int = COMPOSER_TOTAL_TOKENS,
        candidate_limit: int = COMPOSER_CANDIDATE_LIMIT,
    ) -> None:
        self._embedding_func = embedding_func
        self._embedding_owner = embedding_owner
        self._rerank_func = rerank_func
        self._full_pass_tokens = max(0, full_pass_tokens)
        self._current_target_tokens = max(0, current_target_tokens)
        self._history_target_tokens = max(0, history_target_tokens)
        self._total_tokens = max(0, total_tokens)
        self._candidate_limit = max(1, candidate_limit)

    async def aclose(self) -> None:
        """Close only model clients created for the Web Composer lane."""
        seen: set[int] = set()
        for resource in (self._embedding_owner, self._rerank_func):
            if resource is None or id(resource) in seen:
                continue
            seen.add(id(resource))
            close = getattr(resource, "aclose", None)
            if callable(close):
                result = close()
                if inspect.isawaitable(result):
                    await cast(Awaitable[Any], result)

    async def select(
        self,
        *,
        query: str,
        current_rows: list[ContextRow],
        history_rows: list[ContextRow],
    ) -> tuple[list[ContextRow], dict[str, Any]]:
        if not current_rows and not history_rows:
            return [], self._trace("empty", [], [])

        current_tokens = _rows_tokens(current_rows)
        history_tokens = _rows_tokens(history_rows)
        current_budget, history_budget = _allocate_lane_budgets(
            current_tokens,
            history_tokens,
            current_target=self._current_target_tokens,
            history_target=self._history_target_tokens,
            total_budget=self._total_tokens,
        )
        if (
            current_tokens + history_tokens <= self._full_pass_tokens
            and current_tokens <= current_budget
            and history_tokens <= history_budget
        ):
            selected_current = [dict(row) for row in current_rows]
            selected_history = [dict(row) for row in history_rows]
            return [*selected_current, *selected_history], self._trace(
                "full",
                selected_current,
                selected_history,
            )

        selected_current, current_trace = await self._select_lane(
            query=query,
            rows=current_rows,
            token_budget=current_budget,
        )
        selected_history, history_trace = await self._select_lane(
            query=query,
            rows=history_rows,
            token_budget=history_budget,
        )
        trace = self._trace(
            "retrieved",
            selected_current,
            selected_history,
        )
        trace.update(
            {
                "composer_evidence_candidates": (
                    current_trace["candidate_count"] + history_trace["candidate_count"]
                ),
                "composer_evidence_reranked": bool(
                    current_trace["reranked"] or history_trace["reranked"]
                ),
            }
        )
        rerank_error = current_trace.get("rerank_error") or history_trace.get("rerank_error")
        if rerank_error:
            trace["composer_evidence_rerank_error"] = rerank_error
        embedding_error = current_trace.get("embedding_error") or history_trace.get(
            "embedding_error"
        )
        if embedding_error:
            trace["composer_evidence_embedding_error"] = embedding_error
        lexical_error = current_trace.get("lexical_error") or history_trace.get("lexical_error")
        if lexical_error:
            trace["composer_evidence_lexical_error"] = lexical_error
        return [*selected_current, *selected_history], trace

    async def _select_lane(
        self,
        *,
        query: str,
        rows: list[ContextRow],
        token_budget: int,
    ) -> tuple[list[ContextRow], dict[str, Any]]:
        if not rows or token_budget <= 0:
            return [], {"candidate_count": 0, "reranked": False}

        groups = _group_by_document(rows)
        short_ids = {
            reference_id
            for reference_id, group in groups.items()
            if _rows_tokens(group) <= self._full_pass_tokens
        }
        short_rows = [dict(row) for row in rows if str(row.get("reference_id") or "") in short_ids]
        long_rows = [row for row in rows if str(row.get("reference_id") or "") not in short_ids]
        if not long_rows:
            return short_rows, {
                "candidate_count": len(short_rows),
                "reranked": False,
            }
        long_budget = max(0, token_budget - _rows_tokens(short_rows))

        lexical_error: str | None = None
        try:
            lexical = await asyncio.to_thread(
                rank_composer_bm25,
                query,
                long_rows,
                limit=self._candidate_limit,
            )
        except Exception as exc:
            lexical = []
            lexical_error = type(exc).__name__
            logger.warning("Composer BM25 ranking failed", exc_info=True)
        structural = [dict(row) for row in long_rows if _is_structural(row)]
        coverage = _coverage_ranking(long_rows)
        rankings: list[list[ContextRow]] = [lexical, structural, coverage]
        embedding_error: str | None = None
        if self._embedding_func is not None and long_rows:
            try:
                rankings.append(await self._dense_ranking(query, long_rows))
            except Exception as exc:
                embedding_error = type(exc).__name__
                logger.warning("Composer embedding ranking failed", exc_info=True)

        candidates = rrf_fuse(rankings)[: self._candidate_limit]
        ranked = candidates
        reranked = False
        rerank_error: str | None = None
        if self._rerank_func is not None and candidates:
            try:
                ranked = await self._rerank_func(
                    query=query,
                    chunks=candidates,
                    top_k=len(candidates),
                )
                reranked = True
            except Exception as exc:
                rerank_error = type(exc).__name__
                logger.warning("Composer rerank failed; using local fusion", exc_info=True)

        selected = _pack_with_document_guarantees(
            ranked,
            original_rows=long_rows,
            token_budget=long_budget,
        )
        trace: dict[str, Any] = {
            "candidate_count": len(candidates),
            "reranked": reranked,
        }
        if rerank_error:
            trace["rerank_error"] = rerank_error
        if embedding_error:
            trace["embedding_error"] = embedding_error
        if lexical_error:
            trace["lexical_error"] = lexical_error
        selected_ids = {str(row.get("chunk_id") or "") for row in [*short_rows, *selected]}
        ordered = [dict(row) for row in rows if str(row.get("chunk_id") or "") in selected_ids]
        return ordered, trace

    async def _dense_ranking(
        self,
        query: str,
        rows: list[ContextRow],
    ) -> list[ContextRow]:
        import numpy as np

        embedding_func = self._embedding_func
        if embedding_func is None:
            return []
        query_vector = await embedding_func([query], context="query")
        document_vectors = await embedding_func(
            [str(row.get("content") or "") for row in rows],
            context="document",
        )
        query_array = np.asarray(query_vector, dtype=float)[0]
        document_array = np.asarray(document_vectors, dtype=float)
        query_norm = float(np.linalg.norm(query_array))
        document_norms = np.linalg.norm(document_array, axis=1)
        denominator = document_norms * query_norm
        scores = np.divide(
            document_array @ query_array,
            denominator,
            out=np.zeros_like(document_norms, dtype=float),
            where=denominator != 0,
        )
        ranked_indices = np.argsort(-scores, kind="stable")
        return [dict(rows[int(index)]) for index in ranked_indices]

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
        }


def _row_tokens(row: ContextRow) -> int:
    return estimate_tokens(str(row.get("content") or ""))


def _rows_tokens(rows: list[ContextRow]) -> int:
    return sum(_row_tokens(row) for row in rows)


def _allocate_lane_budgets(
    current_demand: int,
    history_demand: int,
    *,
    current_target: int,
    history_target: int,
    total_budget: int,
) -> tuple[int, int]:
    current = min(current_demand, current_target)
    history = min(history_demand, history_target)
    remaining = max(0, total_budget - current - history)
    current_extra = min(max(0, current_demand - current), remaining)
    current += current_extra
    remaining -= current_extra
    history += min(max(0, history_demand - history), remaining)
    return current, history


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
    used = 0

    def add(row: ContextRow) -> None:
        nonlocal used
        chunk_id = str(row.get("chunk_id") or "")
        tokens = _row_tokens(row)
        if not chunk_id or chunk_id in selected_ids or used + tokens > token_budget:
            return
        selected_ids.add(chunk_id)
        used += tokens

    # Every referenced attachment gets its best-ranked chunk and, when room
    # permits, a separate structure/coverage representative.
    coverage_ids = {str(row.get("chunk_id") or "") for row in _coverage_ranking(original_rows)}
    for reference_id in doc_order:
        doc_ranked = ranked_by_doc.get(reference_id) or original_by_doc[reference_id]
        add(doc_ranked[0])
        representative = next(
            (
                row
                for row in doc_ranked
                if str(row.get("chunk_id") or "") in coverage_ids or _is_structural(row)
            ),
            None,
        )
        if representative is not None:
            add(representative)

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
    "COMPOSER_CURRENT_TARGET_TOKENS",
    "COMPOSER_FULL_PASS_TOKENS",
    "COMPOSER_HISTORY_TARGET_TOKENS",
    "COMPOSER_TOTAL_TOKENS",
    "ComposerEvidenceSelector",
]
