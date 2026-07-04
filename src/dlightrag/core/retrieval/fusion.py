# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval score fusion."""

from dlightrag.core.retrieval.protocols import ContextRow


def _chunk_id(row: ContextRow) -> str | None:
    value = row.get("chunk_id") or row.get("id")
    return str(value) if value is not None else None


def rrf_fuse(rankings: list[list[ContextRow]], *, k: int = 60) -> list[ContextRow]:
    """Fuse ranked chunk lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    best: dict[str, ContextRow] = {}

    for ranking in rankings:
        for rank, row in enumerate(ranking):
            cid = _chunk_id(row)
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(cid, dict(row))

    fused: list[ContextRow] = []
    for cid, row in best.items():
        row["chunk_id"] = cid
        row["score"] = scores[cid]
        fused.append(row)
    return sorted(fused, key=lambda row: row["score"], reverse=True)


def format_bm25_top(chunks: list[ContextRow], *, limit: int = 3) -> str:
    parts: list[str] = []
    for chunk in chunks[:limit]:
        chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "?")
        profile = str(chunk.get("bm25_profile") or "?")
        score = chunk.get("score")
        if score is None:
            score_text = "?"
        else:
            try:
                score_text = f"{float(score):.3f}"
            except TypeError, ValueError:
                score_text = "?"
        parts.append(f"{chunk_id}:{profile}:{score_text}")
    if len(chunks) > limit:
        parts.append(f"+{len(chunks) - limit}")
    return ",".join(parts) if parts else "none"


def dedup_chunks_by_content(
    chunks: list[ContextRow],
    *,
    prefix_len: int = 200,
) -> list[ContextRow]:
    """Remove chunks whose content prefix matches an earlier chunk.

    Keeps the first occurrence (highest fusion score, since chunks are
    pre-sorted by score descending). Two chunks with different IDs but
    identical first *prefix_len* characters are considered duplicates.
    """
    seen: set[str] = set()
    result: list[ContextRow] = []
    for chunk in chunks:
        content = str(chunk.get("content", ""))
        key = content[:prefix_len] if len(content) >= prefix_len else content
        if not key:
            result.append(chunk)
            continue
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result
