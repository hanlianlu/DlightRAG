# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval score fusion."""

from __future__ import annotations

from typing import Any


def _chunk_id(row: dict[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("id"))


def rrf_fuse(rankings: list[list[dict[str, Any]]], *, k: int = 60) -> list[dict[str, Any]]:
    """Fuse ranked chunk lists with Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}

    for ranking in rankings:
        for rank, row in enumerate(ranking):
            cid = _chunk_id(row)
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(cid, dict(row))

    fused: list[dict[str, Any]] = []
    for cid, row in best.items():
        row["chunk_id"] = cid
        row["score"] = scores[cid]
        fused.append(row)
    return sorted(fused, key=lambda row: row["score"], reverse=True)
