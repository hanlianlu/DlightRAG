"""Citation index — maps [ref_id-chunk_idx] to actual chunk_ids."""

from __future__ import annotations

import logging
from typing import Any

from .utils import split_source_ids

logger = logging.getLogger(__name__)


class CitationIndexer:
    """Bidirectional index: (ref_id, chunk_idx) <-> chunk_id."""

    def __init__(self) -> None:
        self._index: dict[str, dict[str, int]] = {}
        self._reverse: dict[str, dict[int, str]] = {}

    def build_index(self, contexts: list[dict[str, Any]]) -> None:
        valid_chunk_ids: set[str] = set()
        for ctx in contexts:
            cid = ctx.get("chunk_id")
            if cid and ctx.get("content"):
                valid_chunk_ids.add(cid)

        ref_chunks: dict[str, list[str]] = {}
        for ctx in contexts:
            ref_id = str(ctx.get("reference_id", ""))
            if not ref_id:
                continue
            chunk_id = ctx.get("chunk_id")
            if chunk_id and chunk_id in valid_chunk_ids:
                ref_chunks.setdefault(ref_id, [])
                if chunk_id not in ref_chunks[ref_id]:
                    ref_chunks[ref_id].append(chunk_id)
            else:
                source_id = ctx.get("source_id")
                if source_id:
                    for sid in split_source_ids(source_id):
                        if sid in valid_chunk_ids:
                            ref_chunks.setdefault(ref_id, [])
                            if sid not in ref_chunks[ref_id]:
                                ref_chunks[ref_id].append(sid)

        for ref_id, chunk_ids in ref_chunks.items():
            self._index[ref_id] = {}
            self._reverse[ref_id] = {}
            for idx, cid in enumerate(chunk_ids, start=1):
                self._index[ref_id][cid] = idx
                self._reverse[ref_id][idx] = cid

    def get_chunk_idx(self, ref_id: str | int, chunk_id: str) -> int | None:
        return self._index.get(str(ref_id), {}).get(chunk_id)

    def get_chunk_id(self, ref_id: str | int, chunk_idx: int) -> str | None:
        return self._reverse.get(str(ref_id), {}).get(chunk_idx)

    def get_max_chunk_idx(self, ref_id: str | int) -> int:
        reverse = self._reverse.get(str(ref_id), {})
        return max(reverse.keys()) if reverse else 0

    def inject_chunk_idx(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for ctx in contexts:
            ctx = dict(ctx)
            ref_id = str(ctx.get("reference_id", ""))
            chunk_id = ctx.get("chunk_id")
            if chunk_id and ref_id:
                idx = self.get_chunk_idx(ref_id, chunk_id)
                if idx is not None:
                    ctx["chunk_idx"] = idx
            source_id = ctx.get("source_id")
            if source_id and ref_id:
                idxs = []
                for sid in split_source_ids(source_id):
                    idx = self.get_chunk_idx(ref_id, sid)
                    if idx is not None:
                        idxs.append(idx)
                if idxs:
                    ctx["chunk_idxs"] = idxs
            enriched.append(ctx)
        return enriched

    @staticmethod
    def format_citation(ref_id: str | int, chunk_idx: int) -> str:
        return f"[{ref_id}-{chunk_idx}]"


def build_citation_index(
    contexts: list[dict[str, Any]],
) -> tuple[CitationIndexer, list[dict[str, Any]]]:
    indexer = CitationIndexer()
    indexer.build_index(contexts)
    enriched = indexer.inject_chunk_idx(contexts)
    return indexer, enriched
