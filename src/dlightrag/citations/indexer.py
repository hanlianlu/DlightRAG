"""Citation index — maps [ref_id-chunk_idx] to actual chunk_ids."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .utils import split_source_ids

logger = logging.getLogger(__name__)


class CitationIndexer:
    """Bidirectional index: (ref_id, chunk_idx) <-> chunk_id.

    Also stores per-chunk metadata (file_path, page_idx) so that
    :meth:`format_reference_list` can render the hierarchical reference
    list for VLM prompts — ensuring a single source of truth for the
    ``[ref_id-chunk_idx]`` mapping.
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, int]] = {}
        self._reverse: dict[str, dict[int, str]] = {}
        # chunk_id -> (ref_id, chunk_idx) for source_id lookups
        self._chunk_to_ref: dict[str, tuple[str, int]] = {}
        # Per-ref doc metadata: ref_id -> file_name
        self._doc_names: dict[str, str] = {}
        # Per-chunk metadata: chunk_id -> {"page_idx": int, ...}
        self._chunk_meta: dict[str, dict[str, Any]] = {}

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
                    # Store metadata on first encounter
                    self._chunk_meta[chunk_id] = {
                        "page_idx": ctx.get("page_idx", 0),
                    }
                # Store doc-level metadata (first chunk wins)
                if ref_id not in self._doc_names:
                    fp = ctx.get("file_path", "")
                    self._doc_names[ref_id] = Path(fp).name if fp else f"Source {ref_id}"
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
                self._chunk_to_ref[cid] = (ref_id, idx)

    def get_chunk_idx(self, ref_id: str | int, chunk_id: str) -> int | None:
        return self._index.get(str(ref_id), {}).get(chunk_id)

    def get_chunk_id(self, ref_id: str | int, chunk_idx: int) -> str | None:
        return self._reverse.get(str(ref_id), {}).get(chunk_idx)

    def get_max_chunk_idx(self, ref_id: str | int) -> int:
        reverse = self._reverse.get(str(ref_id), {})
        return max(reverse.keys()) if reverse else 0

    def get_citation_tags(self, source_id: str | None) -> list[str]:
        """Return citation tags for a source_id (single or comma-separated chunk_ids).

        Example: source_id="c1,c2" → ["[1-1]", "[1-2]"]
        """
        if not source_id:
            return []
        tags: list[str] = []
        seen: set[str] = set()
        for cid in split_source_ids(source_id):
            ref_info = self._chunk_to_ref.get(cid)
            if ref_info:
                tag = f"[{ref_info[0]}-{ref_info[1]}]"
                if tag not in seen:
                    seen.add(tag)
                    tags.append(tag)
        return tags

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

    def format_reference_list(self) -> str:
        """Render a hierarchical reference list for VLM prompts.

        Uses the same index built by :meth:`build_index`, guaranteeing
        that ``[n-m]`` markers in the prompt map to the same chunk_ids
        that :meth:`get_chunk_id` resolves during citation processing.

        Example output::

            [1] quarterly_report.pdf
              [1-1] Page 3
              [1-2] Page 7
            [2] spec.pdf
              [2-1] Page 1
        """
        if not self._reverse:
            return "No reference documents available."

        lines: list[str] = []
        for ref_id in self._reverse:
            name = self._doc_names.get(ref_id, f"Source {ref_id}")
            lines.append(f"[{ref_id}] {name}")
            max_idx = max(self._reverse[ref_id])
            for idx in range(1, max_idx + 1):
                cid = self._reverse[ref_id].get(idx)
                if cid is None:
                    continue
                meta = self._chunk_meta.get(cid, {})
                page_idx = meta.get("page_idx", 0)
                page_label = f"Page {page_idx}" if page_idx else f"Chunk {idx}"
                lines.append(f"  [{ref_id}-{idx}] {page_label}")
        return "\n".join(lines)

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
