"""Citation index — maps [ref_id-chunk_idx] to actual chunk_ids."""

import logging
from typing import Any

from .utils import context_chunk_key, split_source_ids

logger = logging.getLogger(__name__)


class CitationIndexer:
    """Bidirectional index: (ref_id, chunk_idx) <-> chunk_id.

    The answer prompt defines every ``[ref_id-chunk_idx]`` marker inline on the
    excerpt it labels, so this index only has to resolve markers back to chunks.
    """

    def __init__(self) -> None:
        self._index: dict[str, dict[str, int]] = {}
        self._reverse: dict[str, dict[int, str]] = {}
        # chunk_id -> (ref_id, chunk_idx) for source_id lookups
        self._chunk_to_ref: dict[str, tuple[str, int]] = {}
        # Per-ref normalized workspace provenance: ref_id -> workspace ID
        self._doc_workspaces: dict[str, str] = {}

    def build_index(self, contexts: list[dict[str, Any]]) -> None:
        self._index.clear()
        self._reverse.clear()
        self._chunk_to_ref.clear()
        self._doc_workspaces.clear()

        valid_chunk_ids: set[str] = set()
        for ctx in contexts:
            cid = ctx.get("chunk_id")
            if cid and (ctx.get("content") or ctx.get("image_data")):
                valid_chunk_ids.add(context_chunk_key(cid, workspace=ctx.get("_workspace")))

        ref_chunks: dict[str, list[str]] = {}
        # Parallel per-ref sets give O(1) membership so build_index is O(C),
        # not O(C^2) over a growing list.
        seen_by_ref: dict[str, set[str]] = {}
        for ctx in contexts:
            ref_id = str(ctx.get("reference_id", ""))
            if not ref_id:
                continue
            chunk_id = ctx.get("chunk_id")
            chunk_key = context_chunk_key(
                chunk_id,
                workspace=ctx.get("_workspace"),
            )
            if chunk_id and chunk_key in valid_chunk_ids:
                seen = seen_by_ref.setdefault(ref_id, set())
                ordered = ref_chunks.setdefault(ref_id, [])
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    ordered.append(chunk_id)
                workspace = str(ctx.get("_workspace") or "").strip()
                if workspace and ref_id not in self._doc_workspaces:
                    self._doc_workspaces[ref_id] = workspace
            else:
                source_id = ctx.get("source_id")
                if source_id:
                    for sid in split_source_ids(source_id):
                        source_key = context_chunk_key(
                            sid,
                            workspace=ctx.get("_workspace"),
                        )
                        if source_key in valid_chunk_ids:
                            seen = seen_by_ref.setdefault(ref_id, set())
                            ordered = ref_chunks.setdefault(ref_id, [])
                            if sid not in seen:
                                seen.add(sid)
                                ordered.append(sid)

        for ref_id, chunk_ids in ref_chunks.items():
            self._index[ref_id] = {}
            self._reverse[ref_id] = {}
            for idx, cid in enumerate(chunk_ids, start=1):
                self._index[ref_id][cid] = idx
                self._reverse[ref_id][idx] = cid
                workspace = self._doc_workspaces.get(ref_id)
                self._chunk_to_ref[context_chunk_key(cid, workspace=workspace)] = (ref_id, idx)

    def get_chunk_idx(self, ref_id: str | int, chunk_id: str) -> int | None:
        return self._index.get(str(ref_id), {}).get(chunk_id)

    def get_chunk_id(self, ref_id: str | int, chunk_idx: int) -> str | None:
        return self._reverse.get(str(ref_id), {}).get(chunk_idx)

    def get_max_chunk_idx(self, ref_id: str | int) -> int:
        reverse = self._reverse.get(str(ref_id), {})
        return max(reverse.keys()) if reverse else 0

    def get_doc_workspace(self, ref_id: str | int) -> str | None:
        """Return the normalized workspace ID recorded for a reference."""
        return self._doc_workspaces.get(str(ref_id))

    def get_doc_tags(self, source_id: str | None, *, workspace: str | None = None) -> list[str]:
        """Return unique doc-level tags for a source_id.

        Returns doc-level ``[n]`` tags (not chunk-level ``[n-m]``) — suitable
        for KG context where the LLM should know the source documents but cite
        specific chunks from the Document Excerpts section instead.

        Example: source_id="c1,c2,c5" (c1,c2 from doc 1, c5 from doc 2)
                 → ["[1]", "[2]"]
        """
        if not source_id:
            return []
        tags: list[str] = []
        seen: set[str] = set()
        for cid in split_source_ids(source_id):
            ref_info = self._chunk_to_ref.get(context_chunk_key(cid, workspace=workspace))
            if ref_info:
                ref_id = ref_info[0]
                if ref_id not in seen:
                    seen.add(ref_id)
                    tags.append(f"[{ref_id}]")
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


def build_citation_index(
    contexts: list[dict[str, Any]],
) -> tuple[CitationIndexer, list[dict[str, Any]]]:
    indexer = CitationIndexer()
    indexer.build_index(contexts)
    enriched = indexer.inject_chunk_idx(contexts)
    return indexer, enriched
