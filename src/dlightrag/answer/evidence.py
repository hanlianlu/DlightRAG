# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One request's factual memory: the sources its tools actually returned."""

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

from dlightrag.answer.citations.indexer import CitationIndexer
from dlightrag.answer.citations.utils import context_chunk_key
from dlightrag.answer.excerpts import build_excerpt_lane_blocks, format_kg_context
from dlightrag.answer.images import AnswerImageBudget
from dlightrag.rag.retrieval import ContextRow, RetrievalContexts

_NO_KG = "No knowledge graph context available."


@dataclass(frozen=True, slots=True)
class EvidenceDelta:
    new_chunks: int = 0
    new_entities: int = 0
    new_relationships: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.new_chunks or self.new_entities or self.new_relationships)


class EvidenceLedger:
    """Accumulate one answer's evidence under stable numeric citation ids.

    The ledger stores only the windows tools or initial retrieval actually
    returned, with stable source and locator identity. ``transform`` renders
    evidence inside the caller's residual request capacity: recent evidence is
    kept verbatim and older evidence collapses to compact re-readable handles
    that still preserve citation identity. There are no per-source quotas or image
    lanes; one shared image budget carries every evidence visual.
    """

    def __init__(self, *, image_budget: AnswerImageBudget | None = None) -> None:
        self.contexts: RetrievalContexts = {
            "chunks": [],
            "entities": [],
            "relationships": [],
        }
        self._source_ids: dict[tuple[str, str, str], str] = {}
        self._seen_chunks: set[str] = set()
        self._seen_rows: dict[str, set[str]] = {}
        self._image_budget = image_budget
        self._image_blocks: dict[str, dict[str, Any]] = {}
        self._pending_image_rows: list[ContextRow] = []
        self._image_budget_lock = asyncio.Lock()

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.contexts.values())

    def ledger_state_json(self) -> str:
        """Return the canonical durable Evidence state for Session settlement.

        Rendered image blocks are deliberately excluded: they are derived from
        the rows under a run-local budget, so recovery re-derives them in the
        same order instead of storing a second copy of every visual. An empty
        ledger serializes as ``{}`` so settlement can skip a no-op write.
        """
        from dlightrag.engine.agent.session.effects import canonical_json

        if not self.row_count and not self._source_ids:
            return canonical_json({})
        return canonical_json(self.durable_state())

    def durable_state(self) -> dict[str, Any]:
        """JSON-ready identity and rows, without derived image blocks."""
        return {
            "contexts": {
                key: [_durable_row(row) for row in rows] for key, rows in self.contexts.items()
            },
            "source_ids": [[list(key), value] for key, value in self._source_ids.items()],
            "seen_chunks": sorted(self._seen_chunks),
            "seen_rows": {key: sorted(values) for key, values in self._seen_rows.items()},
        }

    def citation_handles(self, *, after_chunk_count: int = 0) -> list[str]:
        """Parent-visible citation identities, newest-admitted first after a cursor."""
        seen: set[str] = set()
        handles: list[str] = []
        for row in self.contexts.get("chunks", [])[after_chunk_count:]:
            reference_id = str(row.get("reference_id") or "")
            if not reference_id or reference_id in seen:
                continue
            seen.add(reference_id)
            metadata = row.get("metadata") or {}
            title = (
                str(metadata.get("title") or "")
                or str(row.get("file_path") or "").rsplit("/", 1)[-1]
                or "Source"
            )
            resource_id = str(metadata.get("resource_id") or "")
            suffix = f" [resource: {resource_id}]" if resource_id else ""
            handles.append(f"[{reference_id}] {title}{suffix}")
        return handles

    def restore_ledger_state(self, state: Mapping[str, Any]) -> None:
        """Replace the ledger with durable Session-recovered state."""
        contexts = state.get("contexts")
        if not isinstance(contexts, Mapping):
            raise ValueError("evidence state has no contexts")
        restored: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        for key, rows in cast(Mapping[str, Any], contexts).items():
            restored[key] = [dict(row) for row in cast(list[Any], rows)]
        self.contexts = restored
        self._source_ids = {
            (str(key[0]), str(key[1]), str(key[2])): str(value)
            for key, value in cast(list[Any], state.get("source_ids") or [])
        }
        self._seen_chunks = {
            str(value) for value in cast(list[Any], state.get("seen_chunks") or [])
        }
        self._seen_rows = {
            str(key): {str(value) for value in values}
            for key, values in cast(Mapping[str, Any], state.get("seen_rows") or {}).items()
        }
        self._image_blocks = {}
        self._pending_image_rows = (
            [row for row in self.contexts["chunks"] if row.get("image_data")]
            if self._image_budget is not None
            else []
        )

    def add_rows(self, rows: list[ContextRow]) -> EvidenceDelta:
        return self.add_contexts({"chunks": rows})

    def merge_child_state(
        self,
        state: Mapping[str, Any],
        *,
        child_session_id: str,
        parent_call_id: str,
    ) -> EvidenceDelta:
        """Admit child rows into the parent ledger with explicit lineage.

        The caller invokes this before settling the parent spawn intent, so
        the parent's ordinary fenced evidence settlement persists the merge
        atomically with the ToolResult.
        """
        contexts = state.get("contexts")
        if not isinstance(contexts, Mapping):
            raise ValueError("child evidence state has no contexts")
        merged: RetrievalContexts = {"chunks": [], "entities": [], "relationships": []}
        for key, raw_rows in cast(Mapping[str, Any], contexts).items():
            rows: list[ContextRow] = []
            for raw in cast(list[Any], raw_rows):
                row = dict(cast(Mapping[str, Any], raw))
                metadata = dict(cast(Mapping[str, Any], row.get("metadata") or {}))
                metadata.update(
                    {
                        "merged_from_child": True,
                        "child_session_id": child_session_id,
                        "parent_call_id": parent_call_id,
                    }
                )
                row["metadata"] = metadata
                rows.append(row)
            merged[key] = rows
        return self.add_contexts(merged)

    def add_contexts(self, contexts: RetrievalContexts) -> EvidenceDelta:
        new_chunks = 0
        for row in contexts.get("chunks", []):
            normalized, identity = self._normalize_chunk(row)
            if identity in self._seen_chunks:
                continue
            self._seen_chunks.add(identity)
            self.contexts["chunks"].append(normalized)
            if normalized.get("image_data") and self._image_budget is not None:
                self._pending_image_rows.append(normalized)
            new_chunks += 1

        counts: dict[str, int] = {}
        for key, rows in contexts.items():
            if key == "chunks":
                continue
            target = self.contexts.setdefault(key, [])
            seen = self._seen_rows.setdefault(key, set())
            added = 0
            for row in rows:
                identity = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
                if identity in seen:
                    continue
                seen.add(identity)
                target.append(dict(row))
                added += 1
            counts[key] = added
        return EvidenceDelta(
            new_chunks=new_chunks,
            new_entities=counts.get("entities", 0),
            new_relationships=counts.get("relationships", 0),
        )

    async def aflush_images(self) -> None:
        """Budget newly admitted evidence images without blocking the event loop."""
        if self._image_budget is None:
            return
        async with self._image_budget_lock:
            rows, self._pending_image_rows = self._pending_image_rows, []
            if not rows:
                return
            task = asyncio.create_task(asyncio.to_thread(self._budget_images, rows))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                self._pending_image_rows = [*rows, *self._pending_image_rows]
                await asyncio.gather(task, return_exceptions=True)
                raise
            except Exception:
                self._pending_image_rows = [*rows, *self._pending_image_rows]
                raise

    def render_blocks(
        self,
        *,
        image_blocks_by_context_key: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        """Render every accumulated source as full evidence blocks."""
        chunks = self.contexts["chunks"]
        indexer = CitationIndexer()
        indexer.build_index(chunks)
        image_blocks = (
            image_blocks_by_context_key
            if image_blocks_by_context_key is not None
            else self._image_blocks
        )
        blocks = self._render_chunk_blocks(chunks, indexer, image_blocks)
        return blocks, indexer

    def transform(
        self,
        *,
        residual_tokens: int,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        """Render evidence bounded by the caller's residual request capacity.

        The most recent evidence is rendered verbatim up to the ceiling; older
        evidence collapses to compact re-readable handles.  Citation identities
        are preserved because the indexer always spans every accumulated source,
        so ``[n-m]`` markers still resolve for collapsed sources.
        """
        if residual_tokens < 0:
            raise ValueError("residual_tokens cannot be negative")
        chunks = self.contexts["chunks"]
        indexer = CitationIndexer()
        indexer.build_index(chunks)

        kept_keys: set[str] = set()
        running = 0
        cutoff = False
        for chunk in reversed(chunks):
            cost = _chunk_evidence_cost(chunk)
            if not cutoff and running + cost <= residual_tokens:
                running += cost
                kept_keys.add(self._chunk_identity(chunk))
            else:
                cutoff = True

        kept = [chunk for chunk in chunks if self._chunk_identity(chunk) in kept_keys]
        collapsed = [chunk for chunk in chunks if self._chunk_identity(chunk) not in kept_keys]

        blocks = self._render_chunk_blocks(kept, indexer, self._image_blocks)
        handle_block = _collapsed_handle_block(collapsed)
        if handle_block is not None:
            blocks.append(handle_block)
        return blocks, indexer

    def _render_chunk_blocks(
        self,
        chunks: list[ContextRow],
        indexer: CitationIndexer,
        image_blocks: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        attachments: list[ContextRow] = []
        web: list[ContextRow] = []
        corpus: list[ContextRow] = []
        for row in chunks:
            source_type = str((row.get("metadata") or {}).get("source_type") or "")
            if source_type == "web_attachment":
                attachments.append(row)
            elif source_type == "web_search":
                web.append(row)
            else:
                corpus.append(row)

        blocks: list[dict[str, Any]] = []
        kg = format_kg_context(self.contexts, indexer)
        if kg != _NO_KG:
            blocks.append({"type": "text", "text": f"## Knowledge graph evidence\n{kg}"})
        for title, rows in (
            ("## User-attached documents", attachments),
            ("## Knowledge-base evidence", corpus),
            ("## Open-web evidence", web),
        ):
            if not rows:
                continue
            blocks.append({"type": "text", "text": title})
            blocks.extend(
                build_excerpt_lane_blocks(
                    rows,
                    indexer=indexer,
                    image_blocks_by_context_key=image_blocks,
                )
            )
        return blocks

    def _budget_image(self, row: ContextRow) -> None:
        if not row.get("image_data") or self._image_budget is None:
            return
        chunk_id = str(row.get("chunk_id") or "")
        key = self._chunk_identity(row)
        if not key or key in self._image_blocks:
            return
        block = self._image_budget.add_base64(
            str(row["image_data"]),
            label=chunk_id or str(row.get("file_path") or "evidence_image"),
        )
        if block is not None:
            self._image_blocks[key] = block

    def _budget_images(self, rows: list[ContextRow]) -> None:
        for row in rows:
            self._budget_image(row)

    @staticmethod
    def _chunk_identity(row: ContextRow) -> str:
        return context_chunk_key(str(row.get("chunk_id") or ""), workspace=row.get("_workspace"))

    def _normalize_chunk(self, row: ContextRow) -> tuple[ContextRow, str]:
        normalized = dict(row)
        metadata = dict(row.get("metadata") or {})
        normalized["metadata"] = metadata
        workspace = str(row.get("_workspace") or "")
        original_reference = str(row.get("reference_id") or "")
        source_uri = str(metadata.get("source_uri") or "")
        key_kind, key_value = (
            ("uri", source_uri) if source_uri else ("reference", original_reference)
        )
        source_key = (workspace, key_kind, key_value)
        reference_id = self._source_ids.setdefault(
            source_key,
            str(len(self._source_ids) + 1),
        )
        normalized["_source_reference_id"] = original_reference
        normalized["reference_id"] = reference_id

        if metadata.get("source_type") == "web_search":
            content = str(row.get("content") or "")
            evidence_key = str(row.get("_evidence_key") or content)
            digest = hashlib.sha256(f"{source_uri}\0{evidence_key}".encode()).hexdigest()[:20]
            normalized["chunk_id"] = f"webchunk-{digest}"
            identity = f"web:{digest}"
        else:
            identity = context_chunk_key(row.get("chunk_id"), workspace=workspace)
        return normalized, identity


def _chunk_evidence_cost(row: ContextRow) -> int:
    from dlightrag.engine.ai.tokens import estimate_tokens

    cost = estimate_tokens(str(row.get("content") or ""))
    if row.get("image_data"):
        cost += 85
    return cost


def _collapsed_handle_block(collapsed: list[ContextRow]) -> dict[str, Any] | None:
    if not collapsed:
        return None
    seen: set[str] = set()
    lines: list[str] = []
    for row in collapsed:
        reference_id = str(row.get("reference_id") or "")
        if reference_id in seen:
            continue
        seen.add(reference_id)
        metadata = row.get("metadata") or {}
        title = (
            str(metadata.get("title") or "")
            or str(row.get("file_path") or "").rsplit("/", 1)[-1]
            or "Source"
        )
        resource_id = str(metadata.get("resource_id") or "")
        resource_label = f" [resource: {resource_id}]" if resource_id else ""
        lines.append(
            f"[{reference_id}] {title}{resource_label} - earlier evidence retained; "
            "re-read this source for full detail."
        )
    return {
        "type": "text",
        "text": "## Retained evidence (re-read for detail)\n" + "\n".join(lines),
    }


def _durable_row(row: ContextRow) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("image_data", None)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payload["metadata"] = dict(metadata)
    return payload


__all__ = ["EvidenceDelta", "EvidenceLedger"]
