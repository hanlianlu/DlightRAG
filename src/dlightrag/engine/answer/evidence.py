# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One request's factual memory: the sources its tools actually returned."""

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from dlightrag.engine.answer.citations.indexer import CitationIndexer
from dlightrag.engine.answer.citations.utils import context_chunk_key
from dlightrag.engine.answer.excerpts import (
    build_excerpt_lane_blocks,
    build_image_label,
    chunk_label,
    format_kg_context,
)
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.rag.retrieval import ContextRow, RetrievalContexts

_NO_KG = "No knowledge graph context available."


@dataclass(frozen=True, slots=True)
class EvidenceDelta:
    new_chunks: int = 0
    new_entities: int = 0
    new_relationships: int = 0
    dropped_rows: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.new_chunks or self.new_entities or self.new_relationships)


def has_unrepresentable_text(value: Any) -> bool:
    """Return whether any string in ``value`` holds text a durable store cannot keep.

    PostgreSQL text and jsonb values cannot carry U+0000, and a lone surrogate
    cannot be encoded at all, so admitting either would fail a later durable
    write and end the Run that produced it. Evidence is the one admission path
    every retrieval source shares, which is why the check lives here instead of
    in each Tool or provider adapter.
    """
    if isinstance(value, str):
        if "\x00" in value:
            return True
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return True
        return False
    if isinstance(value, Mapping):
        for key, item in value.items():
            if has_unrepresentable_text(key) or has_unrepresentable_text(item):
                return True
        return False
    if isinstance(value, list | tuple | set | frozenset):
        return any(has_unrepresentable_text(item) for item in value)
    return False


class EvidenceLedger:
    """Accumulate one answer's evidence under stable numeric citation ids.

    The ledger stores only the windows tools or initial retrieval actually
    returned, with stable source and locator identity. ``take_admitted_text``
    renders the rows admitted since the previous call inside one batch budget:
    recent rows are kept verbatim and that batch's older rows collapse to compact
    re-readable handles that still preserve citation identity, and the caller
    freezes the text into the Tool result that admitted them. Evidence text and
    evidence pixels part company here: ``_durable_row`` drops image bytes, so
    ``visual_blocks`` serves them from one shared image budget as their own
    per-request lane instead.
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
        # Rows the model has already been shown as frozen transcript text. The
        # ledger holds this cursor instead of a pending list so an ordinary
        # header skip advances it to "nothing new" without adhoc bookkeeping.
        self._shown_rows: dict[str, int] = dict.fromkeys(("chunks", "entities", "relationships"), 0)
        #: Where each in-flight Effect Intent's freeze began, so re-executing that
        #: intent reproduces the same bytes instead of reporting nothing new.
        self._intent_marks: dict[str, dict[str, int]] = {}

    @property
    def row_count(self) -> int:
        return sum(len(rows) for rows in self.contexts.values())

    def pending_row_count(self) -> int:
        """Return how many admitted rows no Tool result has frozen yet."""
        return sum(
            len(self.contexts.get(key, [])) - self._shown_rows[key]
            for key in ("chunks", "entities", "relationships")
        )

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

    def citation_handles(
        self, *, after_chunk_count: int = 0, matching_chunks: list[ContextRow] | None = None
    ) -> list[str]:
        """Parent-visible citation identities, newest-admitted first after a cursor."""
        identities = (
            {self._chunk_identity(row) for row in matching_chunks}
            if matching_chunks is not None
            else None
        )
        seen: set[str] = set()
        handles: list[str] = []
        for row in self.contexts.get("chunks", [])[after_chunk_count:]:
            if identities is not None and self._chunk_identity(row) not in identities:
                continue
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
        # A recovered ledger was already rendered into the Session transcript by
        # the Tool results that admitted it; nothing is pending on restore.
        self._shown_rows = {key: len(self.contexts.get(key, [])) for key in self._shown_rows}
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
        dropped_rows = 0
        for row in contexts.get("chunks", []):
            if has_unrepresentable_text(row):
                dropped_rows += 1
                continue
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
                if has_unrepresentable_text(row):
                    dropped_rows += 1
                    continue
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
            dropped_rows=dropped_rows,
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

    def take_admitted_text(
        self,
        *,
        budget_tokens: int,
        intent_key: str | None = None,
    ) -> tuple[str, str]:
        """Render the rows admitted since the previous call as model-visible text.

        Durable Research freezes this text into the Tool result that admitted the
        rows, so a passage reaches the model once, at the position it arrived, and
        every later request reuses those exact bytes as a cacheable prefix. The
        previous shape re-rendered the whole ledger *after* the growing Session
        fold on every turn, so the pack sat past every matched cache prefix:
        measured prompts were billed at 88-100% cache miss while the fold itself
        stayed cached.

        Returns ``(labels, rendered)``. A row the admitting Tool marked as already
        carried by its own model-visible result contributes only its ``[n-m]`` label,
        so the passage is not carried twice in one request.

        ``budget_tokens`` bounds one batch exactly as the old pack bounded the
        request: the newest rows render verbatim while the batch's older rows
        collapse to the re-readable handles that already carry citation identity.
        Images never render here; they belong to the run-local visual lane because
        their bytes are not durable (``_durable_row`` drops ``image_data``).

        ``intent_key`` makes the freeze reproduce itself. The caller names the
        durable Effect Intent that admitted the rows, so re-executing that same
        intent — a resume while the effect was still pending — renders those rows
        again instead of reporting nothing new and committing an unlabelled Tool
        result. ``rollback_show`` retracts one such freeze when the caller could not
        use it after all.
        """
        if budget_tokens < 0:
            raise ValueError("evidence render budget cannot be negative")
        marks = self._intent_marks.setdefault(intent_key, {}) if intent_key is not None else None
        start = {
            key: min(self._shown_rows[key], marks[key])
            if marks and key in marks
            else self._shown_rows[key]
            for key in ("chunks", "entities", "relationships")
        }
        if marks is not None:
            for key in ("chunks", "entities", "relationships"):
                marks[key] = start[key]
        chunks = self.contexts["chunks"][start["chunks"] :]
        admitted_kg: RetrievalContexts = {
            "chunks": [],
            "entities": self.contexts.get("entities", [])[start["entities"] :],
            "relationships": self.contexts.get("relationships", [])[start["relationships"] :],
        }
        for key in ("chunks", "entities", "relationships"):
            self._shown_rows[key] = len(self.contexts.get(key, []))
        if not chunks and not admitted_kg["entities"] and not admitted_kg["relationships"]:
            return "", ""

        indexer = CitationIndexer()
        indexer.build_index(self.contexts["chunks"])

        labelled = [chunk for chunk in chunks if _carried_by_tool(chunk)]
        rest = [chunk for chunk in chunks if not _carried_by_tool(chunk)]
        kept_keys: set[str] = set()
        running = 0
        cutoff = False
        for chunk in reversed(rest):
            cost = _chunk_evidence_cost(chunk)
            if not cutoff and running + cost <= budget_tokens:
                running += cost
                kept_keys.add(self._chunk_identity(chunk))
            else:
                cutoff = True
        kept = [chunk for chunk in rest if self._chunk_identity(chunk) in kept_keys]
        collapsed = [chunk for chunk in rest if self._chunk_identity(chunk) not in kept_keys]

        blocks: list[dict[str, Any]] = []
        kg = format_kg_context(admitted_kg, indexer)
        if kg != _NO_KG:
            blocks.append({"type": "text", "text": f"## Knowledge graph evidence\n{kg}"})
        blocks.extend(self._render_chunk_blocks(kept, indexer, {}))
        handle_block = _collapsed_handle_block(collapsed)
        if handle_block is not None:
            blocks.append(handle_block)
        rendered = "\n\n".join(
            text
            for text in (
                str(block.get("text") or "").strip() for block in _drop_empty_headings(blocks)
            )
            if text
        )
        labels = "\n".join(_cited_label(chunk, indexer) for chunk in labelled)
        return labels, rendered

    def rollback_show(self, intent_key: str) -> None:
        """Retract one Tool's freeze because the caller could not use its text.

        The rows stay admitted; the next Tool result renders them the way any
        other pending row is rendered.
        """
        marks = self._intent_marks.pop(intent_key, None)
        if marks is None:
            return
        for key, position in marks.items():
            self._shown_rows[key] = min(self._shown_rows[key], position)

    def visual_blocks(self) -> list[dict[str, Any]]:
        """Return this run's budgeted evidence images as one bounded request lane.

        Evidence pixels are run-local: ``_durable_row`` drops ``image_data``, so a
        recovered Session cannot re-render them from durable state and they cannot
        enter the transcript the way evidence text does. The lane is re-rendered
        per request, which is where this design still pays a cache miss: bounded
        by the resolved image budget and never by the accumulated corpus.
        """
        rows = [row for row in self.contexts["chunks"] if row.get("image_data")]
        if not rows:
            return []
        indexer = CitationIndexer()
        indexer.build_index(self.contexts["chunks"])
        blocks: list[dict[str, Any]] = []
        for row in rows:
            key = context_chunk_key(
                str(row.get("chunk_id") or ""),
                workspace=row.get("_workspace"),
            )
            image_block = self._image_blocks.get(key)
            if image_block is None:
                continue
            ref_id = str(row.get("reference_id") or "")
            chunk_id = str(row.get("chunk_id") or "")
            chunk_index = indexer.get_chunk_idx(ref_id, chunk_id) if ref_id and chunk_id else None
            blocks.append(
                {
                    "type": "text",
                    "text": build_image_label(
                        cite_tag=f"[{ref_id}-{chunk_index}]" if chunk_index is not None else "",
                        chunk=row,
                        filename=str(row.get("file_path") or ""),
                    ),
                }
            )
            blocks.append(image_block)
        return blocks

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
        blocks: list[dict[str, Any]] = []
        kg = format_kg_context(self.contexts, indexer)
        if kg != _NO_KG:
            blocks.append({"type": "text", "text": f"## Knowledge graph evidence\n{kg}"})
        blocks.extend(self._render_chunk_blocks(chunks, indexer, image_blocks))
        return blocks, indexer

    def _render_chunk_blocks(
        self,
        chunks: list[ContextRow],
        indexer: CitationIndexer,
        image_blocks: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Render chunk lanes only. Knowledge-graph evidence is a caller's choice.

        An incremental caller states the graph slice it admits; a full caller states
        the whole ledger. Rendering the whole graph from here would re-copy history
        into every later Tool result.
        """
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


def _carried_by_tool(row: ContextRow) -> bool:
    """Return whether the admitting Tool's own result already carries this row.

    A resource-backed Tool answers with the passage itself, so rendering the row
    again would put the same text in one request twice. The row still needs its
    citation label, which is what the caller splices in front of the body it already
    has. The Tool declares this at admission; nothing infers it from the text.
    """
    return bool(row.get("_carried_by_tool"))


def _cited_label(row: ContextRow, indexer: CitationIndexer) -> str:
    """Return one already-shown row's citation label line."""
    ref_id = str(row.get("reference_id") or "")
    chunk_id = str(row.get("chunk_id") or "")
    chunk_index = indexer.get_chunk_idx(ref_id, chunk_id) if ref_id and chunk_id else None
    cite_tag = f"[{ref_id}-{chunk_index}]" if chunk_index is not None else ""
    file_path = str(row.get("file_path") or "")
    filename = Path(file_path).name if file_path else f"Source {ref_id}"
    return chunk_label(cite_tag=cite_tag, chunk=dict(row), filename=filename)


def _drop_empty_headings(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a section heading whose rows render no text in this batch.

    A visual-only passage renders no text block: its pixels belong to the run-local
    visual lane, so the batch text would otherwise open with a heading and nothing
    under it. A heading is kept when its own section — up to the next heading of the
    same or higher level — contains a block that is not itself a bare heading.
    """
    kept: list[dict[str, Any]] = []
    for position, block in enumerate(blocks):
        level = _bare_heading_level(block)
        if level:
            has_content = False
            for following in blocks[position + 1 :]:
                following_level = _bare_heading_level(following)
                if following_level and following_level <= level:
                    break
                if not following_level:
                    has_content = True
                    break
            if not has_content:
                continue
        kept.append(block)
    return kept


def _bare_heading_level(block: Any) -> int:
    """Return a block's heading depth, or 0 when it is not a heading alone."""
    if not isinstance(block, dict):
        return 0
    text = str(block.get("text") or "").strip()
    if not text.startswith("#") or "\n" in text:
        return 0
    return len(text) - len(text.lstrip("#"))


def _chunk_evidence_cost(row: ContextRow) -> int:
    """Measure text only; image capacity is governed by resolved image policy."""
    from dlightrag.engine.ai.tokens import estimate_tokens

    return estimate_tokens(str(row.get("content") or ""))


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
    # Run-local rendering facts: pixels cannot be re-derived and the label decision
    # is already spent by the freeze that admitted the row.
    payload.pop("image_data", None)
    payload.pop("_carried_by_tool", None)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        payload["metadata"] = dict(metadata)
    return payload


__all__ = ["EvidenceDelta", "EvidenceLedger", "has_unrepresentable_text"]
