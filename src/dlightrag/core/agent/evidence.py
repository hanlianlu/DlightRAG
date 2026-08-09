# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local evidence identity, rendering, and capacity transform."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.utils import context_chunk_key
from dlightrag.core.answer.capacity import AnswerCapacity
from dlightrag.core.answer.excerpts import build_excerpt_lane_blocks, format_kg_context
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import ContextRow, RetrievalContexts

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
    returned, with stable source and locator identity.  ``transform`` renders
    evidence bounded by one :class:`AnswerCapacity`: recent evidence is kept
    verbatim and older evidence collapses to compact re-readable handles that
    still preserve citation identity.  There are no per-source quotas or image
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

    def add_rows(self, rows: list[ContextRow]) -> EvidenceDelta:
        return self.add_contexts({"chunks": rows})

    def add_contexts(self, contexts: RetrievalContexts) -> EvidenceDelta:
        new_chunks = 0
        for row in contexts.get("chunks", []):
            normalized, identity = self._normalize_chunk(row)
            if identity in self._seen_chunks:
                continue
            self._seen_chunks.add(identity)
            self.contexts["chunks"].append(normalized)
            self._budget_image(normalized)
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
        capacity: AnswerCapacity,
        *,
        fixed_input_tokens: int,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        """Render evidence bounded by the capacity evidence ceiling.

        The most recent evidence is rendered verbatim up to the ceiling; older
        evidence collapses to compact re-readable handles.  Citation identities
        are preserved because the indexer always spans every accumulated source,
        so ``[n-m]`` markers still resolve for collapsed sources.
        """
        chunks = self.contexts["chunks"]
        indexer = CitationIndexer()
        indexer.build_index(chunks)
        ceiling = capacity.evidence_ceiling(fixed_input_tokens=fixed_input_tokens)

        kept_keys: set[str] = set()
        running = 0
        cutoff = False
        for chunk in reversed(chunks):
            cost = _chunk_evidence_cost(chunk)
            if not cutoff and running + cost <= ceiling:
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
        composer: list[ContextRow] = []
        web: list[ContextRow] = []
        corpus: list[ContextRow] = []
        for row in chunks:
            source_type = str((row.get("metadata") or {}).get("source_type") or "")
            if source_type == "web_attachment":
                composer.append(row)
            elif source_type == "web_search":
                web.append(row)
            else:
                corpus.append(row)

        blocks: list[dict[str, Any]] = []
        kg = format_kg_context(self.contexts, indexer)
        if kg != _NO_KG:
            blocks.append({"type": "text", "text": f"## Knowledge graph evidence\n{kg}"})
        for title, rows in (
            ("## User-attached documents", composer),
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
            digest = hashlib.sha256(f"{source_uri}\0{content}".encode()).hexdigest()[:20]
            normalized["chunk_id"] = f"webchunk-{digest}"
            identity = f"web:{digest}"
        else:
            identity = context_chunk_key(row.get("chunk_id"), workspace=workspace)
        return normalized, identity


def _chunk_evidence_cost(row: ContextRow) -> int:
    from dlightrag.utils.tokens import estimate_tokens

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
        lines.append(
            f"[{reference_id}] {title} - earlier evidence retained; "
            "re-read this source for full detail."
        )
    return {
        "type": "text",
        "text": "## Retained evidence (re-read for detail)\n" + "\n".join(lines),
    }


__all__ = ["EvidenceDelta", "EvidenceLedger"]
