# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Request-local evidence identity and rendering for agent turns."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.utils import context_chunk_key
from dlightrag.core.answer.excerpts import build_excerpt_lane_blocks, format_kg_context
from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.core.retrieval.protocols import ContextRow, RetrievalContexts


@dataclass(frozen=True, slots=True)
class EvidenceDelta:
    new_chunks: int = 0
    new_entities: int = 0
    new_relationships: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.new_chunks or self.new_entities or self.new_relationships)


class EvidenceSession:
    """Accumulate one answer's evidence under stable numeric citation ids."""

    def __init__(
        self,
        *,
        composer_image_budget: AnswerImageBudget | None = None,
        rag_image_budget: AnswerImageBudget | None = None,
    ) -> None:
        self.contexts: RetrievalContexts = {
            "chunks": [],
            "entities": [],
            "relationships": [],
        }
        self._source_ids: dict[tuple[str, str, str], str] = {}
        self._seen_chunks: set[str] = set()
        self._seen_rows: dict[str, set[str]] = {}
        self._composer_image_budget = composer_image_budget
        self._rag_image_budget = rag_image_budget
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
        chunks = self.contexts["chunks"]
        indexer = CitationIndexer()
        indexer.build_index(chunks)
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
        if kg != "No knowledge graph context available.":
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
                    image_blocks_by_context_key=(
                        image_blocks_by_context_key
                        if image_blocks_by_context_key is not None
                        else self._image_blocks
                    ),
                )
            )
        return blocks, indexer

    def _budget_image(self, row: ContextRow) -> None:
        if not row.get("image_data"):
            return
        source_type = str((row.get("metadata") or {}).get("source_type") or "")
        budget = (
            self._composer_image_budget
            if source_type == "web_attachment"
            else self._rag_image_budget
        )
        if budget is None:
            return
        chunk_id = str(row.get("chunk_id") or "")
        key = context_chunk_key(chunk_id, workspace=row.get("_workspace"))
        if not key or key in self._image_blocks:
            return
        block = budget.add_base64(
            str(row["image_data"]),
            label=chunk_id or str(row.get("file_path") or "evidence_image"),
        )
        if block is not None:
            self._image_blocks[key] = block

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


__all__ = ["EvidenceDelta", "EvidenceSession"]
