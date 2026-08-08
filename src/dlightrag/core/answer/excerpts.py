# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared citation-labelled evidence rendering."""

from pathlib import Path
from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.utils import context_chunk_key
from dlightrag.core.retrieval.protocols import RetrievalContexts
from dlightrag.utils.images import image_data_uri

_INTERNAL_KEYS: frozenset[str] = frozenset(
    {
        "chunk_id",
        "chunk_idx",
        "content",
        "bm25_profile",
        "distance",
        "file_path",
        "full_doc_id",
        "image_data",
        "image_mime_type",
        "image_url",
        "metadata",
        "page_number",
        "pipeline_stage",
        "reference_id",
        "relevance_score",
        "rerank_score",
        "score",
        "sidecar",
        "sidecar_location",
        "thumbnail_url",
        "_answer_image_sent",
        "_workspace",
    }
)


def format_kg_context(
    contexts: RetrievalContexts,
    indexer: CitationIndexer | None = None,
) -> str:
    """Format entities and relationships with document-level citations."""
    parts: list[str] = []
    entities = contexts.get("entities", [])
    if entities:
        parts.append("## Entities")
        for entity in entities[:20]:
            cite = _source_tags(entity, indexer)
            parts.append(
                f"- **{entity.get('entity_name', '')}** "
                f"({entity.get('entity_type', '')}): {entity.get('description', '')}{cite}"
            )
    relationships = contexts.get("relationships", [])
    if relationships:
        parts.append("\n## Relationships")
        for relationship in relationships[:20]:
            cite = _source_tags(relationship, indexer)
            parts.append(
                f"- {relationship.get('src_id', '')} -> {relationship.get('tgt_id', '')}: "
                f"{relationship.get('description', '')}{cite}"
            )
    return "\n".join(parts) if parts else "No knowledge graph context available."


def _source_tags(row: dict[str, Any], indexer: CitationIndexer | None) -> str:
    if indexer is None:
        return ""
    tags = indexer.get_doc_tags(
        row.get("source_id"),
        workspace=row.get("_workspace"),
    )
    return f" (from {', '.join(tags)})" if tags else ""


def build_excerpt_lane_blocks(
    chunks: list[dict[str, Any]],
    *,
    indexer: CitationIndexer | None,
    image_blocks_by_context_key: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Render one evidence lane without changing chunk order."""
    doc_groups: dict[str, list[dict[str, Any]]] = {}
    doc_order: list[str] = []
    for chunk in chunks:
        ref_id = str(chunk.get("reference_id", ""))
        if ref_id not in doc_groups:
            doc_order.append(ref_id)
            doc_groups[ref_id] = [chunk]
        else:
            doc_groups[ref_id].append(chunk)

    blocks: list[dict[str, Any]] = []
    for ref_id in doc_order:
        doc_chunks = doc_groups[ref_id]
        first = doc_chunks[0]
        file_path = first.get("file_path", "")
        filename = Path(file_path).name if file_path else f"Source {ref_id}"
        metadata = first.get("metadata") or {}
        meta_parts = [
            f"{key.removeprefix('doc_').replace('_', ' ')}: {value}"
            for key, value in metadata.items()
            if value is not None and str(value).strip()
        ]
        meta_suffix = f" ({', '.join(meta_parts)})" if meta_parts else ""
        workspace = indexer.get_doc_workspace(ref_id) if indexer is not None else None
        workspace_label = f" [workspace: {workspace}]" if workspace else ""
        blocks.append(
            {
                "type": "text",
                "text": f"### Document [{ref_id}]{workspace_label}: {filename}{meta_suffix}",
            }
        )

        for chunk in doc_chunks:
            content = str(chunk.get("content") or "").strip()
            chunk_id = str(chunk.get("chunk_id") or "")
            page_number = chunk.get("page_number")
            image_data = chunk.get("image_data")
            cite_tag = ""
            if indexer is not None and ref_id and chunk_id:
                chunk_index = indexer.get_chunk_idx(ref_id, chunk_id)
                if chunk_index is not None:
                    cite_tag = f"[{ref_id}-{chunk_index}]"

            if image_data:
                if image_blocks_by_context_key is None:
                    image_block = {
                        "type": "image_url",
                        "image_url": {"url": image_data_uri(image_data)},
                    }
                else:
                    image_block = image_blocks_by_context_key.get(
                        context_chunk_key(chunk_id, workspace=chunk.get("_workspace"))
                    )
                if image_block is not None:
                    blocks.append(
                        {
                            "type": "text",
                            "text": build_image_label(
                                cite_tag=cite_tag,
                                chunk=chunk,
                                filename=filename,
                            ),
                        }
                    )
                    blocks.append(image_block)

            if content:
                if cite_tag:
                    label = (
                        f"{cite_tag} {filename}, Page {page_number}"
                        if page_number
                        else f"{cite_tag} {filename}"
                    )
                else:
                    label = f"[{filename}, Page {page_number}]" if page_number else f"[{filename}]"
                blocks.append({"type": "text", "text": f"{label}\n{content}"})

            metadata_line = format_chunk_metadata(chunk)
            if metadata_line:
                blocks.append({"type": "text", "text": metadata_line})
    return blocks


def format_chunk_metadata(
    chunk: dict[str, Any],
    *,
    internal_keys: frozenset[str] = _INTERNAL_KEYS,
) -> str:
    """Serialize non-internal chunk fields into a compact metadata line."""
    extra = {
        key: value
        for key, value in chunk.items()
        if key not in internal_keys
        and not key.startswith("_")
        and value is not None
        and (not isinstance(value, str) or value.strip())
    }
    parts: list[str] = []
    for key, value in extra.items():
        if isinstance(value, dict):
            parts.extend(
                f"{key}.{subkey}={subvalue}"
                for subkey, subvalue in value.items()
                if subvalue is not None and str(subvalue).strip()
            )
        elif isinstance(value, list):
            items = [str(item) for item in value[:5] if str(item).strip()]
            if len(value) > 5:
                items.append(f"...({len(value)} total)")
            parts.append(f"{key}=[{', '.join(items)}]")
        elif isinstance(value, bool | int):
            parts.append(f"{key}={value}")
        elif isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            text = str(value).strip()
            parts.append(f"{key}={text[:117] + '...' if len(text) > 120 else text}")
    return "[meta: " + ", ".join(parts) + "]" if parts else ""


def build_image_label(*, cite_tag: str, chunk: dict[str, Any], filename: str) -> str:
    """Build an enriched image label with sidecar awareness."""
    metadata = chunk.get("metadata") or {}
    title = metadata.get("title", "")
    page_number = chunk.get("page_number")
    sidecar = chunk.get("sidecar")
    parts: list[str] = []
    if cite_tag:
        parts.append(cite_tag)
    if title:
        parts.append(f'"{title}"')
    if page_number is not None:
        parts.append(f"Page {page_number}")
    elif filename:
        parts.append(filename)
    else:
        parts.append("Page image")
    if isinstance(sidecar, dict):
        sidecar_type = sidecar.get("type", "")
        if sidecar_type == "drawing":
            sidecar_id = sidecar.get("id", "")
            parts.append(
                f"(VLM drawing: {sidecar_id[:24]})" if sidecar_id else "(VLM-generated drawing)"
            )
        elif sidecar_type:
            parts.append(f"(sidecar: {sidecar_type})")
    return " ".join(parts)


__all__ = [
    "build_excerpt_lane_blocks",
    "build_image_label",
    "format_chunk_metadata",
    "format_kg_context",
]
