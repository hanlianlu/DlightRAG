# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared citation-labelled evidence rendering."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dlightrag.engine.answer.citations.indexer import CitationIndexer
from dlightrag.engine.answer.citations.utils import REQUEST_OWNED_WORKSPACES, context_chunk_key
from dlightrag.engine.network_admission import validate_public_web_url
from dlightrag.engine.rag.retrieval import RetrievalContexts

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


def format_kg_context(contexts: RetrievalContexts, indexer: CitationIndexer) -> str:
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


def _source_tags(row: dict[str, Any], indexer: CitationIndexer) -> str:
    tags = indexer.get_doc_tags(
        row.get("source_id"),
        workspace=row.get("_workspace"),
    )
    return f" (from {', '.join(tags)})" if tags else ""


def build_excerpt_lane_blocks(
    chunks: list[dict[str, Any]],
    *,
    indexer: CitationIndexer,
    image_blocks_by_context_key: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Render one evidence lane without changing chunk order.

    A chunk's image is sent only as the block an image budget admitted for it, so a
    row's raw pixels never reach a request unbounded.
    """
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
        heading, filename = _document_heading(ref_id, doc_chunks[0], indexer)
        blocks.append({"type": "text", "text": heading})

        for chunk in doc_chunks:
            content = str(chunk.get("content") or "").strip()
            chunk_id = str(chunk.get("chunk_id") or "")
            cite_tag = ""
            if ref_id and chunk_id:
                chunk_index = indexer.get_chunk_idx(ref_id, chunk_id)
                if chunk_index is not None:
                    cite_tag = f"[{ref_id}-{chunk_index}]"

            if chunk.get("image_data"):
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
                label = chunk_label(cite_tag=cite_tag, chunk=chunk, filename=filename)
                blocks.append({"type": "text", "text": f"{label}\n{content}"})

            metadata_line = format_chunk_metadata(chunk)
            if metadata_line:
                blocks.append({"type": "text", "text": metadata_line})
    return blocks


#: Document metadata a heading leaves out: what citations and storage keep for
#: themselves — where a source is stored, the name it is stored under, how it was
#: acquired — and the resource handle, which the heading prints in its own form.
#: How a row reached the ledger is the row's own ``_`` bookkeeping, never metadata.
_UNLISTED_METADATA: frozenset[str] = frozenset(
    {
        "resource_id",
        "source_download_locator",
        "source_file_name",
        "source_type",
        "resource_kind",
        "admission_origin",
        "acquisition",
        "remote_image_url",
    }
)


def document_name(row: Mapping[str, Any]) -> str:
    """The one name a document is shown, labelled and cited by.

    A corpus document is named by its file. A page or resource the request holds
    itself is named by its title, which is no path to cut at a slash.
    """
    file_path = str(row.get("file_path") or "")
    request_owned = row.get("_workspace") in REQUEST_OWNED_WORKSPACES
    name = file_path if request_owned else Path(file_path).name
    return name or f"Source {row.get('reference_id') or ''}".rstrip()


def source_handle(row: Mapping[str, Any]) -> str:
    """A document's citation number, name and re-readable handle on one line."""
    resource_id = (row.get("metadata") or {}).get("resource_id")
    handle = f" [resource: {resource_id}]" if resource_id else ""
    return f"[{row.get('reference_id') or ''}] {document_name(row)}{handle}"


def _linkable(uri: object) -> bool:
    """Whether an answer may link this address, by the test its citation links pass."""
    try:
        validate_public_web_url(str(uri))
    except ValueError:
        return False
    return True


def _document_heading(
    ref_id: str, chunk: Mapping[str, Any], indexer: CitationIndexer
) -> tuple[str, str]:
    """Return a document's heading and the name its passages are labelled with.

    A corpus document is also labelled with its workspace; a request's own pages and
    resources have none. The metadata describes the document: an address an answer
    may link stays, while any other source uri, such as a ``local://`` locator or a
    private host, is storage. A value the name already gives is not repeated.
    """
    metadata = chunk.get("metadata") or {}
    name = document_name(chunk)
    request_owned = chunk.get("_workspace") in REQUEST_OWNED_WORKSPACES
    workspace = None if request_owned else indexer.get_doc_workspace(ref_id)
    resource_id = metadata.get("resource_id")
    described = [
        f"{key.removeprefix('doc_').replace('_', ' ')}: {value}"
        for key, value in metadata.items()
        if key not in _UNLISTED_METADATA
        and (key != "source_uri" or _linkable(value))
        and value is not None
        and str(value).strip()
        and str(value) != name
    ]
    heading = f"### Document [{ref_id}]"
    if workspace:
        heading += f" [workspace: {workspace}]"
    heading += f": {name}"
    if resource_id:
        heading += f" [resource: {resource_id}]"
    if described:
        heading += f" ({', '.join(described)})"
    return heading, name


def chunk_label(*, cite_tag: str, chunk: dict[str, Any], filename: str) -> str:
    """Return one excerpt's label line: the marker a Citation Contract names.

    The label is rendered in exactly one place because a passage that a Tool result
    already carries is labelled where it stands instead of being rendered again.
    """
    page_number = chunk.get("page_number")
    if cite_tag:
        return (
            f"{cite_tag} {filename}, Page {page_number}"
            if page_number
            else f"{cite_tag} {filename}"
        )
    return f"[{filename}, Page {page_number}]" if page_number else f"[{filename}]"


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
    "document_name",
    "build_image_label",
    "chunk_label",
    "format_chunk_metadata",
    "source_handle",
    "format_kg_context",
]
