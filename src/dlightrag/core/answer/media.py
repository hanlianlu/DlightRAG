# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer media registry and block helpers."""

from collections import Counter, defaultdict
from typing import Any
from urllib.parse import urlparse

from dlightrag.citations.parser import CITATION_PATTERN, DOC_CITATION_PATTERN
from dlightrag.citations.schemas import SourceReference
from dlightrag.citations.utils import context_chunk_key
from dlightrag.core.retrieval.protocols import RetrievalContexts


def answer_images_from_sources(
    sources: list[SourceReference],
    *,
    contexts: RetrievalContexts | None = None,
) -> list[dict[str, Any]]:
    """Return cited visual assets in a transport-neutral shape."""
    sent = _answer_image_sent_by_chunk(contexts)
    chunk_counts = Counter(
        chunk.chunk_id for source in sources for chunk in (source.chunks or []) if chunk.chunk_id
    )
    seen: set[str] = set()
    images: list[dict[str, Any]] = []
    for source in sources:
        source_id = source.id
        label = source.title or source_id
        for chunk in source.chunks or []:
            chunk_id = chunk.chunk_id
            image_id = context_chunk_key(chunk_id, workspace=source.workspace)
            # Transport state no longer gates gallery/citation inclusion: a cited
            # visual chunk renders from its stored URL even when its raw image
            # missed the answer-image budget. Only malformed empty chunks (no URL)
            # are excluded; empty-content unsent chunks are already dropped upstream
            # in answer_context packing so they can never be cited here.
            if not chunk_id or image_id in seen:
                continue
            url = _public_render_url(chunk.image_url)
            thumbnail_url = _public_render_url(chunk.thumbnail_url) or url
            if not url and not thumbnail_url:
                continue
            url = url or thumbnail_url
            thumbnail_url = thumbnail_url or url
            chunk_idx = chunk.chunk_idx
            source_ref = f"{source_id}-{chunk_idx}" if chunk_idx else source_id
            images.append(
                {
                    "id": image_id if chunk_counts[chunk_id] > 1 else chunk_id,
                    "chunk_id": chunk_id,
                    "source_ref": source_ref,
                    "url": url,
                    "thumbnail_url": thumbnail_url,
                    "label": label,
                    "answer_image_sent": sent.get(image_id, sent.get(chunk_id, True)),
                }
            )
            seen.add(image_id)
    return images


def answer_blocks_from_markdown(
    answer: str | None, images: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Split answer text with image references after matching citations."""
    text = answer or ""
    if not text:
        return []
    by_ref: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        source_ref = str(image.get("source_ref") or "")
        if source_ref:
            by_ref[source_ref].append(image)
            source_id = source_ref.rsplit("-", 1)[0]
            by_ref[source_id].append(image)

    blocks: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    pos = 0
    matches = [
        (m.start(), m.end(), f"{m.group(1)}-{m.group(2)}") for m in CITATION_PATTERN.finditer(text)
    ]
    matches.extend((m.start(), m.end(), m.group(1)) for m in DOC_CITATION_PATTERN.finditer(text))
    for start, end, ref in sorted(matches):
        if start < pos:
            continue
        while end < len(text) and text[end] in ".,;:!?，。；：！？":
            end += 1
        segment = text[pos:end]
        if segment:
            blocks.append({"type": "markdown", "text": segment})
        for image in by_ref.get(ref, []):
            image_id = str(image.get("id") or "")
            if image_id and image_id not in seen_images:
                blocks.append({"type": "image_ref", "image_id": image_id})
                seen_images.add(image_id)
        pos = end
    if pos < len(text):
        blocks.append({"type": "markdown", "text": text[pos:]})
    return blocks or [{"type": "markdown", "text": text}]


def _answer_image_sent_by_chunk(contexts: RetrievalContexts | None) -> dict[str, bool]:
    if not contexts:
        return {}
    return {
        context_chunk_key(
            chunk.get("chunk_id") or chunk.get("id"),
            workspace=chunk.get("_workspace"),
        ): chunk.get("_answer_image_sent") is not False
        for chunk in contexts.get("chunks", [])
        if chunk.get("chunk_id") or chunk.get("id")
    }


def _public_render_url(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if candidate.startswith("/") and not candidate.startswith("//"):
        return candidate
    return candidate if urlparse(candidate).scheme in {"http", "https"} else None


__all__ = ["answer_blocks_from_markdown", "answer_images_from_sources"]
