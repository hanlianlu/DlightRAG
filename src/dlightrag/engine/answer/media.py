# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Derive Evidence Images from what a Run cited, and from what it looked at.

Two provenances, one registry. A corpus chunk contributes the visual asset retrieval selected for
it; the Run's own held resources contribute whatever the Answer places by name. Both reach the
reader through the same ``evidence:`` placement grammar, so a surface that can draw one can draw
the other without knowing which is which.
"""

import logging
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from dlightrag.engine.answer.citations.contracts import SourceReference
from dlightrag.engine.answer.citations.sources import can_project_workspace_visual
from dlightrag.engine.answer.citations.utils import context_chunk_key
from dlightrag.engine.answer.reference import markdown_references
from dlightrag.engine.rag.retrieval.results import (
    RetrievalContexts,
)

logger = logging.getLogger(__name__)


def evidence_images_from_sources(
    sources: list[SourceReference],
    *,
    contexts: RetrievalContexts | None = None,
    visual_workspaces: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return cited visual assets in a transport-neutral shape."""
    sent = _answer_image_sent_by_chunk(contexts)
    chunk_counts = Counter(
        chunk.chunk_id for source in sources for chunk in (source.chunks or []) if chunk.chunk_id
    )
    seen: set[str] = set()
    images: list[dict[str, Any]] = []
    for source in sources:
        if not can_project_workspace_visual(source.workspace, visual_workspaces):
            continue
        source_id = source.id
        base_label = source.title or source_id
        for chunk in source.chunks or []:
            chunk_id = chunk.chunk_id
            image_id = context_chunk_key(chunk_id, workspace=source.workspace)
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
            page_number = getattr(chunk, "page_number", None)
            label = f"{base_label} · Page {page_number}" if page_number else base_label
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


def held_evidence_images(
    answer: str,
    *,
    cited: Sequence[Mapping[str, Any]] = (),
    held: Collection[str],
    run_id: str,
) -> list[dict[str, Any]]:
    """Return the Run-held images this Answer places, named by the Resource ids it writes.

    A corpus chunk is evidence because retrieval chose it. A screenshot, or a picture the Agent
    viewed on a page it fetched, is evidence because the Agent looked at it -- and until recently
    the second kind had no way into the Answer at all: a Research Run on an empty corpus could read
    a dozen photographs and place none of them, which left the Model describing pictures in prose
    because the product gave it nowhere to put them.

    Placement is the whole selection rule, and it is the Model's to make: a Run holds every image on
    every page it read -- logos, thumbnails, furniture -- so admitting them all would bury the
    Answer, while naming one keeps the judgement where the reader's question was asked.

    ``cited`` is what retrieval already derived for this Answer, and it decides one thing: the
    ``evidence:`` namespace is shared by both provenances, so an id retrieval has already claimed
    belongs to the corpus and must not be claimed here as well. Two records for one id would collide
    in the projection, and the corpus figure would be addressed as a Run-held Resource -- a route
    that answers not-found, which turns a picture that used to render into one that does not.

    ``held`` is the ids of the images this Run attached for its Model, read from the rows its
    run-resource route serves, so an admitted id is one a reader can open. The Model writes these ids
    itself and can write one wrong, or one it only imagined; such an id gets no record, so the reader
    sees the marker's alt text where a broken image would have been, and the id is logged against
    ``run_id``.
    """
    from_corpus = {str(image.get("id") or "") for image in cited if str(image.get("id") or "")}
    placed = _placed_evidence_ids(answer) - from_corpus
    admitted = placed & set(held)
    if placed > admitted:
        logger.warning(
            "Answer of run %s places evidence ids this Run holds no image for: %s",
            run_id,
            ", ".join(sorted(placed - admitted)),
        )
    images: list[dict[str, Any]] = []
    for evidence_id in sorted(admitted):
        images.append(
            {
                "id": evidence_id,
                "resource_id": evidence_id,
                "label": "",
                "workspace": "",
                "chunk_id": "",
                "source_ref": "",
                # The Run prepared these pixels before attaching them, so there is nothing left for
                # the Answer stage to compress and nothing to describe to a model.
                "answer_image_sent": True,
            }
        )
    return images


def _placed_evidence_ids(answer: str) -> set[str]:
    """Every ``evidence:`` id this Markdown names, as the one placement grammar reads it."""
    ids: set[str] = set()
    for reference in markdown_references(answer):
        scheme, _, evidence_id = reference.target.partition(":")
        if scheme.lower() == "evidence" and evidence_id:
            ids.add(evidence_id)
    return ids


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


__all__ = ["evidence_images_from_sources", "held_evidence_images"]
