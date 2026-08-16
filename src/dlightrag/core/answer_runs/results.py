# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The canonical result of one durable Answer run, stored and projected.

A stored result carries transport-neutral identities -- workspaces, document
ids, chunk ids, and the answer text -- plus, inside the client-safe contexts,
the stable ``/images`` route path, which is a route identity every transport
shares and which authorizes each read on its own. Authorization-dependent
download URLs are never stored, and both image visibility and download links are
re-derived on every authenticated read, so a policy change between execution and
reading is honored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from dlightrag_rag.contracts import Reference
from dlightrag_rag.retrieval import RetrievalContexts

from dlightrag.citations.schemas import SourceReference
from dlightrag.citations.utils import context_chunk_key
from dlightrag.core.answer.media import answer_blocks_from_markdown
from dlightrag.core.answer.projection import can_project_workspace_visual
from dlightrag.core.answer_runs.snapshots import dump_answer_snapshot, load_answer_snapshot
from dlightrag.core.client_payloads import project_contexts_for_client, project_source_payloads
from dlightrag.core.retrieval.source_links import SourceDownloadLinkBuilder

#: Core image route every transport reuses; the route itself authorizes reads.
IMAGE_URL_PREFIX = "/images"


@dataclass
class AnswerResult:
    """Current product Answer result, separate from corpus retrieval output."""

    answer: str
    contexts: RetrievalContexts = field(default_factory=dict)
    references: list[Reference] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    answer_images: list[dict[str, Any]] = field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    image_descriptions: list[str] = field(default_factory=list)


def store_answer_result(
    *,
    answer: str,
    contexts: RetrievalContexts,
    sources: Sequence[SourceReference],
    answer_images: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    image_descriptions: Sequence[str],
) -> dict[str, Any]:
    """Project one finished run into its durable, transport-neutral result.

    ``contexts`` must already be the client-safe projection. Answer blocks are a
    pure function of the answer text and the authorized image set, so they are
    derived on read instead of stored twice.
    """
    workspaces = _image_workspaces(sources)
    return {
        "answer": answer,
        "contexts": dict(contexts),
        "sources": dump_answer_snapshot(list(sources))["sources"],
        "answer_images": [
            _store_answer_image(image, workspaces=workspaces) for image in answer_images
        ],
        "trace": dict(trace),
        "image_descriptions": list(image_descriptions),
    }


def restore_answer_result(stored: Mapping[str, Any]) -> AnswerResult:
    """Rebuild the internal result an in-process caller receives.

    Transports still apply their own visual and download authorization; this
    restores identities and the core image routes, never a download URL.
    """
    sources = load_answer_snapshot(
        {"sources": stored.get("sources") or []}, image_url_prefix=IMAGE_URL_PREFIX
    )
    answer = str(stored.get("answer") or "")
    images = [_public_answer_image(image) for image in stored.get("answer_images") or () if image]
    return AnswerResult(
        answer=answer,
        contexts=dict(stored.get("contexts") or {}),
        references=[Reference(id=source.id, title=source.title or "Source") for source in sources],
        sources=sources,
        answer_images=images,
        answer_blocks=answer_blocks_from_markdown(answer, images),
        trace=dict(stored.get("trace") or {}),
        image_descriptions=list(stored.get("image_descriptions") or ()),
    )


def project_answer_result(
    stored: Mapping[str, Any],
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
) -> dict[str, Any]:
    """Project a stored result for one authenticated reader."""
    sources = load_answer_snapshot(
        {"sources": stored.get("sources") or []}, image_url_prefix=IMAGE_URL_PREFIX
    )
    source_payloads = project_source_payloads(
        sources,
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer = str(stored.get("answer") or "")
    images = [
        _public_answer_image(image)
        for image in stored.get("answer_images") or ()
        if image and can_project_workspace_visual(image.get("workspace"), visual_workspaces)
    ]
    return {
        "answer": answer,
        "contexts": project_contexts_for_client(
            dict(stored.get("contexts") or {}),
            image_url_prefix=IMAGE_URL_PREFIX,
            visual_workspaces=visual_workspaces,
        ),
        "references": [
            {"id": source.id, "title": source.title or "Source"} for source in source_payloads
        ],
        "sources": [source.model_dump() for source in source_payloads],
        "answer_images": images,
        "answer_blocks": answer_blocks_from_markdown(answer, images),
        "trace": dict(stored.get("trace") or {}),
        "image_descriptions": list(stored.get("image_descriptions") or ()),
    }


def _image_workspaces(sources: Sequence[SourceReference]) -> dict[str, str]:
    """Map every answer-image identifier form back to its owning workspace."""
    workspaces: dict[str, str] = {}
    for source in sources:
        for chunk in source.chunks or []:
            workspaces.setdefault(chunk.chunk_id, source.workspace)
            workspaces[context_chunk_key(chunk.chunk_id, workspace=source.workspace)] = (
                source.workspace
            )
    return workspaces


def _store_answer_image(image: Mapping[str, Any], *, workspaces: dict[str, str]) -> dict[str, Any]:
    """Keep the image's identity and transport state, never its rendered URLs."""
    image_id = str(image.get("id") or "")
    chunk_id = str(image.get("chunk_id") or "")
    return {
        "id": image_id,
        "chunk_id": chunk_id,
        "workspace": workspaces.get(image_id) or workspaces.get(chunk_id) or "",
        "source_ref": str(image.get("source_ref") or ""),
        "label": str(image.get("label") or ""),
        "answer_image_sent": image.get("answer_image_sent") is not False,
    }


def _public_answer_image(image: Mapping[str, Any]) -> dict[str, Any]:
    base = (
        f"{IMAGE_URL_PREFIX}/"
        f"{quote(str(image.get('workspace') or ''), safe='')}/"
        f"{quote(str(image.get('chunk_id') or ''), safe='')}"
    )
    return {
        "id": str(image.get("id") or ""),
        "chunk_id": str(image.get("chunk_id") or ""),
        "source_ref": str(image.get("source_ref") or ""),
        "url": f"{base}?size=full",
        "thumbnail_url": f"{base}?size=thumb",
        "label": str(image.get("label") or ""),
        "answer_image_sent": image.get("answer_image_sent") is not False,
    }


__all__ = [
    "AnswerResult",
    "IMAGE_URL_PREFIX",
    "project_answer_result",
    "restore_answer_result",
    "store_answer_result",
]
