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

from pydantic import BaseModel, Field, field_validator

from dlightrag.answer.citations.schemas import SourceReference, SourceReferencePayload
from dlightrag.answer.citations.utils import context_chunk_key
from dlightrag.answer.media import answer_blocks_from_markdown
from dlightrag.answer.runs.snapshots import dump_answer_snapshot, load_answer_snapshot
from dlightrag.answer.sources import (
    SourceDownloadLinkBuilder,
    can_project_workspace_visual,
    project_contexts_for_client,
    project_source_payloads,
)
from dlightrag.rag.retrieval import RetrievalContexts

#: Core image route every transport reuses; the route itself authorizes reads.
IMAGE_URL_PREFIX = "/images"


class Reference(BaseModel):
    """One document-level reference returned with an Answer result."""

    id: str = Field(description="Reference id matching the answer context")
    title: str = Field(description="Document title or filename")

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: object) -> str:
        return str(value)


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
    usage: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


def store_answer_result(
    *,
    answer: str,
    contexts: RetrievalContexts,
    sources: Sequence[SourceReference],
    answer_images: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    image_descriptions: Sequence[str],
    primary_report: str | None = None,
    artifacts: Sequence[Mapping[str, Any]] = (),
    report_sources: Sequence[SourceReference] = (),
) -> dict[str, Any]:
    """Project one finished run into its durable, transport-neutral result.

    ``contexts`` must already be the client-safe projection. Answer blocks are a
    pure function of the answer text and the authorized image set, so they are
    derived on read instead of stored twice.
    """
    workspaces = _image_workspaces(sources)
    usage = trace.get("usage")
    return {
        "answer": answer,
        "contexts": dict(contexts),
        "sources": dump_answer_snapshot(list(sources))["sources"],
        "answer_images": [
            _store_answer_image(image, workspaces=workspaces) for image in answer_images
        ],
        "trace": dict(trace),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
        "evidence": _evidence_summary(contexts, source_count=len(sources)),
        "image_descriptions": list(image_descriptions),
        "primary_report": primary_report,
        "artifacts": [dict(item) for item in artifacts],
        "report_sources": dump_answer_snapshot(list(report_sources))["sources"]
        if report_sources
        else [],
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
        usage=dict(stored.get("usage") or {}),
        evidence=dict(stored.get("evidence") or {}),
    )


def project_answer_result(
    stored: Mapping[str, Any],
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_url_prefix: str = IMAGE_URL_PREFIX,
) -> dict[str, Any]:
    """Project a stored result for one authenticated reader."""
    sources = load_answer_snapshot(
        {"sources": stored.get("sources") or []}, image_url_prefix=image_url_prefix
    )
    source_payloads = project_source_payloads(
        sources,
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer = str(stored.get("answer") or "")
    report_handle = stored.get("primary_report")
    primary_report = report_handle.strip() if isinstance(report_handle, str) else None
    if not primary_report:
        primary_report = None
    images = [
        _public_answer_image(image, image_url_prefix=image_url_prefix)
        for image in stored.get("answer_images") or ()
        if image and can_project_workspace_visual(image.get("workspace"), visual_workspaces)
    ]
    return {
        "answer": answer,
        "contexts": project_contexts_for_client(
            dict(stored.get("contexts") or {}),
            image_url_prefix=image_url_prefix,
            visual_workspaces=visual_workspaces,
        ),
        "references": [
            {"id": source.id, "title": source.title or "Source"} for source in source_payloads
        ],
        "sources": [source.model_dump() for source in source_payloads],
        "answer_images": images,
        "answer_blocks": answer_blocks_from_markdown(answer, images),
        "trace": dict(stored.get("trace") or {}),
        "usage": dict(stored.get("usage") or {}),
        "evidence": dict(stored.get("evidence") or {}),
        "image_descriptions": list(stored.get("image_descriptions") or ()),
        "primary_report": primary_report,
        "artifacts": [
            {
                "resource_id": item.get("resource_id"),
                "kind": item.get("kind"),
                "filename": item.get("filename"),
                "media_type": item.get("media_type"),
            }
            for item in stored.get("artifacts") or ()
            if isinstance(item, Mapping)
        ],
    }


def _evidence_summary(
    contexts: Mapping[str, Sequence[Mapping[str, Any]]], *, source_count: int
) -> dict[str, int]:
    """Return the transport-neutral evidence counts every surface exposes."""
    return {
        "chunks": len(contexts.get("chunks") or ()),
        "entities": len(contexts.get("entities") or ()),
        "relationships": len(contexts.get("relationships") or ()),
        "sources": source_count,
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


def _public_answer_image(
    image: Mapping[str, Any],
    *,
    image_url_prefix: str = IMAGE_URL_PREFIX,
) -> dict[str, Any]:
    base = (
        f"{image_url_prefix.rstrip('/')}/"
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


def project_report_sources(
    stored: Mapping[str, Any],
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_url_prefix: str = IMAGE_URL_PREFIX,
) -> list[SourceReferencePayload]:
    """Project citation sources for the Primary Report without leaking them on REST."""
    return project_source_payloads(
        load_answer_snapshot(
            {"sources": stored.get("report_sources") or []},
            image_url_prefix=image_url_prefix,
        ),
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )


__all__ = [
    "AnswerResult",
    "IMAGE_URL_PREFIX",
    "Reference",
    "project_answer_result",
    "project_report_sources",
    "restore_answer_result",
    "store_answer_result",
]
