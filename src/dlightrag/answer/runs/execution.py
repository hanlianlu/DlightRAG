# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The immutable input and durable turn boundaries of one Answer run.

This is the seam between the coordinator, which owns durability, and the answer
orchestrator, which owns retrieval and synthesis. It holds no lifecycle state of
its own: the run row remains authoritative for status, turns, and cancellation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag_ai.capacity import ModelProfile
from dlightrag_ai.fingerprints import ModelFingerprint

from dlightrag.answer.resources.models import ResourceInput
from dlightrag.runtime.errors import RunExecutionError


@dataclass(frozen=True, slots=True)
class AttachmentReference:
    """One ordered current attachment, addressed by owner artifact digest."""

    digest: str
    filename: str
    mime_type: str
    ordinal: int
    byte_size: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "ordinal": self.ordinal,
            "byte_size": self.byte_size,
        }

    @property
    def resource_id(self) -> str:
        return f"attachment-{self.ordinal}"

    @property
    def history_resource_id(self) -> str:
        return f"history-attachment-{self.ordinal}"


async def _current_attachment_resource(
    reference: AttachmentReference,
    load: Callable[[], Awaitable[bytes]],
) -> ResourceInput:
    """Rebuild one current attachment under its durable declared MIME."""
    if reference.mime_type.strip().casefold().startswith("image/"):
        return ResourceInput(
            filename=reference.filename,
            declared_mime=reference.mime_type,
            content=await load(),
        )
    return ResourceInput(
        filename=reference.filename,
        declared_mime=reference.mime_type,
        loader=load,
    )


@dataclass(frozen=True, slots=True)
class LinkReference:
    """One ordered HTTPS attachment link, kept inert until an explicit read."""

    url: str
    filename: str | None
    ordinal: int
    mime_type: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "filename": self.filename,
            "ordinal": self.ordinal,
            "mime_type": self.mime_type,
        }


def in_memory_attachment_loader(content: bytes) -> Callable[[], Awaitable[bytes]]:
    """Return a stable async loader over bytes already admitted in memory."""

    async def load() -> bytes:
        return content

    return load


async def build_current_answer_resources(
    *,
    links: Sequence[LinkReference],
    attachments: Sequence[AttachmentReference],
    attachment_loaders: Sequence[Callable[[], Awaitable[bytes]]],
) -> list[ResourceInput]:
    """Rebuild links and current attachments exactly as the durable run will."""
    if len(attachments) != len(attachment_loaders):
        raise ValueError("current attachment references and loaders must have equal length")
    resources = [
        ResourceInput(
            filename=link.filename,
            url=link.url,
            declared_mime=link.mime_type,
        )
        for link in links
    ]
    for reference, load in zip(attachments, attachment_loaders, strict=True):
        resources.append(await _current_attachment_resource(reference, load))
    return resources


@dataclass(frozen=True, slots=True)
class PinnedModelProfile:
    """One accepted run's immutable model identity and capacity facts."""

    role: str
    fingerprint: ModelFingerprint
    profile: ModelProfile

    def as_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "fingerprint": {
                "provider": self.fingerprint.provider,
                "model": self.fingerprint.model,
                "endpoint_fingerprint": self.fingerprint.endpoint_fingerprint,
            },
            "profile": {
                "context_window_tokens": self.profile.context_window_tokens,
                "max_input_tokens": self.profile.max_input_tokens,
                "max_output_tokens": self.profile.max_output_tokens,
                "supports_images": self.profile.supports_images,
                "supports_tools": self.profile.supports_tools,
                "supports_reasoning": self.profile.supports_reasoning,
            },
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> PinnedModelProfile:
        fingerprint = value.get("fingerprint")
        profile = value.get("profile")
        if not isinstance(fingerprint, Mapping) or not isinstance(profile, Mapping):
            raise ValueError("pinned model profile requires fingerprint and profile objects")
        return cls(
            role=str(value.get("role") or ""),
            fingerprint=ModelFingerprint(
                provider=str(fingerprint.get("provider") or ""),
                model=str(fingerprint.get("model") or ""),
                endpoint_fingerprint=(
                    str(fingerprint["endpoint_fingerprint"])
                    if fingerprint.get("endpoint_fingerprint") is not None
                    else None
                ),
            ),
            profile=ModelProfile(
                context_window_tokens=int(profile["context_window_tokens"]),
                max_input_tokens=_optional_int(profile.get("max_input_tokens")),
                max_output_tokens=_optional_int(profile.get("max_output_tokens")),
                supports_images=bool(profile.get("supports_images")),
                supports_tools=bool(profile.get("supports_tools")),
                supports_reasoning=bool(profile.get("supports_reasoning")),
            ),
        )


@dataclass(frozen=True, slots=True)
class AnswerRunRequest:
    """Normalized public request before model resolution and history projection."""

    query: str
    workspaces: tuple[str, ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    top_k: int | None = None
    chunk_top_k: int | None = None
    filters: Mapping[str, Any] | None = None
    semantic_highlights: bool = False
    links: tuple[LinkReference, ...] = ()
    attachments: tuple[AttachmentReference, ...] = ()
    history_attachments: tuple[AttachmentReference, ...] = ()

    def as_request(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workspaces": list(self.workspaces),
            "history": [dict(message) for message in self.history],
            "top_k": self.top_k,
            "chunk_top_k": self.chunk_top_k,
            "filters": dict(self.filters) if self.filters else None,
            "semantic_highlights": self.semantic_highlights,
            "links": [item.as_json() for item in self.links],
            "attachments": [item.as_json() for item in self.attachments],
            "history_attachments": [item.as_json() for item in self.history_attachments],
        }

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> AnswerRunRequest:
        """Project the public request fields without parsing execution metadata."""
        filters = request.get("filters")
        return cls(
            query=str(request.get("query") or ""),
            workspaces=tuple(str(value) for value in request.get("workspaces") or ()),
            history=tuple(dict(message) for message in request.get("history") or ()),
            top_k=_optional_int(request.get("top_k")),
            chunk_top_k=_optional_int(request.get("chunk_top_k")),
            filters=dict(filters) if isinstance(filters, Mapping) else None,
            semantic_highlights=bool(request.get("semantic_highlights")),
            links=_link_references(request.get("links")),
            attachments=_attachment_references(request.get("attachments")),
            history_attachments=_attachment_references(request.get("history_attachments")),
        )


@dataclass(frozen=True, slots=True)
class AnswerRunInput:
    """The normalized, immutable request one accepted run executes.

    Workspace authorization is evaluated once before the run is accepted, so the
    stored input carries the resulting workspace set and never a token, mutable
    claim, transport header, temporary path, or authorization-dependent URL.
    """

    query: str
    pinned_models: tuple[PinnedModelProfile, ...]
    context_policy_revision: str
    model_catalog_revision: str
    idempotency_fingerprint: str
    workspaces: tuple[str, ...] = ()
    history: tuple[Mapping[str, Any], ...] = ()
    top_k: int | None = None
    chunk_top_k: int | None = None
    filters: Mapping[str, Any] | None = None
    semantic_highlights: bool = False
    links: tuple[LinkReference, ...] = ()
    attachments: tuple[AttachmentReference, ...] = ()
    #: Earlier conversation uploads this run may read but never sends as a
    #: current-turn image; they point at artifacts an earlier run already stored.
    history_attachments: tuple[AttachmentReference, ...] = ()
    image_descriptions: tuple[str, ...] = ()
    #: The UUIDv7 Research session id pinned at acceptance (M3-D10).
    session_id: str = ""
    #: Whether the accepted run takes the durable research path.
    research: bool = False
    #: The accepted resource manifest, present for research runs.
    resource_manifest: tuple[Mapping[str, Any], ...] = ()

    def as_request(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "workspaces": list(self.workspaces),
            "history": [dict(message) for message in self.history],
            "top_k": self.top_k,
            "chunk_top_k": self.chunk_top_k,
            "filters": dict(self.filters) if self.filters else None,
            "semantic_highlights": self.semantic_highlights,
            "links": [item.as_json() for item in self.links],
            "attachments": [item.as_json() for item in self.attachments],
            "history_attachments": [item.as_json() for item in self.history_attachments],
            "pinned_models": [item.as_json() for item in self.pinned_models],
            "context_policy_revision": self.context_policy_revision,
            "model_catalog_revision": self.model_catalog_revision,
            "idempotency_fingerprint": self.idempotency_fingerprint,
            "image_descriptions": list(self.image_descriptions),
            "session_id": self.session_id,
            "research": self.research,
            "resource_manifest": [dict(item) for item in self.resource_manifest],
        }

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> AnswerRunInput:
        attachments = _attachment_references(request.get("attachments"))
        history_attachments = _attachment_references(request.get("history_attachments"))
        links = _link_references(request.get("links"))
        filters = request.get("filters")
        pinned_models = tuple(
            PinnedModelProfile.from_json(item) for item in request.get("pinned_models") or ()
        )
        context_policy_revision = str(request.get("context_policy_revision") or "")
        model_catalog_revision = str(request.get("model_catalog_revision") or "")
        idempotency_fingerprint = str(request.get("idempotency_fingerprint") or "")
        if (
            not pinned_models
            or not context_policy_revision
            or not model_catalog_revision
            or not idempotency_fingerprint
        ):
            raise ValueError("answer run input is missing pinned model capacity facts")
        return cls(
            query=str(request.get("query") or ""),
            pinned_models=pinned_models,
            context_policy_revision=context_policy_revision,
            model_catalog_revision=model_catalog_revision,
            idempotency_fingerprint=idempotency_fingerprint,
            workspaces=tuple(str(value) for value in request.get("workspaces") or ()),
            history=tuple(dict(message) for message in request.get("history") or ()),
            top_k=_optional_int(request.get("top_k")),
            chunk_top_k=_optional_int(request.get("chunk_top_k")),
            filters=dict(filters) if isinstance(filters, Mapping) else None,
            semantic_highlights=bool(request.get("semantic_highlights")),
            links=links,
            attachments=attachments,
            history_attachments=history_attachments,
            image_descriptions=tuple(str(item) for item in request.get("image_descriptions") or ()),
            session_id=str(request.get("session_id") or ""),
            research=bool(request.get("research")),
            resource_manifest=tuple(dict(item) for item in request.get("resource_manifest") or ()),
        )

    @classmethod
    def from_prepared_input(cls, prepared: Mapping[str, Any] | None) -> AnswerRunInput:
        """Decode one durable prepared input into the immutable run input."""
        if prepared is None:
            raise RunExecutionError(
                "run_execution_failed", "Answer run has no prepared input to execute."
            )
        return cls.from_request(prepared)


def _attachment_references(value: Any) -> tuple[AttachmentReference, ...]:
    return tuple(
        AttachmentReference(
            digest=str(item["digest"]),
            filename=str(item["filename"]),
            mime_type=str(item["mime_type"]),
            ordinal=int(item["ordinal"]),
            byte_size=int(item.get("byte_size") or 0),
        )
        for item in value or ()
    )


def _link_references(value: Any) -> tuple[LinkReference, ...]:
    return tuple(
        LinkReference(
            url=str(item["url"]),
            filename=(str(item["filename"]) if item.get("filename") else None),
            ordinal=int(item["ordinal"]),
            mime_type=(str(item["mime_type"]) if item.get("mime_type") else None),
        )
        for item in value or ()
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "AnswerRunInput",
    "AnswerRunRequest",
    "AttachmentReference",
    "build_current_answer_resources",
    "in_memory_attachment_loader",
    "LinkReference",
    "PinnedModelProfile",
]
