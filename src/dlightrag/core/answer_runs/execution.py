# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The immutable input and durable turn boundaries of one Answer run.

This is the seam between the coordinator, which owns durability, and the answer
orchestrator, which owns retrieval and synthesis. It holds no lifecycle state of
its own: the run row remains authoritative for status, turns, and cancellation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dlightrag.core.answer_runs.models import AgentRunState

#: Encode one control turn's restorable state into its checkpoint envelope.
type CheckpointEncoder = Callable[[AgentRunState], Awaitable[Mapping[str, Any]]]


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


@dataclass(frozen=True, slots=True)
class AnswerRunInput:
    """The normalized, immutable request one accepted run executes.

    Workspace authorization is evaluated once before the run is accepted, so the
    stored input carries the resulting workspace set and never a token, mutable
    claim, transport header, temporary path, or authorization-dependent URL.
    """

    query: str
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
    def from_request(cls, request: Mapping[str, Any]) -> AnswerRunInput:
        attachments = _attachment_references(request.get("attachments"))
        history_attachments = _attachment_references(request.get("history_attachments"))
        links = tuple(
            LinkReference(
                url=str(item["url"]),
                filename=(str(item["filename"]) if item.get("filename") else None),
                ordinal=int(item["ordinal"]),
                mime_type=(str(item["mime_type"]) if item.get("mime_type") else None),
            )
            for item in request.get("links") or ()
        )
        filters = request.get("filters")
        return cls(
            query=str(request.get("query") or ""),
            workspaces=tuple(str(value) for value in request.get("workspaces") or ()),
            history=tuple(dict(message) for message in request.get("history") or ()),
            top_k=_optional_int(request.get("top_k")),
            chunk_top_k=_optional_int(request.get("chunk_top_k")),
            filters=dict(filters) if isinstance(filters, Mapping) else None,
            semantic_highlights=bool(request.get("semantic_highlights")),
            links=links,
            attachments=attachments,
            history_attachments=history_attachments,
        )


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


class SessionBoundaries:
    """Turn the orchestrator's agent boundaries into durable, fenced writes."""

    def __init__(self, session: Any, *, encode: CheckpointEncoder) -> None:
        self._session = session
        self._encode = encode

    async def enter_phase(self, phase: str) -> None:
        await self._session.check_cancelled()
        await self._session.enter_phase(phase)

    async def turn_completed(self, state: AgentRunState) -> None:
        await self._session.commit_checkpoint(await self._encode(state))

    async def check_cancelled(self) -> None:
        await self._session.check_cancelled()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "AnswerRunInput",
    "AttachmentReference",
    "CheckpointEncoder",
    "LinkReference",
    "SessionBoundaries",
]
