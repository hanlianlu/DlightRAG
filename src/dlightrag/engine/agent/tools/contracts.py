# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The contracts one model-visible tool call is made of."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from dlightrag.engine.agent.session.effects import (
    ReplayPolicy,
    schema_digest,
)
from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tool_content import (
    ToolContent,
    ToolTextPart,
    VisualSource,
    tool_content_text,
)
from dlightrag.engine.ai.messages import AssistantTurn, ToolChoice, ToolDefinition


class ToolModelFunc(Protocol):
    """One provider-neutral tool-capable model turn."""

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
        max_tokens: int | None = None,
    ) -> AssistantTurn: ...


class ToolResultCapacityError(RuntimeError):
    """A model-visible tool result cannot preserve its required content."""


@dataclass(frozen=True, slots=True)
class CommittedOutput:
    """One full tool output promoted from staging to durable storage."""

    resource_id: str
    content_digest: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class WorkspacePathFact:
    """One regular workspace path observed after a tool mutation."""

    relative_path: str
    entry_type: str
    size_bytes: int
    mode: int | None = None
    content_digest: str | None = None


@dataclass(frozen=True, slots=True)
class WorkspaceInventoryFacts:
    """Typed workspace upserts/deletes, optionally replacing the full inventory."""

    upserts: tuple[WorkspacePathFact, ...] = ()
    deletes: tuple[str, ...] = ()
    replace_all: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceSourceFact:
    """Typed source identity admitted by a resource-backed tool result."""

    resource_id: str
    source_type: str
    source_uri: str
    title: str
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceAttachmentBytes:
    """One verified original snapshot a tool attached to its result."""

    resource_id: str
    filename: str
    mime_type: str
    source_locator: str
    content: bytes
    resource_kind: str = "tool_attachment"
    source: VisualSource | None = None
    """Another durable handle these bytes are already known by, when one exists."""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolEffects:
    """Typed host facts emitted by a tool and consumed only at settlement."""

    committed_outputs: tuple[CommittedOutput, ...] = ()
    workspace_inventory: WorkspaceInventoryFacts | None = None
    evidence_sources: tuple[EvidenceSourceFact, ...] = ()
    attached_resources: tuple[ResourceAttachmentBytes, ...] = ()


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Typed model content plus transport-private execution facts."""

    parts: ToolContent
    details: dict[str, Any] | None = None
    cached: bool = False
    protected_text: str = ""
    is_error: bool = False
    effects: ToolEffects = ToolEffects()

    @classmethod
    def text(
        cls,
        text: str,
        *,
        details: dict[str, Any] | None = None,
        cached: bool = False,
        protected_text: str = "",
        is_error: bool = False,
        effects: ToolEffects = ToolEffects(),
    ) -> ToolResult:
        """Build the common text-only result without weakening typed content."""
        return cls(
            parts=(ToolTextPart(text),),
            details=details,
            cached=cached,
            protected_text=protected_text,
            is_error=is_error,
            effects=effects,
        )

    @property
    def text_content(self) -> str:
        """Return only model-visible text, excluding attachment metadata."""
        return tool_content_text(self.parts)


type ToolUpdateSink = Callable[["ToolResult"], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ToolRuntime:
    """Explicit identity and live-update channel for one executing tool call."""

    call_id: str
    tool_name: str
    intent_id: IntentId
    execution_scope: str
    _update_sink: ToolUpdateSink
    fencing_epoch: int = 0

    async def emit_update(self, result: ToolResult) -> None:
        """Publish one transient result snapshot without settling the effect."""
        await self._update_sink(result)


type ToolExecute = Callable[[BaseModel, ToolRuntime], Awaitable["ToolResult"]]


@dataclass(frozen=True, slots=True)
class AgentTool:
    """Executable tool with a Pydantic argument contract.

    ``replay_policy``, ``contract_version``, and ``input_schema_digest`` are the
    intent facts replay must match exactly. Replay is fail-closed: tools opt in
    only when identical persisted arguments are safe to execute again.
    The digest is the SHA-256 of the canonicalized input schema, so presentation
    fields and declaration order never change it.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    execute: ToolExecute
    replay_policy: ReplayPolicy = "never"
    contract_version: int = 2
    input_schema_digest: str = ""
    guidance: str = ""

    def __post_init__(self) -> None:
        if self.replay_policy not in {"replayable", "never"}:
            raise ValueError("AgentTool replay policy must be replayable or never")
        if self.contract_version < 1:
            raise ValueError("AgentTool contract_version must be positive")
        object.__setattr__(
            self,
            "input_schema_digest",
            schema_digest(self.input_model.model_json_schema()),
        )

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.input_model.model_json_schema(),
        )


@dataclass(frozen=True, slots=True)
class ExecutedTurn:
    """The one assistant turn a Run's last tool-capable model call produced."""

    assistant: AssistantTurn


__all__ = [
    "AgentTool",
    "CommittedOutput",
    "EvidenceSourceFact",
    "ExecutedTurn",
    "ResourceAttachmentBytes",
    "ToolExecute",
    "ToolModelFunc",
    "ToolResult",
    "ToolResultCapacityError",
    "ToolRuntime",
    "ToolUpdateSink",
    "ToolEffects",
    "WorkspaceInventoryFacts",
    "WorkspacePathFact",
]
