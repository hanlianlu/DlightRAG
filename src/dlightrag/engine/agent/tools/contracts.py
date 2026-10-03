# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The contracts one model-visible tool call is made of."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode

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
    attributes: tuple[tuple[str, str], ...] = ()
    """Facts the Resource's row records beside its kind, such as how the bytes were acquired."""


@dataclass(frozen=True, slots=True)
class ToolEffects:
    """Typed host facts emitted by a tool and consumed only at settlement."""

    committed_outputs: tuple[CommittedOutput, ...] = ()
    workspace_inventory: WorkspaceInventoryFacts | None = None
    evidence_sources: tuple[EvidenceSourceFact, ...] = ()
    attached_resources: tuple[ResourceAttachmentBytes, ...] = ()


TOOL_SUBJECT_MAX_CHARS = 64
"""Longest Tool Subject one producer may report; the browser edge keeps its own cap."""

_SUBJECT_DROPPED_CHARS = {code: None for code in (*range(0x20), 0x7F, *range(0x80, 0xA0))}


def _bounded_tool_subject(value: str) -> str:
    """Normalize one reported subject into the single bounded line the trace shows.

    Control characters are dropped rather than escaped: they are invisible in a one-line
    row, and U+0000 is not representable in the durable JSON payload at all.
    """
    line = " ".join(value.split()).translate(_SUBJECT_DROPPED_CHARS)
    if len(line) <= TOOL_SUBJECT_MAX_CHARS:
        return line
    return line[: TOOL_SUBJECT_MAX_CHARS - 1] + "…"


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Typed model content plus transport-private execution facts."""

    parts: ToolContent
    details: dict[str, Any] | None = None
    subject: str | None = None
    """One bounded line naming what this call acts on, when the tool reports one.

    It reaches a viewer only through an update, never through a settled result: it exists
    for the live activity row. Because it is the one producer-reported fact that crosses to
    the browser, the bound belongs to the field rather than to each producer's discipline.
    """
    cached: bool = False
    protected_text: str = ""
    is_error: bool = False
    effects: ToolEffects = ToolEffects()

    def __post_init__(self) -> None:
        if self.subject is not None:
            object.__setattr__(self, "subject", _bounded_tool_subject(self.subject))

    @classmethod
    def text(
        cls,
        text: str,
        *,
        details: dict[str, Any] | None = None,
        subject: str | None = None,
        cached: bool = False,
        protected_text: str = "",
        is_error: bool = False,
        effects: ToolEffects = ToolEffects(),
    ) -> ToolResult:
        """Build the common text-only result without weakening typed content."""
        return cls(
            parts=(ToolTextPart(text),),
            details=details,
            subject=subject,
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
type SourceOrder = Callable[[], Awaitable[None]]
"""Wait until every earlier call running beside this one has returned."""


async def already_in_source_order() -> None:
    """The source order of a call that runs alone: nothing precedes it."""


@dataclass(frozen=True, slots=True)
class ToolRuntime:
    """Explicit identity and live-update channel for one executing tool call."""

    call_id: str
    tool_name: str
    intent_id: IntentId
    execution_scope: str
    _update_sink: ToolUpdateSink
    fencing_epoch: int = 0
    _in_source_order: SourceOrder = already_in_source_order

    async def emit_update(self, result: ToolResult) -> None:
        """Publish one transient result snapshot without settling the effect."""
        await self._update_sink(result)

    async def in_source_order(self) -> None:
        """Wait until the calls before this one in its batch have returned.

        A read-only call runs beside its read-only neighbours only until it touches
        state they share, such as the Run's evidence, image budget, or trace. It
        awaits this first, so those calls touch that state one at a time in source
        order and each sees exactly what it would have seen running alone. A call
        that runs alone returns at once.
        """
        await self._in_source_order()


type ToolExecute = Callable[[BaseModel, ToolRuntime], Awaitable["ToolResult"]]


def _without_titles(node: Any) -> Any:
    """Drop every ``title`` annotation. A field named ``title`` maps to an object, not a string."""
    if isinstance(node, dict):
        return {
            key: _without_titles(value)
            for key, value in node.items()
            if not (key == "title" and isinstance(value, str))
        }
    if isinstance(node, list):
        return [_without_titles(item) for item in node]
    return node


class _ToolSchema(GenerateJsonSchema):
    """The schema a model is shown for a Tool's arguments.

    Pydantic titles every field and model after its own name, which tells a model
    nothing its field names and descriptions do not. A Connection Tool's schema is
    published, not generated, so this generator never reaches it and the remote
    server's own titles stay.
    """

    def generate(self, schema: Any, mode: JsonSchemaMode = "validation") -> dict[str, Any]:
        return _without_titles(super().generate(schema, mode))


@dataclass(frozen=True, slots=True)
class ToolDeclaration:
    """Pure tool declaration with a Pydantic argument contract.

    ``replay_policy``, ``read_only``, ``contract_version``, and
    ``input_schema_digest`` are the intent facts execution and replay must match
    exactly. Replay is fail-closed: tools opt in only when identical persisted
    arguments are safe to execute again. The digest is the SHA-256 of the
    canonicalized input schema, so presentation fields and declaration order never
    change it. The ``definition`` a model is shown is generated once, with the digest,
    from that same schema.

    ``read_only`` is fail-closed the same way. A read-only call changes nothing
    outside its own Run's record of what it read, so adjacent read-only calls of one
    batch run at once; every other call runs alone and is a barrier between them. A
    read-only Tool awaits ``ToolRuntime.in_source_order`` before it touches Run
    state its neighbours share. Only the Tool's own implementation can declare it:
    nothing infers it from a name or from a remote server's hints.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    replay_policy: ReplayPolicy = "never"
    contract_version: int = 2
    input_schema_digest: str = field(init=False)
    read_only: bool = False
    definition: ToolDefinition = field(init=False, compare=False)

    def __post_init__(self) -> None:
        if self.replay_policy not in {"replayable", "never"}:
            raise ValueError("AgentTool replay policy must be replayable or never")
        if self.contract_version < 1:
            raise ValueError("AgentTool contract_version must be positive")
        schema = self.input_model.model_json_schema(schema_generator=_ToolSchema)
        object.__setattr__(self, "input_schema_digest", schema_digest(schema))
        object.__setattr__(
            self,
            "definition",
            ToolDefinition(name=self.name, description=self.description, parameters=schema),
        )

    def bind(self, execute: ToolExecute) -> AgentTool:
        """Bind this exact declaration to one run's execution capability."""
        return AgentTool(
            self.name,
            self.description,
            self.input_model,
            replay_policy=self.replay_policy,
            contract_version=self.contract_version,
            read_only=self.read_only,
            execute=execute,
        )


@dataclass(frozen=True, slots=True)
class AgentTool(ToolDeclaration):
    """A declared tool with its required, run-local execution binding."""

    execute: ToolExecute = field(kw_only=True)


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
    "SourceOrder",
    "ToolDeclaration",
    "ToolExecute",
    "ToolModelFunc",
    "ToolResult",
    "ToolResultCapacityError",
    "ToolRuntime",
    "ToolUpdateSink",
    "ToolEffects",
    "WorkspaceInventoryFacts",
    "WorkspacePathFact",
    "already_in_source_order",
]
