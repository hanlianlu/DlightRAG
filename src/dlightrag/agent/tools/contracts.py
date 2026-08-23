# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The contracts one model-visible tool call is made of."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from dlightrag.agent.session.effects import (
    EffectIntent,
    ReplayPolicy,
    ToolResultEntry,
    schema_digest,
)
from dlightrag.agent.session.ids import IntentId
from dlightrag.agent.tool_content import ToolContent, ToolTextPart, tool_content_text
from dlightrag.ai.messages import AssistantTurn, ToolCall, ToolChoice, ToolDefinition


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


@dataclass(frozen=True, slots=True)
class ToolEffects:
    """Typed host facts emitted by a tool and consumed only at settlement."""

    committed_outputs: tuple[CommittedOutput, ...] = ()
    workspace_inventory: WorkspaceInventoryFacts | None = None
    evidence_sources: tuple[EvidenceSourceFact, ...] = ()


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

    async def emit_update(self, result: ToolResult) -> None:
        """Publish one transient result snapshot without settling the effect."""
        await self._update_sink(result)


type ToolExecute = Callable[[BaseModel, ToolRuntime], Awaitable["ToolResult"]]


@dataclass(frozen=True, slots=True)
class AgentTool:
    """Executable tool with a Pydantic argument contract.

    ``replay_policy``, ``contract_version``, and ``input_schema_digest`` are the
    intent facts safe replay must match exactly (M3-D13, M3-D18). The digest is
    the SHA-256 of the canonicalized input schema, so presentation fields and
    declaration order never change it.
    """

    name: str
    description: str
    input_model: type[BaseModel]
    execute: ToolExecute
    replay_policy: ReplayPolicy = "safe"
    contract_version: int = 2
    input_schema_digest: str = ""

    def __post_init__(self) -> None:
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
class ToolObservation:
    """What one tool execution did, with nothing it carried.

    Tool payloads can hold attachment text, provider responses, and redacted
    failure detail, so an observation records only the call's shape: which tool
    ran, how long it took, whether the run had already answered that exact call,
    and how it ended.
    """

    tool: str
    call_id: str
    outcome: str
    duration_ms: float
    cached: bool
    is_error: bool
    content_chars: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "call_id": self.call_id,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "cached": self.cached,
            "is_error": self.is_error,
            "content_chars": self.content_chars,
        }


@dataclass(frozen=True, slots=True)
class ToolExecution:
    call: ToolCall
    result: ToolResult
    observation: ToolObservation
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ExecutedTurn:
    assistant: AssistantTurn
    results: tuple[ToolExecution, ...]
    messages: list[dict[str, Any]]
    intents: tuple[EffectIntent, ...] = ()
    validation_results: tuple[ToolResultEntry, ...] = ()


__all__ = [
    "AgentTool",
    "CommittedOutput",
    "EvidenceSourceFact",
    "ExecutedTurn",
    "ToolExecute",
    "ToolExecution",
    "ToolModelFunc",
    "ToolObservation",
    "ToolResult",
    "ToolResultCapacityError",
    "ToolRuntime",
    "ToolUpdateSink",
    "ToolEffects",
    "WorkspaceInventoryFacts",
    "WorkspacePathFact",
]
