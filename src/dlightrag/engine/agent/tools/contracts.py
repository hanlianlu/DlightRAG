# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The contracts one model-visible tool call is made of."""

import json
import math
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaMode

from dlightrag.engine.agent.session.effects import (
    ReplayPolicy,
    canonical_json,
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
from dlightrag.engine.ai.tokens import estimate_tokens


def declaration_tokens(
    definitions: Iterable[ToolDefinition | Mapping[str, Any]],
) -> int:
    """The token estimate of one request's whole tool array.

    It is the exact serialization every accounting site measures: acceptance, Run-plan
    accounting, compaction, and the projections that state what a tool surface costs. A
    pinned Run plan holds its definitions as JSON objects while a live Tool holds the one
    definition it generated, so both shapes measure the same.
    """
    payload: list[dict[str, Any]] = []
    for definition in definitions:
        if isinstance(definition, Mapping):
            payload.append(dict(definition))
        else:
            payload.append(asdict(definition))
    return estimate_tokens(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


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


#: The ``ToolResult.details`` key that records the Runtime's own cut, set only where the
#: residual fit actually shortened a result. It is the Runtime's fact rather than a tool's,
#: so it is never inferred from the text: a tool that pages its output leaves it unset.
OBSERVED_TRUNCATED_DETAIL = "observed_truncated"


@dataclass(frozen=True, slots=True)
class CommittedOutput:
    """One full tool output promoted from staging to durable storage."""

    resource_id: str
    content_digest: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class AgentspacePathFact:
    """One regular workspace path observed after a tool mutation."""

    relative_path: str
    entry_type: str
    size_bytes: int
    mode: int | None = None
    content_digest: str | None = None


@dataclass(frozen=True, slots=True)
class AgentspaceInventoryFacts:
    """Typed workspace upserts/deletes, optionally replacing the full inventory."""

    upserts: tuple[AgentspacePathFact, ...] = ()
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
class DeclaredSource:
    """One file a published product was built from, and the handle it was published under.

    A source is a dependency of the product it belongs to: carried with it, never delivered as a
    product of its own, and named again when a later Run reopens it, so the program that produced
    the deliverable can be read back and run again (ADR 0052).
    """

    path: str
    """Where the file sits, relative to the Artifact root it was published under."""
    resource_id: str
    """The handle its bytes were published under, which is how a later Run reads them back."""

    def __post_init__(self) -> None:
        if not self.path or not self.resource_id:
            raise ValueError("a declared source needs a path and a resource id")
        if self.path.startswith("/") or "\\" in self.path or "\x00" in self.path:
            raise ValueError("a declared source path must be relative to the Artifact root")
        if any(part in {"", ".", ".."} for part in self.path.split("/")):
            raise ValueError("a declared source path must stay beneath the Artifact root")


@dataclass(frozen=True, slots=True)
class AdmittedBytes:
    """The bytes a Host's Resource admitted, as materialize copies them."""

    resource_id: str
    """The canonical handle: after an adoption, this Run's own."""
    media_type: str
    content: bytes
    filename: str | None = None
    """The name the Resource was registered under; a published Artifact's own filename."""
    sources: tuple[DeclaredSource, ...] = ()
    """What a published product was built from. Only a product attached with declared sources
    carries any, and only reopening it reads them (ADR 0052)."""


@dataclass(frozen=True, slots=True)
class ToolEffects:
    """Typed host facts emitted by a tool and consumed only at settlement."""

    committed_outputs: tuple[CommittedOutput, ...] = ()
    workspace_inventory: AgentspaceInventoryFacts | None = None
    evidence_sources: tuple[EvidenceSourceFact, ...] = ()
    attached_resources: tuple[ResourceAttachmentBytes, ...] = ()


TOOL_SUBJECT_MAX_CHARS = 64
"""Longest Tool Subject one producer may report; the browser edge keeps its own cap."""

TOOL_PROGRESS_FACT_KEYS = frozenset(
    {"skill_path", "skill_window_start", "skill_window_end", "skill_document_chars", "sealed_skill"}
)
"""The scalar facts a Tool may report on an update through ``ToolResult.details``.

The Runtime forwards exactly these onto the durable ``tool_progress`` event, beside the Tool's
subject; anything else in ``details`` is never forwarded, so a Tool cannot widen what the event
log holds by putting a key there. The keys are today ``load_skill``'s read fact — the document
a call answered and the window of it (``engine/agent/skill_reads.py`` names and reads them) —
and ``sealed_skill``, which ``attach_artifact`` records when a Skill's gate attests a sealed
artifact: no other event in the log says the gate ran, so a resumed attempt can recover that
conclusion from nowhere else. A key is added here only for a fact a reader of the record needs and
cannot recover from anywhere else. A value must be a string, an integer or a boolean, and a string
is bounded by ``TOOL_PROGRESS_FACT_MAX_CHARS``.
"""

TOOL_PROGRESS_FACT_MAX_CHARS = 512
"""Longest string one progress fact may carry; a longer one is dropped, not cut.

Cut would be a different fact: a document path shortened to its first 512 characters names a
file the Skill does not carry. The bound is the field's, so no producer's discipline is what
keeps the durable event small.
"""

TOOL_FACTS_MAX_KEYS = 16
"""Most facts one Tool result may report, so a step card can state a call in bounded rows."""
TOOL_FACTS_MAX_BYTES = 2048
"""Largest canonical JSON a result's facts may occupy; what does not fit is dropped, never cut."""
TOOL_FACTS_MAX_CHARS = 256
"""Longest string one fact leaf may carry; a longer leaf drops its key rather than shorten it."""
TOOL_FACTS_MAX_ITEMS = 16
"""Longest list one fact may carry."""
TOOL_FACTS_MAX_DEPTH = 2
"""Most nested containers below the fact map: a list of records and a record of lists, no more."""

_DROPPED_FACT = object()


_FACT_MAX_INTEGER_BITS = 64
"""Widest integer a fact may carry.

An integer past this is not a count a step card states; it is a value a producer built, and
Python's own JSON form refuses to render one of thousands of digits — which would turn a
malformed fact into a failed tool call.
"""


def _bounded_fact_scalar(value: Any) -> Any:
    """Return one fact leaf, or the sentinel when it is not a JSON scalar a card can show."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if value.bit_length() <= _FACT_MAX_INTEGER_BITS else _DROPPED_FACT
    if isinstance(value, str):
        if len(value) > TOOL_FACTS_MAX_CHARS:
            return _DROPPED_FACT
        # A lone surrogate cannot be encoded as UTF-8, so it is not a value this entry could
        # hold; a JSON provider can hand one in, and it must cost the fact not the settlement.
        return (
            _DROPPED_FACT
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value)
            else value
        )
    if isinstance(value, float):
        # A non-finite number has no canonical JSON form, so it is not a fact a stored entry
        # could hold; dropping it here is what keeps a writer's bad float from failing commit.
        return _DROPPED_FACT if math.isnan(value) or math.isinf(value) else value
    return _DROPPED_FACT


def _bounded_fact_value(value: Any, *, depth: int) -> Any:
    """Normalize one nested fact value, dropping what a card cannot show.

    ``depth`` counts containers below the fact map already open, so a value at depth 0 may be
    a scalar, a list, or a record; a container at the depth bound is refused rather than
    flattened, because a deeper structure is a producer bug and not something a reader needs.
    A record is kept whole or not at all -- half a source record would state a fact it does not
    hold -- while a list drops the items it cannot keep and stays a list.
    """
    scalar = _bounded_fact_scalar(value)
    if scalar is not _DROPPED_FACT:
        return scalar
    if depth >= TOOL_FACTS_MAX_DEPTH:
        return _DROPPED_FACT
    if isinstance(value, Mapping):
        nested: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                return _DROPPED_FACT
            bounded = _bounded_fact_value(item, depth=depth + 1)
            if bounded is _DROPPED_FACT:
                return _DROPPED_FACT
            nested[key] = bounded
        return nested
    if isinstance(value, (list, tuple)):
        if len(value) > TOOL_FACTS_MAX_ITEMS:
            return _DROPPED_FACT
        items: list[Any] = []
        for item in value:
            bounded = _bounded_fact_value(item, depth=depth + 1)
            if bounded is not _DROPPED_FACT:
                items.append(bounded)
        return items
    return _DROPPED_FACT


def _facts_size(facts: Mapping[str, Any]) -> int:
    """Return the UTF-8 size of one facts map's canonical JSON, or its bound when unrenderable."""
    try:
        return len(canonical_json(dict(facts)).encode("utf-8"))
    except TypeError, ValueError, UnicodeEncodeError:
        # A value canonical JSON or UTF-8 refuses is not a fact; report it as past the bound
        # so the fitting loop drops it instead of raising out of a settlement.
        return TOOL_FACTS_MAX_BYTES + 1


def _fit_tool_facts(facts: dict[str, Any]) -> dict[str, Any] | None:
    """Keep a normalized facts map inside its byte bound by dropping what a card can spare.

    A list is trimmed from its tail before any key is dropped whole, so a Web search keeps its
    provider and hit count beside the sources that still fit; with no list left to trim, the
    largest key goes. Every step is lossy but never fatal: the trace keeps its structure.
    """
    while facts and _facts_size(facts) > TOOL_FACTS_MAX_BYTES:
        lists = [(key, item) for key, item in facts.items() if isinstance(item, list) and item]
        if lists:
            key, item = max(lists, key=lambda pair: len(canonical_json(pair[1])))
            item.pop()
            if not item:
                del facts[key]
            continue
        del facts[max(facts, key=lambda name: len(canonical_json(facts[name])))]
    return facts or None


def bounded_tool_facts(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return one Tool result's structured facts, normalized to the bounds a reader may rely on.

    A fact is what a step card needs to state a call without reading its result: the query it
    searched for, the provider that answered, the file it read and how much of it. It is a
    structure rather than text, so the bounds belong to the field: no producer's discipline is
    what keeps an entry row small. Anything invalid is dropped -- fail-safe, never raised --
    because a malformed fact must cost the card one row, not the Run its settlement.
    """
    if not isinstance(value, Mapping) or not value:
        return None
    facts: dict[str, Any] = {}
    for key, item in value.items():
        if len(facts) >= TOOL_FACTS_MAX_KEYS:
            break
        if not isinstance(key, str):
            continue
        bounded = _bounded_fact_value(item, depth=0)
        if bounded is not _DROPPED_FACT:
            facts[key] = bounded
    if not facts:
        return None
    return _fit_tool_facts(facts)


_SUBJECT_DROPPED_CHARS = {code: None for code in (*range(0x20), 0x7F, *range(0x80, 0xA0))}


def _bounded_tool_subject(value: str) -> str:
    """Normalize one reported subject into the single bounded line the UI update and trace row show.

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

    It has two consumers: the live activity row (the update's own label) and the settled
    ``trace.tool_observations.subject`` row. It is transport-private metadata, never model-visible
    result content. Because it is the one producer-reported fact that crosses to the browser, the
    bound belongs to the field rather than to each producer's discipline.
    """
    cached: bool = False
    protected_text: str = ""
    is_error: bool = False
    effects: ToolEffects = ToolEffects()
    telemetry: Mapping[str, Any] | None = None
    """Facts this call reports about itself, for the span that observed it.

    Bounded and never shown to the model: the Runtime merges them into the Tool span's
    ``output`` beside ``is_error`` and ``cached``, so a dashboard groups a tool's calls by what
    they say about themselves — which Web search provider answered, and what the search cost. A
    key the Runtime already owns is the Runtime's.
    """
    facts: Mapping[str, Any] | None = None
    """Bounded structured facts this call states, for the reader's step card.

    Where ``telemetry`` is for the operator's span and ``details`` for the Runtime's recovery,
    facts are the product's: the settled ToolResult Entry keeps them, and every reader
    projection that shows a tool step shows them. They are structure, not text — the same bounds
    on every producer (``bounded_tool_facts``) — so a card states a search's provider and sources
    or a read's line count without parsing the model-visible result.
    """

    def __post_init__(self) -> None:
        if self.subject is not None:
            object.__setattr__(self, "subject", _bounded_tool_subject(self.subject))
        object.__setattr__(self, "facts", bounded_tool_facts(self.facts))

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
        telemetry: Mapping[str, Any] | None = None,
        facts: Mapping[str, Any] | None = None,
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
            telemetry=telemetry,
            facts=facts,
        )

    @property
    def text_content(self) -> str:
        """Return only model-visible text, excluding attachment metadata."""
        return tool_content_text(self.parts)

    @property
    def progress_facts(self) -> dict[str, str | int | bool]:
        """The part of ``details`` the Runtime forwards onto the durable progress event.

        Only the keys in ``TOOL_PROGRESS_FACT_KEYS``, only scalar values, and no string longer
        than ``TOOL_PROGRESS_FACT_MAX_CHARS``: a Tool reports a fact through ``details`` and the
        event log keeps the facts the vocabulary names, never whatever else a result carries.
        """
        return {
            key: value
            for key, value in (self.details or {}).items()
            if key in TOOL_PROGRESS_FACT_KEYS
            and isinstance(value, (str, int, bool))
            and not (isinstance(value, str) and len(value) > TOOL_PROGRESS_FACT_MAX_CHARS)
        }

    @property
    def observed_truncated(self) -> bool:
        """Whether the Runtime cut this result to fit the model's residual capacity.

        A tool that pages its own output leaves this false; a tool that answered with more
        than the residual holds true, and the model saw a prefix rather than the whole
        answer. The fact lives in ``details`` so a viewer, an audit or a later reader can
        tell a complete result from a cut one without parsing the text.
        """
        return bool(self.details and self.details.get(OBSERVED_TRUNCATED_DETAIL))


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

type AdmittedBytesReader = Callable[[str, ToolRuntime], Awaitable[AdmittedBytes | ToolResult]]
"""A Host's way to the bytes of one resource_id; a ToolResult it returns is its refusal."""


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
    "AdmittedBytes",
    "AdmittedBytesReader",
    "AgentTool",
    "CommittedOutput",
    "EvidenceSourceFact",
    "ExecutedTurn",
    "OBSERVED_TRUNCATED_DETAIL",
    "ResourceAttachmentBytes",
    "SourceOrder",
    "TOOL_FACTS_MAX_BYTES",
    "TOOL_FACTS_MAX_CHARS",
    "TOOL_FACTS_MAX_DEPTH",
    "TOOL_FACTS_MAX_ITEMS",
    "TOOL_FACTS_MAX_KEYS",
    "TOOL_PROGRESS_FACT_KEYS",
    "TOOL_PROGRESS_FACT_MAX_CHARS",
    "ToolDeclaration",
    "ToolExecute",
    "ToolModelFunc",
    "ToolResult",
    "ToolResultCapacityError",
    "ToolRuntime",
    "ToolUpdateSink",
    "ToolEffects",
    "AgentspaceInventoryFacts",
    "AgentspacePathFact",
    "already_in_source_order",
    "bounded_tool_facts",
    "declaration_tokens",
]
