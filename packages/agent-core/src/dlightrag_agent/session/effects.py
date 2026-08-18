# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Effect intents, settlements, replay policy, and tool-contract digests.

An effect is one validated model-visible tool call. Intents persist before
execution in assistant source order; settlements commit results one at a time
in the same source order (M3-D12). Safe replay requires the tool name, replay
policy, contract version, and canonical input-schema digest to match exactly
(M3-D13, M3-D18).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, TypeVar

from dlightrag_agent.session.ids import IntentId

type ReplayPolicy = Literal["safe", "never"]
type EffectOutcome = Literal[
    "succeeded",
    "interrupted",
    "tool_contract_changed",
]
type ToolResultOutcome = Literal[
    "succeeded",
    "interrupted",
    "tool_contract_changed",
    "failed",
    "validation_failed",
    "unknown_tool",
    "invalid_arguments",
]
type JsonValue = Any


def canonical_json(value: JsonValue) -> str:
    """Return canonical UTF-8 JSON with sorted object keys and no NaN."""
    import json

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


# JSON Schema fields that describe presentation, not the accepted input shape.
# Canonicalization removes them so a contract digest never changes when only a
# description, example, or declaration order moves (M3-D18).
_PRESENTATION_SCHEMA_FIELDS = frozenset(
    {
        "$comment",
        "description",
        "examples",
        "markdownDescription",
        "title",
    }
)


def canonical_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return one canonical JSON Schema with keys sorted and presentation removed."""
    cleaned: dict[str, Any] = {}
    for key in sorted(schema):
        if key in _PRESENTATION_SCHEMA_FIELDS:
            continue
        value = schema[key]
        if isinstance(value, Mapping):
            cleaned[key] = canonical_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                canonical_schema(item) if isinstance(item, Mapping) else item for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def schema_digest(schema: Mapping[str, Any]) -> str:
    """Return the SHA-256 of one canonicalized input schema."""
    return sha256(canonical_json(canonical_schema(schema)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EffectIntent:
    """One validated tool call, ordered before execution and settled after it."""

    intent_id: IntentId
    tool_name: str
    replay_policy: ReplayPolicy
    contract_version: int
    input_schema_digest: str
    canonical_input: str
    source_call_id: str | None = None

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("effect intent tool name cannot be empty")
        if self.contract_version < 1:
            raise ValueError("effect intent contract version must be positive")
        if len(self.input_schema_digest) != 64:
            raise ValueError("effect intent schema digest must be a SHA-256 hex digest")
        if self.source_call_id is not None and not self.source_call_id.strip():
            raise ValueError("effect intent source call id cannot be empty")


@dataclass(frozen=True, slots=True)
class ToolResultEntry:
    """One model-visible tool result, kept transport-private beyond its content."""

    tool_name: str
    call_id: str
    outcome: ToolResultOutcome
    content: str
    details: JsonValue | None = None
    cached: bool = False

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ValueError("tool result tool name cannot be empty")
        if not self.call_id.strip():
            raise ValueError("tool result call id cannot be empty")


HostUpdateT = TypeVar("HostUpdateT")


@dataclass(frozen=True, slots=True)
class EffectSettlement[HostUpdateT]:
    """One atomic settlement: outcome, ordered result, and host update."""

    outcome: EffectOutcome
    result: ToolResultEntry
    host_update: HostUpdateT


__all__ = [
    "EffectIntent",
    "EffectOutcome",
    "EffectSettlement",
    "HostUpdateT",
    "JsonValue",
    "ReplayPolicy",
    "ToolResultEntry",
    "ToolResultOutcome",
    "canonical_json",
    "canonical_schema",
    "schema_digest",
]
