# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable context projections, typed compaction summaries, and validity checks.

A projection selects one journal suffix and, when needed, summarizes a
contiguous older prefix. Every committed projection is immutable (M3-D24);
the session row points at the active one. Validity checks here are pure: they
classify candidate projections against ``ContextPolicy`` numbers without
calling a provider or opening a store.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Literal

from dlightrag.agent.session.effects import JsonValue, canonical_json
from dlightrag.agent.session.ids import EntryId, ProjectionId
from dlightrag.ai.capacity import ContextPolicy, ModelProfile

PROJECTION_SCHEMA_VERSION = 1

#: Summary fields are continuation memory, never evidence; keep this schema
#: stable because journal payloads and compaction prompts depend on it.
COMPACTION_SUMMARY_FIELDS: tuple[str, ...] = (
    "goal",
    "constraints_preferences",
    "progress",
    "decisions",
    "next_steps",
    "critical_context",
    "paths",
    "durable_handles",
)


@dataclass(frozen=True, slots=True)
class CompactionSummary:
    """Typed semantic continuation fields for one committed compaction.

    Sequence coverage is not model-authored: the framework chooses the
    compacted contiguous prefix and stores ``covered_through_sequence`` /
    ``first_retained_sequence`` on the projection (M3-D17).
    """

    goal: str
    constraints_preferences: str = ""
    progress: str = ""
    decisions: str = ""
    next_steps: str = ""
    critical_context: str = ""
    paths: JsonValue | None = None
    durable_handles: JsonValue | None = None

    def __post_init__(self) -> None:
        if not self.goal.strip():
            raise ValueError("compaction summary goal cannot be empty")

    def canonical_json(self) -> str:
        return canonical_json({name: getattr(self, name) for name in COMPACTION_SUMMARY_FIELDS})

    @classmethod
    def from_canonical_json(cls, summary_json: str) -> CompactionSummary:
        import json

        try:
            payload = json.loads(summary_json)
        except json.JSONDecodeError as exc:
            raise ValueError("compaction summary is not canonical JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("compaction summary must be a JSON object")
        unknown = set(payload) - set(COMPACTION_SUMMARY_FIELDS)
        if unknown:
            raise ValueError(f"compaction summary has unknown fields: {sorted(unknown)}")
        return cls(**payload)


def render_compaction_summary(summary_json: str | None) -> str:
    """Render one compaction summary deterministically for a model prompt.

    A missing summary renders an explicit empty-continuation note so the fold
    never silently drops a compaction position.
    """
    if summary_json is None:
        return "No prior context summary."
    summary = CompactionSummary.from_canonical_json(summary_json)
    sections: list[str] = []
    for name in ("goal", "constraints_preferences", "progress", "decisions"):
        value = getattr(summary, name)
        if value:
            sections.append(f"{name.replace('_', ' ')}: {value}")
    next_steps = summary.next_steps
    if next_steps:
        sections.append(f"next steps: {next_steps}")
    critical = summary.critical_context
    if critical:
        sections.append(f"critical context: {critical}")
    if summary.paths is not None:
        sections.append(f"paths: {canonical_json(summary.paths)}")
    if summary.durable_handles is not None:
        sections.append(f"durable handles: {canonical_json(summary.durable_handles)}")
    return "Prior context summary:\n" + "\n".join(f"- {section}" for section in sections)


@dataclass(frozen=True, slots=True)
class TokenAnchor:
    """One measured model call anchored at a journal sequence."""

    through_sequence: int
    measured_input_tokens: int
    measured_output_tokens: int

    def __post_init__(self) -> None:
        if self.through_sequence < 0:
            raise ValueError("token anchor sequence cannot be negative")
        if self.measured_input_tokens < 0 or self.measured_output_tokens < 0:
            raise ValueError("token anchor measurements cannot be negative")


@dataclass(frozen=True, slots=True)
class ContextProjection:
    """One immutable projection over a journal suffix plus an optional summary."""

    projection_id: ProjectionId
    first_retained_sequence: int
    covered_through_sequence: int
    summary: str | None
    token_anchors: tuple[TokenAnchor, ...] = field(default_factory=tuple)
    covered_through_entry_id: EntryId | None = None
    first_retained_entry_id: EntryId | None = None
    source_digest: str = ""
    schema_version: int = PROJECTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PROJECTION_SCHEMA_VERSION:
            raise ValueError("projection schema version is not current")
        if self.first_retained_sequence < 1:
            raise ValueError("projection retained start must be positive")
        if self.covered_through_sequence < 0:
            raise ValueError("projection covered-through sequence cannot be negative")
        if self.first_retained_sequence <= self.covered_through_sequence:
            raise ValueError("projection retained start must follow the covered prefix")
        if self.covered_through_sequence == 0 and self.summary is not None:
            raise ValueError("initial projection cannot carry a summary")
        if self.covered_through_sequence > 0 and not self.summary:
            raise ValueError("compacted projection requires a non-empty summary")
        if self.covered_through_sequence > 0:
            if self.covered_through_entry_id is None or not self.source_digest:
                raise ValueError(
                    "compacted projection requires branch Entry identity and source digest"
                )
            if len(self.source_digest) != 64:
                raise ValueError("projection source digest must be SHA-256")
        anchors = list(self.token_anchors)
        if anchors != sorted(anchors, key=lambda anchor: anchor.through_sequence):
            raise ValueError("projection token anchors must be ordered by sequence")


class AgentInputOverflowError(ValueError):
    """The smallest valid context cannot fit the pinned profile's hard limit."""

    def __init__(
        self,
        *,
        input_tokens: int,
        input_limit_tokens: int,
        fixed_input_tokens: int,
    ) -> None:
        self.input_tokens = input_tokens
        self.input_limit_tokens = input_limit_tokens
        self.fixed_input_tokens = fixed_input_tokens
        super().__init__(
            "agent_input_overflow: "
            f"{input_tokens} estimated input tokens exceed the hard limit of "
            f"{input_limit_tokens} after a minimum fixed input of {fixed_input_tokens}"
        )


InputOverflowKind = Literal["context_exhausted", "hard_input_limit_exceeded"]


def require_compactable(
    profile: ModelProfile,
    *,
    input_tokens: int,
    fixed_input_tokens: int,
    context_policy: ContextPolicy | None = None,
) -> None:
    """Raise before any provider call when even the minimal context overflows.

    The caller passes the mandatory fixed input (system prompt, question,
    tool schemas, smallest valid summary) plus the measured active input. A
    request above the hard limit fails as ``agent_input_overflow`` instead of
    looping on compaction (M3 compaction rule).
    """
    if input_tokens < 0 or fixed_input_tokens < 0:
        raise ValueError("token estimates cannot be negative")
    policy = context_policy or ContextPolicy()
    hard_limit = policy.hard_input_limit(profile)
    if fixed_input_tokens > hard_limit:
        raise AgentInputOverflowError(
            input_tokens=fixed_input_tokens,
            input_limit_tokens=hard_limit,
            fixed_input_tokens=fixed_input_tokens,
        )
    if input_tokens > hard_limit:
        raise AgentInputOverflowError(
            input_tokens=input_tokens,
            input_limit_tokens=hard_limit,
            fixed_input_tokens=fixed_input_tokens,
        )


def should_compact(
    profile: ModelProfile,
    *,
    input_tokens: int,
    context_policy: ContextPolicy | None = None,
) -> bool:
    """Return whether the active input crossed the proactive compaction trigger."""
    policy = context_policy or ContextPolicy()
    return input_tokens > policy.compaction_trigger(profile)


def projection_source_digest(entry_ids: Sequence[EntryId]) -> str:
    """Hash one exact branch prefix; Session-global sequence is not coverage."""
    return sha256("\n".join(entry_id.value for entry_id in entry_ids).encode()).hexdigest()


def projection_strictly_reduces(
    previous: ContextProjection | None,
    candidate: ContextProjection,
    *,
    accounted_input_before: int,
    accounted_input_after: int,
) -> bool:
    """Return whether a candidate strictly reduces accounted input.

    The initial projection is always a valid first step. Later projections must
    cover strictly more of the prefix and shrink the accounted input.
    """
    if previous is None:
        return True
    if candidate.covered_through_sequence <= previous.covered_through_sequence:
        return False
    return accounted_input_after < accounted_input_before


def validate_projection_commit(
    previous: ContextProjection | None,
    candidate: ContextProjection,
    *,
    accounted_input_before: int,
    accounted_input_after: int,
) -> str | None:
    """Return a violation reason, or None when the projection may commit.

    A committed projection must be a strictly reducing, covering step. Any
    failed, empty, non-covering, or non-reducing candidate leaves the prior
    projection authoritative.
    """
    if not projection_strictly_reduces(
        previous,
        candidate,
        accounted_input_before=accounted_input_before,
        accounted_input_after=accounted_input_after,
    ):
        return "projection does not strictly reduce accounted input"
    return None


_INPUT_USAGE_KEYS = ("prompt_tokens", "input_tokens", "prompt_token_count")
_OUTPUT_USAGE_KEYS = (
    "completion_tokens",
    "output_tokens",
    "candidates_tokens",
    "candidates_token_count",
)
_OUTPUT_USAGE_EXTRAS = ("thoughts_tokens", "thoughts_token_count", "reasoning_tokens")


def _usage_int(usage: Mapping[str, object], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _first_usage_int(usage: Mapping[str, object], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _usage_int(usage, key)
        if value is not None:
            return value
    return None


def _extra_usage_int(usage: Mapping[str, object], keys: tuple[str, ...]) -> int:
    total = 0
    for key in keys:
        value = _usage_int(usage, key)
        if value is not None:
            total += value
    return total


def token_anchor_from_usage(
    through_sequence: int,
    usage: Mapping[str, object] | None,
) -> TokenAnchor | None:
    """Return a measured anchor, or None when the provider omitted input usage."""
    if usage is None:
        return None
    measured_input = _first_usage_int(usage, _INPUT_USAGE_KEYS)
    if measured_input is None:
        return None
    measured_output = _first_usage_int(usage, _OUTPUT_USAGE_KEYS) or 0
    return TokenAnchor(
        through_sequence=through_sequence,
        measured_input_tokens=measured_input,
        measured_output_tokens=measured_output + _extra_usage_int(usage, _OUTPUT_USAGE_EXTRAS),
    )


def projection_with_anchor(
    previous: ContextProjection,
    anchor: TokenAnchor,
) -> ContextProjection:
    """Return a new projection that records one additional measured call.

    Coverage and summary stay the same; the session row will point at the new
    immutable projection id. An anchor at an existing sequence replaces it.
    """
    kept = tuple(
        existing
        for existing in previous.token_anchors
        if existing.through_sequence != anchor.through_sequence
    )
    anchors = tuple(sorted((*kept, anchor), key=lambda item: item.through_sequence))
    return ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=previous.first_retained_sequence,
        covered_through_sequence=previous.covered_through_sequence,
        summary=previous.summary,
        token_anchors=anchors,
        covered_through_entry_id=previous.covered_through_entry_id,
        first_retained_entry_id=previous.first_retained_entry_id,
        source_digest=previous.source_digest,
        schema_version=previous.schema_version,
    )


def live_anchor(
    projection: ContextProjection,
    *,
    last_retained_sequence: int,
) -> TokenAnchor | None:
    """Return the newest measured anchor still inside the live journal suffix.

    The seed ``through_sequence=0`` placeholder is never live.
    """
    start = projection.first_retained_sequence
    found = [
        anchor
        for anchor in projection.token_anchors
        if start <= anchor.through_sequence <= last_retained_sequence
    ]
    return found[-1] if found else None


def accounted_input_tokens(
    *,
    estimated_input_tokens: int,
    measured_anchor: TokenAnchor | None,
    unanchored_tail_tokens: int,
) -> int:
    """Combine one live measured anchor with the not-yet-anchored tail.

    No live anchor means the whole accounted input is the estimate. A live
    anchor supplies the provider reading through that sequence; add only the
    estimated tokens of messages after it.
    """
    if estimated_input_tokens < 0 or unanchored_tail_tokens < 0:
        raise ValueError("token estimates cannot be negative")
    if measured_anchor is None:
        return estimated_input_tokens
    return measured_anchor.measured_input_tokens + unanchored_tail_tokens


__all__ = [
    "COMPACTION_SUMMARY_FIELDS",
    "PROJECTION_SCHEMA_VERSION",
    "AgentInputOverflowError",
    "CompactionSummary",
    "ContextProjection",
    "InputOverflowKind",
    "TokenAnchor",
    "accounted_input_tokens",
    "live_anchor",
    "projection_source_digest",
    "projection_strictly_reduces",
    "projection_with_anchor",
    "render_compaction_summary",
    "require_compactable",
    "should_compact",
    "token_anchor_from_usage",
    "validate_projection_commit",
]
