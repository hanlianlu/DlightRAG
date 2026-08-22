# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Research compaction collaborator: summarize, validate, commit.

Owns the runtime the grill decided: a tools-disabled summarizer call on the
pinned Research model, a markdown-to-typed-summary parser, framework-extracted
paths/handles, whole-exchange boundaries, hierarchical slices when the
covered prefix does not fit the summarizer window, and a bounded
shrink-and-retry loop with halved retained tails. Pure vocabulary stays in
``dlightrag.agent.session``; prompts live in ``dlightrag.answer.prompts``.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from dlightrag.agent.session.entries import CompactionEntry, EffectIntentEntry, SessionEntry
from dlightrag.agent.session.fold import (
    exchange_starts,
    fold_entries,
    select_compaction_boundary,
)
from dlightrag.agent.session.ids import ProjectionId
from dlightrag.agent.session.projection import (
    CompactionSummary,
    ContextProjection,
    render_compaction_summary,
    should_compact,
    validate_projection_commit,
)
from dlightrag.agent.session.store import AgentSessionSnapshot, SessionCommit
from dlightrag.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.ai.tokens import estimate_messages_tokens, estimate_tokens
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.prompts.compaction import COMPACTION_SYSTEM_PROMPT, compaction_user_prompt

StreamModel = Callable[..., AsyncIterator[str]]

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)
_PATH_LIKE_RE = re.compile(r"^(?:\.{0,2}/|/|[A-Za-z]:[\\/]|~)[^\s]*$")
_URL_RE = re.compile(r"^https?://\S+$")

#: Fixed envelope margin over the rendered prompt texts for one summarizer call.
_SUMMARIZER_ENVELOPE_MARGIN = 16

_KNOWN_HEADINGS = {
    "goal": "goal",
    "constraints & preferences": "constraints_preferences",
    "constraints": "constraints_preferences",
    "constraints_preferences": "constraints_preferences",
    "progress": "progress",
    "key decisions": "decisions",
    "decisions": "decisions",
    "next steps": "next_steps",
    "critical context": "critical_context",
}

_PATH_KEYS = {"path", "paths", "file", "files", "file_path", "workspace", "directory", "dir"}
_HANDLE_KEYS = {
    "url",
    "urls",
    "resource_id",
    "resource_handle",
    "resource_handles",
    "link",
    "links",
}


class CompactionBoundary(Protocol):
    """The durable seams one compaction needs from run boundaries."""

    async def load_snapshot(self) -> AgentSessionSnapshot: ...

    async def commit_compaction(self, *, projection: ContextProjection) -> SessionCommit: ...


@dataclass(frozen=True, slots=True)
class CompactionOutcome:
    """What one committed compaction covered, for the operator trace."""

    covered_through_sequence: int
    first_retained_sequence: int
    accounted_before: int
    accounted_after: int
    summary_chars: int
    hierarchical: bool
    tail_target_tokens: int


class _CompactionAttemptFailed(Exception):
    """One compaction attempt produced no committable projection."""


def parse_compaction_summary(markdown: str) -> CompactionSummary:
    """Parse the summarizer's markdown output into the typed summary.

    ``## `` headings map onto the typed fields; a ``Progress`` section keeps
    its Done/In Progress/Blocked sub-headings verbatim. Unknown headings and
    text before the first heading merge into ``critical_context`` so nothing
    the model wrote is silently dropped. A missing or empty ``Goal`` is a
    failed attempt.
    """
    text = markdown.strip()
    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()
    matches = list(_HEADING_RE.finditer(text))
    sections: list[tuple[str, str]] = []
    if matches:
        if matches[0].start() > 0:
            sections.append(("preamble", text[: matches[0].start()].strip()))
        for position, match in enumerate(matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            sections.append((title, text[body_start:body_end].strip()))
    else:
        sections.append(("preamble", text))
    values: dict[str, str] = {}
    unknown_blocks: list[str] = []
    for title, body in sections:
        field = _KNOWN_HEADINGS.get(title.lower())
        if field is None:
            if title != "preamble":
                unknown_blocks.append(f"## {title}\n{body}".rstrip())
            elif body:
                unknown_blocks.append(body)
            continue
        if body:
            values[field] = (f"{values[field]}\n{body}" if values.get(field) else body).strip()
    if unknown_blocks:
        existing = values.get("critical_context", "")
        values["critical_context"] = "\n\n".join(
            part for part in (*([existing] if existing else []), *unknown_blocks) if part
        )
    goal = values.get("goal", "").strip()
    if not goal:
        raise ValueError("compaction summary has no goal")
    return CompactionSummary(
        goal=goal,
        constraints_preferences=values.get("constraints_preferences", ""),
        progress=values.get("progress", ""),
        decisions=values.get("decisions", ""),
        next_steps=values.get("next_steps", ""),
        critical_context=values.get("critical_context", ""),
    )


def _transcript(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render folded model messages as one role-tagged transcript text."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if isinstance(content, list):
            text = "\n".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("text")
            )
        elif isinstance(content, str):
            text = content
        else:
            text = ""
        call_lines: list[str] = []
        for call in message.get("tool_calls") or ():
            name = call.get("function", {}).get("name") if isinstance(call, Mapping) else ""
            arguments = (
                call.get("function", {}).get("arguments") if isinstance(call, Mapping) else ""
            )
            call_lines.append(f"call {name}({arguments})".rstrip())
        entry = text
        if call_lines:
            entry = f"{entry}\n" + "\n".join(call_lines) if entry else "\n".join(call_lines)
        lines.append(f"[{role}]\n{entry}".rstrip())
    return "\n\n".join(lines)


def _flatten_strings(value: Any) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(item, str):
                pairs.append((str(key), item))
            elif isinstance(item, (list, Mapping)):
                pairs.extend(_flatten_strings(item))
    elif isinstance(value, list):
        for item in value:
            pairs.extend(_flatten_strings(item))
    return pairs


def _extract_paths_and_handles(
    entries: Sequence[SessionEntry],
) -> tuple[list[str] | None, list[str] | None]:
    """Recover paths and durable handles from covered intents, never from the model."""
    paths: set[str] = set()
    handles: set[str] = set()
    for entry in entries:
        if not isinstance(entry, EffectIntentEntry):
            continue
        try:
            payload = json.loads(entry.intent.canonical_input)
        except json.JSONDecodeError, TypeError:
            continue
        for key, value in _flatten_strings(payload):
            lowered = key.lower()
            if lowered in _PATH_KEYS and _PATH_LIKE_RE.match(value):
                paths.add(value)
            elif lowered in _HANDLE_KEYS and (
                _URL_RE.match(value) or value.startswith("resource_")
            ):
                handles.add(value)
    return (sorted(paths) or None, sorted(handles) or None)


class CompactionCoordinator:
    """Run the bounded shrink-and-retry compaction loop for one session."""

    def __init__(
        self,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy,
        stream_model: StreamModel,
        max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._stream_model = stream_model
        self._max_attempts = max_attempts

    async def ensure_fits(
        self,
        *,
        boundaries: CompactionBoundary,
        remeasure: Callable[[], Awaitable[int]],
        trace: dict[str, Any],
        force: bool = False,
    ) -> CompactionOutcome | None:
        """Compact until the proactive trigger clears, at most ``max_attempts``.

        ``force`` compacts at least once even when the gate reads under the
        trigger (the reactive overflow path). Failing attempts shrink the
        retained tail by half each time; exhaustion raises the loud overflow
        error. Summarizer provider errors propagate — compaction never hides
        a dead model.
        """
        outcome: CompactionOutcome | None = None
        for attempt in range(self._max_attempts):
            accounted = await remeasure()
            over = should_compact(
                self._model_profile,
                input_tokens=accounted,
                context_policy=self._context_policy,
            )
            if not over and not (force and outcome is None):
                return outcome
            tail_target = self._context_policy.retained_tail_target(self._model_profile) // (
                2**attempt
            )
            try:
                outcome = await self._compact_once(
                    boundaries,
                    tail_target_tokens=tail_target,
                    accounted_before=accounted,
                    trace=trace,
                )
            except _CompactionAttemptFailed:
                outcome = None
        # The final attempt may have just brought the reading under the trigger.
        accounted = await remeasure()
        if not should_compact(
            self._model_profile,
            input_tokens=accounted,
            context_policy=self._context_policy,
        ):
            return outcome
        trigger = self._context_policy.compaction_trigger(self._model_profile)
        raise AnswerInputOverflowError(
            "Research input still exceeds the proactive compaction threshold after "
            f"{self._max_attempts} compaction attempts: {accounted} > {trigger} "
            "accounted input tokens. Use a larger-context model or shorten the request."
        )

    async def _compact_once(
        self,
        boundaries: CompactionBoundary,
        *,
        tail_target_tokens: int,
        accounted_before: int,
        trace: dict[str, Any],
    ) -> CompactionOutcome:
        snapshot = await boundaries.load_snapshot()
        entries = snapshot.entries
        previous = snapshot.active_projection
        if previous is None:
            raise _CompactionAttemptFailed("no active projection to compact from")
        summarizable = [
            entry
            for entry in entries
            if entry.sequence >= previous.first_retained_sequence
            and not isinstance(entry, CompactionEntry)
        ]
        if not summarizable:
            raise _CompactionAttemptFailed("nothing left to compact")
        tail_index = select_compaction_boundary(
            summarizable, retained_tail_tokens=tail_target_tokens
        )
        target = summarizable[:tail_index]
        if not target:
            raise _CompactionAttemptFailed("no complete exchanges to summarize")
        covered = self._fit_summary_slice(target, previous_summary=previous.summary)
        if covered is None:
            raise _CompactionAttemptFailed(
                "the covered prefix does not fit the summarizer model window"
            )
        hierarchical = len(covered) < len(target)
        if hierarchical:
            first_retained = target[len(covered)].sequence
        elif tail_index < len(summarizable):
            first_retained = summarizable[tail_index].sequence
        else:
            first_retained = covered[-1].sequence + 1
        covered_through = covered[-1].sequence

        summary_text = await self._summarize(covered, previous_summary=previous.summary)
        try:
            parsed = parse_compaction_summary(summary_text)
        except ValueError as exc:
            raise _CompactionAttemptFailed(str(exc)) from exc
        paths, handles = _extract_paths_and_handles(covered)
        summary = _with_framework_fields(parsed, paths=paths, durable_handles=handles)
        summary_json = summary.canonical_json()

        anchors = tuple(
            anchor for anchor in previous.token_anchors if anchor.through_sequence >= first_retained
        )
        accounted_after = self._estimate_retained(entries, first_retained, summary_json)
        # The previous CompactionEntry (sequence == previous.first_retained) is
        # still in the retained set and its fold already renders the old summary.
        estimated_before = self._estimate_retained(entries, previous.first_retained_sequence, None)
        candidate = ContextProjection(
            projection_id=ProjectionId.new(),
            first_retained_sequence=first_retained,
            covered_through_sequence=covered_through,
            summary=summary_json,
            token_anchors=anchors,
        )
        violation = validate_projection_commit(
            previous,
            candidate,
            accounted_input_before=estimated_before,
            accounted_input_after=accounted_after,
        )
        if violation is not None:
            raise _CompactionAttemptFailed(violation)

        await boundaries.commit_compaction(projection=candidate)
        outcome = CompactionOutcome(
            covered_through_sequence=covered_through,
            first_retained_sequence=first_retained,
            accounted_before=accounted_before,
            accounted_after=accounted_after,
            summary_chars=len(summary_json),
            hierarchical=hierarchical,
            tail_target_tokens=tail_target_tokens,
        )
        trace.setdefault("compactions", []).append(
            {
                "covered_through_sequence": covered_through,
                "first_retained_sequence": first_retained,
                "accounted_before": accounted_before,
                "accounted_after": accounted_after,
                "summary_chars": len(summary_json),
                "hierarchical": hierarchical,
                "tail_target_tokens": tail_target_tokens,
            }
        )
        return outcome

    async def _summarize(
        self,
        covered: Sequence[SessionEntry],
        *,
        previous_summary: str | None,
    ) -> str:
        rendered_previous = (
            render_compaction_summary(previous_summary) if previous_summary is not None else None
        )
        messages = [
            {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": compaction_user_prompt(
                    previous_summary=rendered_previous,
                    transcript=_transcript(fold_entries(covered)),
                ),
            },
        ]
        # One single-attempt, thinking-off call capped at the profile output.
        kwargs: dict[str, Any] = {}
        if self._model_profile.max_output_tokens is not None:
            kwargs["max_tokens"] = self._model_profile.max_output_tokens
        stream = self._stream_model(messages=messages, model_kwargs=kwargs, thinking="off")  # type: ignore[call-arg]
        chunks: list[str] = []
        async for chunk in stream:
            chunks.append(chunk)
        return "".join(chunks)

    def _fit_summary_slice(
        self,
        target: Sequence[SessionEntry],
        *,
        previous_summary: str | None,
    ) -> tuple[SessionEntry, ...] | None:
        """Return the oldest whole-exchange prefix that fits the summarizer window.

        Entries before the first exchange (the question and pinned history) are
        one leading slice unit of their own; they are compactable like any
        other old exchange. None when even the first unit does not fit: the
        caller counts a failed attempt instead of silently truncating.
        """
        fixed = (
            estimate_tokens(COMPACTION_SYSTEM_PROMPT)
            + estimate_tokens(
                compaction_user_prompt(
                    previous_summary=(
                        render_compaction_summary(previous_summary)
                        if previous_summary is not None
                        else None
                    ),
                    transcript="",
                )
            )
            + _SUMMARIZER_ENVELOPE_MARGIN
        )
        # The summarizer input fits below the proactive trigger for the pinned
        # role, never just below the hard limit (living spec Proactive
        # Compaction).
        budget = self._context_policy.compaction_trigger(self._model_profile)
        starts = exchange_starts(target)
        if not starts:
            return tuple(target)
        end = 0
        preamble = target[: starts[0]]
        if preamble:
            if estimate_tokens(_transcript(fold_entries(preamble))) + fixed > budget:
                return None
            end = starts[0]
        for position in range(len(starts)):
            boundary = starts[position + 1] if position + 1 < len(starts) else len(target)
            candidate = target[:boundary]
            rendered = _transcript(fold_entries(candidate))
            if estimate_tokens(rendered) + fixed > budget:
                break
            end = boundary
        return tuple(target[:end]) if end > 0 else None

    def _estimate_retained(
        self,
        entries: Sequence[SessionEntry],
        first_retained: int,
        summary_json: str | None,
    ) -> int:
        retained = [entry for entry in entries if entry.sequence >= first_retained]
        total = estimate_messages_tokens(fold_entries(retained))
        if summary_json is not None:
            total += estimate_messages_tokens(
                [
                    {
                        "role": "user",
                        "content": render_compaction_summary(summary_json),
                    }
                ]
            )
        return total


def _with_framework_fields(
    summary: CompactionSummary,
    *,
    paths: list[str] | None,
    durable_handles: list[str] | None,
) -> CompactionSummary:
    return CompactionSummary(
        goal=summary.goal,
        constraints_preferences=summary.constraints_preferences,
        progress=summary.progress,
        decisions=summary.decisions,
        next_steps=summary.next_steps,
        critical_context=summary.critical_context,
        paths=paths,
        durable_handles=durable_handles,
    )


__all__ = [
    "CompactionBoundary",
    "CompactionCoordinator",
    "CompactionOutcome",
    "parse_compaction_summary",
]
