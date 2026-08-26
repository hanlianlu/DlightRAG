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

import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag.agent.session.entries import CompactionEntry, SessionEntry
from dlightrag.agent.session.fold import (
    exchange_starts,
    fold_entries,
    select_compaction_boundary,
)
from dlightrag.agent.session.ids import ProjectionId
from dlightrag.agent.session.projection import (
    CompactionSummary,
    ContextProjection,
    projection_source_digest,
    render_compaction_summary,
    validate_projection_commit,
)
from dlightrag.agent.session.store import AgentSessionSnapshot
from dlightrag.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.ai.tokens import estimate_messages_tokens, estimate_tokens
from dlightrag.answer.prompts.compaction import COMPACTION_SYSTEM_PROMPT, compaction_user_prompt

StreamModel = Callable[..., AsyncIterator[str]]

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.DOTALL)
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


def _extract_paths_and_handles(
    entries: Sequence[SessionEntry],
) -> tuple[list[str] | None, list[str] | None]:
    """Return no inferred authority after transient Tool Arguments are deleted."""
    del entries
    return None, None


class CompactionCoordinator:
    """Prepare one bounded automatic compaction effect for Runtime settlement."""

    def __init__(
        self,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy,
        stream_model: StreamModel,
    ) -> None:
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._stream_model = stream_model

    async def prepare(
        self,
        snapshot: AgentSessionSnapshot,
        *,
        tail_target_tokens: int,
        accounted_before: int,
        trace: dict[str, Any],
    ) -> tuple[ContextProjection, CompactionOutcome]:
        """Prepare one projection effect result; Runtime owns its atomic commit."""
        entries = snapshot.graph.ancestry()
        previous = snapshot.active_projection
        if previous is None:
            raise _CompactionAttemptFailed("no active projection to compact from")
        branch_entries = [entry for entry in entries if not isinstance(entry, CompactionEntry)]
        if previous.first_retained_entry_id is None:
            summarizable = [
                entry
                for entry in branch_entries
                if entry.sequence >= previous.first_retained_sequence
            ]
        else:
            retained_index = next(
                (
                    index
                    for index, entry in enumerate(branch_entries)
                    if entry.entry_id == previous.first_retained_entry_id
                ),
                None,
            )
            if retained_index is None:
                raise _CompactionAttemptFailed(
                    "active projection retained Head is not on this branch"
                )
            summarizable = branch_entries[retained_index:]
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
        covered_entry_id = covered[-1].entry_id
        first_retained_entry_id = next(
            (entry.entry_id for entry in branch_entries if entry.sequence == first_retained),
            None,
        )
        covered_index = next(
            index
            for index, entry in enumerate(branch_entries)
            if entry.entry_id == covered_entry_id
        )
        source_digest = projection_source_digest(
            [entry.entry_id for entry in branch_entries[: covered_index + 1]]
        )

        summary_text = await self._summarize(covered, previous_summary=previous.summary)
        try:
            parsed = parse_compaction_summary(summary_text)
        except ValueError as exc:
            raise _CompactionAttemptFailed(str(exc)) from exc
        paths, handles = _extract_paths_and_handles(covered)
        summary = _with_framework_fields(parsed, paths=paths, durable_handles=handles)
        summary_json = summary.canonical_json()

        # Replacing the active summary changes every provider input represented
        # by the prior measured anchors. They cannot remain live after this
        # projection even when their sequence lies in the retained suffix.
        anchors: tuple[Any, ...] = ()
        accounted_after = self._estimate_retained(entries, first_retained, summary_json)
        estimated_before = self._estimate_retained(
            entries,
            previous.first_retained_sequence,
            previous.summary,
        )
        candidate = ContextProjection(
            projection_id=ProjectionId.new(),
            first_retained_sequence=first_retained,
            covered_through_sequence=covered_through,
            summary=summary_json,
            token_anchors=anchors,
            covered_through_entry_id=covered_entry_id,
            first_retained_entry_id=first_retained_entry_id,
            source_digest=source_digest,
        )
        violation = validate_projection_commit(
            previous,
            candidate,
            accounted_input_before=estimated_before,
            accounted_input_after=accounted_after,
        )
        if violation is not None:
            raise _CompactionAttemptFailed(violation)

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
        return candidate, outcome

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
        # One single-attempt, thinking-off call bounded by the same physical
        # input/output policy as every other provider call.
        input_tokens = estimate_messages_tokens(messages)
        max_tokens = self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=input_tokens,
        )
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
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
    "CompactionCoordinator",
    "CompactionOutcome",
    "parse_compaction_summary",
]
