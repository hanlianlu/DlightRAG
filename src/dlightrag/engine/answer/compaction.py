# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Mode-neutral Answer compaction: summarize and validate one projection.

The coordinator is shared by Fast Host turns and Research Agent operations. It
performs a tools-disabled call on the pinned query model, parses the typed
summary, preserves whole-exchange boundaries, and returns an uncommitted
projection. The caller that owns the Session lane remains responsible for the
atomic ``CompactionEntry`` and projection-register commit.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dlightrag.engine.agent.session.entries import CompactionEntry, SessionEntry
from dlightrag.engine.agent.session.fold import (
    exchange_starts,
    fold_entries,
    select_compaction_boundary,
)
from dlightrag.engine.agent.session.ids import ProjectionId
from dlightrag.engine.agent.session.projection import (
    CompactionSummary,
    ContextProjection,
    projection_source_digest,
    render_compaction_summary,
    validate_projection_commit,
)
from dlightrag.engine.agent.session.repository import AgentSessionSnapshot
from dlightrag.engine.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.engine.ai.reasoning import cheapest_supported_reasoning
from dlightrag.engine.ai.tokens import estimate_messages_tokens, estimate_tokens
from dlightrag.engine.answer.continuation_handles import MAX_NAMED_SESSION_NOTES
from dlightrag.engine.answer.prompts.compaction import (
    COMPACTION_SYSTEM_PROMPT,
    compaction_user_prompt,
)

StreamModel = Callable[..., AsyncIterator[str]]
ExchangeStarts = Callable[[Sequence[SessionEntry]], Sequence[int]]

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


def _durable_handles(handles: Sequence[str]) -> list[str] | None:
    """Bound the re-readable handles one compaction keeps for the model.

    The list arrives composed and ordered by the caller: the run's Evidence
    ledger supplies its citation identities, and its committed spill rows supply
    the outputs whose bytes outlive the covered prefix. Neither source is
    inferred from message text. The former source was the covered entries' Tool
    Arguments, and that inference was withdrawn with the temporary Arguments
    themselves — correctly, because a replayed or deleted argument is not
    evidence. A compacted transcript otherwise loses every passage it had shown,
    and the handles are what let the next turn read a source again by identity.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for handle in handles:
        text = handle.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered[:_MAX_DURABLE_HANDLES] or None


#: One summary's handle list is continuation memory; this bounds its prose.
_MAX_DURABLE_HANDLES = 40


def _run_notes(notes: Sequence[str]) -> list[str] | None:
    """Return the Run Note lines one summary keeps for the model.

    A note is a file the Run owns, so unlike a handle this is a locator and its
    content is whatever the file holds now. Nothing is inferred from the covered
    entries: the caller reads the Workspace Inventory, which is the framework's own
    observation of what the Run wrote. The cap is applied here as well as where the
    lines are composed, so no caller can spend the summary's budget on notes.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    for note in notes:
        text = note.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered[:MAX_NAMED_SESSION_NOTES] or None


class CompactionCoordinator:
    """Prepare one bounded automatic compaction effect for Runtime settlement."""

    def __init__(
        self,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy,
        stream_model: StreamModel,
        exchange_starts_func: ExchangeStarts = exchange_starts,
    ) -> None:
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._stream_model = stream_model
        self._exchange_starts = exchange_starts_func

    async def prepare(
        self,
        snapshot: AgentSessionSnapshot,
        *,
        tail_target_tokens: int,
        accounted_before: int,
        durable_handles: Sequence[str] = (),
        run_notes: Sequence[str] = (),
        trace: dict[str, Any],
    ) -> tuple[ContextProjection, CompactionOutcome]:
        """Prepare one projection effect result; Runtime owns its atomic commit."""
        entries = snapshot.graph.ancestry()
        previous = snapshot.active_projection or ContextProjection(
            projection_id=ProjectionId.new(),
            first_retained_sequence=1,
            covered_through_sequence=0,
            summary=None,
        )
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
            summarizable,
            retained_tail_tokens=tail_target_tokens,
            starts=self._exchange_starts(summarizable),
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
        summary = _with_framework_fields(
            parsed,
            paths=None,
            durable_handles=_durable_handles(durable_handles),
            run_notes=_run_notes(run_notes),
        )
        summary_json = summary.canonical_json()

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
                "durable_handles": len(summary.durable_handles or ()),
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
        # One single-attempt call at the cheapest supported reasoning level,
        # bounded by the same physical input/output policy as every other call.
        input_tokens = estimate_messages_tokens(messages)
        max_tokens = self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=input_tokens,
        )
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        stream = self._stream_model(
            messages=messages,
            model_kwargs=kwargs,
            reasoning=cheapest_supported_reasoning(self._model_profile.reasoning),
            model_profile=self._model_profile,
        )  # type: ignore[call-arg]
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
        starts = tuple(self._exchange_starts(target))
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
    run_notes: list[str] | None = None,
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
        run_notes=run_notes,
    )


__all__ = [
    "CompactionCoordinator",
    "CompactionOutcome",
    "parse_compaction_summary",
]
