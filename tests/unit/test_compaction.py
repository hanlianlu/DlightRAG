# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Compaction summarizer parsing and the shrink-and-retry loop."""

from datetime import UTC, datetime
from typing import Any

import pytest
from dlightrag_agent.session.effects import EffectIntent, ToolResultEntry
from dlightrag_agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    EffectIntentEntry,
    EffectResultEntry,
    UserMessageEntry,
)
from dlightrag_agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag_agent.session.memory import InMemoryAgentSessionStore
from dlightrag_agent.session.projection import CompactionSummary, ContextProjection, TokenAnchor
from dlightrag_agent.session.store import SessionCommit
from dlightrag_ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag_ai.messages import ToolCall
from dlightrag_ai.providers.base import is_provider_context_overflow

from dlightrag.answer.agent.compaction import CompactionCoordinator, parse_compaction_summary
from dlightrag.answer.errors import AnswerInputOverflowError

_MARKDOWN = """\
## Goal
Answer the question.

## Constraints & Preferences
- Cite every source.
- Keep it short.

## Progress
### Done
- [x] Read three sources.
### In Progress
- [ ] Cross-check citation 2.
### Blocked
- Missing page 4.

## Key Decisions
- **Use schema A**: it fits the data.

## Next Steps
1. Verify citation 2.

## Critical Context
- source-uuid is the main file.

## Leftovers
Model chatter that must not be dropped.
"""


def _profile() -> ModelProfile:
    return ModelProfile(context_window_tokens=100_000, max_input_tokens=85_000)


def _user(session_id: SessionId, content: str) -> UserMessageEntry:
    return UserMessageEntry(
        entry_id=EntryId.new(), session_id=session_id, timestamp=datetime.now(UTC), content=content
    )


def _assistant(session_id: SessionId, content: str) -> AssistantMessageEntry:
    return AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content=content,
        stop_reason="tool_use",
        tool_calls=(ToolCall(id="c1", name="read_file", arguments={}),),
    )


def _intent(session_id: SessionId, canonical_input: str) -> EffectIntentEntry:
    return EffectIntentEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        intent=EffectIntent(
            intent_id=IntentId.new(),
            tool_name="read_file",
            replay_policy="safe",
            contract_version=1,
            input_schema_digest="a" * 64,
            canonical_input=canonical_input,
            source_call_id="c1",
        ),
    )


def _result(session_id: SessionId, content: str) -> EffectResultEntry:
    return EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        intent_id=IntentId.new(),
        result=ToolResultEntry(
            tool_name="read_file", call_id="c1", outcome="succeeded", content=content
        ),
    )


def _seed_projection() -> ContextProjection:
    return ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=1,
        covered_through_sequence=0,
        summary=None,
        token_anchors=(
            TokenAnchor(through_sequence=0, measured_input_tokens=0, measured_output_tokens=0),
        ),
    )


class _FakeBoundary:
    """A durable boundary over an in-memory journal."""

    def __init__(self, entries: list[Any]) -> None:
        self.store = InMemoryAgentSessionStore()
        self.session_id = SessionId.new()
        self.entries = list(entries)
        self.commits: list[dict[str, Any]] = []
        self.seeded = False

    async def seed(self, *, projection: ContextProjection) -> None:
        commit = await self.store.append(
            session_id=self.session_id,
            expected_version=0,
            entries=self.entries,
            projection=projection,
        )
        assert isinstance(commit, SessionCommit)
        self.seeded = True

    async def load_snapshot(self) -> Any:
        return await self.store.load(self.session_id)

    async def commit_compaction(
        self,
        *,
        covered_through_sequence: int,
        first_retained_sequence: int,
        summary_json: str,
        token_anchors: tuple[TokenAnchor, ...],
    ) -> Any:
        self.commits.append(
            {
                "covered_through_sequence": covered_through_sequence,
                "first_retained_sequence": first_retained_sequence,
                "summary_json": summary_json,
                "token_anchors": token_anchors,
            }
        )
        projection = ContextProjection(
            projection_id=ProjectionId.new(),
            first_retained_sequence=first_retained_sequence,
            covered_through_sequence=covered_through_sequence,
            summary=summary_json,
            token_anchors=token_anchors,
        )
        entry = CompactionEntry(
            entry_id=EntryId.new(),
            session_id=self.session_id,
            timestamp=datetime.now(UTC),
            projection_id=projection.projection_id,
            summary=summary_json,
            covered_through_sequence=covered_through_sequence,
            first_retained_sequence=first_retained_sequence,
        )
        snapshot = await self.store.load(self.session_id)
        return await self.store.append(
            session_id=self.session_id,
            expected_version=snapshot.version,
            entries=[entry],
            projection=projection,
        )


def _stream_once(text: str) -> Any:
    async def stream(*, messages: list[dict[str, Any]]):
        yield text

    return stream


def _two_exchange_entries() -> tuple[SessionId, list[Any]]:
    """One big first exchange (≈20k tokens) plus a small newest exchange."""
    session_id = SessionId.new()
    big = "read " + "x" * 80_000
    return session_id, [
        _user(session_id, "Question"),
        _assistant(session_id, big),
        _intent(session_id, '{"path": "./notes.txt"}'),
        _result(session_id, "found notes"),
        _assistant(session_id, "second angle"),
        _intent(session_id, '{"path": "./other.txt"}'),
        _result(session_id, "found other"),
    ]


class TestParseCompactionSummary:
    def test_headings_map_onto_typed_fields(self) -> None:
        summary = parse_compaction_summary(_MARKDOWN)
        assert summary.goal == "Answer the question."
        assert "Cite every source." in summary.constraints_preferences
        assert "Read three sources." in summary.progress
        assert "Cross-check citation 2." in summary.progress
        assert "Use schema A" in summary.decisions
        assert "Verify citation 2." in summary.next_steps
        assert "source-uuid" in summary.critical_context
        assert "Leftovers" in summary.critical_context
        assert summary.paths is None
        assert summary.durable_handles is None

    def test_round_trip_is_canonical_json(self) -> None:
        summary = parse_compaction_summary(_MARKDOWN)
        assert CompactionSummary.from_canonical_json(summary.canonical_json()) == summary

    def test_missing_goal_is_a_failed_attempt(self) -> None:
        with pytest.raises(ValueError, match="goal"):
            parse_compaction_summary("## Next Steps\n1. Go")

    def test_unknown_headings_merge_into_critical_context(self) -> None:
        summary = parse_compaction_summary("## Goal\ng\n## Mystery\ndetail")
        assert summary.critical_context.strip() == "## Mystery\ndetail"

    def test_preamble_merges_into_critical_context(self) -> None:
        summary = parse_compaction_summary("Preamble line.\n\n## Goal\ng")
        assert "Preamble line." in summary.critical_context

    def test_whole_output_fence_is_stripped(self) -> None:
        summary = parse_compaction_summary(f"```markdown\n{_MARKDOWN}\n```")
        assert summary.goal == "Answer the question."
        assert "Leftovers" in summary.critical_context


class TestOverflowClassifier:
    def test_provider_text_classifies(self) -> None:
        assert is_provider_context_overflow(RuntimeError("prompt is too long: 500 > 100"))
        assert is_provider_context_overflow(
            RuntimeError("The input (500 tokens) is longer than the model's context length")
        )
        assert not is_provider_context_overflow(RuntimeError("network down"))

    def test_status_code_alone_does_not_classify(self) -> None:
        exc = RuntimeError("upstream error")
        exc.status_code = 413  # type: ignore[attr-defined]
        assert not is_provider_context_overflow(exc)


class TestCoordinatorLoop:
    async def test_force_compacts_once_even_under_the_trigger(self) -> None:
        session_id, entries = _two_exchange_entries()
        boundary = _FakeBoundary(entries)
        await boundary.seed(projection=_seed_projection())
        coordinator = CompactionCoordinator(
            model_profile=_profile(),
            context_policy=CONTEXT_POLICY,
            stream_model=_stream_once(_MARKDOWN),
        )
        trace: dict[str, Any] = {}

        async def remeasure() -> int:
            return 10_000  # under the trigger

        outcome = await coordinator.ensure_fits(
            boundaries=boundary, remeasure=remeasure, trace=trace, force=True
        )
        assert outcome is not None
        assert len(boundary.commits) == 1
        commit = boundary.commits[0]
        assert commit["covered_through_sequence"] == 4
        assert commit["first_retained_sequence"] == 5
        summary = CompactionSummary.from_canonical_json(commit["summary_json"])
        assert summary.paths == ["./notes.txt"]
        assert trace["compactions"][0]["hierarchical"] is False
        assert trace["compactions"][0]["summary_chars"] > 0

    async def test_exhausted_attempts_fail_loudly(self) -> None:
        boundary = _FakeBoundary([_user(SessionId.new(), "question")])
        coordinator = CompactionCoordinator(
            model_profile=_profile(),
            context_policy=CONTEXT_POLICY,
            stream_model=_stream_once(_MARKDOWN),
            max_attempts=2,
        )
        trace: dict[str, Any] = {}

        async def remeasure() -> int:
            return 40_000

        with pytest.raises(AnswerInputOverflowError, match="larger-context model"):
            await coordinator.ensure_fits(
                boundaries=boundary, remeasure=remeasure, trace=trace, force=True
            )

    async def test_invalid_summary_is_a_failed_attempt_that_never_commits(self) -> None:
        session_id, entries = _two_exchange_entries()
        boundary = _FakeBoundary(entries)
        await boundary.seed(projection=_seed_projection())
        coordinator = CompactionCoordinator(
            model_profile=_profile(),
            context_policy=CONTEXT_POLICY,
            stream_model=_stream_once("no headings at all"),
            max_attempts=2,
        )
        trace: dict[str, Any] = {}

        async def remeasure() -> int:
            return 40_000

        with pytest.raises(AnswerInputOverflowError):
            await coordinator.ensure_fits(
                boundaries=boundary, remeasure=remeasure, trace=trace, force=True
            )
        assert boundary.commits == []

    async def test_non_reducing_summary_is_discarded(self) -> None:
        # A summary larger than a tiny covered prefix must never commit.
        session_id = SessionId.new()
        boundary = _FakeBoundary(
            [
                _user(session_id, "Question"),
                _assistant(session_id, "checking"),
                _intent(session_id, '{"path": "./notes.txt"}'),
                _result(session_id, "found"),
            ]
        )
        await boundary.seed(projection=_seed_projection())
        coordinator = CompactionCoordinator(
            model_profile=_profile(),
            context_policy=CONTEXT_POLICY,
            stream_model=_stream_once("## Goal\n" + "long " * 4000),
            max_attempts=1,
        )
        trace: dict[str, Any] = {}

        async def remeasure() -> int:
            return 40_000

        with pytest.raises(AnswerInputOverflowError):
            await coordinator.ensure_fits(
                boundaries=boundary, remeasure=remeasure, trace=trace, force=True
            )
        assert boundary.commits == []
