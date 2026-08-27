# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for context projection records and compaction validity checks."""

import pytest

from dlightrag.engine.agent.session.ids import EntryId, ProjectionId
from dlightrag.engine.agent.session.projection import (
    AgentInputOverflowError,
    CompactionSummary,
    ContextProjection,
    projection_strictly_reduces,
    render_compaction_summary,
    require_compactable,
    should_compact,
    validate_projection_commit,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile

PROFILE = ModelProfile(context_window_tokens=100_000)


def _initial_projection() -> ContextProjection:
    return ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=1,
        covered_through_sequence=0,
        summary=None,
    )


def _compacted_projection(
    *,
    covered: int,
    first_retained: int,
    summary: str,
) -> ContextProjection:
    return ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=first_retained,
        covered_through_sequence=covered,
        summary=summary,
        covered_through_entry_id=EntryId.new(),
        first_retained_entry_id=EntryId.new(),
        source_digest="a" * 64,
    )


class TestCompactionSummary:
    def test_canonical_round_trip(self) -> None:
        summary = CompactionSummary(
            goal="answer the question",
            progress="three sources reviewed",
            next_steps="verify citation 2",
        )
        encoded = summary.canonical_json()
        assert CompactionSummary.from_canonical_json(encoded) == summary

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValueError):
            CompactionSummary.from_canonical_json('{"goal":"g","made_up":1}')

    def test_rejects_empty_goal(self) -> None:
        with pytest.raises(ValueError):
            CompactionSummary(goal="   ")

    def test_render_is_deterministic(self) -> None:
        summary = CompactionSummary(goal="g", progress="p").canonical_json()
        first = render_compaction_summary(summary)
        assert first == render_compaction_summary(summary)
        assert "g" in first and "p" in first
        assert render_compaction_summary(None) == "No prior context summary."


class TestProjectionRecord:
    def test_initial_projection_is_valid(self) -> None:
        projection = _initial_projection()
        assert projection.covered_through_sequence == 0
        assert projection.summary is None

    def test_compacted_projection_requires_summary(self) -> None:
        with pytest.raises(ValueError):
            ContextProjection(
                projection_id=ProjectionId.new(),
                first_retained_sequence=3,
                covered_through_sequence=2,
                summary=None,
            )

    def test_initial_projection_cannot_carry_summary(self) -> None:
        with pytest.raises(ValueError):
            ContextProjection(
                projection_id=ProjectionId.new(),
                first_retained_sequence=1,
                covered_through_sequence=0,
                summary='{"goal":"g"}',
            )

    def test_retained_start_must_follow_covered_prefix(self) -> None:
        with pytest.raises(ValueError):
            ContextProjection(
                projection_id=ProjectionId.new(),
                first_retained_sequence=2,
                covered_through_sequence=2,
                summary='{"goal":"g"}',
            )

    def test_projection_stores_no_derived_model_messages(self) -> None:
        projection = _compacted_projection(covered=2, first_retained=3, summary='{"goal":"g"}')
        fields = {
            name
            for name in (
                "projection_id",
                "first_retained_sequence",
                "covered_through_sequence",
                "summary",
                "covered_through_entry_id",
                "first_retained_entry_id",
                "source_digest",
                "schema_version",
            )
        }
        assert set(projection.__dataclass_fields__) == fields  # type: ignore[attr-defined]
        assert "messages" not in fields


class TestCompactionValidity:
    def test_strictly_reducing_requires_progress_and_smaller_input(self) -> None:
        previous = _compacted_projection(covered=2, first_retained=3, summary='{"goal":"g"}')
        # Same coverage is not progress.
        assert not projection_strictly_reduces(
            previous,
            _compacted_projection(covered=2, first_retained=5, summary='{"goal":"g2"}'),
            accounted_input_before=1000,
            accounted_input_after=900,
        )
        # More coverage but larger accounted input is not reducing.
        assert not projection_strictly_reduces(
            previous,
            _compacted_projection(covered=5, first_retained=6, summary='{"goal":"g2"}'),
            accounted_input_before=1000,
            accounted_input_after=1100,
        )
        # More coverage and smaller input is a valid step.
        assert projection_strictly_reduces(
            previous,
            _compacted_projection(covered=5, first_retained=6, summary='{"goal":"g2"}'),
            accounted_input_before=1000,
            accounted_input_after=900,
        )

    def test_initial_projection_always_starts_valid(self) -> None:
        assert (
            validate_projection_commit(
                None,
                _initial_projection(),
                accounted_input_before=0,
                accounted_input_after=100,
            )
            is None
        )

    def test_non_reducing_candidate_is_rejected(self) -> None:
        previous = _compacted_projection(covered=2, first_retained=3, summary='{"goal":"g"}')
        reason = validate_projection_commit(
            previous,
            _compacted_projection(covered=3, first_retained=4, summary='{"goal":"g2"}'),
            accounted_input_before=1000,
            accounted_input_after=1001,
        )
        assert reason is not None
        assert "strictly reduce" in reason

    def test_compaction_trigger_and_hard_limit_classification(self) -> None:
        profile = PROFILE
        hard = CONTEXT_POLICY.hard_input_limit(profile)
        trigger = CONTEXT_POLICY.compaction_trigger(profile)
        assert trigger < hard
        assert should_compact(profile, input_tokens=trigger + 1)
        assert not should_compact(profile, input_tokens=trigger)

    def test_minimal_summary_overflow_fails_without_provider_call(self) -> None:
        hard = CONTEXT_POLICY.hard_input_limit(PROFILE)
        with pytest.raises(AgentInputOverflowError) as raised:
            require_compactable(
                PROFILE,
                input_tokens=hard + 1,
                fixed_input_tokens=hard - 10,
            )
        assert raised.value.input_limit_tokens == hard

    def test_fixed_input_above_hard_limit_also_fails(self) -> None:
        hard = CONTEXT_POLICY.hard_input_limit(PROFILE)
        with pytest.raises(AgentInputOverflowError):
            require_compactable(
                PROFILE,
                input_tokens=hard - 1,
                fixed_input_tokens=hard + 1,
            )
