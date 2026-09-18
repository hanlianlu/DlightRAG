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

    def test_render_without_the_tools_drops_the_calls_and_keeps_the_summary(self) -> None:
        """A Run that owns no `read` tool must not be told to make a re-read call.

        The stored summary is the Run that wrote it; rendering is a projection. Fast
        composes no tools, so its view of a Research Run's summary states the content
        and omits calls it cannot make.
        """
        summary = CompactionSummary(
            goal="Keep the decision.",
            decisions="Decided to keep the spill handles.",
            durable_handles=[
                '[spill] spill_read_ab12 (4096 bytes) — re-read with read(resource_id="spill_read_ab12")'
            ],
            run_notes=[
                "[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md') before a step that needs a value this summary does not state"
            ],
        ).canonical_json()

        rendered = render_compaction_summary(summary, re_readable_handles=False)

        assert "Keep the decision." in rendered
        assert "Decided to keep the spill handles." in rendered
        assert "re-read with" not in rendered
        assert "durable handles" not in rendered
        assert "Run Notes" not in rendered
        # The record itself is untouched: the same summary still renders its calls.
        with_calls = render_compaction_summary(summary)
        assert "re-read with read(resource_id=" in with_calls
        assert "re-read with read(path=" in with_calls

    def test_render_states_run_notes_as_paths_to_read_again(self) -> None:
        summary = CompactionSummary(
            goal="g",
            run_notes=[
                '[note] notes/plan.md (1240 bytes) — re-read with read(path="notes/plan.md") before a step that needs a value this summary does not state'
            ],
        ).canonical_json()

        rendered = render_compaction_summary(summary)

        assert "Run Notes (re-readable, not evidence):" in rendered
        assert "  - [note] notes/plan.md (1240 bytes)" in rendered
        assert "{" not in rendered

    def test_re_readable_handles_render_before_the_plan_they_serve(self) -> None:
        """A step that needs a value the summary omits reads the handle while planning.

        The live experiment this ordering answers: a Run was told twice that a note
        held its values and re-derived them with fresh searches both times, because the
        plan was written before the call that reads the note was seen.
        """
        summary = CompactionSummary(
            goal="Keep the decision.",
            next_steps="3. Divide the Stage 1 number by the Stage 2 number.",
            run_notes=[
                "[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md')"
            ],
            durable_handles=[
                '[spill] spill_read_ab12 (4096 bytes) — re-read with read(resource_id="spill_read_ab12")'
            ],
        ).canonical_json()

        rendered = render_compaction_summary(summary)

        assert rendered.index("Run Notes") < rendered.index("next steps:")
        assert rendered.index("durable handles") < rendered.index("next steps:")
        assert rendered.index("Run Notes") < rendered.index("durable handles")

    def test_a_summary_without_run_notes_still_decodes(self) -> None:
        """Adding a field is backward compatible; removing one is not.

        A projection committed before this field existed carries no key for it, and
        `from_canonical_json` rejects a key it does not know — so the decoder must
        keep accepting the older payload while the field's default stands in.
        """
        legacy = (
            '{"constraints_preferences":"","critical_context":"","decisions":"",'
            '"durable_handles":null,"goal":"g","next_steps":"","paths":null,"progress":""}'
        )

        summary = CompactionSummary.from_canonical_json(legacy)

        assert summary.goal == "g"
        assert summary.run_notes is None
        assert render_compaction_summary(legacy).endswith("goal: g")

    def test_render_states_durable_handles_as_a_re_readable_list(self) -> None:
        summary = CompactionSummary(
            goal="g",
            durable_handles=["[1] report.pdf", "[resource: res-1] memo.docx"],
        ).canonical_json()

        rendered = render_compaction_summary(summary)

        assert "durable handles (re-readable, not evidence):" in rendered
        assert "  - [1] report.pdf" in rendered
        assert "  - [resource: res-1] memo.docx" in rendered
        assert "{" not in rendered


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
