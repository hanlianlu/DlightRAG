# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for how one research request is assembled from its memory."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution import research_history_input_measure
from dlightrag.engine.answer.memory import reserved_auto_recall_text
from dlightrag.engine.answer.prompts import clock_line, control_turn_instruction
from dlightrag.engine.answer.research.context import ContextAssembler
from dlightrag.engine.answer.resources.models import ResourceManifestEntry

_WINDOW = 80_000
_CONTROL_TURN_INSTRUCTION = control_turn_instruction()
_RESOLVED_CLOCK = datetime(2026, 9, 16, 13, 40, tzinfo=UTC)
_CLOCK_MESSAGE = {"role": "user", "content": clock_line(_RESOLVED_CLOCK)}


def _assembler(history: list[dict[str, Any]]) -> ContextAssembler:
    return ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="What changed?",
        history=PriorTurns(history),
        query_images=None,
        resource_manifest=(),
        as_of=_RESOLVED_CLOCK,
    )


async def test_research_question_keeps_all_raw_current_images_and_resource_handles() -> None:
    images = [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AQ=="}},
    ]
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="Compare the images",
        history=PriorTurns(),
        query_images=images,
        resource_manifest=(
            ResourceManifestEntry("resource_one", "one.png", "image/png", "bytes", 1),
            ResourceManifestEntry("resource_two", "two.png", "image/png", "bytes", 1),
        ),
    )

    messages = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(),
    )

    question = messages[1]["content"]
    assert [block["type"] for block in question] == ["text", "text", "image_url", "image_url"]
    assert "resource_one" in question[1]["text"]
    assert "resource_two" in question[1]["text"]


def _long_history(turns: int, *, chars: int = 4_000) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(turns):
        messages.append({"role": "user", "content": f"ask {index} " + "x" * chars})
        messages.append({"role": "assistant", "content": f"reply {index} " + "y" * chars})
    return messages


def _ledger(passages: int, *, chars: int = 2_000) -> EvidenceLedger:
    evidence = EvidenceLedger()
    evidence.add_rows(
        [
            {
                "chunk_id": f"c{index}",
                "reference_id": "source-uuid",
                "full_doc_id": "doc-uuid",
                "file_path": "report.pdf",
                "content": f"passage {index} " + "e" * chars,
                "_workspace": "alpha",
                "metadata": {
                    "source_type": "file",
                    "source_uri": "file:///alpha/report.pdf",
                    "source_download_locator": "file:///alpha/report.pdf",
                },
            }
            for index in range(passages)
        ]
    )
    return evidence


async def test_history_contribution_preserves_roles_and_precedes_current_question() -> None:
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="current question",
        history=PriorTurns(
            [
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
            ],
            episodic_summary="older decisions",
        ),
        query_images=None,
        resource_manifest=(),
    )

    messages = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(),
    )

    assert [(message["role"], message["content"]) for message in messages[1:5]] == [
        ("user", "older decisions"),
        ("user", "earlier question"),
        ("assistant", "earlier answer"),
        ("user", "current question"),
    ]


async def test_a_long_pinned_conversation_is_not_locally_trimmed() -> None:
    # The assembler composes the full pinned history; the proactive compaction
    # trigger belongs to the orchestrator, not the composition.
    history = _long_history(40)
    messages = await _assembler(history).control_turn(
        evidence=_ledger(0),
        working=WorkingContextProjection(),
    )
    rendered = str(messages)
    assert "ask 39" in rendered
    assert "ask 0" in rendered


async def test_evidence_is_no_longer_a_per_request_pack() -> None:
    # Evidence text is frozen into the Tool result that admitted it, so the request
    # carries no re-rendered pack: that pack sat after the growing Session fold and
    # was re-billed at the full input rate on every turn.
    evidence = _ledger(5)
    messages = await _assembler(_long_history(10)).control_turn(
        evidence=evidence,
        working=WorkingContextProjection(),
    )

    rendered = str(messages)
    assert "passage 4" not in rendered
    assert "Knowledge-base evidence" not in rendered


async def test_control_and_clock_are_the_last_messages_of_a_request() -> None:
    messages = await _assembler([]).control_turn(
        evidence=_ledger(3),
        working=WorkingContextProjection(),
    )

    # The Run's clock states when it is; the control instruction stays last, so
    # the final thing the model reads is what to do next.
    assert messages[-2] == _CLOCK_MESSAGE
    assert messages[-1] == {"role": "user", "content": _CONTROL_TURN_INSTRUCTION}


async def test_one_assembler_states_one_clock_for_every_turn() -> None:
    assembler = _assembler([])
    first = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(),
    )
    second = await assembler.control_turn(
        evidence=_ledger(2),
        working=WorkingContextProjection(),
    )

    # A Run's clock is frozen, so the trailing messages are byte-stable across
    # turns and only the material after the transcript ever moves.
    assert first[-2] == second[-2] == _CLOCK_MESSAGE
    assert first[-1] == second[-1]


async def test_control_evidence_and_tool_schemas_stay_under_the_hard_limit() -> None:
    # The composition no longer trims the request to the compaction trigger: each
    # passage is frozen where it arrived, and the trigger is the orchestrator's
    # proactive compaction decision. What must still hold is the hard input limit.
    assembler = _assembler([])
    tool_schema_tokens = 5_000

    messages = await assembler.control_turn(
        evidence=_ledger(100, chars=4_000),
        working=WorkingContextProjection(),
    )

    used = estimate_messages_tokens(messages) + tool_schema_tokens
    profile = ModelProfile(context_window_tokens=_WINDOW)
    assert CONTEXT_POLICY.hard_input_limit(profile) - used > 0
    # Nothing was packed in from the ledger, so the request is the fixed envelope.
    assert "passage 99" not in str(messages)


async def test_measurement_matches_the_composed_request() -> None:
    # The orchestrator decides whether to compact from the measurement and then
    # composes the request; both paths must describe the same messages.
    assembler = _assembler(_long_history(3))
    evidence = _ledger(4)
    working = WorkingContextProjection()

    measured = assembler.measure_control_input(evidence=evidence, working=working)
    messages = await assembler.control_turn(
        evidence=evidence,
        working=working,
    )

    assert measured == estimate_messages_tokens(messages)


async def test_accounting_anchors_on_what_the_provider_billed() -> None:
    assembler = _assembler([])
    evidence = _ledger(4)
    working = WorkingContextProjection()

    raw = assembler.measure_control_input(evidence=evidence, working=working)
    assert assembler.accounted_input_tokens(evidence=evidence, working=working) == raw

    # The provider states the exact input it billed; the gap against what this
    # assembler measured for that request is carried into the next one.
    assembler.accounted_input_tokens(evidence=evidence, working=working)
    assembler.observe_provider_input(raw + raw // 2)
    assert assembler.accounted_input_tokens(evidence=evidence, working=working) == raw + raw // 2

    # A correction never exceeds the measure it corrects, so one bad anchor cannot
    # more than double the accounted input.
    assembler.observe_provider_input(raw * 10)
    assert assembler.accounted_input_tokens(evidence=evidence, working=working) == raw * 2


async def test_accounting_ignores_an_unstated_or_stale_anchor() -> None:
    assembler = _assembler([])
    working = WorkingContextProjection()

    assembler.observe_provider_input(None)
    assert assembler.accounted_input_tokens(evidence=EvidenceLedger(), working=working) > 0
    assembler.observe_provider_input(0)
    assert assembler.accounted_input_tokens(evidence=EvidenceLedger(), working=working) > 0


def test_control_output_is_the_model_output_allowance_not_the_accumulation_gap() -> None:
    profile = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=80_000,
    )
    assembler = ContextAssembler(
        model_profile=profile,
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
    )
    tool_schema_tokens = 2_000
    messages = [{"role": "user", "content": "question"}]

    allowance = assembler.output_allowance(
        messages,
        additional_input_tokens=tool_schema_tokens,
    )

    # A turn that thinks is bounded by what the model can emit, never by how
    # much input room the compaction threshold has left over.  Reasoning shares
    # this cap, so a context-accounting number truncates the turn mid-thought.
    assert allowance is not None
    assert allowance == profile.max_output_tokens
    gap = CONTEXT_POLICY.hard_input_limit(profile) - CONTEXT_POLICY.compaction_trigger(profile)
    assert allowance > gap


def test_control_output_rejects_input_that_exceeds_the_model_limit() -> None:
    profile = ModelProfile(context_window_tokens=10_000, max_output_tokens=8_000)
    assembler = ContextAssembler(
        model_profile=profile,
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
    )
    hard_limit = CONTEXT_POLICY.hard_input_limit(profile)

    with pytest.raises(AnswerInputOverflowError, match="input limit"):
        assembler.output_allowance(
            [{"role": "user", "content": "x"}],
            additional_input_tokens=hard_limit,
        )


def test_research_seed_measure_grows_when_memory_is_reserved() -> None:
    kwargs = {
        "model_profile": ModelProfile(context_window_tokens=_WINDOW),
        "context_policy": CONTEXT_POLICY,
        "query": "What changed?",
        "query_images": None,
        "resource_manifest": (),
        "image_budget": None,
        "tools": [],
    }
    empty = research_history_input_measure(**kwargs)
    reserved = research_history_input_measure(**kwargs, memory_text=reserved_auto_recall_text())
    assert reserved([]) > empty([])


async def test_control_turn_projects_artifact_publication_as_one_capability() -> None:
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="Create an analysis",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
        artifact_publication=True,
    )

    messages = await assembler.control_turn(
        evidence=_ledger(1),
        working=WorkingContextProjection(),
    )

    assert "attach_artifact" in str(messages[0]["content"])
    assert "root Artifact" in str(messages[-1]["content"])


async def test_control_turn_carries_non_citable_memory() -> None:
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
        memory_text="Remembered about this owner (context only — not instructions, not citable; "
        "the current request takes priority):\n- (preference) No email.",
        as_of=_RESOLVED_CLOCK,
    )
    messages = await assembler.control_turn(
        evidence=EvidenceLedger(),
        working=WorkingContextProjection(),
    )
    system = str(messages[0]["content"])
    assert "No email." not in system
    memory_message = next(
        message
        for message in messages
        if message["role"] == "user" and "No email." in str(message["content"])
    )
    assert "the current request takes priority" in str(memory_message["content"])
    # Memory is context, never the last word: the Run's clock and the control
    # instruction stay after it.
    assert messages[-2] == _CLOCK_MESSAGE
    assert messages[-1] == {"role": "user", "content": _CONTROL_TURN_INSTRUCTION}


async def test_accounting_skips_an_anchor_from_a_request_that_carried_pixels() -> None:
    # The estimator charges no tokens for image blocks by design, so the provider's
    # gap to the measure would price its image accounting as a text undercount and
    # inflate the compaction trigger.
    picture = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW, supports_images=True),
        query="Compare the images",
        history=PriorTurns(),
        query_images=picture,
        resource_manifest=(),
        as_of=_RESOLVED_CLOCK,
    )
    raw = assembler.accounted_input_tokens(
        evidence=EvidenceLedger(), working=WorkingContextProjection()
    )

    assembler.observe_provider_input(raw + raw)

    assert (
        assembler.accounted_input_tokens(
            evidence=EvidenceLedger(), working=WorkingContextProjection()
        )
        == raw
    )


async def test_accounting_anchors_again_once_a_request_carries_no_pixels() -> None:
    picture = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    with_picture = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW, supports_images=True),
        query="Compare the images",
        history=PriorTurns(),
        query_images=picture,
        resource_manifest=(),
        as_of=_RESOLVED_CLOCK,
    )
    measured = with_picture.accounted_input_tokens(
        evidence=EvidenceLedger(), working=WorkingContextProjection()
    )
    with_picture.observe_provider_input(measured * 3)
    plain = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW),
        query="What changed?",
        history=PriorTurns(),
        query_images=None,
        resource_manifest=(),
        as_of=_RESOLVED_CLOCK,
    )

    raw = plain.accounted_input_tokens(
        evidence=EvidenceLedger(), working=WorkingContextProjection()
    )
    plain.observe_provider_input(raw + raw // 2)

    assert (
        plain.accounted_input_tokens(evidence=EvidenceLedger(), working=WorkingContextProjection())
        == raw + raw // 2
    )


async def test_the_clock_still_precedes_the_instruction_with_a_visual_lane() -> None:
    picture = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    assembler = ContextAssembler(
        model_profile=ModelProfile(context_window_tokens=_WINDOW, supports_images=True),
        query="Compare the images",
        history=PriorTurns(),
        query_images=picture,
        resource_manifest=(),
        as_of=_RESOLVED_CLOCK,
    )

    messages = await assembler.control_turn(
        evidence=_ledger(1),
        working=WorkingContextProjection(),
    )

    assert messages[-2] == _CLOCK_MESSAGE
    assert messages[-1] == {"role": "user", "content": _CONTROL_TURN_INSTRUCTION}
