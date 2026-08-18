# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the final M3 answer result validator and read-side derivation."""

import pytest

from dlightrag.answer.final_result import (
    DurableSourceIdentity,
    derive_answer_blocks,
    derive_references,
    validate_final_result,
)


def _valid_payload() -> dict:
    return {
        "answer": "The answer is 42 [1-1].",
        "answer_sources": [
            {"id": "s1", "source_uri": "local://default/report.pdf", "title": "report"},
        ],
        "report_sources": None,
        "primary_report": None,
        "artifacts": [],
        "answer_images": [],
        "trace": {"agent_turns": 2},
        "usage": {"input_tokens": 10, "output_tokens": 4},
        "image_descriptions": [],
    }


def test_accepts_declared_shape_and_derives_references_on_read() -> None:
    result = validate_final_result(_valid_payload())

    assert result.answer == "The answer is 42 [1-1]."
    assert result.answer_sources[0].id == "s1"
    assert result.usage == {"input_tokens": 10, "output_tokens": 4}

    references = derive_references(result)
    assert references == [{"id": "s1", "title": "report"}]

    # contexts/references/blocks are derived, never durable fields.
    assert "contexts" not in result.__dataclass_fields__
    assert "references" not in result.__dataclass_fields__
    assert "answer_blocks" not in result.__dataclass_fields__


def test_derives_answer_blocks_from_markdown_and_images() -> None:
    payload = _valid_payload()
    payload["answer"] = "look at this [1-1] and that [2-1]"
    payload["answer_images"] = [{"id": "img-1", "source_ref": "1-1"}]
    result = validate_final_result(payload)

    blocks = derive_answer_blocks(result)
    assert any(block["type"] == "image_ref" for block in blocks)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(answer=None),
        lambda p: p.update(answer_sources="nope"),
        lambda p: p.update(answer_sources=[{"id": "s1"}]),  # missing source_uri
        lambda p: p.update(trace="nope"),
        lambda p: p.update(primary_report=3),
        lambda p: p.update(image_descriptions=[1]),
        lambda p: p.update(artifacts=[1]),
        lambda p: p.update(unknown_field=True),
        lambda p: p.update(usage=[1]),
    ],
)
def test_malformed_payloads_fail_validation(mutate) -> None:
    payload = _valid_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        validate_final_result(payload)


def test_report_sources_and_nullable_fields_round_trip() -> None:
    payload = _valid_payload()
    payload["report_sources"] = [{"id": "r1", "source_uri": "local://default/r.pdf"}]
    payload["primary_report"] = "report-body"
    result = validate_final_result(payload)

    assert result.report_sources[0].id == "r1"
    assert result.primary_report == "report-body"


def test_durable_source_identity_is_strict() -> None:
    with pytest.raises(ValueError):
        DurableSourceIdentity.from_payload({"id": "", "source_uri": "x"})
    with pytest.raises(ValueError):
        DurableSourceIdentity.from_payload({"id": "s", "source_uri": 3})
    source = DurableSourceIdentity.from_payload({"id": "s", "source_uri": "x"})
    assert source.as_payload() == {"id": "s", "source_uri": "x"}


def test_final_result_is_frozen() -> None:
    from dataclasses import FrozenInstanceError
    from typing import Any, cast

    result = validate_final_result(_valid_payload())
    mutable = cast(Any, result)
    with pytest.raises(FrozenInstanceError):
        mutable.answer = "mutated"
    assert result == validate_final_result(_valid_payload())


def test_evidence_digests_are_deterministic_and_conflict_exactly() -> None:
    from dlightrag.answer.evidence_digests import (
        digests_conflict_reason,
        digests_match,
        evidence_digests,
    )

    first = evidence_digests(content={"text": "found"}, locator={"uri": "a"})
    same = evidence_digests(content={"text": "found"}, locator={"uri": "a"})
    assert first == same
    assert digests_match(first, same)

    other_content = evidence_digests(content={"text": "changed"}, locator={"uri": "a"})
    other_locator = evidence_digests(content={"text": "found"}, locator={"uri": "b"})
    assert not digests_match(first, other_content)
    assert not digests_match(first, other_locator)
    conflict = digests_conflict_reason(first, other_content)
    assert conflict is not None
    assert "evidence_settlement_conflict" in conflict
    assert digests_conflict_reason(first, same) is None


def test_canonical_identity_is_stable() -> None:
    from dlightrag.answer.evidence_digests import canonical_evidence_identity

    identity = canonical_evidence_identity(
        owner_id="o",
        run_id="r",
        session_id="s",
        intent_id="i",
        result_ordinal=2,
    )
    assert identity == canonical_evidence_identity(
        owner_id="o", run_id="r", session_id="s", intent_id="i", result_ordinal=2
    )
    assert identity != canonical_evidence_identity(
        owner_id="o", run_id="r", session_id="s", intent_id="i", result_ordinal=3
    )
