# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one rule that shapes a reader's judgement before it is kept."""

import pytest

from dlightrag.engine.answer.feedback import (
    FEEDBACK_NOTE_LIMIT,
    FEEDBACK_REASONS,
    normalize_feedback,
)


def test_a_thumbs_up_is_kept_as_the_thumb_alone() -> None:
    assert normalize_feedback("up", (), None) == ("up", (), None)
    assert normalize_feedback("up", [], " \n ") == ("up", (), None), "a note of nothing is none"


def test_a_thumbs_down_keeps_each_reason_once_in_the_order_they_are_offered() -> None:
    assert normalize_feedback("down", ["off_topic", "unsupported", "off_topic"], None) == (
        "down",
        ("unsupported", "off_topic"),
        None,
    )
    assert normalize_feedback("down", reversed(FEEDBACK_REASONS), None)[1] == FEEDBACK_REASONS


def test_a_note_is_trimmed_and_is_none_when_nothing_is_left() -> None:
    assert normalize_feedback("down", ["too_verbose"], "  too long \n")[2] == "too long"
    assert normalize_feedback("down", ["too_verbose"], "   ")[2] is None
    assert normalize_feedback("down", ["too_verbose"], None)[2] is None


def test_a_thumb_down_says_why_in_words_alone_where_no_reason_is_chosen() -> None:
    assert normalize_feedback("down", (), "  cites nothing \n") == ("down", (), "cites nothing")
    assert normalize_feedback("down", [], "cites nothing") == ("down", (), "cites nothing")
    # A reason and words are not alternatives: both are kept when the reader gives both.
    assert normalize_feedback("down", ["off_topic"], "cites nothing") == (
        "down",
        ("off_topic",),
        "cites nothing",
    )


def test_the_note_limit_counts_the_characters_left_after_trimming() -> None:
    at_limit = "字" * FEEDBACK_NOTE_LIMIT

    assert normalize_feedback("down", ["off_topic"], f"  {at_limit}\n")[2] == at_limit
    with pytest.raises(ValueError, match=str(FEEDBACK_NOTE_LIMIT)):
        normalize_feedback("down", ["off_topic"], at_limit + "字")


@pytest.mark.parametrize(
    ("value", "reasons", "note", "says"),
    [
        ("sideways", [], None, "value must be one of up, down"),
        ("up", ["unsupported"], None, "carries no reasons"),
        ("up", [], "great", "carries no reasons and no note"),
        ("down", [], None, "at least one reason or a note"),
        ("down", [], "  \n ", "at least one reason or a note"),
        ("down", [], "字" * (FEEDBACK_NOTE_LIMIT + 1), "note is limited"),
        ("down", ["boring"], None, "reasons must come from"),
    ],
)
def test_a_judgement_that_cannot_be_kept_says_what_to_change(
    value: str, reasons: list[str], note: str | None, says: str
) -> None:
    with pytest.raises(ValueError, match=says):
        normalize_feedback(value, reasons, note)


def test_an_unknown_reason_is_answered_with_those_on_offer_and_never_echoed() -> None:
    with pytest.raises(ValueError) as refused:
        normalize_feedback("down", ["secret-token-xyz"], None)

    assert all(reason in str(refused.value) for reason in FEEDBACK_REASONS)
    assert "secret-token-xyz" not in str(refused.value)
