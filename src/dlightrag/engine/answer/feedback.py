# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a reader thought of one answered message: its vocabulary, its record, and its one rule.

A reader judges a message with a thumb. A thumb down says why, in one of the few reasons the
reader is offered or in words of their own, and never in neither. The vocabulary is stated here
once: the service and the Web take every word of it from this module. The store adds no rule of
its own, and the table's constraints restate the shape only as the last word.
"""

import datetime
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, get_args

#: The agent a Run's own answer comes from. A Child Session is named by its own id instead.
MAIN_AGENT = "main"

#: The two thumbs a reader gives.
FeedbackValue = Literal["up", "down"]
FEEDBACK_VALUES: tuple[FeedbackValue, ...] = get_args(FeedbackValue)

#: The reasons a thumb down is given, in the order the reader is offered them.
FEEDBACK_REASONS: tuple[str, ...] = (
    "unsupported",
    "factual_error",
    "too_verbose",
    "off_topic",
    "irrelevant_citations",
)

#: The longest note a reader may add, in characters, once it is trimmed.
FEEDBACK_NOTE_LIMIT = 2000


@dataclass(frozen=True, slots=True)
class AnswerFeedback:
    """One reader's judgement of one message of one agent of a Run, as it is kept.

    ``agent`` is ``main`` for the Run's own agent, or the id of one of its Child Sessions, and
    ``sequence`` is the message's own place in that agent's Session, which never moves.
    ``updated_at`` is when the judgement was last given; when it was first given is kept in the
    table, for the training record, and is not part of what a reader sees.
    """

    agent: str
    sequence: int
    value: str
    reasons: tuple[str, ...]
    note: str | None
    updated_at: datetime.datetime


@dataclass(frozen=True, slots=True)
class Judgement:
    """What a reader judged one message to be: the vote and, for a thumb down, why.

    It is the part of an ``AnswerFeedback`` that travels with the message it judges, wherever
    that message is drawn: the message names its own place, and when the vote was given is the
    record's. Every client is told a judgement in the shape ``payload`` states.
    """

    value: str
    reasons: tuple[str, ...]
    note: str | None

    def payload(self) -> dict[str, Any]:
        """The judgement as clients read it: ``{value, reasons, note}``."""
        return {"value": self.value, "reasons": list(self.reasons), "note": self.note}


def normalize_feedback(
    value: str, reasons: Iterable[str], note: str | None
) -> tuple[str, tuple[str, ...], str | None]:
    """The judgement as it is kept, or a ValueError saying what the reader has to change.

    A thumb up carries no reasons and no note. A thumb down says why in at least one reason, each
    one of those the reader is offered, or in a note of their own words: a reason is not required
    where the words are there, and a thumb down that says neither is refused. Reasons are kept once
    each, in the order they are offered, so the same judgement is always kept the same way; a note
    is trimmed, and one left empty is none. The messages state what to change and never the words
    they were given.
    """
    if value not in FEEDBACK_VALUES:
        raise ValueError(f"value must be one of {', '.join(FEEDBACK_VALUES)}")
    chosen = set(reasons)
    if not chosen <= set(FEEDBACK_REASONS):
        raise ValueError(f"reasons must come from {', '.join(FEEDBACK_REASONS)}")
    kept = tuple(reason for reason in FEEDBACK_REASONS if reason in chosen)
    text = (note or "").strip()
    if value == "up":
        if kept or text:
            raise ValueError("a thumbs-up carries no reasons and no note")
        return value, (), None
    if len(text) > FEEDBACK_NOTE_LIMIT:
        raise ValueError(f"note is limited to {FEEDBACK_NOTE_LIMIT} characters")
    if not kept and not text:
        raise ValueError("a thumbs-down needs at least one reason or a note")
    return value, kept, text or None


__all__ = [
    "FEEDBACK_NOTE_LIMIT",
    "FEEDBACK_REASONS",
    "FEEDBACK_VALUES",
    "MAIN_AGENT",
    "AnswerFeedback",
    "FeedbackValue",
    "Judgement",
    "normalize_feedback",
]
