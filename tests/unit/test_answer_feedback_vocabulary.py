# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The feedback vocabulary the Web parses is the vocabulary the server keeps.

The Engine states the reasons a thumb down is given, in the order the reader is offered them and
the server keeps them, and the longest note it keeps. `frontend/api/feedback.ts` cannot import
them, and the browser parses every turn and Child row with a picklist of those reasons. A reason
added on one side only makes each history page that carries it unparseable, so the two are locked
here instead of being trusted.
"""

import re
from pathlib import Path

from dlightrag.engine.answer.feedback import FEEDBACK_NOTE_LIMIT, FEEDBACK_REASONS

_FEEDBACK = Path(__file__).resolve().parents[2] / "frontend/api/feedback.ts"


def _frontend_constant(pattern: str) -> str:
    """The text one pattern captures from the frontend's feedback module."""
    match = re.search(pattern, _FEEDBACK.read_text(encoding="utf-8"))
    assert match is not None, f"{pattern!r} is not in {_FEEDBACK.name}"
    return match.group(1)


def test_the_frontend_offers_the_reasons_the_server_keeps_in_the_order_it_keeps_them() -> None:
    reasons = _frontend_constant(r"FEEDBACK_REASONS = \[([^\]]*)\] as const")

    assert tuple(re.findall(r"'([a-z_]+)'", reasons)) == FEEDBACK_REASONS


def test_the_frontend_states_the_longest_note_the_server_keeps() -> None:
    assert int(_frontend_constant(r"FEEDBACK_NOTE_LIMIT = (\d+);")) == FEEDBACK_NOTE_LIMIT
