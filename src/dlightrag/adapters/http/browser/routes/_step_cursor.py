# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The opaque cursor one Run step page states, shared by every page that walks steps.

A page states where it stopped and the next page is read from that token, so the token names the
step itself — its agent, its sequence, and its kind — rather than the position it happened to hold:
the roster can gain a Child between two reads, and a position would then name a different step. The
kind is part of the identity because a ``thinking`` step and the ``say``/``answer`` it produced
share one agent and sequence. The instant stays for a step that has left the Run.

This module is the format's one statement: the step search is its only writer and reader, so
the cursor a page states and the cursor it reads back cannot drift apart.
"""

from __future__ import annotations

import json

from dlightrag.application.answer_runs.timeline import (
    TIMELINE_KINDS,
    RunTimelinePosition,
    RunTimelineStep,
    parse_instant,
)


def encode_step_cursor(step: RunTimelineStep) -> str:
    """Where one page ended, as the opaque token the next page is read from."""
    return json.dumps(
        [step.instant.isoformat(), step.agent, step.sequence, step.kind],
        separators=(",", ":"),
    )


def decode_step_cursor(value: str) -> RunTimelinePosition | None:
    """One page cursor, or None when the token is not one these routes stated."""
    try:
        decoded = json.loads(value)
    except ValueError:
        return None
    if not (
        isinstance(decoded, list)
        and len(decoded) == 4
        and isinstance(decoded[0], str)
        and isinstance(decoded[1], str)
        and isinstance(decoded[2], int)
        and decoded[3] in TIMELINE_KINDS
    ):
        return None
    instant = parse_instant(decoded[0])
    if instant is None:
        return None
    return RunTimelinePosition(
        instant=instant, agent=decoded[1], sequence=decoded[2], kind=decoded[3]
    )


__all__ = ["decode_step_cursor", "encode_step_cursor"]
