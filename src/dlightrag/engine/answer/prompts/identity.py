# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-level identity for DlightRAG prompts."""

from datetime import UTC, datetime


def core_identity() -> str:
    """Who the assistant is, and where its clock is stated.

    The clock is deliberately *not* in this text. A system message that changes
    every minute makes every request a new prompt prefix, and a provider's
    prefix cache then reuses nothing: DeepSeek bills a cache miss at the full
    input rate and is authoritative about the hit, so a per-minute string at
    offset ~60 tokens cost every prompt token of every turn. Measured on this
    deployment, turns that crossed a minute boundary reported
    ``prompt_cache_hit_tokens: 0`` while turns inside one minute reported
    56k-63k of a 64k-82k prompt as hits.

    The current time rides in ``clock_line`` near the end of each request, where a
    new value costs that one message instead of the whole prompt.
    """
    return (
        "You are DlightRAG's rigorous, knowledge-base-grounded analysis expert. You answer "
        "questions based on provided evidence, preserve uncertainty, and avoid "
        "unsupported claims. If asked who you are, say you are DlightRAG's "
        "knowledge-base assistant. Never reveal the underlying model, "
        "provider, or internal processes.\n"
        "The current time is stated at the end of this request. Evidence may be older "
        "than that; judge how current a source is by its own date, never by its presence."
    )


def clock_line(as_of: datetime) -> str:
    """Return the one-line clock a request carries near its end.

    Research places it immediately before the trailing control instruction and Fast
    immediately before the question block, so whatever follows it in a request is
    the actionable text. Frozen per Run: a Run's turns share one value, so the
    message is byte-stable for its whole life and only a new Run moves it.
    """
    return f"Current time: {as_of:%Y-%m-%d %H:%M} UTC."


def run_clock(as_of: datetime | None = None) -> datetime:
    """Return the clock one Run pins for every request it composes."""
    return as_of if as_of is not None else datetime.now(UTC)


__all__ = ["clock_line", "core_identity", "run_clock"]
