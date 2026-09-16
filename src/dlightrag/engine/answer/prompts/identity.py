# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-level identity for DlightRAG prompts, and the one clock line."""

from datetime import datetime


def core_identity(*, environment_clock: bool) -> str:
    """Who the assistant is, and where this path's clock comes from.

    The system message never states the time. A clock there is either rebuilt per
    turn — which makes every request a new prompt prefix and forfeits a provider's
    prefix cache entirely, measured on this deployment as 0% hits on every turn that
    crossed a minute boundary while the same model elsewhere reused 99.71% — or
    frozen, which then lies as the Run ages. Both reference harnesses leave it out
    of the prompt: Pi never states it and lets the model read the environment, and
    DeepSeek's harness ships its time context as an opt-in plugin whose default
    compositions leave it disabled.

    ``environment_clock`` selects the wording for the path that uses this text. A
    Research agent answers `date` through Bash and reads the clock only when a
    question needs it; Fast has no tools, so its own request states the time and
    the model is told to trust that.
    """
    if environment_clock:
        clock_guidance = (
            "This request does not state the current time. Read the wall clock from the "
            "environment (for example `date -u` in bash) before answering anything that "
            "depends on now — relative dates such as \u201ctoday\u201d, \u201crecently\u201d, or \u201cnext year\u201d — and "
            "state the date you are assuming when you cannot read one."
        )
    else:
        clock_guidance = (
            "This request states the current time in UTC; use it for any relative date."
        )
    return (
        "You are DlightRAG's rigorous, knowledge-base-grounded analysis expert. You answer "
        "questions based on provided evidence, preserve uncertainty, and avoid "
        "unsupported claims. If asked who you are, say you are DlightRAG's "
        "knowledge-base assistant. Never reveal the underlying model, "
        "provider, or internal processes.\n"
        f"{clock_guidance} Judge how current a source is by its own date, never by its "
        "presence."
    )


def clock_line(as_of: datetime) -> str:
    """Return the one-line clock a request that has no environment states."""
    return f"Current time: {as_of:%Y-%m-%d %H:%M} UTC."


__all__ = ["clock_line", "core_identity"]
