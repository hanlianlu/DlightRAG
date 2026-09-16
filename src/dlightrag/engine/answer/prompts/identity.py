# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Product-level identity for DlightRAG prompts."""


def core_identity() -> str:
    """Who the assistant is, and where its clock comes from.

    No request states the current time. A clock in the prompt is either rebuilt per
    turn — which makes every request a new prompt prefix and forfeits a provider's
    prefix cache entirely, measured on this deployment as 0% hits on every turn that
    crossed a minute boundary while the same model elsewhere reused 99.71% — or
    frozen, which then lies as the Run ages. Both reference harnesses leave it out:
    Pi never states it and lets the model read the environment, and DeepSeek's
    harness ships its time context as an opt-in plugin whose default compositions
    leave it disabled.

    DlightRAG's Research environment already answers `date` through Bash, so the
    clock is a tool call the model makes when a question needs it, and the sentence
    below is the only prompt cost: one static line, identical for every request.
    """
    return (
        "You are DlightRAG's rigorous, knowledge-base-grounded analysis expert. You answer "
        "questions based on provided evidence, preserve uncertainty, and avoid "
        "unsupported claims. If asked who you are, say you are DlightRAG's "
        "knowledge-base assistant. Never reveal the underlying model, "
        "provider, or internal processes.\n"
        "This request does not state the current time. Judge how current a source is by "
        "its own date, never by its presence, and read the wall clock from the "
        "environment (for example `date -u` in bash) before answering anything that "
        "depends on now — relative dates such as \u201ctoday\u201d, \u201crecently\u201d, or \u201cnext year\u201d. When "
        "no clock is available to you, state the date you are assuming instead of "
        "guessing one silently."
    )


__all__ = ["core_identity"]
