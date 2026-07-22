# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer generation and evaluation prompts."""

from .guidance import (
    ANSWER_CONTEXT_GUIDANCE,
    CITATION_GUIDANCE,
    HIGHLIGHT_GUIDANCE,
)
from .identity import CORE_IDENTITY

# --- Answer Generation ---

ANSWER_CORE = "\n\n".join(
    [
        CORE_IDENTITY,
        ANSWER_CONTEXT_GUIDANCE,
        CITATION_GUIDANCE,
    ]
)


def get_answer_system_prompt() -> str:
    """Return the single unified system prompt for answer generation.

    The LLM only generates the answer with inline ``[n]`` / ``[n-m]``
    citation markers. References are extracted programmatically by
    CitationProcessor -- the LLM is NOT asked to produce a References
    section.
    """
    return ANSWER_CORE


# --- Semantic Highlighting ---

HIGHLIGHT_SYSTEM_PROMPT = HIGHLIGHT_GUIDANCE
