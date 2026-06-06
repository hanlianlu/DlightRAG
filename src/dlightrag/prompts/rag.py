# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer generation and evaluation prompts."""

from .guidance import (
    ANSWER_CITATION_EXAMPLE,
    ANSWER_CONTEXT_GUIDANCE,
    CITATION_GUIDANCE,
    HIGHLIGHT_GUIDANCE,
    HIGHLIGHT_RESPONSE_FORMAT,
    RERANK_GUIDANCE,
    VISUAL_RERANK_PROMPT_TEMPLATE,
)
from .identity import CORE_IDENTITY

# --- Answer Generation ---

ANSWER_CORE = "\n\n".join(
    [
        CORE_IDENTITY,
        ANSWER_CONTEXT_GUIDANCE,
        CITATION_GUIDANCE,
        ANSWER_CITATION_EXAMPLE,
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


# --- Reranking ---

VISUAL_RERANK_PROMPT = VISUAL_RERANK_PROMPT_TEMPLATE.format(
    query="{query}",
    rerank_guidance=RERANK_GUIDANCE,
)


# --- Semantic Highlighting ---

HIGHLIGHT_SYSTEM_PROMPT = "\n\n".join(
    [
        CORE_IDENTITY,
        HIGHLIGHT_GUIDANCE,
        HIGHLIGHT_RESPONSE_FORMAT,
    ]
)
