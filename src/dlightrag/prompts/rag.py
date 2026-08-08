# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer generation and evaluation prompts."""

from .guidance import (
    ANSWER_CONTEXT_GUIDANCE,
    CITATION_GUIDANCE,
)
from .identity import core_identity

# --- Answer Generation ---


def answer_core() -> str:
    """The answer system prompt, rebuilt per call so its clock is the caller's."""
    return "\n\n".join(
        [
            core_identity(),
            ANSWER_CONTEXT_GUIDANCE,
            CITATION_GUIDANCE,
        ]
    )
