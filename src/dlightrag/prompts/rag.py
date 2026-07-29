# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer generation and evaluation prompts."""

from .guidance import (
    ANSWER_CONTEXT_GUIDANCE,
    CITATION_GUIDANCE,
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
