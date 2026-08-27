# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Forward the Docling request options LightRAG never sends.

LightRAG's Docling client builds a fixed multipart body, so any docling-serve
option outside that body is unreachable no matter how the service is
configured. Two of them change the corpus materially:

``code_formula_preset`` names the model that transcribes detected formula
regions when ``do_formula_enrichment`` is on. Omitting it makes docling-serve
bypass its preset registry and fall back to the pipeline's built-in
``codeformulav2``, which ships only a Transformers engine whose supported
devices exclude MPS -- so enrichment hard-fails on Apple Silicon.

``do_pdf_heading_hierarchy`` infers section-header levels from PDF bookmarks,
outline numbering and font style. docling-serve defaults it off, and without it
Docling leaves every heading at level 1, so LightRAG's deliberately faithful IR
builder collapses every chunk's ``parent_headings`` to the document title alone.
It is always forwarded: turning it off only degrades the corpus. Needs
docling-serve >= 1.30.0 (docling-jobkit >= 3.3.0); older services accept the
field and drop it silently. See docs/configuration.md.

A forwarded option also joins LightRAG's fixed pipeline constants, which exist
so that a change to the request shape invalidates every cached bundle on its
own. Without that, a repointed preset or a newly inferred hierarchy would leave
already-ingested workspace documents on stale parses.

Delete this module once LightRAG forwards these itself.
"""

import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_dlightrag_forwards_docling_options"


def apply_docling_request_options(*, code_formula_preset: str | None = None) -> bool:
    """Patch LightRAG's Docling client to send the options it omits.

    Call only when Docling is the active parser. Idempotent; returns True when
    it installs.
    """
    from lightrag.parser.external.docling.client import (
        FIXED_CONSTANTS,
        DoclingRawClient,
        _bool_form,
    )

    original = DoclingRawClient._build_multipart_data
    if getattr(original, _PATCH_ATTR, False):
        return False

    @wraps(original)
    def patched_build_multipart_data(self: Any) -> dict[str, Any]:
        data = original(self)
        if code_formula_preset and self.do_formula_enrichment:
            data.setdefault("code_formula_preset", code_formula_preset)
        data.setdefault("do_pdf_heading_hierarchy", _bool_form(True))
        return data

    setattr(patched_build_multipart_data, _PATCH_ATTR, True)
    DoclingRawClient._build_multipart_data = patched_build_multipart_data
    # Both cache-signature call sites read this dict by reference; setdefault
    # mirrors the body patch so an upstream value always wins.
    if code_formula_preset:
        FIXED_CONSTANTS.setdefault("code_formula_preset", code_formula_preset)
    FIXED_CONSTANTS.setdefault("do_pdf_heading_hierarchy", True)
    logger.info(
        "Applied LightRAG Docling option forwarding: code_formula_preset=%s",
        code_formula_preset,
    )
    return True


__all__ = ["apply_docling_request_options"]
