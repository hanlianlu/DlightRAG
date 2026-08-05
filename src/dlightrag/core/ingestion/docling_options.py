# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Forward the Docling code/formula preset choice to docling-serve.

Docling transcribes detected formula regions only when ``do_formula_enrichment``
is on, using whichever code/formula model the request names. When the request
omits ``code_formula_preset`` docling-serve bypasses its preset registry and
falls back to the pipeline's built-in ``codeformulav2``, so no server setting
can redirect it -- and because that model ships only a Transformers engine whose
supported devices exclude MPS, enrichment hard-fails on Apple Silicon. LightRAG
1.5.5 never sends the field.

Leaving the setting unset installs no patch, so the request stays exactly what
upstream sends. Naming a preset also requires the service to allow it; see
docs/configuration.md.

The preset also joins LightRAG's fixed pipeline constants, which exist so that a
change to the request shape invalidates every cached bundle on its own. Without
that, a repointed preset would re-parse Web Composer attachments -- whose
signature already covers it -- but leave workspace documents on stale
transcriptions.

Delete this module once LightRAG forwards the preset itself.
"""

import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_dlightrag_sends_code_formula_preset"


def apply_docling_code_formula_preset(preset: str | None) -> bool:
    """Patch LightRAG's Docling client to send *preset*.

    No preset means no patch, so the request stays byte-identical to upstream's.
    Idempotent; returns True when it installs.
    """
    if not preset:
        return False
    try:
        from lightrag.parser.external.docling.client import (
            FIXED_CONSTANTS,
            DoclingRawClient,
        )
    except Exception:  # pragma: no cover - defensive import guard
        return False

    original = DoclingRawClient._build_multipart_data
    if getattr(original, _PATCH_ATTR, False):
        return False

    @wraps(original)
    def patched_build_multipart_data(self: Any) -> dict[str, Any]:
        data = original(self)
        if self.do_formula_enrichment:
            data.setdefault("code_formula_preset", preset)
        return data

    setattr(patched_build_multipart_data, _PATCH_ATTR, True)
    patched_build_multipart_data._dlightrag_original = original  # type: ignore[attr-defined]
    DoclingRawClient._build_multipart_data = patched_build_multipart_data
    # Both cache-signature call sites read this dict by reference; setdefault
    # mirrors the body patch so an upstream value always wins.
    FIXED_CONSTANTS.setdefault("code_formula_preset", preset)
    logger.info("Applied LightRAG Docling code/formula preset forwarding: %s", preset)
    return True


__all__ = ["apply_docling_code_formula_preset"]
