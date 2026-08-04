# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Forward the Docling code/formula preset choice to docling-serve.

Docling detects formula regions during layout but only transcribes them when
``do_formula_enrichment`` is on, and it transcribes them with whichever
code/formula model the request selects. docling-serve resolves that model from
the request's ``code_formula_preset``; when the field is absent it bypasses the
preset registry entirely and falls back to the pipeline's built-in default, so
``DOCLING_SERVE_DEFAULT_CODE_FORMULA_PRESET`` never applies. LightRAG 1.5.5
sends no preset field, which pins every deployment to that built-in default —
a model with no MLX engine override, so it hard-fails on Apple Silicon.

Sending the ``"default"`` alias instead of a concrete preset id hands the choice
back to the docling-serve operator: the alias resolves through
``DOCLING_SERVE_DEFAULT_CODE_FORMULA_PRESET``, so an MPS host can point it at an
MLX-capable model while a CUDA host keeps the stock one, with no DlightRAG
change. The alias is only sent when enrichment is on, so deployments that leave
it off never load a code/formula model.

Caveat: because the wire value is a constant, LightRAG's Docling bundle cache
cannot see a server-side preset change. Re-parse with ``force_reparse`` after
repointing the alias.

Delete this module once LightRAG forwards the preset itself.
"""

import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

# docling-serve's admin-controlled alias, resolved by
# DOCLING_SERVE_DEFAULT_CODE_FORMULA_PRESET.
_OPERATOR_PRESET_ALIAS = "default"
_PATCH_ATTR = "_dlightrag_sends_code_formula_preset"


def apply_docling_code_formula_preset() -> bool:
    """Patch LightRAG's Docling client to send the operator preset alias.

    Idempotent; returns True when it installs.
    """
    try:
        from lightrag.parser.external.docling.client import DoclingRawClient
    except Exception:  # pragma: no cover - defensive import guard
        return False

    original = DoclingRawClient._build_multipart_data
    if getattr(original, _PATCH_ATTR, False):
        return False

    @wraps(original)
    def patched_build_multipart_data(self: Any) -> dict[str, Any]:
        data = original(self)
        if self.do_formula_enrichment:
            data.setdefault("code_formula_preset", _OPERATOR_PRESET_ALIAS)
        return data

    setattr(patched_build_multipart_data, _PATCH_ATTR, True)
    patched_build_multipart_data._dlightrag_original = original  # type: ignore[attr-defined]
    DoclingRawClient._build_multipart_data = patched_build_multipart_data
    logger.info("Applied LightRAG Docling code/formula preset forwarding")
    return True


__all__ = ["apply_docling_code_formula_preset"]
