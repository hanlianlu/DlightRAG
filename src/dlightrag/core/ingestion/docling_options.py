# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Forward the Docling code/formula preset choice to docling-serve.

Docling detects formula regions during layout but only transcribes them when
``do_formula_enrichment`` is on, and it transcribes them with whichever
code/formula model the request selects. docling-serve resolves that model from
the request's ``code_formula_preset``; when the field is absent it bypasses the
preset registry entirely and falls back to the pipeline's built-in default, so
no amount of server configuration applies. LightRAG 1.5.5 sends no preset field,
which pins every deployment to that built-in default — a model with no MLX
engine override, so it hard-fails on Apple Silicon.

The preset is therefore an explicit DlightRAG setting rather than a constant.
Leaving it unset installs no patch at all, so the request stays exactly what
upstream sends: nothing new can break. Setting it is what opts a deployment into
choosing the model, and both forms of that choice need one matching setting on
the parser service:

* ``default`` is docling-serve's admin alias, resolved by
  ``DOCLING_SERVE_DEFAULT_CODE_FORMULA_PRESET``. That server setting is
  mandatory, because docling-serve ships the alias pointing at the literal id
  ``"default"`` while docling registers only ``codeformulav2`` and
  ``granite_docling`` — the stock value resolves to a preset that does not exist
  and raises ``KeyError``. Nothing upstream sends a preset, which is why the
  broken default goes unnoticed there. The upside is that one DlightRAG config
  serves hosts with different accelerators.
* A concrete id such as ``granite_docling`` names the model in DlightRAG config,
  but the service must list it in ``DOCLING_SERVE_ALLOWED_CODE_FORMULA_PRESETS``
  or it is rejected as not allowed.

Caveat: because the wire value is a constant per deployment, LightRAG's Docling
bundle cache cannot see a server-side preset change. Delete and re-ingest the
affected documents after repointing the alias.

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
            data.setdefault("code_formula_preset", preset)
        return data

    setattr(patched_build_multipart_data, _PATCH_ATTR, True)
    patched_build_multipart_data._dlightrag_original = original  # type: ignore[attr-defined]
    DoclingRawClient._build_multipart_data = patched_build_multipart_data
    logger.info("Applied LightRAG Docling code/formula preset forwarding: %s", preset)
    return True


__all__ = ["apply_docling_code_formula_preset"]
