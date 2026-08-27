# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime-gated LightRAG compatibility patches.

MinerU parser hygiene: current LightRAG MinerU IR builder serializes unknown
content-list item types as body text. Two consequences are patched at the parser
boundary: (1) figure blocks MinerU emits as ``chart`` are aliased to ``image``
so they reach the drawing/VLM sidecar path instead of being dropped;
(2) headers and footers are removed so they do not pollute chunks, KG
extraction, BM25, and citations. LightRAG already removes printed page-number
items.

Docling request options: LightRAG's Docling client builds a fixed multipart
body, so the code/formula preset and PDF heading-level inference are unreachable
no matter how the parser service is configured. The patch forwards both and is
installed only when Docling is the active parser.

Keep this module small and delete patches as upstream covers them.
"""

import logging

logger = logging.getLogger(__name__)


def apply(
    *,
    docling_active: bool = False,
    docling_code_formula_preset: str | None = None,
) -> None:
    """Apply all LightRAG patches. Idempotent."""
    applied = []
    if docling_active:
        from dlightrag.engine.rag.corpus.ingestion.docling_options import (
            apply_docling_request_options,
        )

        if apply_docling_request_options(
            code_formula_preset=docling_code_formula_preset,
        ):
            applied.append("docling_request_options")
    else:
        from dlightrag.engine.rag.corpus.ingestion.parser_hygiene import (
            apply_mineru_content_list_hygiene,
        )

        if apply_mineru_content_list_hygiene():
            applied.append("mineru_content_list_hygiene")
    if applied:
        logger.info("Applied LightRAG patches: %s", ", ".join(applied))
    else:
        logger.debug("LightRAG patches already installed or covered upstream")
