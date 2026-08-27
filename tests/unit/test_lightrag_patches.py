# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for active-parser LightRAG patch selection."""

import pytest

from dlightrag.engine.rag.lightrag import patches as _lightrag_patches


def test_docling_mode_does_not_install_the_mineru_patch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed: list[str] = []
    monkeypatch.setattr(
        "dlightrag.engine.rag.corpus.ingestion.parser_hygiene.apply_mineru_content_list_hygiene",
        lambda: installed.append("mineru") or True,
    )
    monkeypatch.setattr(
        "dlightrag.engine.rag.corpus.ingestion.docling_options.apply_docling_request_options",
        lambda **_kwargs: installed.append("docling") or True,
    )

    _lightrag_patches.apply(docling_active=True)

    assert installed == ["docling"]
