# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for active-parser LightRAG patch selection."""

import pytest

from dlightrag.engine.rag.lightrag import patches as _lightrag_patches


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    installed: list[str] = []
    monkeypatch.setattr(
        "dlightrag.engine.rag.corpus.ingestion.parser_hygiene.apply_mineru_content_list_hygiene",
        lambda: installed.append("mineru") or True,
    )
    monkeypatch.setattr(
        "dlightrag.engine.rag.corpus.ingestion.docling_options.apply_docling_request_options",
        lambda **_kwargs: installed.append("docling") or True,
    )
    monkeypatch.setattr(
        "dlightrag.engine.rag.corpus.ingestion.parser_transport.apply_parser_outage_reporting",
        lambda: installed.append("outage") or True,
    )
    return installed


def test_docling_mode_does_not_install_the_mineru_patch(installed: list[str]) -> None:
    _lightrag_patches.apply(docling_active=True)

    assert installed == ["docling", "outage"]


def test_either_mode_reports_parser_outages(installed: list[str]) -> None:
    _lightrag_patches.apply(docling_active=False)

    assert installed == ["mineru", "outage"]
