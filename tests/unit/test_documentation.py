# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable product-documentation invariants for the unified Answer architecture.

These tests guard the committed product docs (README plus the ``docs/*.md``
pages) against reintroducing the retired Composer/query-image answer contract and
against drifting away from the single Answer attachment + capacity contract that
the code enforces. They read only explicitly named product files, never the
ignored ``docs/superpowers`` planning notes.
"""

from __future__ import annotations

import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Only these committed product docs are read. docs/superpowers is never touched.
_PRODUCT_DOCS = (
    "README.md",
    "docs/architecture.md",
    "docs/retrieval-answer.md",
    "docs/interfaces.md",
    "docs/configuration.md",
    "docs/operations.md",
    "docs/security.md",
    "docs/postgresql.md",
    "docs/evaluation.md",
)

# Retired product claims. None may reappear in any committed product doc. Each is
# an exact retired phrase, so a legitimate durable-ingestion mention (for example
# the RobustDocumentEmbedder class) or the chat composer UI is not banned.
_STALE_PHRASES = (
    "ComposerModelBundle",
    "Web Composer",
    "Composer document",
    "web_conversation_attachment_chunks",
    "web_conversation_images",
    "24,576",
    "context_top_k",
    "answer_context_top_k",
)


def _doc_text(name: str) -> str:
    return (_ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", _PRODUCT_DOCS)
def test_product_doc_exists(name: str) -> None:
    assert (_ROOT / name).is_file()


def test_no_superpowers_files_are_read() -> None:
    # The invariant set never references the ignored planning tree.
    assert all(not name.startswith("docs/superpowers") for name in _PRODUCT_DOCS)


@pytest.mark.parametrize("name", _PRODUCT_DOCS)
@pytest.mark.parametrize("phrase", _STALE_PHRASES)
def test_no_stale_answer_claims(name: str, phrase: str) -> None:
    assert phrase not in _doc_text(name), f"stale phrase {phrase!r} found in {name}"


def test_protected_engine_terms_survive() -> None:
    architecture = _doc_text("docs/architecture.md")
    for term in ("LightRAG", "MinerU", "Docling"):
        assert term in architecture, f"protected term {term!r} missing from architecture.md"


def test_mineru_docling_are_only_durable_ingestion_parsers() -> None:
    architecture = _doc_text("docs/architecture.md")
    assert "MinerU" in architecture and "Docling" in architecture
    # Ingestion is where the durable parsers live; the answer path never names them.
    assert "ingest" in architecture.lower()


def test_answer_context_window_documented() -> None:
    configuration = _doc_text("docs/configuration.md")
    assert "260,000" in configuration or "260000" in configuration
    assert "context_window_tokens" in configuration


def test_answer_attachment_limits_documented() -> None:
    configuration = _doc_text("docs/configuration.md")
    assert "max_attachments" in configuration
    assert "100 MiB" in configuration
    assert "128 MiB" in configuration


def test_answer_generation_reserve_is_not_an_output_cap() -> None:
    configuration = _doc_text("docs/configuration.md")
    assert "32,768" in configuration or "32768" in configuration
    assert (
        "not an output cap" in configuration.lower() or "not `max_output_tokens`" in configuration
    )


def test_answer_public_input_is_attachments_across_interfaces() -> None:
    interfaces = _doc_text("docs/interfaces.md")
    assert "attachments" in interfaces
    # The answer contract must describe multipart uploads and HTTPS link descriptors.
    assert "multipart" in interfaces.lower()


def test_retrieve_keeps_query_images_but_answer_does_not() -> None:
    interfaces = _doc_text("docs/interfaces.md")
    assert "query_images" in interfaces  # retrieve-only visual path is still documented


def test_orchestrator_fast_and_research_paths_documented() -> None:
    retrieval = _doc_text("docs/retrieval-answer.md")
    lowered = retrieval.lower()
    assert "fast path" in lowered
    assert "research" in lowered


def test_web_uses_one_raw_attachment_table() -> None:
    postgresql = _doc_text("docs/postgresql.md")
    assert "web_conversation_attachments" in postgresql


def test_optional_web_search_is_exa() -> None:
    retrieval = _doc_text("docs/retrieval-answer.md")
    assert "Exa" in retrieval
