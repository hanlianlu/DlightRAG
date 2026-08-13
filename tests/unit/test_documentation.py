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
    "web_conversation_images",
    "24,576",
    "context_top_k",
    "answer_context_top_k",
    "web_conversation_attachments",
    # Superseded by the durable Answer run contract.
    "answer_acquire_timeout",
    "read replica",
    "read-replica",
    "streaming replication",
    "hot standby",
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


def test_web_turns_link_to_durable_runs_instead_of_copying_answers() -> None:
    postgresql = _doc_text("docs/postgresql.md")
    assert "dlightrag_answer_runs" in postgresql
    assert "dlightrag_answer_run_events" in postgresql
    assert "dlightrag_answer_artifacts" in postgresql
    assert "dlightrag_answer_run_artifacts" in postgresql
    assert "answer_run_id" in postgresql


def test_durable_answer_run_contract_is_documented() -> None:
    interfaces = _doc_text("docs/interfaces.md")
    for term in ("202", "Last-Event-ID", "Idempotency-Key", "410", "run_id"):
        assert term in interfaces, f"durable run term {term!r} missing from interfaces.md"
    # The ephemeral answer mode and its request field are gone.
    assert "stream: true" not in interfaces


def test_reader_role_is_corpus_read_only_not_process_read_only() -> None:
    postgresql = _doc_text("docs/postgresql.md")
    assert "corpus-read-only" in postgresql
    assert "same primary endpoint" in postgresql


def test_multi_host_requires_one_shared_posix_working_dir() -> None:
    postgresql = _doc_text("docs/postgresql.md")
    assert "same absolute `working_dir` path" in postgresql


def test_ingress_owns_rate_limiting_and_the_application_does_not() -> None:
    security = _doc_text("docs/security.md")
    assert "ships no in-process rate limiter" in security
    assert "not the inline WAF" in security


def test_error_contract_names_its_stable_kinds() -> None:
    interfaces = _doc_text("docs/interfaces.md")
    for kind in ("invalid_tool_configuration", "checkpoint_corrupt", "run_abandoned"):
        assert kind in interfaces, f"error kind {kind!r} missing from interfaces.md"
    assert "`configuration`" in interfaces


def test_optional_web_search_is_exa() -> None:
    retrieval = _doc_text("docs/retrieval-answer.md")
    assert "Exa" in retrieval
