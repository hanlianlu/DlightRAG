# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pin the private LightRAG parser-owner and sidecar-builder contracts.

The Composer document parse-owner shim reuses LightRAG-internal parser hooks.
These tests fail loudly if a future LightRAG upgrade renames or reshapes one
of the private methods the shim depends on, so the coupling is never silent.
"""

import inspect

from lightrag import LightRAG

from dlightrag.core.request.attachments import _ParseOwnerShim


def test_lightrag_exposes_parser_owner_contract() -> None:
    resolver = LightRAG._resolve_source_file_for_parser
    params = inspect.signature(resolver).parameters
    assert "source_file" in params
    assert "parser_engine" in params
    assert hasattr(LightRAG, "_persist_parsed_full_docs")

    persist = LightRAG._persist_parsed_full_docs
    persist_params = inspect.signature(persist).parameters
    assert "doc_id" in persist_params
    assert "record" in persist_params
    assert inspect.iscoroutinefunction(persist)


def test_lightrag_exposes_multimodal_sidecar_builder_contract() -> None:
    method = LightRAG._build_mm_chunks_from_sidecars
    params = inspect.signature(method).parameters
    for name in ("doc_id", "file_path", "blocks_path", "base_order_index", "process_options"):
        assert name in params


def test_lightrag_exposes_multimodal_analyzer_contract() -> None:
    method = LightRAG.analyze_multimodal
    params = inspect.signature(method).parameters
    assert inspect.iscoroutinefunction(method)
    assert {
        "doc_id",
        "file_path",
        "parsed_data",
        "process_options",
        "pipeline_status",
        "pipeline_status_lock",
    } <= set(params)


def test_shim_satisfies_owner_contract() -> None:
    shim = _ParseOwnerShim()
    assert shim._resolve_source_file_for_parser("/tmp/x.pdf") == "/tmp/x.pdf"
    assert hasattr(shim, "_persist_parsed_full_docs")
    assert inspect.iscoroutinefunction(shim._persist_parsed_full_docs)

    # The shim's resolver signature must remain call-compatible with the
    # LightRAG source-file resolver dispatch (source_file / parser_engine kw).
    shim_params = inspect.signature(shim._resolve_source_file_for_parser).parameters
    assert "source_file" in shim_params
    assert "parser_engine" in shim_params
