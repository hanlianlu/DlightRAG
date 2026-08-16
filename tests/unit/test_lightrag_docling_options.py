# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for forwarding the Docling request options LightRAG omits."""

import sys
from types import ModuleType
from typing import Any

import pytest
from dlightrag_rag.ingestion.docling_options import apply_docling_request_options


def _client(
    *,
    do_formula_enrichment: bool,
    preset: str | None = "granite_docling",
) -> Any:
    from lightrag.parser.external.docling.client import DoclingRawClient

    apply_docling_request_options(code_formula_preset=preset)
    client = DoclingRawClient.__new__(DoclingRawClient)
    client.do_ocr = True
    client.force_ocr = False
    client.ocr_engine = "auto"
    client.ocr_preset = "auto"
    client.ocr_lang_raw = ""
    client.do_formula_enrichment = do_formula_enrichment
    return client


def test_configured_preset_sent_only_when_enrichment_is_on() -> None:
    on = _client(do_formula_enrichment=True)._build_multipart_data()
    off = _client(do_formula_enrichment=False)._build_multipart_data()

    assert on["code_formula_preset"] == "granite_docling"
    assert "code_formula_preset" not in off


def test_heading_hierarchy_sent_as_a_form_boolean() -> None:
    data = _client(do_formula_enrichment=False)._build_multipart_data()

    assert data["do_pdf_heading_hierarchy"] == "true"


def test_patch_adds_only_the_forwarded_fields_and_is_idempotent() -> None:
    from lightrag.parser.external.docling.client import DoclingRawClient

    client = _client(do_formula_enrichment=True)
    assert apply_docling_request_options(code_formula_preset="granite_docling") is False

    original = getattr(DoclingRawClient._build_multipart_data, "__wrapped__", None)
    assert original is not None, "Docling option patch did not install"

    patched = client._build_multipart_data()
    upstream = original(client)
    assert patched.keys() - upstream.keys() == {
        "code_formula_preset",
        "do_pdf_heading_hierarchy",
    }
    assert all(patched[key] == value for key, value in upstream.items())


def test_forwarded_options_join_the_bundle_cache_signature() -> None:
    from lightrag.parser.external.docling.client import FIXED_CONSTANTS

    apply_docling_request_options(code_formula_preset="granite_docling")

    assert FIXED_CONSTANTS["code_formula_preset"] == "granite_docling"
    assert FIXED_CONSTANTS["do_pdf_heading_hierarchy"] is True


def test_active_docling_fails_closed_when_upstream_contract_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incompatible_client = ModuleType("lightrag.parser.external.docling.client")
    monkeypatch.setitem(
        sys.modules,
        "lightrag.parser.external.docling.client",
        incompatible_client,
    )

    with pytest.raises(ImportError):
        apply_docling_request_options(code_formula_preset="granite_docling")


def test_do_formula_enrichment_defaults_on_like_mineru() -> None:
    from lightrag.parser.external.mineru.cache import DEFAULT_MINERU_ENABLE_FORMULA

    from dlightrag.config import DoclingSidecarConfig

    assert DoclingSidecarConfig().do_formula_enrichment is DEFAULT_MINERU_ENABLE_FORMULA


def test_do_formula_enrichment_reaches_lightrag_env() -> None:
    from dlightrag.config import DlightragConfig, DoclingSidecarConfig, ParserSidecarsConfig

    config = DlightragConfig(
        parser_sidecars=ParserSidecarsConfig(
            docling=DoclingSidecarConfig(do_formula_enrichment=True),
        ),
    )
    assert config._lightrag_sidecar_env_map()["DOCLING_DO_FORMULA_ENRICHMENT"] == "true"


def test_force_ocr_reaches_lightrag_env() -> None:
    from dlightrag.config import DlightragConfig, DoclingSidecarConfig, ParserSidecarsConfig

    config = DlightragConfig(
        parser_sidecars=ParserSidecarsConfig(
            docling=DoclingSidecarConfig(force_ocr=False),
        ),
    )
    assert config._lightrag_sidecar_env_map()["DOCLING_FORCE_OCR"] == "false"
