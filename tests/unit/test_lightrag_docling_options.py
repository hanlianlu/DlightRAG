# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for forwarding the Docling code/formula preset to docling-serve."""

from typing import Any

from dlightrag.core.ingestion.docling_options import apply_docling_code_formula_preset


def _client(*, do_formula_enrichment: bool) -> Any:
    from lightrag.parser.external.docling.client import DoclingRawClient

    apply_docling_code_formula_preset()
    client = DoclingRawClient.__new__(DoclingRawClient)
    client.do_ocr = True
    client.force_ocr = False
    client.ocr_engine = "auto"
    client.ocr_preset = "auto"
    client.ocr_lang_raw = ""
    client.do_formula_enrichment = do_formula_enrichment
    return client


def test_preset_alias_sent_only_when_enrichment_is_on() -> None:
    on = _client(do_formula_enrichment=True)._build_multipart_data()
    off = _client(do_formula_enrichment=False)._build_multipart_data()

    assert on["code_formula_preset"] == "default"
    assert "code_formula_preset" not in off


def test_patch_adds_one_field_and_is_idempotent() -> None:
    from lightrag.parser.external.docling.client import DoclingRawClient

    client = _client(do_formula_enrichment=True)
    assert apply_docling_code_formula_preset() is False

    original = getattr(DoclingRawClient._build_multipart_data, "_dlightrag_original", None)
    assert original is not None, "Docling preset patch did not install"

    patched = client._build_multipart_data()
    upstream = original(client)
    assert patched.keys() - upstream.keys() == {"code_formula_preset"}
    assert all(patched[key] == value for key, value in upstream.items())


def test_do_formula_enrichment_reaches_lightrag_env() -> None:
    from dlightrag.config import DlightragConfig, DoclingSidecarConfig, ParserSidecarsConfig

    config = DlightragConfig(
        parser_sidecars=ParserSidecarsConfig(
            docling=DoclingSidecarConfig(do_formula_enrichment=True),
        ),
    )
    assert config._lightrag_sidecar_env_map()["DOCLING_DO_FORMULA_ENRICHMENT"] == "true"
