# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dlightrag.models.schemas import Reference, StructuredAnswer

# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


class TestReference:
    def test_valid_reference(self) -> None:
        ref = Reference(id=1, title="report.pdf")
        assert ref.id == 1
        assert ref.title == "report.pdf"

    def test_json_roundtrip(self) -> None:
        ref = Reference(id=2, title="spec.pdf")
        data = ref.model_dump()
        assert data == {"id": 2, "title": "spec.pdf"}


# ---------------------------------------------------------------------------
# StructuredAnswer
# ---------------------------------------------------------------------------


class TestStructuredAnswer:
    def test_valid_structured_answer(self) -> None:
        sa = StructuredAnswer(
            answer="The findings are... [1-1]",
            references=[Reference(id=1, title="report.pdf")],
        )
        assert sa.answer == "The findings are... [1-1]"
        assert len(sa.references) == 1

    def test_empty_references_default(self) -> None:
        sa = StructuredAnswer(answer="No refs")
        assert sa.references == []

    def test_parse_from_json(self) -> None:
        raw = '{"answer": "text [1-1]", "references": [{"id": 1, "title": "doc.pdf"}]}'
        sa = StructuredAnswer.model_validate_json(raw)
        assert sa.answer == "text [1-1]"
        assert sa.references[0].title == "doc.pdf"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValidationError):
            StructuredAnswer.model_validate_json('{"answer": 123}')

    def test_json_schema_generation(self) -> None:
        schema = StructuredAnswer.model_json_schema()
        assert "answer" in schema["properties"]
        assert "references" in schema["properties"]
