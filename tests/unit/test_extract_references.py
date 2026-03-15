# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for extract_references() 3-level fallback."""

from __future__ import annotations

from dlightrag.citations.parser import extract_references


class TestLevel1JsonBlock:
    def test_full_structured_json(self) -> None:
        """Full structured JSON — extract answer and refs."""
        raw = '{"answer": "Growth is 15% [1-1].", "references": [{"id": 1, "title": "report.pdf"}]}'
        answer, refs = extract_references(raw)
        assert answer == "Growth is 15% [1-1]."
        assert len(refs) == 1
        assert refs[0].title == "report.pdf"

    def test_freetext_then_json_block(self) -> None:
        """Freetext followed by JSON block — strip JSON, extract refs."""
        raw = (
            "Based on the analysis, growth is 15% [1-1].\n\n"
            '{"answer": "Growth is 15% [1-1].", '
            '"references": [{"id": 1, "title": "report.pdf"}]}'
        )
        answer, refs = extract_references(raw)
        assert "Based on the analysis" in answer
        assert '{"answer"' not in answer
        assert len(refs) == 1

    def test_json_with_code_fence(self) -> None:
        """JSON wrapped in ```json fence."""
        raw = '```json\n{"answer": "Hello.", "references": [{"id": 1, "title": "doc.pdf"}]}\n```'
        answer, refs = extract_references(raw)
        assert answer == "Hello."
        assert len(refs) == 1

    def test_json_no_answer_key(self) -> None:
        """JSON block has only references, no answer key — strip JSON from raw."""
        raw = (
            "The analysis shows growth.\n\n"
            '{"references": [{"id": 1, "title": "report.pdf"}]}'
        )
        answer, refs = extract_references(raw)
        assert "The analysis shows growth." in answer
        assert '{"references"' not in answer
        assert len(refs) == 1

    def test_code_fenced_json_no_answer_key(self) -> None:
        """Code-fenced JSON without answer key — strip fence and JSON."""
        raw = (
            "The analysis shows growth.\n\n"
            '```json\n{"references": [{"id": 1, "title": "report.pdf"}]}\n```'
        )
        answer, refs = extract_references(raw)
        assert "```" not in answer
        assert "The analysis shows growth." in answer
        assert len(refs) == 1

    def test_malformed_references_array(self) -> None:
        """JSON exists but references is not a list — use answer from JSON."""
        raw = '{"answer": "Text.", "references": "not-a-list"}'
        answer, refs = extract_references(raw)
        assert answer == "Text."
        assert refs == []

    def test_json_with_empty_references(self) -> None:
        """JSON with empty references array — use answer from JSON."""
        raw = '{"answer": "No refs.", "references": []}'
        answer, refs = extract_references(raw)
        assert answer == "No refs."
        assert refs == []

    def test_json_with_invalid_ref_entries(self) -> None:
        """Some ref entries are invalid — skip bad ones, keep good ones."""
        raw = '{"answer": "Text.", "references": [{"id": 1, "title": "a.pdf"}, {"bad": true}]}'
        answer, refs = extract_references(raw)
        assert len(refs) == 1
        assert refs[0].title == "a.pdf"


class TestLevel2ReferencesSection:
    def test_standard_references_section(self) -> None:
        """Standard ### References section at end of text."""
        raw = (
            "Growth is 15% [1].\n\n"
            "### References\n"
            "- [1] report.pdf\n"
            "- [2] summary.docx"
        )
        answer, refs = extract_references(raw)
        assert answer == "Growth is 15% [1]."
        assert len(refs) == 2
        assert refs[0].title == "report.pdf"
        assert refs[1].title == "summary.docx"

    def test_references_heading_case_insensitive(self) -> None:
        """Heading case variants work."""
        raw = "Answer text.\n\n## references\n[1] doc.pdf"
        answer, refs = extract_references(raw)
        assert len(refs) == 1


class TestLevel3Empty:
    def test_plain_text_no_refs(self) -> None:
        """Pure text without any refs structure — return as-is."""
        raw = "This is a plain answer with no references."
        answer, refs = extract_references(raw)
        assert answer == raw
        assert refs == []

    def test_empty_string(self) -> None:
        raw = ""
        answer, refs = extract_references(raw)
        assert answer == ""
        assert refs == []
