# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Ownership contract for storage-neutral RAG retrieval results."""

from dataclasses import fields

from dlightrag.rag.retrieval import RetrievalResult


def test_rag_result_carries_no_answer_projection_fields() -> None:
    assert {field.name for field in fields(RetrievalResult)} == {
        "contexts",
        "references",
        "image_descriptions",
        "trace",
    }
