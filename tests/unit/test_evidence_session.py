# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for request-local agent evidence."""

from dlightrag.core.agent.evidence import EvidenceSession
from dlightrag.core.retrieval.web_search import WebSearchHit, web_context_rows


def _corpus_row(*, workspace: str = "alpha", chunk: str = "c1") -> dict[str, object]:
    return {
        "chunk_id": chunk,
        "reference_id": "source-uuid",
        "full_doc_id": "doc-uuid",
        "file_path": "report.pdf",
        "content": f"corpus {workspace} {chunk}",
        "_workspace": workspace,
        "metadata": {
            "source_type": "file",
            "source_uri": f"file:///{workspace}/report.pdf",
            "source_download_locator": f"file:///{workspace}/report.pdf",
        },
    }


def _web(text: str) -> list[dict[str, object]]:
    return web_context_rows([WebSearchHit(url="https://example.com/a", title="Page A", text=text)])


def test_sources_receive_stable_numeric_request_local_ids() -> None:
    session = EvidenceSession()

    session.add_contexts({"chunks": [_corpus_row(chunk="c1")], "entities": [], "relationships": []})
    session.add_contexts({"chunks": [_corpus_row(chunk="c2")], "entities": [], "relationships": []})
    session.add_rows(_web("current passage"))

    rows = session.contexts["chunks"]
    assert [row["reference_id"] for row in rows] == ["1", "1", "2"]
    assert rows[0]["_source_reference_id"] == "source-uuid"
    assert rows[2]["_source_reference_id"].startswith("web-")


def test_same_page_same_passage_is_ignored_but_a_fresh_passage_survives() -> None:
    session = EvidenceSession()

    first = session.add_rows(_web("first angle"))
    duplicate = session.add_rows(_web("first angle"))
    fresh = session.add_rows(_web("second angle"))

    assert first.new_chunks == 1
    assert duplicate.new_chunks == 0
    assert fresh.new_chunks == 1
    rows = session.contexts["chunks"]
    assert len(rows) == 2
    assert rows[0]["reference_id"] == rows[1]["reference_id"]
    assert rows[0]["chunk_id"] != rows[1]["chunk_id"]


def test_equal_upstream_ids_in_different_workspaces_are_distinct_sources() -> None:
    session = EvidenceSession()

    session.add_rows([_corpus_row(workspace="alpha")])
    session.add_rows([_corpus_row(workspace="beta")])

    assert [row["reference_id"] for row in session.contexts["chunks"]] == ["1", "2"]


def test_non_chunk_context_is_deduplicated_without_losing_new_facts() -> None:
    session = EvidenceSession()
    entity = {
        "entity_name": "Inflation",
        "entity_type": "concept",
        "description": "A sustained rise in prices.",
        "source_id": "c1",
        "_workspace": "alpha",
    }

    first = session.add_contexts(
        {"chunks": [_corpus_row()], "entities": [entity], "relationships": []}
    )
    second = session.add_contexts(
        {"chunks": [_corpus_row()], "entities": [entity], "relationships": []}
    )

    assert first.changed is True
    assert second.changed is False
    assert session.contexts["entities"] == [entity]


def test_rendering_labels_knowledge_base_and_open_web_separately() -> None:
    session = EvidenceSession()
    session.add_rows([_corpus_row()])
    session.add_rows(_web("current passage"))

    blocks, _ = session.render_blocks()
    text = "\n".join(str(block["text"]) for block in blocks if block["type"] == "text")

    assert "## Knowledge-base evidence" in text
    assert "## Open-web evidence" in text
    assert "[1-1]" in text
    assert "[2-1]" in text
    assert text.index("Knowledge-base evidence") < text.index("corpus alpha c1")
    assert text.index("Open-web evidence") < text.index("current passage")
