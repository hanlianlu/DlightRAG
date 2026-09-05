# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for request-local agent evidence (EvidenceLedger)."""

import asyncio
import threading

import pytest

from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.tools.web_search import web_context_rows
from dlightrag.engine.answer.web_sources import WebSearchHit


def _corpus_row(
    *, workspace: str = "alpha", chunk: str = "c1", content: str | None = None
) -> dict[str, object]:
    return {
        "chunk_id": chunk,
        "reference_id": "source-uuid",
        "full_doc_id": "doc-uuid",
        "file_path": "report.pdf",
        "content": content if content is not None else f"corpus {workspace} {chunk}",
        "_workspace": workspace,
        "metadata": {
            "source_type": "file",
            "source_uri": f"file:///{workspace}/report.pdf",
            "source_download_locator": f"file:///{workspace}/report.pdf",
        },
    }


def _web(text: str, *, resource_id: str | None = None) -> list[dict[str, object]]:
    rows = web_context_rows([WebSearchHit(url="https://example.com/a", title="Page A", text=text)])
    if resource_id is not None:
        rows[0]["metadata"]["resource_id"] = resource_id
    return rows


def test_sources_receive_stable_numeric_request_local_ids() -> None:
    ledger = EvidenceLedger()

    ledger.add_contexts({"chunks": [_corpus_row(chunk="c1")], "entities": [], "relationships": []})
    ledger.add_contexts({"chunks": [_corpus_row(chunk="c2")], "entities": [], "relationships": []})
    ledger.add_rows(_web("current passage"))

    rows = ledger.contexts["chunks"]
    assert [row["reference_id"] for row in rows] == ["1", "1", "2"]
    assert rows[0]["_source_reference_id"] == "source-uuid"
    assert rows[2]["_source_reference_id"].startswith("web-")


def test_same_page_same_passage_is_ignored_but_a_fresh_passage_survives() -> None:
    ledger = EvidenceLedger()

    first = ledger.add_rows(_web("first angle"))
    duplicate = ledger.add_rows(_web("first angle"))
    fresh = ledger.add_rows(_web("second angle"))

    assert first.new_chunks == 1
    assert duplicate.new_chunks == 0
    assert fresh.new_chunks == 1
    rows = ledger.contexts["chunks"]
    assert len(rows) == 2
    assert rows[0]["reference_id"] == rows[1]["reference_id"]
    assert rows[0]["chunk_id"] != rows[1]["chunk_id"]


def test_web_resource_cursor_does_not_create_new_evidence_for_the_same_window() -> None:
    ledger = EvidenceLedger()
    first = _web("same window\n[more text available; cursor=first]")[0]
    second = _web("same window\n[more text available; cursor=second]")[0]
    first["_evidence_key"] = second["_evidence_key"] = "lines 1-20"

    assert ledger.add_rows([first]).new_chunks == 1
    assert ledger.add_rows([second]).new_chunks == 0


def test_equal_upstream_ids_in_different_workspaces_are_distinct_sources() -> None:
    ledger = EvidenceLedger()

    ledger.add_rows([_corpus_row(workspace="alpha")])
    ledger.add_rows([_corpus_row(workspace="beta")])

    assert [row["reference_id"] for row in ledger.contexts["chunks"]] == ["1", "2"]


def test_non_chunk_context_is_deduplicated_without_losing_new_facts() -> None:
    ledger = EvidenceLedger()
    entity = {
        "entity_name": "Inflation",
        "entity_type": "concept",
        "description": "A sustained rise in prices.",
        "source_id": "c1",
        "_workspace": "alpha",
    }

    first = ledger.add_contexts(
        {"chunks": [_corpus_row()], "entities": [entity], "relationships": []}
    )
    second = ledger.add_contexts(
        {"chunks": [_corpus_row()], "entities": [entity], "relationships": []}
    )

    assert first.changed is True
    assert second.changed is False
    assert ledger.contexts["entities"] == [entity]


def test_rendering_labels_knowledge_base_and_open_web_separately() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row()])
    ledger.add_rows(_web("current passage"))

    blocks, _ = ledger.render_blocks()
    text = "\n".join(str(block["text"]) for block in blocks if block["type"] == "text")

    assert "## Knowledge-base evidence" in text
    assert "## Open-web evidence" in text
    assert "[1-1]" in text
    assert "[2-1]" in text
    assert text.index("Knowledge-base evidence") < text.index("corpus alpha c1")
    assert text.index("Open-web evidence") < text.index("current passage")


def test_rendering_keeps_a_web_source_resource_handle() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows(_web("current passage", resource_id="res-web-page"))

    blocks, _ = ledger.render_blocks()
    text = "\n".join(str(block["text"]) for block in blocks if block["type"] == "text")

    assert "resource id: res-web-page" in text


def test_images_are_never_rendered_without_an_explicit_transport_budget() -> None:
    row = _corpus_row()
    row["image_data"] = "raw-unbounded-payload"
    ledger = EvidenceLedger()
    ledger.add_rows([row])

    blocks, _ = ledger.render_blocks()

    assert all(block["type"] != "image_url" for block in blocks)


async def test_evidence_images_consume_the_single_supplied_budget_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )
    row = _corpus_row()
    row["image_data"] = png
    ledger = EvidenceLedger(image_budget=budget)
    loop_thread = threading.get_ident()
    budget_threads: list[int] = []
    add_base64 = budget.add_base64

    def capture_budget(value: str, *, label: str):
        budget_threads.append(threading.get_ident())
        return add_base64(value, label=label)

    budget.add_base64 = capture_budget  # type: ignore[method-assign]

    ledger.add_rows([row])
    await ledger.aflush_images()
    first, _ = ledger.render_blocks()
    second, _ = ledger.render_blocks()

    assert len([block for block in first if block["type"] == "image_url"]) == 1
    assert len([block for block in second if block["type"] == "image_url"]) == 1
    assert budget.count == 1
    assert budget_threads
    assert all(thread_id != loop_thread for thread_id in budget_threads)


async def test_failed_evidence_image_worker_restores_pending_rows() -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )
    add_base64 = budget.add_base64
    first = True

    def fail_once(value: str, *, label: str):
        nonlocal first
        if first:
            first = False
            raise RuntimeError("worker failed")
        return add_base64(value, label=label)

    budget.add_base64 = fail_once  # type: ignore[method-assign]
    row = _corpus_row()
    row["image_data"] = png
    ledger = EvidenceLedger(image_budget=budget)
    ledger.add_rows([row])

    with pytest.raises(RuntimeError, match="worker failed"):
        await ledger.aflush_images()
    await ledger.aflush_images()
    blocks, _ = ledger.render_blocks()

    assert len([block for block in blocks if block["type"] == "image_url"]) == 1


async def test_cancelled_evidence_flush_restores_rows_before_join() -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    budget = AnswerImageBudget(
        max_images=1,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )
    started = threading.Event()
    release = threading.Event()
    add_base64 = budget.add_base64

    def blocked_budget(value: str, *, label: str):
        started.set()
        release.wait()
        return add_base64(value, label=label)

    budget.add_base64 = blocked_budget  # type: ignore[method-assign]
    row = _corpus_row()
    row["image_data"] = png
    ledger = EvidenceLedger(image_budget=budget)
    ledger.add_rows([row])
    flush = asyncio.create_task(ledger.aflush_images())
    await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=1)

    try:
        flush.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await flush
        await ledger.aflush_images()
        blocks, _ = ledger.render_blocks()

        assert len([block for block in blocks if block["type"] == "image_url"]) == 1
        assert budget.count == 1
    finally:
        release.set()
        await asyncio.gather(flush, return_exceptions=True)


def test_transform_keeps_recent_evidence_and_collapses_older_to_handles() -> None:
    ledger = EvidenceLedger()
    # Oldest observation: large enough to exceed a tight ceiling on its own.
    ledger.add_rows([_corpus_row(chunk="old", content="OLD-EVIDENCE " + ("filler " * 200))])
    # Recent observation: small and must be retained verbatim.
    ledger.add_rows([_corpus_row(workspace="beta", chunk="new", content="RECENT-EVIDENCE key")])

    # A residual window that only fits the small recent observation.
    blocks, indexer = ledger.transform(residual_tokens=60)

    text = "\n".join(str(b["text"]) for b in blocks if b.get("type") == "text")
    assert "RECENT-EVIDENCE key" in text
    assert "OLD-EVIDENCE" not in text
    # The collapsed older source remains a re-readable handle preserving its id.
    assert "Retained evidence (re-read for detail)" in text
    assert "[1]" in text
    # Citation identity for the collapsed source still resolves.
    assert indexer.get_max_chunk_idx("1") > 0


def test_transform_does_not_guess_an_image_token_cost() -> None:
    ledger = EvidenceLedger()
    visual = _corpus_row(chunk="visual", content="")
    visual["image_data"] = "AAAA"
    ledger.add_rows([visual])

    blocks, indexer = ledger.transform(residual_tokens=0)

    text = "\n".join(str(block["text"]) for block in blocks if block["type"] == "text")
    assert "Retained evidence (re-read for detail)" not in text
    assert indexer.get_chunk_id("1", 1) == "visual"


def test_transform_keeps_a_collapsed_web_resource_re_readable() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows(
        _web(
            "OLD-WEB-EVIDENCE " + ("filler " * 200),
            resource_id="res-web-page",
        )
    )
    ledger.add_rows([_corpus_row(workspace="beta", chunk="new", content="RECENT")])

    blocks, _ = ledger.transform(residual_tokens=60)

    text = "\n".join(str(block["text"]) for block in blocks if block["type"] == "text")
    assert "OLD-WEB-EVIDENCE" not in text
    assert "[resource: res-web-page]" in text


def test_transform_preserves_stable_citation_ids_across_full_render() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row(chunk="c1", content="alpha evidence")])
    ledger.add_rows(_web("web evidence"))

    blocks, indexer = ledger.transform(residual_tokens=1_000_000)

    text = "\n".join(str(b["text"]) for b in blocks if b.get("type") == "text")
    # Nothing collapses when the whole window is available.
    assert "Retained evidence (re-read for detail)" not in text
    assert "alpha evidence" in text
    assert "web evidence" in text
    assert "[1-1]" in text
    assert "[2-1]" in text


def test_empty_ledger_state_is_empty_object() -> None:
    assert EvidenceLedger().ledger_state_json() == "{}"


def test_child_evidence_adoption_is_citable_idempotent_and_records_lineage() -> None:
    child = EvidenceLedger()
    child.add_rows([_corpus_row(chunk="child-c1", content="child finding")])
    parent = EvidenceLedger()

    first = parent.merge_child_state(
        child.durable_state(),
        child_session_id="child-session",
        parent_call_id="spawn-call",
    )
    second = parent.merge_child_state(
        child.durable_state(),
        child_session_id="child-session",
        parent_call_id="spawn-call",
    )

    assert first.new_chunks == 1
    assert second.new_chunks == 0
    row = parent.contexts["chunks"][0]
    assert row["metadata"]["child_session_id"] == "child-session"
    assert row["metadata"]["parent_call_id"] == "spawn-call"
    assert parent.render_blocks()[1].get_chunk_id("1", 1) == "child-c1"


def test_ledger_state_round_trips_identities_without_image_bytes() -> None:
    import json

    source = EvidenceLedger()
    source.add_rows([_corpus_row(chunk="c1", content="keep me")])
    source.add_rows(_web("web keep", resource_id="res-a"))
    source.contexts["chunks"][0]["image_data"] = "AAAA"

    payload = json.loads(source.ledger_state_json())
    assert "image_data" not in payload["contexts"]["chunks"][0]
    restored = EvidenceLedger()
    restored.restore_ledger_state(payload)
    assert [row["content"] for row in restored.contexts["chunks"]] == [
        row["content"] for row in source.contexts["chunks"]
    ]
    assert restored.citation_handles() == [
        "[1] report.pdf",
        "[2] Page A [resource: res-a]",
    ]
