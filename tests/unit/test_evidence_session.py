# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for request-local agent evidence (EvidenceLedger)."""

import asyncio
import threading
from typing import cast

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


def test_admitted_evidence_keeps_recent_rows_and_collapses_older_to_handles() -> None:
    ledger = EvidenceLedger()
    # Oldest observation: large enough to exceed a tight ceiling on its own.
    ledger.add_rows([_corpus_row(chunk="old", content="OLD-EVIDENCE " + ("filler " * 200))])
    # Recent observation: small and must be retained verbatim.
    ledger.add_rows([_corpus_row(workspace="beta", chunk="new", content="RECENT-EVIDENCE key")])

    # A window that only fits the small recent observation. Both rows were
    # admitted inside one batch here, so both are offered to the one render.
    _labels, text = ledger.take_admitted_text(budget_tokens=60)

    assert "RECENT-EVIDENCE key" in text
    assert "OLD-EVIDENCE" not in text
    # The collapsed older source remains a re-readable handle preserving its id.
    assert "Retained evidence (re-read for detail)" in text
    assert "[1]" in text
    # A second take has nothing left: the transcript already carries this text.
    assert ledger.take_admitted_text(budget_tokens=60) == ("", "")


def test_admitted_evidence_does_not_guess_an_image_token_cost() -> None:
    ledger = EvidenceLedger()
    visual = _corpus_row(chunk="visual", content="")
    visual["image_data"] = "AAAA"
    ledger.add_rows([visual])

    # A zero-token window still renders nothing for a body-less visual row: the
    # text estimator charges no tokens for pixels, and the row's pixels ride in
    # the run-local visual lane instead of the transcript.
    assert ledger.take_admitted_text(budget_tokens=0) == ("", "")
    assert ledger.visual_blocks() == []


def test_admitted_evidence_keeps_a_collapsed_web_resource_re_readable() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows(
        _web(
            "OLD-WEB-EVIDENCE " + ("filler " * 200),
            resource_id="res-web-page",
        )
    )
    ledger.add_rows([_corpus_row(workspace="beta", chunk="new", content="RECENT")])

    _labels, text = ledger.take_admitted_text(budget_tokens=60)

    assert "OLD-WEB-EVIDENCE" not in text
    assert "[resource: res-web-page]" in text


def test_admitted_evidence_preserves_stable_citation_ids_across_batches() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row(chunk="c1", content="alpha evidence")])
    _first_labels, first = ledger.take_admitted_text(budget_tokens=1_000_000)
    ledger.add_rows(_web("web evidence"))
    _second_labels, second = ledger.take_admitted_text(budget_tokens=1_000_000)

    # Nothing collapses when the whole window is available, and the batch that
    # arrived later keeps numbering from the ledger rather than restarting.
    assert "Retained evidence (re-read for detail)" not in first + second
    assert "alpha evidence" in first
    assert "web evidence" in second
    assert "alpha evidence" not in second
    assert "[1-1]" in first
    assert "[2-1]" in second


def test_admitted_evidence_renders_only_what_the_previous_take_left() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row(chunk="c1", content="alpha evidence")])
    ledger.take_admitted_text(budget_tokens=1_000_000)

    assert ledger.take_admitted_text(budget_tokens=1_000_000) == ("", "")


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


def test_unrepresentable_web_passage_is_dropped_and_counted() -> None:
    ledger = EvidenceLedger()

    delta = ledger.add_rows(_web("poison\x00passage"))

    assert delta.new_chunks == 0
    assert delta.changed is False
    assert delta.dropped_rows == 1
    assert ledger.contexts["chunks"] == []


def test_lone_surrogate_passage_is_dropped() -> None:
    ledger = EvidenceLedger()

    delta = ledger.add_rows(_web("bad\ud800passage"))

    assert delta.dropped_rows == 1
    assert ledger.contexts["chunks"] == []


def test_unrepresentable_corpus_and_graph_rows_are_dropped() -> None:
    ledger = EvidenceLedger()

    delta = ledger.add_contexts(
        {
            "chunks": [_corpus_row(chunk="c1", content="bad\x00corpus")],
            "entities": [{"entity_name": "Volvo\x00Cars", "description": "layoffs"}],
            "relationships": [],
        }
    )

    assert (delta.new_chunks, delta.new_entities, delta.dropped_rows) == (0, 0, 2)
    assert ledger.contexts["chunks"] == []
    assert ledger.contexts["entities"] == []


def test_checkable_content_keeps_admitting_after_a_dropped_passage() -> None:
    ledger = EvidenceLedger()

    dropped = ledger.add_rows(_web("poison\x00passage"))
    kept = ledger.add_rows(_web("clean passage"))

    assert dropped.dropped_rows == 1
    assert kept.new_chunks == 1
    assert [row["content"] for row in ledger.contexts["chunks"]] == ["clean passage"]


def test_a_body_the_tool_already_carried_is_labelled_not_repeated() -> None:
    read_body = "Read window " + ("quoted page text " * 8)
    carried = _corpus_row(chunk="read-1", content=read_body)
    # What `read`/`view` declare at admission: this Tool's own result already
    # carries the row's body (its text, or its pixels).
    carried["_carried_by_tool"] = True
    ledger = EvidenceLedger()
    ledger.add_rows(
        [
            carried,
            _corpus_row(chunk="search-1", content="passage only the ledger carries"),
        ]
    )

    labels, rendered = ledger.take_admitted_text(budget_tokens=1_000_000)

    # The label is what the Citation Contract asks the model to reuse; the body is
    # not carried twice in one request.
    assert labels == "[1-1] report.pdf"
    assert read_body not in rendered
    assert "passage only the ledger carries" in rendered
    assert "[1-2] report.pdf" in rendered


def test_visual_evidence_keeps_its_own_lane_with_a_citation_label() -> None:
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    budget = AnswerImageBudget(
        max_images=4,
        max_total_bytes=10_000,
        max_bytes_per_image=10_000,
        max_pixels=40_000_000,
        max_px=64,
        min_px=32,
        quality=85,
        min_quality=72,
    )
    ledger = EvidenceLedger(image_budget=budget)
    visual = _corpus_row(chunk="p1", content="caption for the page")
    visual["image_data"] = png
    visual["page_number"] = 7
    ledger.add_rows([visual])
    asyncio.run(ledger.aflush_images())

    blocks = ledger.visual_blocks()

    assert [block["type"] for block in blocks] == ["text", "image_url"]
    assert blocks[0]["text"].startswith("[1-1]")
    assert "Page 7" in blocks[0]["text"]


def test_visual_evidence_renders_nothing_without_a_budgeted_image() -> None:
    ledger = EvidenceLedger()
    visual = _corpus_row(chunk="p1", content="caption")
    visual["image_data"] = "AAAA"
    ledger.add_rows([visual])

    assert ledger.visual_blocks() == []


def test_an_incremental_batch_states_the_graph_once_and_never_again() -> None:
    from dlightrag.engine.rag.retrieval import RetrievalContexts

    ledger = EvidenceLedger()
    ledger.add_contexts(
        cast(
            RetrievalContexts,
            {
                "chunks": [_corpus_row(chunk="c1", content="grounded passage")],
                "entities": [{"entity_name": "Acme", "entity_type": "ORG", "description": "d"}],
                "relationships": [],
            },
        )
    )

    labels, first = ledger.take_admitted_text(budget_tokens=1_000_000)
    # A chunk-only second batch must not re-state the graph the first batch already
    # froze: that would copy history into the uncached suffix of every later result.
    ledger.add_rows([_corpus_row(chunk="c2", content="later passage")])
    _labels, second = ledger.take_admitted_text(budget_tokens=1_000_000)

    assert labels == ""
    assert first.count("## Knowledge graph evidence") == 1
    assert "Acme" in first
    assert "## Knowledge graph evidence" not in second
    assert "Acme" not in second
    assert "later passage" in second


def test_a_restored_ledger_renders_nothing_it_already_froze() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row(chunk="c1", content="already in the transcript")])
    ledger.take_admitted_text(budget_tokens=1_000_000)

    restored = EvidenceLedger()
    restored.restore_ledger_state(ledger.durable_state())

    # The committed Tool results already carry that text; a recovery must not
    # render it a second time into a later result.
    assert restored.take_admitted_text(budget_tokens=1_000_000) == ("", "")


def test_re_executing_one_intent_freeze_reproduces_the_same_bytes() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row(chunk="c1", content="passage the tool admitted")])

    first = ledger.take_admitted_text(budget_tokens=1_000_000, intent_key="intent-1")
    # A resume while the effect was still pending re-executes the same intent. It
    # must render the same text again rather than report nothing new and commit an
    # unlabelled Tool result.
    replay = ledger.take_admitted_text(budget_tokens=1_000_000, intent_key="intent-1")
    # A different Tool is not entitled to it.
    other = ledger.take_admitted_text(budget_tokens=1_000_000, intent_key="intent-2")

    assert first == replay
    assert first != ("", "")
    assert other == ("", "")


def test_rollback_show_returns_the_rows_to_the_pending_set() -> None:
    ledger = EvidenceLedger()
    ledger.add_rows([_corpus_row(chunk="c1", content="passage the tool admitted")])
    ledger.take_admitted_text(budget_tokens=1_000_000, intent_key="intent-1")

    ledger.rollback_show("intent-1")
    _labels, rendered = ledger.take_admitted_text(budget_tokens=1_000_000, intent_key="intent-2")

    assert "passage the tool admitted" in rendered


def test_a_body_no_tool_carried_is_always_rendered_however_short() -> None:
    ledger = EvidenceLedger()
    # No Tool declared this row carried, so nothing about its text can suppress it:
    # a snippet that merely resembles a Tool's status line still reaches the model.
    ledger.add_rows([_corpus_row(chunk="c1", content="added")])

    labels, rendered = ledger.take_admitted_text(budget_tokens=1_000_000)

    assert labels == ""
    assert "added" in rendered


def test_a_carried_declaration_is_a_run_local_rendering_fact() -> None:
    carried = _corpus_row(chunk="read-1", content="body the tool result already shows")
    carried["_carried_by_tool"] = True
    ledger = EvidenceLedger()
    ledger.add_rows([carried])

    # It is spent where it was made and never enters the durable ledger row.
    state = ledger.durable_state()
    assert all("_carried_by_tool" not in row for row in state["contexts"]["chunks"])
    restored = EvidenceLedger()
    restored.restore_ledger_state(state)
    restored.add_rows([])
    assert restored.row_count == 1


def test_the_visual_lane_never_exceeds_its_image_budget() -> None:
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
    ledger = EvidenceLedger(image_budget=budget)
    rows = []
    for index in range(3):
        row = _corpus_row(chunk=f"p{index}", content=f"caption {index}")
        row["image_data"] = png
        rows.append(row)
    ledger.add_rows(rows)
    asyncio.run(ledger.aflush_images())

    blocks = ledger.visual_blocks()

    # One image budget, one lane: the third row's pixels are dropped, not queued.
    assert sum(block["type"] == "image_url" for block in blocks) == 1
    assert len(blocks) == 2
