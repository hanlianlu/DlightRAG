# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one door every Tool Subject passes through, where its wire spelling may appear, and the
scalar facts a Tool may report beside it."""

from pathlib import Path

from dlightrag.engine.agent.tool_content import ToolTextPart
from dlightrag.engine.agent.tools.contracts import (
    TOOL_PROGRESS_FACT_KEYS,
    TOOL_PROGRESS_FACT_MAX_CHARS,
    ToolResult,
)
from tests.support.source_tree import python_files, source_text

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WIRE_SPELLING = "object_label"
# The Engine sink that projects a reported subject onto the run event, the browser edge that
# keeps its own display cap, the Application derivation that reads the same durable payload
# back to learn which Skill a Run loaded and what a call ran, the Engine module that reads it
# back for which document a ``load_skill`` call answered (the derivation and the in-loop ledger
# share it), and the Postgres run store that reads the same payload back for the subjects
# historical trace steps showed. No other Engine module may name the wire field, and every owner
# except the sink only reads it.
_WIRE_SPELLING_OWNERS = {
    Path("src/dlightrag/engine/answer/research/runtime.py"),
    Path("src/dlightrag/adapters/http/browser/answer_events.py"),
    Path("src/dlightrag/application/answer/provenance.py"),
    Path("src/dlightrag/engine/agent/skill_reads.py"),
    Path("src/dlightrag/adapters/postgres/runtime/run_store.py"),
}


def test_a_reported_subject_becomes_one_bounded_line() -> None:
    collapsed = ToolResult.text("", subject="  quarterly\n\trevenue  ")

    assert collapsed.subject == "quarterly revenue"
    # Control characters are invisible in a one-line row, and U+0000 cannot be stored in
    # the durable JSON payload at all.
    assert ToolResult.text("", subject="a\x00b\x07c").subject == "abc"
    assert ToolResult.text("", subject="x" * 64).subject == "x" * 64
    assert ToolResult.text("", subject="x" * 65).subject == "x" * 63 + "…"


def test_the_bound_holds_on_every_construction_path() -> None:
    constructed = ToolResult(parts=(ToolTextPart(""),), subject="y" * 200)

    assert constructed.subject == "y" * 63 + "…"


def test_a_subject_never_reaches_the_model_visible_text() -> None:
    result = ToolResult.text("body", subject="quarterly revenue")

    assert result.text_content == "body"
    assert ToolResult.text("body").subject is None


def test_the_wire_spelling_stays_at_the_engine_sink_and_the_browser_edge() -> None:
    owners = {
        path.relative_to(_REPO_ROOT)
        for path in python_files(_REPO_ROOT / "src")
        if _WIRE_SPELLING in source_text(path)
    }

    assert owners == _WIRE_SPELLING_OWNERS


def test_only_the_named_scalar_facts_reach_the_progress_event() -> None:
    """``details`` carries more than the event log should hold, so the vocabulary is the filter."""
    result = ToolResult.text(
        "",
        details={
            "skill_path": "references/gotchas.md",
            "skill_window_start": 0,
            "skill_window_end": 400,
            "skill_document_chars": 1000,
            "observed_truncated": True,  # the Runtime's own key: not a progress fact
            "anything_else": "never forwarded",
        },
    )

    assert result.progress_facts == {
        "skill_path": "references/gotchas.md",
        "skill_window_start": 0,
        "skill_window_end": 400,
        "skill_document_chars": 1000,
    }
    assert set(result.progress_facts) <= TOOL_PROGRESS_FACT_KEYS
    assert ToolResult.text("").progress_facts == {}


def test_a_fact_that_is_not_a_scalar_or_is_too_long_is_not_forwarded() -> None:
    result = ToolResult.text(
        "",
        details={
            "skill_path": ["references/gotchas.md"],
            "skill_window_start": {"nested": 1},
            "skill_window_end": 12,
            "skill_document_chars": "x" * (TOOL_PROGRESS_FACT_MAX_CHARS + 1),
        },
    )

    assert result.progress_facts == {"skill_window_end": 12}
    assert ToolResult.text(
        "", details={"skill_path": "x" * TOOL_PROGRESS_FACT_MAX_CHARS}
    ).progress_facts == {"skill_path": "x" * TOOL_PROGRESS_FACT_MAX_CHARS}
