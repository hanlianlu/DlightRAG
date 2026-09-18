# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The continuation identities one compaction summary carries forward.

Two of them are handles the next turn reads by identity, and a Session Note is a path
it reads again — the file itself is what outlives the summary.
"""

from dlightrag.engine.answer.continuation_handles import (
    MAX_ARTIFACT_HANDLES,
    MAX_NAMED_SESSION_NOTES,
    MAX_SESSION_NOTE_PATH_CHARS,
    MAX_SPILL_HANDLES,
    SESSION_NOTE_DIRECTORY,
    compose_durable_handles,
    compose_session_notes,
    is_session_note,
    published_artifact_handle,
    session_note_handle,
    session_notes_message,
    spill_handle,
)
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import CommittedSpillRecord, SessionNoteRecord


def _spill(
    resource_id: str, *, size_bytes: int = 4_096, intent_id: str = "i"
) -> CommittedSpillRecord:
    return CommittedSpillRecord(
        resource_id=resource_id,
        content_digest="a" * 64,
        size_bytes=size_bytes,
        session_id="session",
        intent_id=intent_id,
    )


def test_spill_handle_spells_out_the_call_the_next_turn_must_make() -> None:
    """The receipt that taught the call is inside the covered prefix by then.

    A cursor is deliberately absent rather than rendered as ``0``: the Tool takes
    a string cursor and an omitted one already starts at the first line, so a
    handle that showed an integer would invite a rejected call.
    """
    handle = spill_handle(_spill("spill_web_search_ab12", size_bytes=924_133))

    assert handle == (
        "[spill] spill_web_search_ab12 (924133 bytes) — "
        're-read with read(resource_id="spill_web_search_ab12")'
    )
    assert "cursor=0" not in handle


def test_newest_spills_precede_evidence_and_evidence_survives_them() -> None:
    handles = compose_durable_handles(
        spills=[_spill("spill_new"), _spill("spill_old")],
        evidence_handles=["[1] report.pdf"],
    )

    assert handles == [
        spill_handle(_spill("spill_new")),
        spill_handle(_spill("spill_old")),
        "[1] report.pdf",
    ]


def test_spills_claim_a_bounded_share_and_never_evict_evidence_entirely() -> None:
    """A share, not priority: a retrieval-heavy Run keeps its citation handles."""
    spills = [_spill(f"spill_{index:03d}") for index in range(MAX_SPILL_HANDLES + 5)]

    handles = compose_durable_handles(spills=spills, evidence_handles=["[1] report.pdf"])

    assert len(handles) == MAX_SPILL_HANDLES + 1
    assert handles[:MAX_SPILL_HANDLES] == [
        spill_handle(spill) for spill in spills[:MAX_SPILL_HANDLES]
    ]
    assert handles[-1] == "[1] report.pdf"


def test_either_class_alone_still_composes() -> None:
    assert compose_durable_handles(spills=[], evidence_handles=[]) == []
    assert compose_durable_handles(spills=[], evidence_handles=["[1] memo"]) == ["[1] memo"]
    assert compose_durable_handles(spills=[_spill("spill_a")], evidence_handles=[]) == [
        spill_handle(_spill("spill_a"))
    ]


def _inventory(
    relative_path: str, *, size_bytes: int = 1_240, entry_type: str = "file"
) -> InventoryPathRecord:
    return InventoryPathRecord(
        relative_path=relative_path,
        entry_type=entry_type,
        size_bytes=size_bytes,
        content_digest="c" * 64,
    )


def test_only_the_notes_directory_holds_run_notes() -> None:
    """`artifacts/` is a publication surface, and a sibling prefix is not a note."""
    assert is_session_note("notes/plan.md")
    assert is_session_note("notes/2026/decisions.md")
    assert not is_session_note("notes")
    assert not is_session_note("notes/")
    assert not is_session_note("artifacts/notes.md")
    assert not is_session_note("noteset/plan.md")
    assert not is_session_note("notes/../artifacts/report.md")


def test_session_note_handle_names_the_path_call_and_not_a_frozen_digest() -> None:
    """A note is a live file: the read that follows serves its current bytes.

    The recorded digest is absent on purpose. The Inventory holds one only for
    paths the framework wrote itself — a `bash` call re-observes the whole
    workspace without digests — so a handle that printed one would be claiming a
    guarantee the inventory cannot keep.
    """
    handle = session_note_handle(_inventory("notes/plan.md"))

    assert handle == (
        "[note] notes/plan.md (1240 bytes) — "
        "re-read with read(path='notes/plan.md') before a step that needs a value this summary does not state"
    )
    assert "c" * 64 not in handle


def test_compose_keeps_inventory_order_bounds_the_list_and_skips_directories() -> None:
    records = [
        _inventory("artifacts/report.md"),
        _inventory("notes/a.md"),
        _inventory("notes", entry_type="directory"),
        _inventory("readme.md"),
        *(_inventory(f"notes/{index:02d}.md") for index in range(MAX_NAMED_SESSION_NOTES + 3)),
    ]

    notes = compose_session_notes(records)

    assert len(notes) == MAX_NAMED_SESSION_NOTES
    assert notes[0] == (
        "[note] notes/a.md (1240 bytes) — re-read with read(path='notes/a.md') before a step that needs a value this summary does not state"
    )
    assert all("artifacts" not in note and "readme" not in note for note in notes)


def test_a_note_name_that_needs_quoting_still_renders_a_call_that_parses() -> None:
    """A name may hold a quote; a call the model cannot reproduce is worse than none."""
    handle = session_note_handle(_inventory('notes/say "hello".md'))

    assert """re-read with read(path='notes/say "hello".md')""" in handle


def test_an_absurd_note_path_is_not_named_at_all() -> None:
    """The count cap does not bound prose, and the summary's budget is not a note's."""
    records = [
        _inventory(f"notes/{'x' * MAX_SESSION_NOTE_PATH_CHARS}.md"),
        _inventory("notes/plan.md"),
    ]

    notes = compose_session_notes(records)

    assert notes == [
        "[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md') before a step that needs a value this summary does not state"
    ]


def test_the_reserved_directory_the_prompt_teaches_is_the_one_the_rule_reads() -> None:
    """Two modules know the prefix, so one test has to hold them together.

    The prompt states the directory in prose (prompts stay import-light), and the
    filter has the constant; a rename that moved only one of them would leave the
    habit pointing at a directory nothing recognizes.
    """
    from dlightrag.engine.answer.prompts.agent import agent_control_prompt

    taught = agent_control_prompt(run_notes=True)

    assert f"`{SESSION_NOTE_DIRECTORY}/`" in taught
    assert f"`{SESSION_NOTE_DIRECTORY}/`" not in agent_control_prompt(run_notes=False)


def test_a_published_artifact_renders_the_handle_a_later_turn_reads_it_with() -> None:
    """One handle family: the address is the path hashed, so it exists before the row."""
    from dlightrag.engine.answer.publication import artifact_resource_id
    from dlightrag.engine.runtime.workspace import RunArtifactRecord

    record = RunArtifactRecord(
        relative_path="reports/analysis.md",
        label="analysis.md",
        size_bytes=1_234,
        content_digest="d" * 64,
        presentation="markdown",
    )

    handle = published_artifact_handle(record)

    assert handle.startswith("[artifact] reports/analysis.md (1234 bytes)")
    assert f"read(resource_id={artifact_resource_id('reports/analysis.md')!r})" in handle
    assert "editing it again" in handle


def test_a_product_whose_type_needs_conversion_is_taught_the_view_call() -> None:
    """The taught call has to be one that works for that product's type.

    A Markdown report is decoded, so `read` reaches it. A PDF is not converted
    retrospectively — the earlier Run never recorded that view — so the call a later
    turn can actually make is `view`, and teaching `read` there would teach a refusal.
    """
    from dlightrag.engine.answer.publication import artifact_read_call
    from dlightrag.engine.runtime.workspace import RunArtifactRecord

    markdown = artifact_read_call("reports/analysis.md", mime_type="text/markdown")
    assert markdown.startswith("read(resource_id='artifact-")
    assert artifact_read_call("reports/data.csv", mime_type="text/csv").startswith(
        "view(resource_id='artifact-"
    )

    record = RunArtifactRecord(
        relative_path="reports/report.pdf",
        label="report.pdf",
        size_bytes=2_048,
        content_digest="e" * 64,
        presentation="pdf",
    )
    handle = published_artifact_handle(record)

    assert "view(resource_id='artifact-" in handle
    assert "read(resource_id=" not in handle


def test_the_handle_list_reserves_a_share_for_each_non_evidence_class() -> None:
    """Spills keep their place, published products keep theirs, Evidence fills the rest."""
    from dlightrag.engine.runtime.workspace import RunArtifactRecord

    artifacts = tuple(
        RunArtifactRecord(
            relative_path=f"reports/{index:02d}.md",
            label=f"{index:02d}.md",
            size_bytes=10,
            content_digest="d" * 64,
            presentation="markdown",
        )
        for index in range(MAX_ARTIFACT_HANDLES + 3)
    )

    handles = compose_durable_handles(
        spills=(_spill("spill-1"),),
        evidence_handles=("[evidence] res-1",),
        artifacts=artifacts,
    )

    assert handles[0].startswith("[spill] spill-1")
    assert len([handle for handle in handles if handle.startswith("[artifact]")]) == (
        MAX_ARTIFACT_HANDLES
    )
    assert handles[-1] == "[evidence] res-1"


def test_the_notes_message_is_empty_when_the_session_holds_no_memory() -> None:
    assert session_notes_message(()) == ""


def test_the_notes_message_names_paths_once_without_a_clock_or_run_id() -> None:
    message = session_notes_message(
        (
            SessionNoteRecord(relative_path="notes/plan.md", content=b"x" * 1240),
            SessionNoteRecord(relative_path="notes/decisions.md", content=b"y" * 310),
        )
    )

    assert message.count("already in this workspace") == 1
    assert "notes/plan.md (1240 bytes)" in message
    assert "notes/decisions.md (310 bytes)" in message
    assert "do not re-derive" in message
