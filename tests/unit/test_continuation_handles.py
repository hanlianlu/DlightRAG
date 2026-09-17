# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The continuation identities one compaction summary carries forward.

Two of them are handles the next turn reads by identity, and a Run Note is a path
it reads again — the file itself is what outlives the summary.
"""

from dlightrag.engine.answer.continuation_handles import (
    MAX_CARRIED_RUN_NOTE_BYTES,
    MAX_RUN_NOTE_PATH_CHARS,
    MAX_RUN_NOTES,
    MAX_SPILL_HANDLES,
    RUN_NOTE_DIRECTORY,
    carried_run_notes_message,
    compose_durable_handles,
    compose_run_notes,
    is_run_note,
    run_note_handle,
    select_carried_run_notes,
    spill_handle,
)
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import CommittedSpillRecord


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
    assert is_run_note("notes/plan.md")
    assert is_run_note("notes/2026/decisions.md")
    assert not is_run_note("notes")
    assert not is_run_note("notes/")
    assert not is_run_note("artifacts/notes.md")
    assert not is_run_note("noteset/plan.md")
    assert not is_run_note("notes/../artifacts/report.md")


def test_run_note_handle_names_the_path_call_and_not_a_frozen_digest() -> None:
    """A note is a live file: the read that follows serves its current bytes.

    The recorded digest is absent on purpose. The Inventory holds one only for
    paths the framework wrote itself — a `bash` call re-observes the whole
    workspace without digests — so a handle that printed one would be claiming a
    guarantee the inventory cannot keep.
    """
    handle = run_note_handle(_inventory("notes/plan.md"))

    assert handle == ("[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md')")
    assert "c" * 64 not in handle


def test_compose_keeps_inventory_order_bounds_the_list_and_skips_directories() -> None:
    records = [
        _inventory("artifacts/report.md"),
        _inventory("notes/a.md"),
        _inventory("notes", entry_type="directory"),
        _inventory("readme.md"),
        *(_inventory(f"notes/{index:02d}.md") for index in range(MAX_RUN_NOTES + 3)),
    ]

    notes = compose_run_notes(records)

    assert len(notes) == MAX_RUN_NOTES
    assert notes[0] == ("[note] notes/a.md (1240 bytes) — re-read with read(path='notes/a.md')")
    assert all("artifacts" not in note and "readme" not in note for note in notes)


def test_a_note_name_that_needs_quoting_still_renders_a_call_that_parses() -> None:
    """A name may hold a quote; a call the model cannot reproduce is worse than none."""
    handle = run_note_handle(_inventory('notes/say "hello".md'))

    assert handle.endswith("""re-read with read(path='notes/say "hello".md')""")


def test_an_absurd_note_path_is_not_named_at_all() -> None:
    """The count cap does not bound prose, and the summary's budget is not a note's."""
    records = [
        _inventory(f"notes/{'x' * MAX_RUN_NOTE_PATH_CHARS}.md"),
        _inventory("notes/plan.md"),
    ]

    notes = compose_run_notes(records)

    assert notes == ["[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md')"]


def test_the_reserved_directory_the_prompt_teaches_is_the_one_the_rule_reads() -> None:
    """Two modules know the prefix, so one test has to hold them together.

    The prompt states the directory in prose (prompts stay import-light), and the
    filter has the constant; a rename that moved only one of them would leave the
    habit pointing at a directory nothing recognizes.
    """
    from dlightrag.engine.answer.prompts.agent import agent_control_prompt

    taught = agent_control_prompt(run_notes=True)

    assert f"`{RUN_NOTE_DIRECTORY}/`" in taught
    assert f"`{RUN_NOTE_DIRECTORY}/`" not in agent_control_prompt(run_notes=False)


def test_select_carried_run_notes_reuses_the_summary_caps_and_skips_long_names() -> None:
    records = [
        _inventory("artifacts/report.md"),
        _inventory(f"notes/{'x' * MAX_RUN_NOTE_PATH_CHARS}.md"),
        _inventory("notes/a.md", size_bytes=12),
        _inventory("notes/b.md", size_bytes=20),
        *(_inventory(f"notes/{index:02d}.md") for index in range(MAX_RUN_NOTES)),
    ]

    selected = select_carried_run_notes(records)

    assert [item.relative_path for item in selected] == [
        "notes/a.md",
        "notes/b.md",
        *[f"notes/{index:02d}.md" for index in range(MAX_RUN_NOTES - 2)],
    ]


def test_select_carried_run_notes_stops_at_the_byte_ceiling() -> None:
    first = _inventory("notes/a.md", size_bytes=MAX_CARRIED_RUN_NOTE_BYTES - 10)
    second = _inventory("notes/b.md", size_bytes=20)

    assert select_carried_run_notes((first, second)) == (first,)


def test_the_carry_message_is_empty_when_nothing_was_carried() -> None:
    assert carried_run_notes_message(()) == ""


def test_the_carry_message_names_paths_once_without_a_clock_or_run_id() -> None:
    message = carried_run_notes_message(
        (_inventory("notes/plan.md"), _inventory("notes/decisions.md", size_bytes=310))
    )

    assert message.count("already in this workspace") == 1
    assert "notes/plan.md (1240 bytes)" in message
    assert "notes/decisions.md (310 bytes)" in message
    assert "do not re-derive" in message
