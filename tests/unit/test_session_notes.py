# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Session's notes plane: admission, promotion, and what degrades instead.

Memory belongs to the Agent Session (ADR 0022). A Run materializes a working copy and
promotes its differences back; a plane that cannot take a note refuses that note and
never evicts an older one, and no failure here reaches the Run as a failure.
"""

import hashlib
from pathlib import Path

import pytest

from dlightrag.engine.answer.session_notes import (
    SESSION_NOTES_PLANE_UNREADABLE,
    SESSION_NOTES_PROMOTION_REFUSED,
    SESSION_NOTES_WORKING_COPY_UNREADABLE,
    SessionNotesPlane,
    read_legacy_notes,
    read_working_copy,
    read_working_copy_notes,
)
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import (
    SESSION_NOTES_BUDGET_REFUSED,
    SESSION_NOTES_LEASE_LOST,
    SESSION_NOTES_MAX_BYTES,
    SESSION_NOTES_MAX_COUNT,
    InMemoryWorkspaceStore,
    SessionNoteRecord,
    select_promotable_session_notes,
)

SESSION = "01930000-0000-7000-8000-0000000000aa"


def _note(path: str, content: bytes = b"value") -> SessionNoteRecord:
    return SessionNoteRecord(relative_path=path, content=content)


def _inventory(relative_path: str, *, size_bytes: int = 1_240, digest: str | None = None):
    return InventoryPathRecord(
        relative_path=relative_path,
        entry_type="file",
        size_bytes=size_bytes,
        content_digest=digest,
    )


def _plane_with(workspace: Path, store: InMemoryWorkspaceStore) -> SessionNotesPlane:
    plane = SessionNotesPlane(store=store, session_id=SESSION)
    plane.rebind(workspace=workspace, records=())
    return plane


# -- admission -----------------------------------------------------------------


def test_admission_is_by_path_and_replaces_in_place() -> None:
    accepted, refused = select_promotable_session_notes(
        existing={"notes/b.md": 4},
        upserts=(_note("notes/b.md", b"longer"), _note("notes/a.md")),
        deletes=(),
    )

    assert [note.relative_path for note in accepted] == ["notes/a.md", "notes/b.md"]
    assert refused == ()


def test_a_deletion_prices_its_own_budget_in_the_same_promotion() -> None:
    """A Run that removes a note may use the space it freed, in one promotion."""
    existing = {"notes/old.md": SESSION_NOTES_MAX_BYTES}
    accepted, refused = select_promotable_session_notes(
        existing=existing,
        upserts=(_note("notes/new.md", b"x" * 64),),
        deletes=("notes/old.md",),
    )

    assert [note.relative_path for note in accepted] == ["notes/new.md"]
    assert refused == ()


def test_the_byte_budget_refuses_a_note_rather_than_evicting_one() -> None:
    existing = {"notes/kept.md": SESSION_NOTES_MAX_BYTES - 8}
    accepted, refused = select_promotable_session_notes(
        existing=existing,
        upserts=(_note("notes/new.md", b"x" * 64),),
        deletes=(),
    )

    assert accepted == ()
    assert refused == ("notes/new.md",)


def test_the_count_cap_refuses_the_note_that_would_exceed_it() -> None:
    existing = {f"notes/{index:02d}.md": 8 for index in range(SESSION_NOTES_MAX_COUNT)}

    accepted, refused = select_promotable_session_notes(
        existing=existing,
        upserts=(_note("notes/one-more.md"),),
        deletes=(),
    )

    assert accepted == ()
    assert refused == ("notes/one-more.md",)


# -- the working copy ----------------------------------------------------------


def test_the_working_copy_reads_notes_only_and_skips_what_it_cannot_promote(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes" / "nested").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"plan")
    (workspace / "notes" / "nested" / "deep.md").write_bytes(b"deep")
    (workspace / "artifacts").mkdir()
    (workspace / "artifacts" / "report.md").write_bytes(b"published")
    (workspace / "notes" / "link.md").symlink_to(workspace / "notes" / "plan.md")
    (workspace / "notes" / "huge.md").write_bytes(b"x" * (SESSION_NOTES_MAX_BYTES + 1))
    absurd = workspace / "notes" / ("a" * 200) / ("b" * 200) / ("c" * 200)
    absurd.mkdir(parents=True)
    (absurd / "plan.md").write_bytes(b"absurd name")

    notes = read_working_copy(workspace)

    assert [note.relative_path for note in notes] == ["notes/nested/deep.md", "notes/plan.md"]


def test_the_working_copy_is_empty_without_a_notes_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    assert read_working_copy(workspace) == ()


# -- promotion through the plane -----------------------------------------------


@pytest.mark.asyncio
async def test_reconcile_promotes_new_changed_and_deleted_notes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"first")
    store = InMemoryWorkspaceStore()
    plane = _plane_with(workspace, store)

    assert await plane.reconcile() is None
    assert await store.load_session_notes(session_id=SESSION) == (_note("notes/plan.md", b"first"),)

    (workspace / "notes" / "plan.md").write_bytes(b"second")
    (workspace / "notes" / "findings.md").write_bytes(b"found")
    assert await plane.reconcile() is None
    assert [note.relative_path for note in await store.load_session_notes(session_id=SESSION)] == [
        "notes/findings.md",
        "notes/plan.md",
    ]
    assert store.session_notes[SESSION]["notes/plan.md"] == b"second"

    (workspace / "notes" / "plan.md").unlink()
    assert await plane.reconcile() is None
    assert [note.relative_path for note in await store.load_session_notes(session_id=SESSION)] == [
        "notes/findings.md",
    ]


@pytest.mark.asyncio
async def test_reconcile_promotes_nothing_when_the_copy_is_unchanged(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"same")
    store = InMemoryWorkspaceStore()
    await store.promote_session_notes(
        session_id=SESSION, upserts=(_note("notes/plan.md", b"same"),), deletes=()
    )
    plane = _plane_with(workspace, store)
    plane.materialized(await store.load_session_notes(session_id=SESSION))

    promoted = len(store.session_notes[SESSION])
    assert await plane.reconcile() is None
    assert len(store.session_notes[SESSION]) == promoted


@pytest.mark.asyncio
async def test_a_lane_never_deletes_a_note_it_did_not_materialize(tmp_path: Path) -> None:
    """Deletion is priced against the baseline, not against whatever the plane holds.

    A Fork shares its Session's memory, so two Lanes may write the same plane. One
    Lane's working copy is not evidence that another Lane's note should go.
    """
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    store = InMemoryWorkspaceStore()
    plane = _plane_with(workspace, store)
    await store.promote_session_notes(
        session_id=SESSION, upserts=(_note("notes/sibling.md", b"hers"),), deletes=()
    )

    assert await plane.reconcile() is None
    assert [note.relative_path for note in await store.load_session_notes(session_id=SESSION)] == [
        "notes/sibling.md"
    ]


@pytest.mark.asyncio
async def test_a_note_over_the_whole_budget_is_refused_by_name(tmp_path: Path) -> None:
    """A note the plane could never hold is stated, not silently dropped.

    Its bytes are not read at all — the working copy keeps them for this Run's turns —
    but the Run's trace has to say that the write was not remembered.
    """
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "huge.md").write_bytes(b"x" * (SESSION_NOTES_MAX_BYTES + 1))
    (workspace / "notes" / "small.md").write_bytes(b"value")
    store = InMemoryWorkspaceStore()
    plane = _plane_with(workspace, store)

    assert await plane.reconcile() == SESSION_NOTES_BUDGET_REFUSED
    assert [note.relative_path for note in await store.load_session_notes(session_id=SESSION)] == [
        "notes/small.md"
    ]
    read = read_working_copy_notes(workspace)
    assert read.oversized == ("notes/huge.md",)


@pytest.mark.asyncio
async def test_a_refused_promotion_reports_a_reason_and_keeps_the_note_for_retry(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "small.md").write_bytes(b"value")
    store = InMemoryWorkspaceStore()
    await store.promote_session_notes(
        session_id=SESSION,
        upserts=(_note("notes/fills.md", b"x" * SESSION_NOTES_MAX_BYTES),),
        deletes=(),
    )
    plane = _plane_with(workspace, store)

    reason = await plane.reconcile()

    assert reason == SESSION_NOTES_BUDGET_REFUSED
    assert "notes/small.md" not in store.session_notes[SESSION]
    # The baseline still lacks it, so the next settlement tries again.
    assert await plane.reconcile() == SESSION_NOTES_BUDGET_REFUSED


@pytest.mark.asyncio
async def test_a_symlinked_notes_root_is_refused_rather_than_followed(tmp_path: Path) -> None:
    """A linked notes root neither reads outside the workspace nor wipes the plane.

    Every other surface in this system refuses a symbolic link at a workspace path;
    the promotion read is the one that must not follow one.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "notes").mkdir(parents=True)
    (elsewhere / "notes" / "secret.md").write_bytes(b"outside the workspace")
    (workspace / "notes").symlink_to(elsewhere / "notes")

    store = InMemoryWorkspaceStore()
    await store.promote_session_notes(
        session_id=SESSION, upserts=(_note("notes/plan.md", b"memory"),), deletes=()
    )
    plane = _plane_with(workspace, store)
    plane.materialized(await store.load_session_notes(session_id=SESSION))

    assert await plane.reconcile() == SESSION_NOTES_WORKING_COPY_UNREADABLE
    # Nothing was promoted, and above all nothing was deleted.
    assert [note.relative_path for note in await store.load_session_notes(session_id=SESSION)] == [
        "notes/plan.md"
    ]


@pytest.mark.asyncio
async def test_a_baseline_note_that_grows_past_the_budget_is_not_evicted(tmp_path: Path) -> None:
    """A note the plane already holds is refused when it outgrows the cap, never deleted.

    Priced as a deletion it would leave memory entirely, which is the eviction ADR 0022
    forbids: the plane refuses the new bytes and keeps the note it has.
    """
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"v" * (SESSION_NOTES_MAX_BYTES + 1))
    store = InMemoryWorkspaceStore()
    await store.promote_session_notes(
        session_id=SESSION, upserts=(_note("notes/plan.md", b"remembered"),), deletes=()
    )
    plane = _plane_with(workspace, store)
    plane.materialized(await store.load_session_notes(session_id=SESSION))

    assert await plane.reconcile() == SESSION_NOTES_BUDGET_REFUSED
    assert await store.load_session_notes(session_id=SESSION) == (
        _note("notes/plan.md", b"remembered"),
    )


@pytest.mark.asyncio
async def test_a_delete_is_confirmed_against_the_planes_current_bytes(tmp_path: Path) -> None:
    """A Lane removes a note only while the plane still holds the copy it materialized.

    A sibling Lane's newer rewrite is not this Lane's to delete: memory is
    last-settled-wins, and a delete is confirmed before it is sent.
    """
    workspace = tmp_path / "workspace"
    store = InMemoryWorkspaceStore()
    await store.promote_session_notes(
        session_id=SESSION, upserts=(_note("notes/plan.md", b"original"),), deletes=()
    )
    plane = SessionNotesPlane(store=store, session_id=SESSION)
    plane.rebind(
        workspace=workspace,
        records=await store.load_session_notes(session_id=SESSION),
    )
    # The sibling Lane rewrites the note, and this Lane never had it on disk.
    await store.promote_session_notes(
        session_id=SESSION, upserts=(_note("notes/plan.md", b"sibling rewrite"),), deletes=()
    )

    assert await plane.reconcile() is None
    assert await store.load_session_notes(session_id=SESSION) == (
        _note("notes/plan.md", b"sibling rewrite"),
    )


@pytest.mark.asyncio
async def test_an_unreadable_plane_degrades_instead_of_raising(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = InMemoryWorkspaceStore()

    async def broken(*, session_id: str):
        raise RuntimeError("plane is gone")

    store.load_session_notes = broken  # type: ignore[method-assign]
    plane = SessionNotesPlane(store=store, session_id=SESSION)

    assert (await plane.load()).degraded_reason == SESSION_NOTES_PLANE_UNREADABLE


@pytest.mark.asyncio
async def test_a_plane_that_refuses_the_write_degrades(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"value")
    store = InMemoryWorkspaceStore()

    async def lease_lost(*, session_id: str, upserts, deletes):
        from dlightrag.engine.runtime.workspace import SessionNotesPromotion

        return SessionNotesPromotion(degraded_reason=SESSION_NOTES_LEASE_LOST)

    store.promote_session_notes = lease_lost  # type: ignore[method-assign]
    plane = _plane_with(workspace, store)

    assert await plane.reconcile() == SESSION_NOTES_LEASE_LOST


@pytest.mark.asyncio
async def test_a_plane_that_raises_degrades(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"value")
    store = InMemoryWorkspaceStore()

    async def exploding(*, session_id: str, upserts, deletes):
        raise RuntimeError("write refused")

    store.promote_session_notes = exploding  # type: ignore[method-assign]
    plane = _plane_with(workspace, store)

    assert await plane.reconcile() == SESSION_NOTES_PROMOTION_REFUSED


def test_an_unreadable_working_copy_degrades(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "notes").mkdir(parents=True)
    (workspace / "notes" / "plan.md").write_bytes(b"value")

    def exploding(path: Path):
        raise OSError("working copy is gone")

    monkeypatch.setattr("dlightrag.engine.answer.session_notes.read_working_copy_notes", exploding)
    store = InMemoryWorkspaceStore()
    plane = _plane_with(workspace, store)

    import asyncio

    assert asyncio.run(plane.reconcile()) == SESSION_NOTES_WORKING_COPY_UNREADABLE


# -- the one last carry --------------------------------------------------------


def test_legacy_notes_are_read_only_when_their_bytes_still_match(tmp_path: Path) -> None:
    """A registration that disagrees with the bytes is left behind, not migrated."""
    source = tmp_path / "parent"
    (source / "notes").mkdir(parents=True)
    (source / "notes" / "plan.md").write_bytes(b"stale")
    (source / "notes" / "findings.md").write_bytes(b"fresh")
    (source / "notes" / "gone.md").write_bytes(b"x")

    notes = read_legacy_notes(
        source_workspace=source,
        records=(
            _inventory(
                "notes/plan.md",
                size_bytes=len(b"expected"),
                digest=hashlib.sha256(b"expected").hexdigest(),
            ),
            _inventory(
                "notes/findings.md",
                size_bytes=len(b"fresh"),
                digest=hashlib.sha256(b"fresh").hexdigest(),
            ),
            _inventory("notes/gone.md", size_bytes=0),
            _inventory("artifacts/report.md", size_bytes=0),
        ),
    )

    assert [note.relative_path for note in notes] == ["notes/findings.md"]


def test_legacy_notes_stop_at_the_planes_own_bounds(tmp_path: Path) -> None:
    source = tmp_path / "parent"
    (source / "notes").mkdir(parents=True)
    (source / "notes" / "big.md").write_bytes(b"x" * (SESSION_NOTES_MAX_BYTES + 1))

    notes = read_legacy_notes(
        source_workspace=source,
        records=(_inventory("notes/big.md", size_bytes=SESSION_NOTES_MAX_BYTES + 1),),
    )

    assert notes == ()


def test_legacy_notes_are_empty_without_a_source_tree(tmp_path: Path) -> None:
    assert read_legacy_notes(source_workspace=None, records=(_inventory("notes/a.md"),)) == ()
