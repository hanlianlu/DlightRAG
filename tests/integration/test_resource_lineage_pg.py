# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lineage adoption against real PostgreSQL rows and blobs."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore, write_blob_content
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.agent.environment.access import AccessScheduler
from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tools import ResourceAttachmentBytes, ToolResult
from dlightrag.engine.agent.tools.files import read_tool, view_tool
from dlightrag.engine.answer.execution.executor import AnswerExecutor
from dlightrag.engine.answer.execution.lineage import RetainedResourceLoader
from dlightrag.engine.answer.resources.converters import ExtractedVisual
from dlightrag.engine.answer.resources.lineage import (
    ASSET_KIND,
    LINEAGE_ADOPTION_KIND,
    SNAPSHOT_KIND,
    LineageAdoptionConflict,
    adopt_lineage_resource,
)
from dlightrag.engine.answer.resources.registry import ResourceEffectOwner, ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
from dlightrag.engine.runtime.coordinator import LeaseLostError
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope
from tests.support.pg import delete_runs
from tests.support.resources import printed_handle
from tests.tool_helpers import tool_runtime

OWNER = "lineage-owner"
DOCUMENT = b"%PDF-1.7 an earlier run's document"
PAGE = b"\x89PNG\r\n\x1a\npage-one"
EARLIER_TEXT = "Text the earlier run already extracted."


async def _store(db: Any) -> PGRunStore:
    """Establish the complete operational schema exactly as a real process does."""
    created = PGRunStore(pool=db)
    await created.initialize()
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore

    await PGWebConversationStore(pool=db, run_store=created).initialize()
    return created


async def _seed_origin_run(
    db: Any,
    store: PGRunStore,
    *,
    session_id: str | None = None,
    document_id: str = "res-earlier-document",
    text: str = EARLIER_TEXT,
    document: bytes = DOCUMENT,
) -> tuple[str, str]:
    """Write one earlier Run's document, its stored view, and its page asset.

    The origin Run exists as a real accepted Run, because a Resource row is only
    meaningful for a Run that owns it.
    """
    session_id = session_id or str(uuid.uuid4())
    accepted = await store.accept_run(
        envelope=run_envelope("answer", key=f"lineage-{uuid.uuid4().hex[:8]}", owner=OWNER),
        run_id=str(uuid.uuid7()),
        connection_bindings=(),
    )
    origin_run = str(accepted.run.run_id)
    snapshot = ConversionSnapshot(
        resource_id=document_id,
        input_digest=hashlib.sha256(document).hexdigest(),
        text=text,
        visuals=(
            ExtractedVisual(
                handle_id=f"page-1-of-{document_id}",
                anchor="page 1",
                origin_part=None,
                media_type="image/png",
                data=PAGE,
            ),
        ),
        extraction_status="complete",
        converter="fixture",
        converter_version="1",
    )
    rows = [
        (document_id, "tool_attachment", "earlier.pdf", "application/pdf", document, document_id)
    ]
    for effect in snapshot.effects():
        rows.append(
            (
                effect.resource_id,
                effect.resource_kind,
                effect.filename,
                effect.mime_type,
                effect.content,
                document_id,
            )
        )
    async with db.acquire() as conn, conn.transaction():
        for resource_id, kind, name, mime, content, locator in rows:
            digest = hashlib.sha256(content).hexdigest()
            await write_blob_content(conn, owner_id=OWNER, digest=digest, content=content)
            await conn.execute(
                """
                INSERT INTO dlightrag_answer_resources (
                    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
                    ordinal, blob_digest, locator_digest, source_locator, session_id, intent_id
                ) VALUES ($1,$2,$3,'fetched_blob',$4,$5,$6::jsonb,0,$7,$8,$9,$10,$11)
                """,
                OWNER,
                uuid.UUID(origin_run),
                resource_id,
                name,
                mime,
                json.dumps({"resource_kind": kind}),
                digest,
                hashlib.sha256(locator.encode()).hexdigest(),
                locator.encode(),
                uuid.UUID(session_id),
                uuid.uuid7(),
            )
    return session_id, origin_run


@dataclass(frozen=True)
class _Claim:
    run_id: str
    worker_id: str
    fencing_epoch: int


#: For loaders that only read: a claim no Run holds, so a write through it is refused.
_UNCLAIMED = _Claim(run_id=str(uuid.UUID(int=0)), worker_id="unclaimed", fencing_epoch=0)


async def _claimed_run(store: PGRunStore, worker_id: str = "lineage-worker") -> _Claim:
    """Accept one consuming Run and claim it, as the worker that executes it does."""
    accepted = await store.accept_run(
        envelope=run_envelope("answer", key=f"consumer-{uuid.uuid4().hex[:8]}", owner=OWNER),
        run_id=str(uuid.uuid7()),
        connection_bindings=(),
    )
    run_id = str(accepted.run.run_id)
    while (claim := await store.claim_next(worker_id=worker_id)) is not None:
        if str(claim.run.run_id) == run_id:
            return _Claim(run_id, worker_id, claim.run.fencing_epoch)
    raise AssertionError("the consuming Run was never claimed")


def _loader(
    store: PGRunStore,
    db: Any,
    session_id: str,
    claim: _Claim = _UNCLAIMED,
    *,
    owner_id: str = OWNER,
) -> RetainedResourceLoader:
    return RetainedResourceLoader(
        store=store,
        blobs=PGRunBlobStore(pool=db),
        owner_id=owner_id,
        session_id=session_id,
        run_id=claim.run_id,
        worker_id=claim.worker_id,
        fencing_epoch=claim.fencing_epoch,
    )


def _tools(registry: ResourceRegistry, lineage: RetainedResourceLoader):
    access = AccessScheduler()
    return (
        read_tool(
            None,
            access,
            resource_reader=make_resource_reader(registry, 1000, lineage=lineage),
        ),
        view_tool(
            None,
            access,
            resource_viewer=make_resource_viewer(registry, lineage=lineage),
            image_preparer=lambda _data, _label: None,
        ),
    )


async def _call(tool: Any, session_id: str, **args: Any) -> ToolResult:
    """Run one Tool call of the Agent Session ``session_id``, as its Host would."""
    return await tool.execute(
        tool.input_model.model_validate(args),
        tool_runtime(tool_name=tool.name, execution_scope=session_id),
    )


async def _resumed(
    store: PGRunStore,
    db: Any,
    run_id: str,
    *,
    resource_secret: bytes | None = None,
    max_attachments: int = 6,
) -> ResourceRegistry:
    """The Run's registry after a resume: its own rows only, no lineage loader at all."""
    executor = object.__new__(AnswerExecutor)
    executor._store = store
    executor._blob_store = PGRunBlobStore(pool=db)
    registry = ResourceRegistry(resource_secret=resource_secret, max_attachments=max_attachments)
    try:
        await executor._restore_registry_fetches(registry, owner_id=OWNER, run_id=run_id)
    except BaseException:
        await registry.aclose()
        raise
    return registry


async def _run_rows(db: Any, run_id: str) -> list[Any]:
    async with db.acquire() as conn:
        return list(
            await conn.fetch(
                "SELECT resource_id, blob_digest, source_locator, session_id::text AS session_id,"
                " intent_id::text AS intent_id, capabilities"
                " FROM dlightrag_answer_resources WHERE owner_id = $1 AND run_id = $2"
                " ORDER BY resource_id",
                OWNER,
                uuid.UUID(run_id),
            )
        )


def _kind(row: Any) -> str:
    return str(json.loads(row["capabilities"])["resource_kind"])


async def test_adopts_an_earlier_runs_document_and_its_stored_view() -> None:
    async with isolated_run_runtime("resource_lineage") as (_, db):
        store = await _store(db)
        session_id, origin_run = await _seed_origin_run(db, store)
        claim = await _claimed_run(store)
        loader = _loader(store, db, session_id, claim)

        loaded = await loader.load("res-earlier-document")

        assert loaded is not None
        assert loaded.content == DOCUMENT
        assert loaded.origin_run_id == origin_run
        assert loaded.filename == "earlier.pdf"
        assert loaded.conversion_snapshot is not None
        assert loaded.assets == {"page-1-of-res-earlier-document": PAGE}

        # A Child's call adopts: the rows still carry the conversation's Session.
        child_session = str(uuid.uuid4())
        owner = ResourceEffectOwner(execution_scope=child_session, intent_id=IntentId.new())
        async with ResourceRegistry() as registry:
            adopted = await adopt_lineage_resource(
                registry, loaded, record=partial(loader.record, owner=owner)
            )
            assert adopted != "res-earlier-document"
            text = await registry.read("res-earlier-document", max_window_tokens=1000)
            assert text.resource_id == adopted
            assert EARLIER_TEXT in text.content

        rows = await _run_rows(db, claim.run_id)
        assert sorted(_kind(row) for row in rows) == [
            ASSET_KIND,
            SNAPSHOT_KIND,
            LINEAGE_ADOPTION_KIND,
        ]
        assert {bytes(row["source_locator"]) for row in rows} == {adopted.encode()}
        assert {row["session_id"] for row in rows} == {session_id}
        assert {row["intent_id"] for row in rows} == {owner.intent_id.value}


async def test_an_adoption_is_durable_before_its_call_settles() -> None:
    """The adopting call's own result never settles here, and the Run resumes anyway.

    The adoption row and its view are this Run's, written under its lease, so a resume
    restores both handles and the view without reading the lineage.
    """
    async with isolated_run_runtime("resource_lineage_durable") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)
        claim = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            _, view = _tools(registry, _loader(store, db, session_id, claim))
            # The fixture document has no renderable page, so the retried view fails
            # after the adoption, and its refusal settles nothing on its behalf.
            refused = await _call(view, session_id, resource_id="res-earlier-document")
            assert refused.is_error is True
            assert refused.effects.attached_resources == ()
            earlier = await registry.read("res-earlier-document", max_window_tokens=1000)
            canonical = earlier.resource_id

        rows = await _run_rows(db, claim.run_id)
        adoption = next(row for row in rows if _kind(row) == LINEAGE_ADOPTION_KIND)
        assert adoption["resource_id"] == canonical
        assert json.loads(adoption["capabilities"])["resource_aliases"] == ["res-earlier-document"]
        assert f"{canonical}-conversion" in {row["resource_id"] for row in rows}

        resumed = await _resumed(store, db, claim.run_id)
        try:
            for handle in ("res-earlier-document", canonical):
                text = await resumed.read(handle, max_window_tokens=1000)
                assert text.resource_id == canonical
                assert EARLIER_TEXT in text.content
        finally:
            await resumed.aclose()


async def test_a_lost_lease_records_nothing() -> None:
    async with isolated_run_runtime("resource_lineage_lease") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)
        claim = await _claimed_run(store)
        stale = _Claim(claim.run_id, claim.worker_id, claim.fencing_epoch + 1)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, stale))
            with pytest.raises(LeaseLostError):
                await _call(read, session_id, resource_id="res-earlier-document")
            assert registry.manifest() == ()

        assert await _run_rows(db, claim.run_id) == []


class _LandsThenFails:
    """The store, except that its first adoption commits and then loses its connection."""

    def __init__(self, store: PGRunStore) -> None:
        self._store = store
        self._landed = False

    async def lineage_resource_rows(
        self, *, owner_id: str, session_id: str, resource_id: str
    ) -> tuple[Any, ...]:
        return await self._store.lineage_resource_rows(
            owner_id=owner_id, session_id=session_id, resource_id=resource_id
        )

    async def record_lineage_adoption(self, **write: Any) -> None:
        await self._store.record_lineage_adoption(**write)
        if not self._landed:
            self._landed = True
            raise ConnectionResetError("connection lost after COMMIT")


async def test_an_adoption_that_landed_despite_an_error_is_restored_without_refusal() -> None:
    """An adoption whose COMMIT landed unseen is durable, even past the allowance.

    The live Run gave the slot back and adopted another document with it; the resume
    restores both recorded adoptions as this Run's durable state, refusing neither.
    """
    async with isolated_run_runtime("resource_lineage_ambiguous") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store, document_id="res-earlier-a")
        await _seed_origin_run(
            db,
            store,
            session_id=session_id,
            document_id="res-earlier-b",
            text="The second document.",
            document=b"%PDF-1.7 a second earlier document",
        )
        claim = await _claimed_run(store)
        loader = RetainedResourceLoader(
            store=_LandsThenFails(store),
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=session_id,
            run_id=claim.run_id,
            worker_id=claim.worker_id,
            fencing_epoch=claim.fencing_epoch,
        )
        async with ResourceRegistry(max_attachments=1) as registry:
            read, _ = _tools(registry, loader)
            with pytest.raises(ConnectionResetError):
                await _call(read, session_id, resource_id="res-earlier-a")
            assert registry.manifest() == ()
            second = await _call(read, session_id, resource_id="res-earlier-b")
            assert second.is_error is False, "the slot was given back and used again"

        rows = await _run_rows(db, claim.run_id)
        assert [_kind(row) for row in rows].count(LINEAGE_ADOPTION_KIND) == 2
        resumed = await _resumed(store, db, claim.run_id, max_attachments=1)
        try:
            first = await resumed.read("res-earlier-a", max_window_tokens=1000)
            assert EARLIER_TEXT in first.content
            again = await resumed.read("res-earlier-b", max_window_tokens=1000)
            assert "The second document." in again.content
        finally:
            await resumed.aclose()


async def test_two_handles_for_one_document_record_one_row_with_both_aliases() -> None:
    """Identical bytes printed under two handles are one Resource with one view.

    The second adoption merges its alias into the row the first wrote; recording an
    adoption again changes nothing; a resume serves both handles the first view.
    """
    async with isolated_run_runtime("resource_lineage_aliases") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store, document_id="res-earlier-one")
        await _seed_origin_run(
            db, store, session_id=session_id, document_id="res-earlier-two", text="View two."
        )
        claim = await _claimed_run(store)
        loader = _loader(store, db, session_id, claim)
        # A Run mints the same handles after a resume: its Resource secret is its own.
        run_secret = b"one run's resource secret"
        async with ResourceRegistry(resource_secret=run_secret) as registry:
            read, _ = _tools(registry, loader)
            one = await _call(read, session_id, resource_id="res-earlier-one")
            two = await _call(read, session_id, resource_id="res-earlier-two")
            assert (one.is_error, two.is_error) == (False, False)
            assert EARLIER_TEXT in two.text_content, "the first view serves both handles"
            canonical = printed_handle(one)
            assert printed_handle(two) == canonical

        before = await _run_rows(db, claim.run_id)
        adoptions = [row for row in before if _kind(row) == LINEAGE_ADOPTION_KIND]
        assert [row["resource_id"] for row in adoptions] == [canonical]
        assert json.loads(adoptions[0]["capabilities"])["resource_aliases"] == [
            "res-earlier-one",
            "res-earlier-two",
        ]
        assert [_kind(row) for row in before].count(SNAPSHOT_KIND) == 1

        # The same adoption recorded again, as a retried write would, changes nothing.
        loaded = await loader.load("res-earlier-one")
        assert loaded is not None
        owner = ResourceEffectOwner(execution_scope=session_id, intent_id=IntentId.new())
        async with ResourceRegistry(resource_secret=run_secret) as registry:
            again = await adopt_lineage_resource(
                registry, loaded, record=partial(loader.record, owner=owner)
            )
            assert again == canonical
        after = await _run_rows(db, claim.run_id)
        assert [(row["resource_id"], row["blob_digest"]) for row in after] == [
            (row["resource_id"], row["blob_digest"]) for row in before
        ]

        resumed = await _resumed(store, db, claim.run_id, resource_secret=run_secret)
        try:
            for handle in ("res-earlier-one", "res-earlier-two"):
                text = await resumed.read(handle, max_window_tokens=1000)
                assert text.resource_id == canonical
                assert EARLIER_TEXT in text.content
        finally:
            await resumed.aclose()


async def test_a_second_view_for_one_resource_rolls_the_whole_adoption_back() -> None:
    """The store keeps one view per Resource: a different one refuses, row and all."""
    async with isolated_run_runtime("resource_lineage_conflict") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)
        claim = await _claimed_run(store)
        loader = _loader(store, db, session_id, claim)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, loader)
            shown = await _call(read, session_id, resource_id="res-earlier-document")
            assert shown.is_error is False
            canonical = printed_handle(shown)
        before = await _run_rows(db, claim.run_id)

        loaded = await loader.load("res-earlier-document")
        assert loaded is not None
        other = ConversionSnapshot(
            resource_id=canonical,
            input_digest=hashlib.sha256(DOCUMENT).hexdigest(),
            text="Another view of the same bytes.",
            visuals=(),
            extraction_status="complete",
            converter="fixture",
            converter_version="2",
        )
        adoption = ResourceAttachmentBytes(
            resource_id=canonical,
            filename=loaded.filename,
            mime_type=loaded.media_type,
            source_locator=canonical,
            content=loaded.content,
            resource_kind=LINEAGE_ADOPTION_KIND,
            aliases=("res-another-handle",),
        )
        owner = ResourceEffectOwner(execution_scope=session_id, intent_id=IntentId.new())
        with pytest.raises(LineageAdoptionConflict):
            await loader.record((adoption, *other.effects()), owner)

        assert await _run_rows(db, claim.run_id) == before, "the alias merge rolled back too"


async def test_a_later_turn_adopts_through_the_handle_an_adoption_printed() -> None:
    """Turn 3 names the handle turn 2's read printed, and resumes without the lineage.

    Turn 2 printed its own canonical handle for the document turn 1 attached. That
    adoption's row and view are turn 2's, stamped with the Session, so turn 3 adopts
    them by that handle like any other Resource the conversation holds.
    """
    async with isolated_run_runtime("resource_lineage_chain") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)

        second = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, second))
            shown = await _call(read, session_id, resource_id="res-earlier-document")
        assert shown.is_error is False
        handle = printed_handle(shown)
        assert handle != "res-earlier-document"

        third = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, third))
            result = await _call(read, session_id, resource_id=handle)
            assert result.is_error is False, result.text_content
            assert EARLIER_TEXT in result.text_content
            canonical = printed_handle(result)
            assert canonical not in {handle, "res-earlier-document"}

        resumed = await _resumed(store, db, third.run_id)
        try:
            for name in (handle, canonical):
                text = await resumed.read(name, max_window_tokens=1000)
                assert EARLIER_TEXT in text.content
        finally:
            await resumed.aclose()


async def test_a_later_turn_adopts_what_a_child_adopted_for_the_conversation() -> None:
    """A Child's adoption is the conversation's: the next turn adopts it by the handle.

    The Child adopts first, the parent reads the same document and prints this Run's
    handle, and the next turn names that handle. The rows carry the Agent Session
    whose lineage admitted the read, not the Child Session that asked.
    """
    async with isolated_run_runtime("resource_lineage_child") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)

        second = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, second))
            by_child = await _call(read, str(uuid.uuid4()), resource_id="res-earlier-document")
            by_parent = await _call(read, session_id, resource_id="res-earlier-document")
        assert (by_child.is_error, by_parent.is_error) == (False, False)
        handle = printed_handle(by_parent)

        third = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, third))
            result = await _call(read, session_id, resource_id=handle)
        assert result.is_error is False, result.text_content
        assert EARLIER_TEXT in result.text_content


async def test_an_adoption_outlives_the_run_it_came_from() -> None:
    """The adopting Run holds its own references, so deleting the origin keeps the bytes."""
    async with isolated_run_runtime("resource_lineage_origin") as (_, db):
        store = await _store(db)
        session_id, origin_run = await _seed_origin_run(db, store)
        claim = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, claim))
            assert (
                await _call(read, session_id, resource_id="res-earlier-document")
            ).is_error is False

        deletion = await delete_runs(db, store, owner_id=OWNER, run_ids=[origin_run])
        assert deletion.runs == 1

        resumed = await _resumed(store, db, claim.run_id)
        try:
            text = await resumed.read("res-earlier-document", max_window_tokens=1000)
            assert EARLIER_TEXT in text.content
        finally:
            await resumed.aclose()


async def test_a_purge_of_the_origin_during_the_adoption_write_keeps_the_blobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A purge that commits while the adoption is written cannot take its Blobs.

    The purge sees no committed row naming the adopted Blobs yet; without the hold it
    deleted the bytes the adoption's rows then named, and the resume failed.
    """
    from dlightrag.adapters.postgres.answer import session_repository

    async with isolated_run_runtime("resource_lineage_purge") as (_, db):
        store = await _store(db)
        session_id, origin_run = await _seed_origin_run(db, store)
        claim = await _claimed_run(store)
        written = session_repository.write_fetched_resources
        purges: list[Any] = []

        async def write_then_purge_the_origin(conn: Any, **write: Any) -> None:
            await written(conn, **write)
            # Retention purges the origin Run in its own transaction before this commit.
            purges.append(await delete_runs(db, store, owner_id=OWNER, run_ids=[origin_run]))

        monkeypatch.setattr(
            session_repository, "write_fetched_resources", write_then_purge_the_origin
        )
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, claim))
            assert (
                await _call(read, session_id, resource_id="res-earlier-document")
            ).is_error is False
        assert [purge.runs for purge in purges] == [1]

        resumed = await _resumed(store, db, claim.run_id)
        try:
            text = await resumed.read("res-earlier-document", max_window_tokens=1000)
            assert EARLIER_TEXT in text.content
        finally:
            await resumed.aclose()


async def test_adopts_the_artifact_a_conversation_published_earlier() -> None:
    """A published product is a Resource: the next turn reads the version it published.

    This is the whole point of registering it — the address is the path hashed, so the
    Tool can hand the model a handle before the publication exists, and the bytes a
    later Run of the same Session adopts are the version that was published.
    """
    from dlightrag.engine.answer.publication import artifact_resource_id
    from dlightrag.engine.runtime.records import PendingPublication

    async with isolated_run_runtime("resource_lineage_artifact") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        accepted = await store.accept_run(
            envelope=run_envelope("answer", key=f"artifact-{uuid.uuid4().hex[:8]}", owner=OWNER),
            run_id=str(uuid.uuid7()),
            connection_bindings=(),
        )
        run_id = str(accepted.run.run_id)
        claim = await store.claim_next(worker_id="artifact-worker")
        assert claim is not None
        resource_id = artifact_resource_id("reports/analysis.md")
        content = b"# analysis\nversion one\n"

        outcome = await store.finish_success(
            owner_id=OWNER,
            run_id=run_id,
            worker_id="artifact-worker",
            fencing_epoch=claim.run.fencing_epoch,
            result={"answer": "published"},
            publications=(
                PendingPublication(
                    resource_id=resource_id,
                    reference_kind="published_artifact",
                    filename="analysis.md",
                    mime_type="text/markdown",
                    content=content,
                    session_id=session_id,
                    relative_path="reports/analysis.md",
                    presentation="markdown",
                    label="analysis.md",
                ),
            ),
        )
        assert outcome is not None

        async with db.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT kind, blob_digest, session_id::text AS session_id, capabilities"
                " FROM dlightrag_answer_resources WHERE owner_id = $1 AND resource_id = $2",
                OWNER,
                resource_id,
            )
        assert row is not None
        assert row["kind"] == "published_artifact"
        assert row["blob_digest"] == hashlib.sha256(content).hexdigest()
        assert row["session_id"] == session_id
        capabilities = json.loads(row["capabilities"])
        assert capabilities["resource_kind"] == "published_artifact"
        assert capabilities["artifact_path"] == "reports/analysis.md"
        assert capabilities["presentation"] == "markdown"

        loader = _loader(store, db, session_id, await _claimed_run(store))
        loaded = await loader.load(resource_id)
        assert loaded is not None
        assert loaded.content == content
        assert loaded.origin_run_id == run_id
        assert loaded.filename == "analysis.md"

        # The call the Tool teaches must actually read it: a product has no conversion
        # route, so the reader decodes the adopted bytes instead of demanding a view.
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, loader)
            result = await _call(read, session_id, resource_id=resource_id)
        assert result.is_error is False, result.text_content
        assert "version one" in result.text_content

        # Another conversation cannot reach it, and neither can another owner.
        assert await _loader(store, db, str(uuid.uuid4())).load(resource_id) is None
        assert (
            await _loader(store, db, session_id, owner_id="another-owner").load(resource_id) is None
        )


async def test_adoption_reads_the_newest_published_version_of_one_path() -> None:
    """Republishing a path makes a new version; the handle reads the newest one.

    The address is derived from the path, so every version of one Artifact shares it.
    Without a recency tie-break the read could return the version the conversation
    published first, which is the opposite of iterating on a deliverable.
    """
    from dlightrag.engine.answer.publication import artifact_resource_id
    from dlightrag.engine.runtime.records import PendingPublication

    async with isolated_run_runtime("resource_lineage_versions") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        resource_id = artifact_resource_id("reports/analysis.md")

        async def publish(version: int) -> str:
            accepted = await store.accept_run(
                envelope=run_envelope(
                    "answer", key=f"version-{version}-{uuid.uuid4().hex[:8]}", owner=OWNER
                ),
                run_id=str(uuid.uuid7()),
                connection_bindings=(),
            )
            run_id = str(accepted.run.run_id)
            claim = await store.claim_next(worker_id=f"version-worker-{version}")
            assert claim is not None
            await store.finish_success(
                owner_id=OWNER,
                run_id=run_id,
                worker_id=f"version-worker-{version}",
                fencing_epoch=claim.run.fencing_epoch,
                result={"answer": f"version {version}"},
                publications=(
                    PendingPublication(
                        resource_id=resource_id,
                        reference_kind="published_artifact",
                        filename="analysis.md",
                        mime_type="text/markdown",
                        content=f"# version {version}\n".encode(),
                        session_id=session_id,
                        relative_path="reports/analysis.md",
                        presentation="markdown",
                        label=f"version {version}",
                    ),
                ),
            )
            return run_id

        await publish(1)
        newest_run = await publish(2)

        loaded = await _loader(store, db, session_id).load(resource_id)

        assert loaded is not None
        assert loaded.content == b"# version 2\n"
        assert loaded.origin_run_id == newest_run


async def test_the_session_stamp_decides_admission() -> None:
    async with isolated_run_runtime("resource_lineage_scope") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)
        other_session = _loader(store, db, str(uuid.uuid4()))
        other_owner = _loader(store, db, session_id, owner_id="another-owner")

        assert await other_session.load("res-earlier-document") is None
        assert await other_owner.load("res-earlier-document") is None

        # Only the document and its own view rows come back: no unrelated Resource
        # of the same Session can be reached by naming it.
        rows = await store.lineage_resource_rows(
            owner_id=OWNER, session_id=session_id, resource_id="res-unrelated"
        )
        assert rows == ()

        document_rows = await store.lineage_resource_rows(
            owner_id=OWNER, session_id=session_id, resource_id="res-earlier-document"
        )
        kinds = {str(row.capabilities.get("resource_kind")) for row in document_rows}
        assert kinds == {"tool_attachment", SNAPSHOT_KIND, "conversion_asset"}


async def test_a_document_without_a_stored_view_is_still_adoptable() -> None:
    async with isolated_run_runtime("resource_lineage_no_view") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        accepted = await store.accept_run(
            envelope=run_envelope("answer", key=f"no-view-{uuid.uuid4().hex[:8]}", owner=OWNER),
            run_id=str(uuid.uuid7()),
            connection_bindings=(),
        )
        async with db.acquire() as conn, conn.transaction():
            digest = hashlib.sha256(DOCUMENT).hexdigest()
            await write_blob_content(conn, owner_id=OWNER, digest=digest, content=DOCUMENT)
            await conn.execute(
                """
                INSERT INTO dlightrag_answer_resources (
                    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
                    ordinal, blob_digest, locator_digest, source_locator, session_id
                ) VALUES ($1,$2,'res-no-view','fetched_blob','plain.pdf','application/pdf',
                          $3::jsonb,0,$4,$5,$6,$7)
                """,
                OWNER,
                uuid.UUID(str(accepted.run.run_id)),
                json.dumps({"resource_kind": "web", "admission_origin": "search"}),
                digest,
                hashlib.sha256(b"https://example.com/plain.pdf").hexdigest(),
                b"https://example.com/plain.pdf",
                uuid.UUID(session_id),
            )

        loaded = await _loader(store, db, session_id).load("res-no-view")

        assert loaded is not None
        assert loaded.conversion_snapshot is None
        assert loaded.source_url == "https://example.com/plain.pdf"


@pytest.mark.parametrize("resource_id", ["", "res-unknown"])
async def test_unknown_handles_load_nothing(resource_id: str) -> None:
    async with isolated_run_runtime("resource_lineage_unknown") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)

        assert await _loader(store, db, session_id).load(resource_id) is None


def _product(path: str, media_type: str, content: bytes) -> Any:
    from dlightrag.engine.answer.publication import StagedArtifact, artifact_resource_id

    return StagedArtifact(
        relative_path=path,
        media_type=media_type,
        size_bytes=len(content),
        resource_id=artifact_resource_id(path),
        filename=path.rsplit("/", 1)[-1],
        digest=hashlib.sha256(content).hexdigest(),
        presentation="download",
        content=content,
    )


def _workbook(revenue: int) -> bytes:
    import io

    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.append(["quarter", "revenue"])
    sheet.append(["Q1", revenue])
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


async def _publish(store: PGRunStore, session_id: str, products: tuple[Any, ...]) -> str:
    """Publish products the way the executor does: views built, staged, committed."""
    from dlightrag.engine.answer.execution.executor import (
        _publication_views,
        _stage_publications,
    )
    from dlightrag.engine.answer.publication import PublicationPlan

    claim = await _claimed_run(store, worker_id=f"publisher-{uuid.uuid4().hex[:8]}")
    async with ResourceRegistry() as registry:
        views = await _publication_views(products, registry=registry)
    publications, _descriptors, _sources = _stage_publications(
        plan=PublicationPlan(answer="published", artifacts=products),
        answer="published",
        session_id=session_id,
        views=views,
    )
    outcome = await store.finish_success(
        owner_id=OWNER,
        run_id=claim.run_id,
        worker_id=claim.worker_id,
        fencing_epoch=claim.fencing_epoch,
        result={"answer": "published"},
        publications=tuple(publications),
    )
    assert outcome is not None
    return claim.run_id


async def test_a_later_turn_reads_a_convertible_product_through_its_published_view() -> None:
    """The view publication stored is the view a later turn adopts and reads."""
    async with isolated_run_runtime("resource_lineage_product_view") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        product = _product("reports/model.xlsx", _XLSX, _workbook(11))
        publishing_run = await _publish(store, session_id, (product,))

        rows = await _run_rows(db, publishing_run)
        kinds = sorted(json.loads(row["capabilities"])["resource_kind"] for row in rows)
        assert kinds == ["conversion_snapshot", "published_artifact"]
        view_row = next(row for row in rows if row["resource_id"].endswith("-conversion"))
        assert bytes(view_row["source_locator"]) == product.resource_id.encode()
        assert (view_row["session_id"], view_row["intent_id"]) == (session_id, None)

        claim = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, claim))
            result = await _call(read, session_id, resource_id=product.resource_id)

        assert result.is_error is False, result.text_content
        assert "Q1" in result.text_content and "11" in result.text_content
        # The reading turn now holds the product and its view as its own Resources.
        adopted = {
            json.loads(row["capabilities"])["resource_kind"]
            for row in await _run_rows(db, claim.run_id)
        }
        assert adopted == {LINEAGE_ADOPTION_KIND, SNAPSHOT_KIND}


async def test_a_product_that_cannot_be_converted_publishes_and_keeps_the_refusal() -> None:
    async with isolated_run_runtime("resource_lineage_product_no_view") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        product = _product("reports/model.xlsx", _XLSX, b"not a workbook")
        publishing_run = await _publish(store, session_id, (product,))

        rows = await _run_rows(db, publishing_run)
        assert [json.loads(row["capabilities"])["resource_kind"] for row in rows] == [
            "published_artifact"
        ]

        claim = await _claimed_run(store)
        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, _loader(store, db, session_id, claim))
            refused = await _call(read, session_id, resource_id=product.resource_id)

        assert refused.is_error is True
        assert "never extracted text from model.xlsx" in refused.text_content


async def test_a_newer_version_published_without_a_view_never_reads_an_older_ones() -> None:
    """Every version of one path shares its handle; each keeps only its own view."""
    async with isolated_run_runtime("resource_lineage_product_versions") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        older = _product("reports/model.xlsx", _XLSX, _workbook(11))
        newer = _product("reports/model.xlsx", _XLSX, b"not a workbook")
        await _publish(store, session_id, (older,))
        newest_run = await _publish(store, session_id, (newer,))

        loader = _loader(store, db, session_id, await _claimed_run(store))
        loaded = await loader.load(newer.resource_id)
        assert loaded is not None
        assert (loaded.content, loaded.origin_run_id) == (newer.content, newest_run)
        assert loaded.conversion_snapshot is None

        async with ResourceRegistry() as registry:
            read, _ = _tools(registry, loader)
            refused = await _call(read, session_id, resource_id=newer.resource_id)

        # The refusal that says what is true of the newest version, not an older view
        # that does not belong to its bytes.
        assert refused.is_error is True
        assert "never extracted text" in refused.text_content


async def test_one_publication_writes_every_blob_in_one_order_and_its_views_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two commits that share bytes must never lock them in crossed orders.

    Every Blob the commit names, products and views alike, is written in one sorted
    pass, and the views of all its products are written in one call.
    """
    import base64
    import io as _io

    from PIL import Image

    from dlightrag.adapters.postgres.runtime import run_store as run_store_module

    written: list[str] = []
    view_writes: list[int] = []
    write_blob_content = run_store_module.write_blob_content
    write_fetched_resources = run_store_module.write_fetched_resources

    async def recording_blob(conn: Any, *, owner_id: str, digest: str, content: bytes) -> None:
        written.append(digest)
        await write_blob_content(conn, owner_id=owner_id, digest=digest, content=content)

    async def recording_views(conn: Any, **write: Any) -> None:
        view_writes.append(len(write["updates"]))
        await write_fetched_resources(conn, **write)

    monkeypatch.setattr(run_store_module, "write_blob_content", recording_blob)
    monkeypatch.setattr(run_store_module, "write_fetched_resources", recording_views)
    buffer = _io.BytesIO()
    Image.new("RGB", (6, 6), (20, 90, 200)).save(buffer, "PNG")
    chart = base64.b64encode(buffer.getvalue()).decode()
    report = _product(
        "reports/report.html",
        "text/html",
        (
            f'<html><body><p>Q1</p><img alt="chart" src="data:image/png;base64,{chart}"></body></html>'
        ).encode(),
    )
    model = _product("reports/model.xlsx", _XLSX, _workbook(11))

    async with isolated_run_runtime("resource_lineage_product_order") as (_, db):
        store = await _store(db)
        session_id = str(uuid.uuid4())
        publishing_run = await _publish(store, session_id, (report, model))

        rows = await _run_rows(db, publishing_run)
        kinds = sorted(json.loads(row["capabilities"])["resource_kind"] for row in rows)
        assert kinds == [
            "conversion_asset",
            "conversion_snapshot",
            "conversion_snapshot",
            "published_artifact",
            "published_artifact",
        ]
        view_digests = {row["blob_digest"] for row in rows} - {report.digest, model.digest}
        # One sorted pass over every Blob, the views' among them, before any row.
        assert written == sorted(set(written))
        assert view_digests <= set(written)
        # The three view rows of both products, written in one call.
        assert view_writes == [3]
