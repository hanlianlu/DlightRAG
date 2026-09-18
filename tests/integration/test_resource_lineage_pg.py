# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lineage adoption against real PostgreSQL rows and blobs."""

import hashlib
import json
import uuid
from typing import Any

import pytest

from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore, write_blob_content
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.answer.execution.lineage import RetainedResourceLoader
from dlightrag.engine.answer.resources.converters import ExtractedVisual
from dlightrag.engine.answer.resources.lineage import SNAPSHOT_KIND
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope

OWNER = "lineage-owner"
DOCUMENT = b"%PDF-1.7 an earlier run's document"
PAGE = b"\x89PNG\r\n\x1a\npage-one"


async def _store(db: Any) -> PGRunStore:
    """Establish the complete operational schema exactly as a real process does."""
    created = PGRunStore(pool=db)
    await created.initialize()
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore

    await PGWebConversationStore(pool=db, run_store=created).initialize()
    return created


async def _seed_origin_run(db: Any, store: PGRunStore) -> tuple[str, str]:
    """Write one earlier Run's document, its stored view, and its page asset.

    The origin Run exists as a real accepted Run, because a Resource row is only
    meaningful for a Run that owns it.
    """
    session_id = str(uuid.uuid4())
    accepted = await store.accept_run(
        envelope=run_envelope("answer", key=f"lineage-{uuid.uuid4().hex[:8]}", owner=OWNER),
        run_id=str(uuid.uuid7()),
        connection_bindings=(),
    )
    origin_run = str(accepted.run.run_id)
    document_id = "res-earlier-document"
    snapshot = ConversionSnapshot(
        resource_id=document_id,
        input_digest=hashlib.sha256(DOCUMENT).hexdigest(),
        text="Text the earlier run already extracted.",
        visuals=(
            ExtractedVisual(
                handle_id="page-1",
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
        (document_id, "tool_attachment", "earlier.pdf", "application/pdf", DOCUMENT, document_id)
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


async def test_adopts_an_earlier_runs_document_and_its_stored_view() -> None:
    async with isolated_run_runtime("resource_lineage") as (_, db):
        store = await _store(db)
        session_id, origin_run = await _seed_origin_run(db, store)
        loader = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=session_id,
        )

        loaded = await loader.load("res-earlier-document")

        assert loaded is not None
        assert loaded.content == DOCUMENT
        assert loaded.origin_run_id == origin_run
        assert loaded.filename == "earlier.pdf"
        assert loaded.conversion_snapshot is not None
        assert loaded.assets == {"page-1": PAGE}

        async with ResourceRegistry() as registry:
            from dlightrag.engine.answer.resources.lineage import adopt_lineage_resource

            adopted = adopt_lineage_resource(registry, loaded)
            assert adopted.resource_id != "res-earlier-document"
            assert registry.canonical_resource_id("res-earlier-document") == adopted.resource_id
            assert adopted.snapshot is not None
            text = await registry.read(adopted.resource_id, max_window_tokens=1000)
            assert "Text the earlier run already extracted." in text.content


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

        loader = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=session_id,
        )
        loaded = await loader.load(resource_id)
        assert loaded is not None
        assert loaded.content == content
        assert loaded.origin_run_id == run_id
        assert loaded.filename == "analysis.md"

        # The call the Tool teaches must actually read it: a product has no conversion
        # route, so the reader decodes the adopted bytes instead of demanding a view.
        from dlightrag.engine.agent.environment.access import AccessScheduler
        from dlightrag.engine.agent.tools.files import read_tool
        from dlightrag.engine.answer.resources.models import TextWindowBudget
        from dlightrag.engine.answer.resources.registry import ResourceRegistry
        from dlightrag.engine.answer.tools.resources import make_resource_reader
        from tests.tool_helpers import tool_runtime

        async with ResourceRegistry() as registry:
            tool = read_tool(
                None,
                AccessScheduler(),
                resource_reader=make_resource_reader(
                    registry, TextWindowBudget(1000), lineage=loader
                ),
            )
            read = await tool.execute(
                tool.input_model.model_validate({"resource_id": resource_id}),
                tool_runtime(tool_name=tool.name),
            )
        assert read.is_error is False, read.text_content
        assert "version one" in read.text_content

        # Another conversation cannot reach it, and neither can another owner.
        other_session = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=str(uuid.uuid4()),
        )
        assert await other_session.load(resource_id) is None
        other_owner = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id="another-owner",
            session_id=session_id,
        )
        assert await other_owner.load(resource_id) is None


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

        loader = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=session_id,
        )
        loaded = await loader.load(resource_id)

        assert loaded is not None
        assert loaded.content == b"# version 2\n"
        assert loaded.origin_run_id == newest_run


async def test_the_session_stamp_decides_admission() -> None:
    async with isolated_run_runtime("resource_lineage_scope") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)
        other_session = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=str(uuid.uuid4()),
        )
        other_owner = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id="another-owner",
            session_id=session_id,
        )

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
        loader = RetainedResourceLoader(
            store=store, blobs=PGRunBlobStore(pool=db), owner_id=OWNER, session_id=session_id
        )

        loaded = await loader.load("res-no-view")

        assert loaded is not None
        assert loaded.conversion_snapshot is None
        assert loaded.source_url == "https://example.com/plain.pdf"


@pytest.mark.parametrize("resource_id", ["", "res-unknown"])
async def test_unknown_handles_load_nothing(resource_id: str) -> None:
    async with isolated_run_runtime("resource_lineage_unknown") as (_, db):
        store = await _store(db)
        session_id, _ = await _seed_origin_run(db, store)
        loader = RetainedResourceLoader(
            store=store,
            blobs=PGRunBlobStore(pool=db),
            owner_id=OWNER,
            session_id=session_id,
        )

        assert await loader.load(resource_id) is None
