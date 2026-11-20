# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The acceptance test ADR 0049 asks for: export, then erase with zero residue.

Seeded across the real stores — a Run with an Artifact, a Web conversation, a Memory
record, the contribution switch, a workspace tree and an owner Skills shard — the
erase is then checked against every plane the schema declares, not against a list,
so a table that gains an `owner_id` tomorrow fails this test until it is covered.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from dlightrag_memory import Memory, MemoryOperation, MemoryProvenance
from dlightrag_memory.postgres import PostgresMemoryStore

from dlightrag.adapters.postgres.corpus.workspaces import PGWorkspaceRegistry
from dlightrag.adapters.postgres.owner_data import (
    created_corpus_workspaces,
    erase_owner,
    export_owner,
)
from dlightrag.adapters.postgres.owner_data_corpora import corpus_deletion_deliverer
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
from dlightrag.engine.agent.skills import owner_skill_root
from dlightrag.engine.answer.agentspace import run_root
from dlightrag.engine.runtime.records import (
    PendingArtifact,
    PreparedRunEnvelope,
    RunAccessScope,
)
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

OWNER = "c" * 64
_MEMORY_BODY = "The owner prefers charts in a dark theme."


def _envelope(prepared_input: Mapping[str, Any], *, submission_key: str) -> PreparedRunEnvelope:
    return PreparedRunEnvelope(
        run_kind="answer",
        lane="query",
        submitted_by=OWNER,
        access_scope=RunAccessScope(kind="owner", scope_id=OWNER),
        submission_key=submission_key,
        request_fingerprint="f" * 64,
        payload=prepared_input,
        accepted_input=prepared_input,
        retention_seconds=365 * 24 * 60 * 60,
    )


async def _seed(pool: Any, *, workspace_root: Path, skills_root: Path) -> str:
    """Put one row of each personal plane in place, and return the Run id."""
    store = PGRunStore(pool=pool)
    run_id = str(uuid.uuid4())
    await store.accept_run(
        envelope=_envelope(
            {
                "agent_session_id": str(uuid.uuid4()),
                "agent_lane_id": "default",
                "source_lane_id": None,
                "fingerprint": "f" * 64,
                "query": "which chart theme?",
                "workspaces": ["default"],
                "schema_version": 1,
            },
            submission_key=run_id,
        ),
        run_id=run_id,
        artifacts=[PendingArtifact(content=b"artifact bytes")],
    )
    # An owner erase refuses while a Run is in flight, so the seed finishes its Run
    # the way the coordinator would before the request arrives.
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE dlightrag_runs
            SET status = 'cancelled', finished_at = NOW(), prepared_input_json = NULL
            WHERE run_id = $1::uuid
            """,
            run_id,
        )
    conversations = PGWebConversationStore(pool=pool)
    await conversations.create_conversation(OWNER)
    memory = Memory(PostgresMemoryStore(pool=pool))
    await memory.remember(
        owner_id=OWNER,
        kind="preference",
        body=_MEMORY_BODY,
        provenance=MemoryProvenance(origin_kind="management", origin_id=run_id),
        idempotency_key=f"seed:{run_id}",
    )
    await store.write_privacy_settings(owner_id=OWNER, contribute=True)
    async with pool.acquire() as conn:
        # An Agent account is sealed credentials plus metadata; the metadata is data
        # the subject is owed, and the envelope is what an export withholds.
        await conn.execute(
            """
            INSERT INTO dlightrag_agent_accounts
                (owner_id, site, account_id, email, username, key_id, encrypted_envelope)
            VALUES ($1, 'example.test', 'acct-1', 'owner@example.test', 'owner', 'key-1', 'sealed')
            """,
            OWNER,
        )
    tree = run_root(workspace_root, OWNER, run_id)
    tree.mkdir(parents=True)
    shard = tree.parent
    # A tree whose Run row is already gone: nothing attributes it to this owner, so
    # the erase names it instead of deleting a neighbour's tree.
    (shard / "orphan-tree").mkdir()
    skill = owner_skill_root(skills_root, OWNER) / "mine" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: mine\ndescription: the owner's own skill\n---\n\nBody.\n")
    return run_id


async def independent_residue(conn: Any, owner_id: str) -> dict[str, int]:
    """Where the owner still appears, asked of every column rather than of our list.

    A discovery hole cannot hide here: this walks `information_schema` itself, so a
    future column that names a person some other way still fails the test.
    """
    columns = await conn.fetch(
        """
        SELECT table_name, column_name, data_type FROM information_schema.columns
        WHERE table_schema = 'public' ORDER BY table_name, ordinal_position
        """
    )
    found: dict[str, int] = {}
    for row in columns:
        table, column, kind = str(row["table_name"]), str(row["column_name"]), str(row["data_type"])
        if table == "dlightrag_owner_data_requests":
            continue  # the ledger is the evidence, and naming the owner is its job
        if kind in {"text", "character varying", "uuid"}:
            predicate = f'"{column}"::text = $1'
        elif kind in {"json", "jsonb"} or kind == "ARRAY":
            predicate = f""" "{column}"::text LIKE concat('%', $1::text, '%')""".strip()
        else:
            continue
        count = await conn.fetchval(
            f'SELECT count(*) FROM "{table}" WHERE {predicate}',  # noqa: S608
            owner_id,
        )
        if count:
            found[f"{table}.{column}"] = int(count)
    return found


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()
    database = f"dlightrag_owner_data_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=4
    )
    try:
        await PGRunStore(pool=created).initialize()
        await PostgresMemoryStore(pool=created).initialize()
        yield created
    finally:
        await created.close()
        await drop_database(database)


async def test_export_and_erase_leave_no_residue(pool: Any, tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspaces"
    skills_root = tmp_path / "skills"
    await _seed(pool, workspace_root=workspace_root, skills_root=skills_root)

    async with pool.acquire() as conn:
        exported = await export_owner(
            conn, owner_id=OWNER, out_dir=tmp_path / "package", owner_skills_root=skills_root
        )
    assert exported.rows["dlightrag_runs"] == 1
    assert exported.rows["web_conversations"] == 1
    assert exported.rows["dlightrag_memory_records"] == 1
    assert exported.blob_count == 1
    assert _MEMORY_BODY in (exported.path / "dlightrag_memory_records.jsonl").read_text()
    assert (exported.path / "skills" / "mine" / "SKILL.md").is_file()
    assert "sealed" not in (exported.path / "dlightrag_agent_accounts.jsonl").read_text()
    manifest = json.loads((exported.path / "manifest.json").read_text())
    assert manifest["skill_files"] == 1
    assert "key ring" in manifest["note"]

    erased_owners: list[str] = []

    async def telemetry(*, owner_id: str) -> int:
        erased_owners.append(owner_id)
        return 4

    async with pool.acquire() as conn:
        erased = await erase_owner(
            conn,
            owner_id=OWNER,
            telemetry=telemetry,
            workspace_root=workspace_root,
            owner_skills_root=skills_root,
            line="b2c",
            region="eu",
        )
        residue = await independent_residue(conn, OWNER)
        receipts = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_owner_data_requests WHERE owner_id = $1", OWNER
        )

    assert residue == {}
    assert erased.deleted["dlightrag_runs"] >= 1
    assert erased.deleted["dlightrag_agent_accounts"] == 1
    assert erased.telemetry_rows == 4
    assert erased_owners == [OWNER]
    # The ledger of the two requests is the evidence, and it is all that stays.
    assert int(receipts) == 2
    assert not owner_skill_root(skills_root, OWNER).exists()
    # The orphan tree is named, not deleted: a shard is a hash prefix owners share.
    assert any("orphan" in entry for entry in erased.left_in_place)
    assert (run_root(workspace_root, OWNER, "any").parent / "orphan-tree").is_dir()


async def test_the_owners_words_behind_a_memory_are_exported_with_it_and_erased_with_it(
    pool: Any, tmp_path: Path
) -> None:
    """A quote is the owner's own text, stored beside the record and in the journal's snapshots."""
    memory = Memory(PostgresMemoryStore(pool=pool))
    provenance = MemoryProvenance(origin_kind="session_pass", origin_id="s:1-2", run_id="run-1")

    def operation(key: str, body: str, quote: str, supersedes_id: str | None = None):
        return MemoryOperation(
            owner_id=OWNER,
            idempotency_key=key,
            action="remember",
            provenance=provenance,
            kind="preference",
            body=body,
            supersedes_id=supersedes_id,
            quote=quote,
        )

    first = await memory.apply(operation("one", "Prefers dark charts.", "WORDS-ABOUT-DARK-CHARTS"))
    await memory.apply(
        operation("two", "Prefers dark charts, no pies.", "WORDS-ABOUT-NO-PIES", first.memory_id)
    )

    async with pool.acquire() as conn:
        exported = await export_owner(conn, owner_id=OWNER, out_dir=tmp_path / "package")
        before = {
            needle: await independent_residue(conn, needle)
            for needle in ("WORDS-ABOUT-DARK-CHARTS", "WORDS-ABOUT-NO-PIES")
        }
        await erase_owner(
            conn,
            owner_id=OWNER,
            telemetry=None,
            workspace_root=tmp_path / "workspaces",
            owner_skills_root=tmp_path / "skills",
        )
        after = {
            needle: await independent_residue(conn, needle)
            for needle in ("WORDS-ABOUT-DARK-CHARTS", "WORDS-ABOUT-NO-PIES")
        }

    records = (exported.path / "dlightrag_memory_records.jsonl").read_text()
    rows = [json.loads(line) for line in records.splitlines()]
    # Both rows of the record's history go out with the words each stood on, and the journal's
    # snapshot of the replaced one carries its words too.
    assert {row["quote"] for row in rows} == {"WORDS-ABOUT-DARK-CHARTS", "WORDS-ABOUT-NO-PIES"}
    assert (
        "WORDS-ABOUT-DARK-CHARTS"
        in (exported.path / "dlightrag_memory_operations.jsonl").read_text()
    )
    # Where they are stored is where the erase finds them, with no list of columns to keep.
    assert before["WORDS-ABOUT-NO-PIES"] == {"dlightrag_memory_records.quote": 1}
    assert set(before["WORDS-ABOUT-DARK-CHARTS"]) == {
        "dlightrag_memory_records.quote",
        "dlightrag_memory_operations.before_records",
    }
    assert after == {"WORDS-ABOUT-DARK-CHARTS": {}, "WORDS-ABOUT-NO-PIES": {}}


async def test_erase_refuses_while_a_run_is_queued(pool: Any, tmp_path: Path) -> None:
    """A Run still in flight would write back what the erase removed."""
    from dlightrag.adapters.postgres.owner_data import OwnerEraseRefused

    store = PGRunStore(pool=pool)
    run_id = str(uuid.uuid4())
    await store.accept_run(
        envelope=_envelope(
            {
                "agent_session_id": str(uuid.uuid4()),
                "agent_lane_id": "default",
                "source_lane_id": None,
                "fingerprint": "f" * 64,
                "query": "still running?",
                "workspaces": ["default"],
                "schema_version": 1,
            },
            submission_key=run_id,
        ),
        run_id=run_id,
    )
    async with pool.acquire() as conn:
        with pytest.raises(OwnerEraseRefused):
            await erase_owner(
                conn,
                owner_id=OWNER,
                telemetry=None,
                workspace_root=tmp_path,
                owner_skills_root=tmp_path / "skills",
            )
        outcome = await conn.fetchval(
            "SELECT outcome FROM dlightrag_owner_data_requests WHERE owner_id = $1", OWNER
        )
        still_there = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_runs WHERE owner_id = $1", OWNER
        )
    assert outcome == "refused"
    assert int(still_there) == 1


OTHER_OWNER = "d" * 64
_CREATED_WORKSPACE = "ws_requester_created"
_SOMEONE_ELSES_WORKSPACE = "ws_someone_else"


async def _seed_workspace(pool: Any, *, workspace: str, created_by: str, documents: int) -> None:
    """Register one Corpus Workspace and say who created it and how much is in it."""
    registry = PGWorkspaceRegistry(pool=pool)
    await registry.initialize()
    await registry.upsert(workspace=workspace, display_name=workspace, embedding_model="m")
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE dlightrag_workspace_meta
            SET created_by = $1, ingested_docs_total = $2
            WHERE workspace = $3
            """,
            created_by,
            documents,
            workspace,
        )


async def _corpus_mutation_runs(conn: Any) -> list[Any]:
    """The lane's Runs: `owner_id` is the Run's access scope, which here is the Workspace."""
    return list(
        await conn.fetch(
            """
            SELECT run_id::text AS run_id, owner_id, submitted_by, access_scope_kind, status
            FROM dlightrag_runs WHERE run_kind = 'corpus_mutation' ORDER BY run_id
            """
        )
    )


async def test_erase_covers_the_corpus_workspaces_the_requester_created(
    pool: Any, tmp_path: Path
) -> None:
    """A covered Workspace is delivered as a Run, not deleted here: the lane's write
    fence is what keeps a concurrent ingest from writing the documents back."""
    from dlightrag.application.config.loading import get_config

    workspace_root = tmp_path / "workspaces"
    skills_root = tmp_path / "skills"
    await _seed(pool, workspace_root=workspace_root, skills_root=skills_root)
    await _seed_workspace(pool, workspace=_CREATED_WORKSPACE, created_by=OWNER, documents=3)
    await _seed_workspace(
        pool, workspace=_SOMEONE_ELSES_WORKSPACE, created_by=OTHER_OWNER, documents=5
    )
    deliverer = corpus_deletion_deliverer(pool=pool, config=get_config())

    async with pool.acquire() as conn:
        planned = await created_corpus_workspaces(conn, owner_id=OWNER)
        erased = await erase_owner(
            conn,
            owner_id=OWNER,
            telemetry=None,
            workspace_root=workspace_root,
            owner_skills_root=skills_root,
            line="b2c",
            corpora=deliverer,
        )
        runs = await _corpus_mutation_runs(conn)
        workspaces = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_workspace_meta WHERE workspace = ANY($1::text[])",
            [_CREATED_WORKSPACE, _SOMEONE_ELSES_WORKSPACE],
        )
        counts = json.loads(
            await conn.fetchval(
                "SELECT counts FROM dlightrag_owner_data_requests WHERE owner_id = $1", OWNER
            )
        )
        left = await conn.fetchval(
            "SELECT left_in_place FROM dlightrag_owner_data_requests WHERE owner_id = $1",
            OWNER,
        )
        residue = await independent_residue(conn, OWNER)

    # Only creation lets an erase reach a corpus: membership would not.
    assert [(item.workspace, item.documents) for item in planned] == [(_CREATED_WORKSPACE, 3)]
    assert len(runs) == 1
    delivered = runs[0]
    assert delivered["owner_id"] == _CREATED_WORKSPACE  # a workspace-scoped Run
    assert delivered["access_scope_kind"] == "workspace"
    assert delivered["status"] == "queued"  # honest about being asynchronous
    assert erased.corpora_deletion_runs == (delivered["run_id"],)
    assert counts["corpora_deletion_runs"] == [delivered["run_id"]]
    assert counts["customer_approval"] is None
    # The Run is the deployment's act: submitting it as the subject would leave the
    # subject's id in a row the erase must not delete, and in the residue walk below.
    assert delivered["submitted_by"] == "operator"
    # The one residue is the delivery's own latency, and it is bounded: the Run's own
    # `unregister_workspace` step is what clears the row, and that is the step below.
    # The receipt's run id is what makes the wait checkable rather than open-ended.
    assert residue == {"dlightrag_workspace_meta.created_by": 1}
    assert await PGWorkspaceRegistry(pool=pool).delete(_CREATED_WORKSPACE)
    async with pool.acquire() as conn:
        assert await independent_residue(conn, OWNER) == {}
    # The documents are still there, because nothing has executed the Run yet. An
    # erase that claimed otherwise would be lying about its own asynchrony.
    assert int(workspaces) == 2
    assert not any("created" in entry for entry in json.loads(left))
    assert not any("created" in entry for entry in erased.left_in_place)


async def test_keep_corpora_leaves_the_documents_and_names_the_workspace(
    pool: Any, tmp_path: Path
) -> None:
    """Kept, the Workspace is a named fact in the receipt rather than a silent miss."""
    from dlightrag.application.config.loading import get_config

    await _seed(pool, workspace_root=tmp_path / "workspaces", skills_root=tmp_path / "skills")
    await _seed_workspace(pool, workspace=_CREATED_WORKSPACE, created_by=OWNER, documents=3)
    deliverer = corpus_deletion_deliverer(pool=pool, config=get_config())

    async with pool.acquire() as conn:
        erased = await erase_owner(
            conn,
            owner_id=OWNER,
            telemetry=None,
            workspace_root=tmp_path / "workspaces",
            owner_skills_root=tmp_path / "skills",
            line="b2c",
            corpora=deliverer,
            keep_corpora=True,
        )
        runs = await _corpus_mutation_runs(conn)
        documents = await conn.fetchval(
            "SELECT ingested_docs_total FROM dlightrag_workspace_meta WHERE workspace = $1",
            _CREATED_WORKSPACE,
        )
        counts = json.loads(
            await conn.fetchval(
                "SELECT counts FROM dlightrag_owner_data_requests WHERE owner_id = $1", OWNER
            )
        )

    assert runs == [] and erased.corpora_deletion_runs == ()
    assert int(documents) == 3
    assert any(_CREATED_WORKSPACE in entry for entry in erased.left_in_place)
    assert counts["corpora_deletion_runs"] == []
