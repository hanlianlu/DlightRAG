# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for one conversation's workspace across four bounded Runs (S1.3).

A Skill that builds a deep document in four bounded phases runs each as a
separate Run. This test drives the real head pointer and workspace trees through four
such rounds: a file written in round one is readable in round four, a failed round
carries nothing, and the model file lives in the workspace rather than in `notes/`.
Only the Agent is absent; every durable and filesystem step is the production path.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.answer.agentspace import PGAgentspaceStore
from dlightrag.engine.answer.agentspace import (
    AgentspaceSeed,
    bind_run_agentspace,
    session_heads_root,
    stage_session_head,
)
from dlightrag.engine.runtime.agentspace import AgentspaceHeadCommit
from tests.conftest import FingerprintingRunStore
from tests.support.pg import (
    PG_CONN_KWARGS,
    drop_database,
    skip_without_postgres,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_OWNER = "owner-alpha"
_WORKER = "worker-1"
_SESSION_ID = "00000000-0000-7000-8000-000000000001"
_REQUEST: dict[str, Any] = {
    "query": "build the case",
    "workspaces": ["alpha"],
    "agent_session_id": _SESSION_ID,
    "agent_lane_id": "main",
}


@pytest.fixture
async def db_name() -> AsyncIterator[str]:
    await skip_without_postgres()
    name = f"dlightrag_wshead_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        yield name
    finally:
        await drop_database(name)


@pytest.fixture
async def pool(db_name: str) -> AsyncIterator[Any]:
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()


@pytest.fixture
async def runs(pool: Any) -> FingerprintingRunStore:
    store = FingerprintingRunStore(pool=pool)
    await store.initialize()
    return store


class _Runs:
    """One bounded round of the case: seed, write, settle."""

    def __init__(self, runs: FingerprintingRunStore, pool: Any, root: Any) -> None:
        self.runs = runs
        self.pool = pool
        self.root = root
        self.head_run_id: str | None = None
        self.claim_epoch = 0

    async def open(self) -> tuple[str, Any]:
        creation = await self.runs.create_run(owner_id=_OWNER, request=_REQUEST)
        claim = await self.runs.claim_next(worker_id=_WORKER)
        assert claim is not None
        self.claim_epoch = claim.run.fencing_epoch
        reader = PGAgentspaceStore(
            pool=self.pool,
            owner_id=_OWNER,
            run_id=uuid.UUID(creation.run.run_id),
            worker_id=_WORKER,
            lease_owner=_WORKER,
            fencing_epoch=claim.run.fencing_epoch,
        )
        head = await reader.load_session_agentspace_head(session_id=_SESSION_ID)
        self.head_run_id = head.head_run_id if head is not None else None
        seed = None
        if head is not None:
            seed = AgentspaceSeed(
                head_run_id=head.head_run_id,
                manifest_digest=head.manifest_digest,
                workspace=session_heads_root(self.root, _OWNER, _SESSION_ID)
                / head.head_run_id
                / "workspace",
            )
        bound = await bind_run_agentspace(
            workspace_root=self.root,
            owner_id=_OWNER,
            run_id=creation.run.run_id,
            fencing_epoch=claim.run.fencing_epoch,
            recorded_epoch=None,
            store=None,
            seed=seed,
        )
        return creation.run.run_id, bound

    async def succeed(self, run_id: str, *, expected_head: str | None, workspace: Any) -> None:
        staged = stage_session_head(
            workspace=workspace,
            heads_root=session_heads_root(self.root, _OWNER, _SESSION_ID),
            run_id=run_id,
        )
        outcome = await self.runs.finish_success(
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            fencing_epoch=self.claim_epoch,
            result={"answer": "phase done"},
            workspace_head=AgentspaceHeadCommit(
                session_id=_SESSION_ID,
                head_run_id=run_id,
                manifest_digest=staged.manifest_digest,
                expected_head_run_id=expected_head,
                parent_run_id=expected_head,
                manifest=staged.files,
            ),
        )
        assert outcome.committed

    async def fail(self, run_id: str) -> None:
        await self.runs.finish_failure(
            owner_id=_OWNER,
            run_id=run_id,
            worker_id=_WORKER,
            fencing_epoch=self.claim_epoch,
            error_kind="ANSWER_STREAM_FAILED",
            error_message="phase failed",
        )


def _write(workspace: Any, relative: str, text: str) -> None:
    path = workspace / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _notes_files(workspace: Any) -> list[str]:
    notes = workspace / "notes"
    return [] if not notes.is_dir() else _files(notes)


def _files(workspace: Any) -> list[str]:
    return sorted(
        path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()
    )


async def test_a_case_builds_across_four_bounded_runs_and_carries_its_files(
    runs: FingerprintingRunStore, pool: Any, tmp_path: Any
) -> None:
    (tmp_path / "workspace").mkdir(parents=True)
    rounds = _Runs(runs, pool, tmp_path / "workspace")
    expected_head = None

    # Round 1 — evidence: the dated facts and the model's source are workspace files.
    run_one, one = await rounds.open()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, fencing_epoch)"
            " VALUES ($1, $2, $3, 1)",
            _OWNER,
            uuid.UUID(_SESSION_ID),
            uuid.UUID(run_one),
        )
    assert _files(one.workspace) == []
    _write(one.workspace, "scripts/model.py", "model = {}\n")
    _write(one.workspace, "artifacts/evidence-facts.md", "the facts\n")
    _write(one.workspace, "notes/plan.md", "memory, not staging\n")
    await rounds.succeed(run_one, expected_head=expected_head, workspace=one.workspace)
    expected_head = run_one

    # Round 2 — model: the previous round's files are already in the fresh epoch.
    run_two, two = await rounds.open()
    assert (two.workspace / "scripts/model.py").read_text() == "model = {}\n"
    assert (two.workspace / "artifacts/evidence-facts.md").is_file()
    assert _notes_files(two.workspace) == [], "notes come from the memory plane, not the head"
    _write(two.workspace, "case.draft.json", '{"phase": "model"}\n')
    await rounds.succeed(run_two, expected_head=expected_head, workspace=two.workspace)
    expected_head = run_two

    # Round 3 — a failed phase writes draft files but must carry none of them.
    run_three, three = await rounds.open()
    assert (three.workspace / "case.draft.json").is_file()
    _write(three.workspace, "case.partial.json", '{"half": true}\n')
    await rounds.fail(run_three)

    # Round 4 — report: the committed state is still round 2's, and the failed tail is gone.
    run_four, four = await rounds.open()
    assert (four.workspace / "scripts/model.py").is_file()
    assert (four.workspace / "case.draft.json").is_file()
    assert not (four.workspace / "case.partial.json").exists()
    assert _notes_files(four.workspace) == []
    _write(four.workspace, "artifacts/report.html", "<html>sealed</html>\n")
    await rounds.succeed(run_four, expected_head=expected_head, workspace=four.workspace)

    reader = PGAgentspaceStore(
        pool=pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(run_four),
        worker_id=_WORKER,
        lease_owner=_WORKER,
        fencing_epoch=1,
    )
    head = await reader.load_session_agentspace_head(session_id=_SESSION_ID)
    assert head is not None
    assert (head.head_run_id, head.lane_id, head.revision) == (run_four, "main", 3)
    head_workspace = (
        session_heads_root(tmp_path / "workspace", _OWNER, _SESSION_ID) / run_four / "workspace"
    )
    async with pool.acquire() as conn:
        snapshots = await conn.fetch(
            "SELECT run_id::text AS run_id, parent_run_id::text AS parent_run_id,"
            "       manifest_digest, manifest_json"
            " FROM dlightrag_session_workspace_snapshots"
            " WHERE owner_id = $1 AND session_id = $2::uuid ORDER BY created_at",
            _OWNER,
            uuid.UUID(_SESSION_ID),
        )
    # Every succeeded phase kept a restorable snapshot; the failed phase kept none, and each
    # snapshot names the one it was built on.
    assert [str(row["run_id"]) for row in snapshots] == [run_one, run_two, run_four]
    assert dict(
        zip(
            (str(row["run_id"]) for row in snapshots),
            (str(row["parent_run_id"]) if row["parent_run_id"] else None for row in snapshots),
            strict=True,
        )
    ) == {run_one: None, run_two: run_one, run_four: run_two}
    final_manifest = snapshots[-1]["manifest_json"]
    if isinstance(final_manifest, str):
        import json as _json

        final_manifest = _json.loads(final_manifest)
    assert {entry["path"] for entry in final_manifest} == {
        "artifacts/evidence-facts.md",
        "artifacts/report.html",
        "case.draft.json",
        "scripts/model.py",
    }
    # Every settled Run's tree is still restorable from the working copy, not only the head.
    for snapshot_run_id in (run_one, run_two, run_four):
        tree = (
            session_heads_root(tmp_path / "workspace", _OWNER, _SESSION_ID)
            / snapshot_run_id
            / "workspace"
        )
        assert tree.is_dir(), f"snapshot tree for {snapshot_run_id} was pruned"
    assert _files(head_workspace) == [
        "artifacts/evidence-facts.md",
        "artifacts/report.html",
        "case.draft.json",
        "scripts/model.py",
    ]
