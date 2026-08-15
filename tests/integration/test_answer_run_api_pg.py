# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The durable Answer REST contract over a live PostgreSQL run store.

Exercises what an in-memory fake cannot prove: that a created run and its
uploaded bytes are persisted in one transaction, that idempotency replay and
conflict are owner-scoped, that another owner's run is indistinguishable from an
unknown one, that a reconnecting subscriber replays the durable sequence without
gaps or duplicates, and that cancellation and event trimming return the exact
documented statuses.

Every test runs inside a throwaway database, so the developer's ``dlightrag``
database is never mutated. Requires PostgreSQL at localhost:5432; skipped
otherwise.
"""

import json
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.postgres.answer_runs import PGAnswerRunStore
from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.principal import owner_id_from_user
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig, EmbeddingConfig, LLMConfig, ModelConfig, set_config
from tests.unit.conftest import prepare_test_answer_run_input

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)

_ANON = UserContext(user_id="anonymous", auth_mode="none")
_ALICE = UserContext(user_id="alice", auth_mode="jwt", claims={"iss": "https://issuer.test"})
_BOB = UserContext(user_id="bob", auth_mode="jwt", claims={"iss": "https://issuer.test"})


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    db_name = f"dlightrag_run_api_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture
async def store(pool: Any) -> PGAnswerRunStore:
    created = PGAnswerRunStore(pool=pool)
    await created.initialize()
    return created


class _StoreBackedManager:
    """The durable surface REST uses, wired straight to the real store."""

    def __init__(self, store: PGAnswerRunStore, config: DlightragConfig) -> None:
        self._store = store
        self.config = config

    async def aprepare_answer_run_input(
        self,
        request: Any,
        *,
        resources: Any,
        idempotency_fingerprint: str,
    ) -> Any:
        pinned = await prepare_test_answer_run_input(
            request,
            resources=resources,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return pinned

    async def astart_answer_run(
        self,
        *,
        owner_id: str,
        request: Any,
        idempotency_key: str | None = None,
        attachment_bytes: Any = (),
    ) -> Any:
        from dlightrag.runtime import PendingArtifact, PendingArtifactReference

        return await self._store.create_run(
            owner_id=owner_id,
            request=request.as_request(),
            idempotency_fingerprint=request.idempotency_fingerprint,
            idempotency_key=idempotency_key,
            artifacts=[PendingArtifact(content=content) for content in attachment_bytes],
            references=[
                PendingArtifactReference(
                    resource_id=item.resource_id,
                    reference_kind="current_attachment",
                    ordinal=item.ordinal,
                    digest=item.digest,
                    filename=item.filename,
                    mime_type=item.mime_type,
                )
                for item in request.attachments
            ],
        )

    async def areplay_answer_run(
        self,
        *,
        owner_id: str,
        idempotency_key: str,
        idempotency_fingerprint: str,
    ) -> Any:
        replay = await self._store.replay_run(
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            idempotency_fingerprint=idempotency_fingerprint,
        )
        return replay.run if replay is not None else None

    async def aget_answer_run(self, *, owner_id: str, run_id: str) -> Any:
        return await self._store.get_run(owner_id=owner_id, run_id=run_id)

    async def acancel_answer_run(self, *, owner_id: str, run_id: str) -> Any:
        return await self._store.request_cancellation(owner_id=owner_id, run_id=run_id)

    async def asubscribe_answer_run(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[Any]:
        events = await self._store.read_event_page(
            owner_id=owner_id, run_id=run_id, after_sequence=after_sequence
        )

        async def _iterate() -> AsyncIterator[Any]:
            for event in events:
                yield event

        return _iterate()


@pytest.fixture
def app(store: PGAnswerRunStore, tmp_path) -> Iterator[FastAPI]:
    config = DlightragConfig(
        working_dir=str(tmp_path / "dlightrag_storage"),
        llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="test")),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="voyage-multimodal-3.5",
            api_key="test",
            startup_probe=False,
        ),
    )
    set_config(config)
    application = create_app(include_web_app=False)
    application.state.manager = _StoreBackedManager(store, config)
    application.dependency_overrides[get_current_user] = lambda: _ANON
    yield application
    application.dependency_overrides.clear()


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as api:
        yield api


def _as_user(app: FastAPI, user: UserContext) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


async def _claim(store: PGAnswerRunStore, owner: str, run_id: str) -> Any:
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    assert claimed.run.run_id == run_id
    assert claimed.run.owner_id == owner
    return claimed


async def test_create_persists_the_run_and_its_uploaded_bytes(
    client: AsyncClient, store: PGAnswerRunStore
) -> None:
    response = await client.post(
        "/answer",
        data={"request": json.dumps({"query": "summarize"})},
        files=[("attachments", ("notes.txt", b"durable-bytes", "text/plain"))],
    )

    assert response.status_code == 202
    run_id = response.json()["run_id"]
    owner = owner_id_from_user(_ANON)
    record = await store.get_run(owner_id=owner, run_id=run_id)
    assert record is not None
    assert record.status == "queued"
    assert record.request["query"] == "summarize"
    references = await store.list_run_artifacts(owner_id=owner, run_id=run_id)
    assert [reference.filename for reference in references] == ["notes.txt"]
    assert (
        await store.load_artifact(owner_id=owner, digest=references[0].digest) == b"durable-bytes"
    )


async def test_idempotency_replays_and_conflicts_within_one_owner(
    client: AsyncClient, app: FastAPI
) -> None:
    headers = {"Idempotency-Key": "key-1"}
    first = await client.post("/answer", json={"query": "q"}, headers=headers)
    replay = await client.post("/answer", json={"query": "q"}, headers=headers)
    conflict = await client.post("/answer", json={"query": "different"}, headers=headers)

    assert first.status_code == replay.status_code == 202
    assert first.json()["run_id"] == replay.json()["run_id"]
    assert conflict.status_code == 409

    # The same key belongs to a different owner's namespace.
    _as_user(app, _ALICE)
    other = await client.post("/answer", json={"query": "different"}, headers=headers)
    assert other.status_code == 202
    assert other.json()["run_id"] != first.json()["run_id"]


@pytest.mark.parametrize("key", ["", "   "])
async def test_a_blank_idempotency_key_never_replays_or_conflicts(
    client: AsyncClient, key: str
) -> None:
    headers = {"Idempotency-Key": key}
    first = await client.post("/answer", json={"query": "one"}, headers=headers)
    second = await client.post("/answer", json={"query": "two"}, headers=headers)

    assert first.status_code == second.status_code == 202
    assert first.json()["run_id"] != second.json()["run_id"]


async def test_another_owner_cannot_read_cancel_or_follow_a_run(
    client: AsyncClient, app: FastAPI
) -> None:
    _as_user(app, _ALICE)
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]

    _as_user(app, _BOB)
    assert (await client.get(f"/answer/{run_id}")).status_code == 404
    assert (await client.get(f"/answer/{run_id}/events")).status_code == 404
    assert (await client.delete(f"/answer/{run_id}")).status_code == 404


async def test_reconnect_replays_the_durable_sequence_without_gaps(
    client: AsyncClient, store: PGAnswerRunStore
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    worker = str(claimed.run.lease_owner)
    epoch = int(claimed.run.fencing_epoch)
    await store.record_phase(
        owner_id=owner, run_id=run_id, worker_id=worker, fencing_epoch=epoch, phase="planning"
    )
    await store.append_token_batch(
        owner_id=owner, run_id=run_id, worker_id=worker, fencing_epoch=epoch, text="first"
    )
    await store.append_token_batch(
        owner_id=owner, run_id=run_id, worker_id=worker, fencing_epoch=epoch, text="second"
    )

    full = await client.get(f"/answer/{run_id}/events")
    resumed = await client.get(f"/answer/{run_id}/events", headers={"Last-Event-ID": "1"})

    assert [line for line in full.text.splitlines() if line.startswith("id: ")] == [
        "id: 1",
        "id: 2",
        "id: 3",
    ]
    assert [line for line in resumed.text.splitlines() if line.startswith("id: ")] == [
        "id: 2",
        "id: 3",
    ]


async def test_cancellation_status_matrix(client: AsyncClient, store: PGAnswerRunStore) -> None:
    queued = (await client.post("/answer", json={"query": "queued"})).json()["run_id"]
    running = (await client.post("/answer", json={"query": "running"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)

    cancelled = await client.delete(f"/answer/{queued}")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert (await client.delete(f"/answer/{queued}")).status_code == 200

    await _claim(store, owner, running)
    pending = await client.delete(f"/answer/{running}")
    assert pending.status_code == 202
    assert pending.json()["cancel_requested"] is True


async def test_a_trimmed_event_log_is_gone_but_the_result_remains(
    client: AsyncClient, store: PGAnswerRunStore, pool: Any
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    await store.finish_success(
        owner_id=owner,
        run_id=run_id,
        worker_id=str(claimed.run.lease_owner),
        fencing_epoch=int(claimed.run.fencing_epoch),
        result={"answer": "durable", "contexts": {"chunks": []}, "sources": []},
    )
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_answer_runs SET finished_at = NOW() - INTERVAL '31 days' "
            "WHERE run_id = $1",
            uuid.UUID(run_id),
        )
    assert await store.trim_expired_event_logs() == 1

    events = await client.get(f"/answer/{run_id}/events")
    status = await client.get(f"/answer/{run_id}")

    assert events.status_code == 410
    assert status.status_code == 200
    assert status.json()["result"]["answer"] == "durable"
