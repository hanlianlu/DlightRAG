# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The durable Answer REST contract over a live PostgreSQL run store.

Exercises what an in-memory fake cannot prove: that a created run and its
uploaded bytes are persisted in one transaction, that idempotency replay and
conflict are owner-scoped, that another owner's run is indistinguishable from an
unknown one, that a reconnecting subscriber replays the durable sequence without
gaps or duplicates, and that cancellation and event trimming return the exact
documented statuses.

Every test runs inside a throwaway database, so the configured administrative
database is never mutated. Connection settings come from the shared
integration-test PostgreSQL environment; skipped if unavailable.
"""

import asyncio
import datetime
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.application.access.principal import (
    UserContext,
    owner_id_from_user,
)
from dlightrag.application.answer_runs.service import AnswerService
from dlightrag.application.config.loading import set_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.policies.service import (
    ACTIVE_ANSWER_RUNS,
    AGENT_EFFORT_CEILING,
    RuntimePolicies,
)
from dlightrag.application.runs import RunService
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.settings import (
    CHAT_MODEL_SELECTORS,
    ChatModelSelector,
    EmbeddingSettings,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.engine.answer.capabilities import AnswerCapabilities, RequestModelContext
from dlightrag.engine.answer.errors import ACTIVE_RUN_LIMIT_MESSAGE
from tests.support.agent_browser import never_registers
from tests.support.application_double import application_double, delegate
from tests.support.child_sessions import settle_child, spawn_child
from tests.support.pg import PG_CONN_KWARGS, drop_database, skip_without_postgres
from tests.support.session_entries import (
    Entry,
    answer,
    question,
    tool_result,
    tool_use,
    write_entries,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = PG_CONN_KWARGS

_ANON = UserContext(user_id="anonymous", auth_mode="none")
_ALICE = UserContext(user_id="alice", auth_mode="jwt", claims={"iss": "https://issuer.test"})
_BOB = UserContext(user_id="bob", auth_mode="jwt", claims={"iss": "https://issuer.test"})
_PROFILE = ModelProfile(
    context_window_tokens=200_000,
    max_input_tokens=180_000,
    max_output_tokens=16_000,
)


@pytest.fixture
async def pool() -> AsyncIterator[Any]:
    await skip_without_postgres()

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
        await drop_database(db_name)


@pytest.fixture
async def store(pool: Any) -> PGRunStore:
    created = PGRunStore(pool=pool)
    await created.initialize()
    return created


class _StoreScheduler:
    """Wake-free scheduler whose subscriptions replay the real PG event page."""

    def __init__(self, store: PGRunStore) -> None:
        self._store = store
        self.is_started = True

    def wake(self) -> None:
        return None

    def cancel_local(self, owner_id: str, run_id: str) -> None:
        return None

    @asynccontextmanager
    async def admission(self) -> AsyncIterator[bool]:
        yield self.is_started

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[Any]:
        async def _iterate() -> AsyncIterator[Any]:
            events = await self._store.read_event_page(
                owner_id=owner_id,
                run_id=run_id,
                after_sequence=after_sequence,
            )
            for event in events:
                yield event

        return _iterate()


class _Resources:
    async def pin_current_image_links(
        self, request: Any, attachment_bytes: Any
    ) -> tuple[Any, list[bytes]]:
        return request, list(attachment_bytes)

    async def resolve(
        self,
        _resources: Any,
        /,
        *,
        models: RequestModelContext,
        confirm_image_context: Any,
        resolved_mode: str,
        resource_identity: str | None = None,
    ) -> Any:
        del confirm_image_context, resolved_mode, resource_identity
        return SimpleNamespace(
            models=models,
            registry=None,
            current_images=(),
            resource_manifest=(),
            web_sources=None,
            image_budget=None,
            query_images=(),
        )


class _Capabilities:
    async def refresh_vlm(self) -> AnswerCapabilities:
        return AnswerCapabilities(answer=None, vlm_status="unknown")

    def current_profiles(self) -> dict[ChatModelSelector, ModelProfile]:
        return {role: _PROFILE for role in CHAT_MODEL_SELECTORS}

    def request_model_context(
        self, profiles: dict[ChatModelSelector, ModelProfile] | None, /
    ) -> RequestModelContext:
        selected = profiles or self.current_profiles()
        return RequestModelContext(
            extract=selected["extract"],
            query=selected["query"],
            vlm=selected["vlm"],
        )

    def answer_image_policy(self, _profile: ModelProfile, /) -> Any:
        return MagicMock()

    async def confirmed_live_answer_context(
        self, models: RequestModelContext, /
    ) -> tuple[RequestModelContext, None]:
        return models, None


class _Retrieval:
    def planner_for(self, _profile: ModelProfile | None = None) -> Any:
        planner = MagicMock()
        planner.history_input_measure.return_value = lambda messages, _summary="": (
            10 * len(messages)
        )
        return planner

    def warm(self, _workspaces: Any) -> None:
        return None

    async def schema_for(self, _workspaces: Any) -> dict[str, Any]:
        return {}


class _CapabilityView:
    async def read(self) -> AnswerCapabilities:
        return AnswerCapabilities(answer=None, vlm_status="unknown")


def _fingerprint(role: ChatModelSelector) -> ModelInvocationFingerprint:
    return ModelInvocationFingerprint(
        provider="test",
        model=f"model-{role}",
        endpoint_fingerprint=None,
        api_family="chat_completion",
    )


def _store_backed_application(
    store: PGRunStore, config: DlightragConfig, *, bind_research: Any = None
) -> Any:
    """The strict Application double over a real RunService and AnswerService on the store.

    AnswerService plans through the Application's own autospecced RetrievalService, which
    delegates to _Retrieval, so its calls still meet the real signatures.
    """
    application = application_double(config)
    application.corpora.alist_workspace_records.return_value = [{"workspace": "default"}]
    delegate(application.retrieval, _Retrieval(), "planner_for", "warm", "schema_for")
    scheduler = _StoreScheduler(store)
    application.runs = RunService(store=store, scheduler=scheduler)
    application.answers = AnswerService(
        store=store,
        blob_store=PGRunBlobStore(pool=store._operation_pool),  # noqa: SLF001
        coordinator=cast(Any, scheduler),
        retrieval=application.retrieval,
        capabilities=cast(Any, _Capabilities()),
        capability_view=cast(Any, _CapabilityView()),
        models=cast(
            Any,
            SimpleNamespace(
                query_image_describer=lambda: MagicMock(),
                model_settings=lambda role: ModelSettings(model="test"),
            ),
        ),
        resources=cast(Any, _Resources()),
        model_invocation_fingerprint_for_role=_fingerprint,
        bind_research=bind_research,
        agent_may_register=never_registers,
        policies=RuntimePolicies(store=store),
        child_roster_cursor_secret=b"answer-run-api-child-roster-test",
        protected_skill_documents=lambda: frozenset(),
    )
    return application


@pytest.fixture
def app(store: PGRunStore, tmp_path) -> Iterator[FastAPI]:
    config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={
            "working_dir": str(tmp_path / "dlightrag_storage"),
        },
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="test",
                startup_probe=False,
            ),
        },
    )
    set_config(config)
    application = create_app()
    application.state.application = _store_backed_application(store, config)
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


async def _claim(store: PGRunStore, owner: str, run_id: str) -> Any:
    claimed = await store.claim_next(worker_id="worker-1")
    assert claimed is not None
    assert claimed.run.run_id == run_id
    assert claimed.run.owner_id == owner
    return claimed


async def test_create_persists_the_run_and_its_uploaded_bytes(
    client: AsyncClient, store: PGRunStore, pool: Any
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
    assert (record.prepared_input or {})["query"] == "summarize"
    references = await store.list_run_artifacts(owner_id=owner, run_id=run_id)
    assert [reference.filename for reference in references] == ["notes.txt"]
    assert (
        await PGRunBlobStore(pool=pool).read(owner_id=owner, digest=references[0].digest)
        == b"durable-bytes"
    )


async def test_an_accepted_upload_reads_back_by_its_id_from_its_artifact_reference(
    app: FastAPI, client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    """The upload's artifact reference is its only record; no resource row repeats it."""
    response = await client.post(
        "/answer",
        data={"request": json.dumps({"query": "summarize"})},
        files=[("attachments", ("notes.txt", b"durable-bytes", "text/plain"))],
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    owner = owner_id_from_user(_ANON)
    [reference] = await store.list_run_artifacts(owner_id=owner, run_id=run_id)
    answers = app.state.application.answers

    read = await answers.read_run_resource(
        owner_id=owner, run_id=run_id, resource_id=reference.resource_id
    )
    assert read is not None
    descriptor, content = read
    assert (descriptor.registry, descriptor.reference_kind) == ("artifact", "current_attachment")
    assert (descriptor.filename, descriptor.digest) == ("notes.txt", reference.digest)
    assert content == b"durable-bytes"
    stream = await answers.open_run_resource(
        owner_id=owner, run_id=run_id, resource_id=reference.resource_id
    )
    assert stream is not None
    assert b"".join([piece async for piece in stream]) == b"durable-bytes"
    assert await answers.run_resource_size(
        owner_id=owner, run_id=run_id, resource_id=reference.resource_id
    ) == len(b"durable-bytes")
    assert (
        await pool.fetchval(
            "SELECT count(*) FROM dlightrag_answer_resources WHERE run_id = $1",
            uuid.UUID(run_id),
        )
        == 0
    )


async def test_an_upload_no_run_can_read_is_refused_by_kind_and_stores_nothing(
    client: AsyncClient, pool: Any
) -> None:
    response = await client.post(
        "/answer",
        data={"request": json.dumps({"query": "summarize"})},
        files=[
            ("attachments", ("notes.md", b"# notes", "text/markdown")),
            ("attachments", ("draft.odt", b"PK\x03\x04", "application/octet-stream")),
        ],
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error_type"] == "validation"
    assert body["error_kind"] == "UNSUPPORTED_ATTACHMENT_TYPE"
    assert body["detail"].startswith("Attachment type .odt cannot be read by an answer.")
    assert await pool.fetchval("SELECT count(*) FROM dlightrag_runs") == 0


async def test_a_text_upload_of_an_unlisted_type_is_accepted_by_its_bytes(
    client: AsyncClient, store: PGRunStore
) -> None:
    # curl and requests name no type for such a file: application/octet-stream.
    response = await client.post(
        "/answer",
        data={"request": json.dumps({"query": "review the build"})},
        files=[
            ("attachments", ("main.go", b"package main\n", "application/octet-stream")),
            ("attachments", ("Makefile", b"build:\n\tgo build\n", "application/octet-stream")),
        ],
    )

    assert response.status_code == 202
    references = await store.list_run_artifacts(
        owner_id=owner_id_from_user(_ANON), run_id=response.json()["run_id"]
    )
    assert sorted(reference.filename for reference in references) == ["Makefile", "main.go"]


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
    assert (await client.get(f"/runs/{run_id}")).status_code == 404
    assert (await client.get(f"/runs/{run_id}/events")).status_code == 404
    assert (await client.delete(f"/runs/{run_id}")).status_code == 404


async def test_reconnect_replays_the_durable_sequence_without_gaps(
    client: AsyncClient, store: PGRunStore
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    worker = str(claimed.run.lease_owner)
    epoch = int(claimed.run.fencing_epoch)
    await store.record_phase(
        owner_id=owner, run_id=run_id, worker_id=worker, fencing_epoch=epoch, phase="planning"
    )
    await store.append_event(
        owner_id=owner,
        run_id=run_id,
        worker_id=worker,
        fencing_epoch=epoch,
        phase=None,
        event_type="token",
        payload={"text": "first"},
    )
    await store.append_event(
        owner_id=owner,
        run_id=run_id,
        worker_id=worker,
        fencing_epoch=epoch,
        phase=None,
        event_type="token",
        payload={"text": "second"},
    )

    full = await client.get(f"/runs/{run_id}/events")
    resumed = await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "1"})

    assert [line for line in full.text.splitlines() if line.startswith("id: ")] == [
        "id: 1",
        "id: 2",
        "id: 3",
    ]
    assert [line for line in resumed.text.splitlines() if line.startswith("id: ")] == [
        "id: 2",
        "id: 3",
    ]


async def test_cancellation_status_matrix(client: AsyncClient, store: PGRunStore) -> None:
    queued = (await client.post("/answer", json={"query": "queued"})).json()["run_id"]
    running = (await client.post("/answer", json={"query": "running"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)

    cancelled = await client.delete(f"/runs/{queued}")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert (await client.delete(f"/runs/{queued}")).status_code == 200

    await _claim(store, owner, running)
    pending = await client.delete(f"/runs/{running}")
    assert pending.status_code == 202
    assert pending.json()["cancel_requested"] is True


async def test_a_trimmed_event_log_is_gone_but_the_result_remains(
    client: AsyncClient, store: PGRunStore, pool: Any
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
            "UPDATE dlightrag_runs SET finished_at = NOW() - INTERVAL '370 days', "
            "purge_after = NOW() - INTERVAL '5 days' WHERE run_id = $1",
            uuid.UUID(run_id),
        )
    assert await store.trim_expired_event_logs() == 1

    events = await client.get(f"/runs/{run_id}/events")
    status = await client.get(f"/runs/{run_id}")

    assert events.status_code == 410
    assert status.status_code == 200
    assert status.json()["result"]["answer"] == "durable"


async def _operation_times(pool: Any, child_session_id: str) -> list[tuple[Any, Any]]:
    """Each Operation's stored creation and last-change time, oldest first."""
    rows = await pool.fetch(
        "SELECT created_at, updated_at FROM dlightrag_answer_child_operations "
        "WHERE child_session_id = $1 ORDER BY operation_sequence",
        uuid.UUID(child_session_id),
    )
    return [(row["created_at"], row["updated_at"]) for row in rows]


def _instant(text: str) -> datetime.datetime:
    assert text.endswith("Z")
    return datetime.datetime.fromisoformat(text)


async def test_the_child_roster_carries_operation_times_and_waiting_questions(
    client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    worker, epoch = str(claimed.run.lease_owner), int(claimed.run.fencing_epoch)
    fenced = {"owner_id": owner, "run_id": run_id, "worker_id": worker, "fencing_epoch": epoch}
    child = await spawn_child(store, **fenced)

    async def ask(question: str) -> str:
        request_id = str(uuid.uuid7())
        assert await store.create_child_guidance(
            owner_id=owner,
            run_id=run_id,
            request_id=request_id,
            child_session_id=child.child_session_id,
            child_operation_id=child.operation_id,
            parent_session_id=child.parent_session_id,
            question=question,
            expires_after_seconds=300,
            worker_id=worker,
            fencing_epoch=epoch,
            child_fencing_epoch=child.child_epoch,
        )
        return request_id

    async def lapse(request_id: str) -> None:
        await pool.execute(
            "UPDATE dlightrag_answer_child_guidance "
            "SET expires_at = NOW() - INTERVAL '1 minute' WHERE request_id = $1",
            uuid.UUID(request_id),
        )

    await ask("still waiting")
    answered = await ask("answered")
    gave_up = await ask("given up on")
    lapsed = await ask("lapsed unseen")
    reply = await store.reply_child_guidance(
        owner_id=owner,
        run_id=run_id,
        request_id=answered,
        content="use the report",
        submission_key="reply-one",
    )
    assert reply is not False and reply["outcome"] == "replied"
    await lapse(gave_up)
    assert await store.expire_child_guidance(
        **fenced,
        request_id=gave_up,
        child_session_id=child.child_session_id,
        child_operation_id=child.operation_id,
        child_fencing_epoch=child.child_epoch,
    )
    await lapse(lapsed)

    async def listed() -> dict[str, Any]:
        roster = (await client.get(f"/answer/{run_id}/children")).json()["children"]
        [row] = [row for row in roster if row["child_session_id"] == child.child_session_id]
        observed = (await client.get(f"/answer/{run_id}/children/{child.child_session_id}")).json()
        assert observed["child"] == row
        return row

    [(opened, _)] = await _operation_times(pool, child.child_session_id)
    running = await listed()
    assert _instant(running["started_at"]) == opened
    assert running["finished_at"] is None
    assert running["pending_questions"] == 1

    await settle_child(store, child, **fenced)
    [(_, settled)] = await _operation_times(pool, child.child_session_id)
    done = await listed()
    assert _instant(done["started_at"]) == opened
    assert _instant(done["finished_at"]) == settled
    assert _instant(done["finished_at"]) >= _instant(done["started_at"])

    continuation = await store.continue_child_session(
        owner_id=owner,
        run_id=run_id,
        child_session_id=child.child_session_id,
        content="follow up",
        submission_key="continue-one",
    )
    assert continuation is not False and continuation["outcome"] == "accepted"
    [_, (reopened, _)] = await _operation_times(pool, child.child_session_id)
    continued = await listed()
    assert reopened > opened
    assert _instant(continued["started_at"]) == reopened
    assert continued["finished_at"] is None


async def test_every_roster_page_carries_the_same_child_timing(
    client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    worker, epoch = str(claimed.run.lease_owner), int(claimed.run.fencing_epoch)
    full = await spawn_child(
        store, owner_id=owner, run_id=run_id, worker_id=worker, fencing_epoch=epoch
    )
    assert await store.create_child_guidance(
        owner_id=owner,
        run_id=run_id,
        request_id=str(uuid.uuid7()),
        child_session_id=full.child_session_id,
        child_operation_id=full.operation_id,
        parent_session_id=full.parent_session_id,
        question="which source?",
        expires_after_seconds=300,
        worker_id=worker,
        fencing_epoch=epoch,
        child_fencing_epoch=full.child_epoch,
    )
    # A Child created without an Operation yet, and the newer of the two.
    sparse_id = str(uuid.uuid7())
    assert await store.upsert_child_session(
        owner_id=owner,
        run_id=run_id,
        child_session_id=sparse_id,
        parent_session_id=str(uuid.uuid7()),
        parent_call_id="call-sparse",
        worker_id=worker,
        fencing_epoch=epoch,
    )

    first = (await client.get(f"/answer/{run_id}/children", params={"limit": 1})).json()
    second = (
        await client.get(
            f"/answer/{run_id}/children", params={"limit": 1, "cursor": first["next_cursor"]}
        )
    ).json()

    [newest] = first["children"]
    [oldest] = second["children"]
    assert (newest["child_session_id"], oldest["child_session_id"]) == (
        sparse_id,
        full.child_session_id,
    )
    sparse_created = await pool.fetchval(
        "SELECT created_at FROM dlightrag_answer_child_sessions WHERE child_session_id = $1",
        uuid.UUID(sparse_id),
    )
    assert newest["operation_id"] is None
    assert _instant(newest["started_at"]) == sparse_created
    assert newest["finished_at"] is None
    assert newest["pending_questions"] == 0
    [(opened, _)] = await _operation_times(pool, full.child_session_id)
    assert _instant(oldest["started_at"]) == opened
    assert oldest["finished_at"] is None
    assert oldest["pending_questions"] == 1


async def test_every_read_of_a_child_names_the_message_it_answered_with_and_the_judgement_of_it(
    client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    run_id = (await client.post("/answer", json={"query": "q"})).json()["run_id"]
    owner = owner_id_from_user(_ANON)
    claimed = await _claim(store, owner, run_id)
    fenced = {
        "owner_id": owner,
        "run_id": run_id,
        "worker_id": str(claimed.run.lease_owner),
        "fencing_epoch": int(claimed.run.fencing_epoch),
    }
    judged, plain, working, continued = [await spawn_child(store, **fenced) for _ in range(4)]

    async def wrote(child: Any, *entries: Entry) -> list[int]:
        return await write_entries(
            pool,
            owner=owner,
            session=child.child_session_id,
            run_id=run_id,
            entries=entries,
        )

    _, judged_tool_call, _, judged_answer = await wrote(
        judged, question(), tool_use(), tool_result(), answer()
    )
    _, plain_answer = await wrote(plain, question(), answer())
    await wrote(working, question(), tool_use())
    # A Child told to go on after it answered is working again: it ended on a call to a tool.
    await wrote(continued, question(), answer(), question("go on"), tool_use())

    async def judge(
        child: Any, sequence: int, value: str, *reasons: str, note: str | None = None
    ) -> None:
        kept = await store.put_answer_feedback(
            owner_id=owner,
            run_id=run_id,
            agent=child.child_session_id,
            sequence=sequence,
            value=value,
            reasons=reasons,
            note=note,
        )
        assert kept is not None

    async def paged() -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        cursor = None
        while True:
            params = {"limit": 1, **({"cursor": cursor} if cursor else {})}
            page = (await client.get(f"/answer/{run_id}/children", params=params)).json()
            rows += page["children"]
            cursor = page["next_cursor"]
            if cursor is None:
                return rows

    async def stated() -> dict[str, tuple[Any, Any]]:
        """What each read of the Children says, which every other read must say as well."""
        roster = (await client.get(f"/answer/{run_id}/children")).json()["children"]
        by_child = {row["child_session_id"]: row for row in roster}
        assert sorted(by_child) == sorted(
            child.child_session_id for child in (judged, plain, working, continued)
        )
        assert sorted(await paged(), key=lambda row: row["child_session_id"]) == sorted(
            roster, key=lambda row: row["child_session_id"]
        )
        for child_id, row in by_child.items():
            observed = (await client.get(f"/answer/{run_id}/children/{child_id}")).json()
            assert observed["child"] == row
        listed = await store.list_child_sessions(owner_id=owner, run_id=run_id)
        for row in listed:
            kept = by_child[row["child_session_id"]]
            judgement = row["feedback"]
            assert row["result_sequence"] == kept["result_sequence"]
            assert (judgement.payload() if judgement else None) == kept["feedback"]
        return {
            child_id: (row["result_sequence"], row["feedback"])
            for child_id, row in by_child.items()
        }

    def expected(judgement: dict[str, Any] | None) -> dict[str, tuple[Any, Any]]:
        return {
            judged.child_session_id: (judged_answer, judgement),
            plain.child_session_id: (plain_answer, None),
            working.child_session_id: (None, None),
            continued.child_session_id: (None, None),
        }

    # A Child that ended on a message with no tool call has answered, judged or not. One that
    # ended on a call to a tool has not, though it answered before it was told to go on.
    assert await stated() == expected(None)

    # The judgement shown is the answer's own: another message of the same Child is not.
    await judge(judged, judged_tool_call, "up")
    assert await stated() == expected(None)
    await judge(judged, judged_answer, "down", "off_topic", note="not what I asked")
    given = {"value": "down", "reasons": ["off_topic"], "note": "not what I asked"}
    assert await stated() == expected(given)

    # A judgement is replaced whole, and one taken back is gone.
    await judge(judged, judged_answer, "up")
    assert await stated() == expected({"value": "up", "reasons": [], "note": None})
    await store.delete_answer_feedback(
        owner_id=owner, run_id=run_id, agent=judged.child_session_id, sequence=judged_answer
    )
    assert await stated() == expected(None)


# ---------------------------------------------------------------------------
# Runtime Policies: what an Answer is accepted under, and what it carries
# ---------------------------------------------------------------------------

_INTERN = UserContext(
    user_id="intern",
    auth_mode="jwt",
    claims={"iss": "https://issuer.test", "email": "intern@example.com"},
)
_LEAD = UserContext(
    user_id="lead",
    auth_mode="jwt",
    claims={"iss": "https://issuer.test", "email": "lead@example.com"},
)
_LOW_FOR_THE_INTERN = {
    "default": "max",
    "rules": [{"claim": "email", "value": "intern@example.com", "then": "low"}],
}


async def _administrator_writes(pool: Any, key: str, document: dict[str, Any]) -> None:
    """What an administrator does on another process: a policy service over a store of its own."""
    await RuntimePolicies(store=PGRunStore(pool=pool)).write(key, document, "admin")


async def _submit(client: AsyncClient, query: str, *, key: str, **extra: Any) -> Any:
    return await client.post(
        "/answer", json={"query": query, **extra}, headers={"Idempotency-Key": key}
    )


async def test_an_owners_active_limit_is_enforced_per_owner_and_follows_the_administrator(
    app: FastAPI, client: AsyncClient, pool: Any
) -> None:
    await _administrator_writes(pool, ACTIVE_ANSWER_RUNS, {"default": 2})
    _as_user(app, _INTERN)

    # Five submissions at once, through the real route and the real lane lock: two are
    # accepted and three are told it is their own limit.
    responses = await asyncio.gather(*(_submit(client, f"q{n}", key=f"key-{n}") for n in range(5)))

    assert sorted(response.status_code for response in responses) == [202, 202, 429, 429, 429]
    refused = next(response for response in responses if response.status_code == 429)
    assert refused.json() == {
        "detail": ACTIVE_RUN_LIMIT_MESSAGE,
        "error_type": "unavailable",
        "error_kind": "active_run_limit",
    }
    assert await pool.fetchval("SELECT count(*) FROM dlightrag_runs") == 2

    # Another owner holds none, so it is not limited by this one's.
    _as_user(app, _LEAD)
    assert (await _submit(client, "mine", key="key-lead")).status_code == 202

    # A retry of a key that was accepted is answered while its owner is at the limit.
    _as_user(app, _INTERN)
    accepted = next(
        number for number, response in enumerate(responses) if response.status_code == 202
    )
    replay = await _submit(client, f"q{accepted}", key=f"key-{accepted}")
    assert replay.status_code == 202
    assert replay.json()["run_id"] == responses[accepted].json()["run_id"]

    # The administrator raises the limit on another process, and the very next request meets it.
    await _administrator_writes(pool, ACTIVE_ANSWER_RUNS, {"default": 3})
    assert (await _submit(client, "one more", key="key-more")).status_code == 202
    assert (await _submit(client, "and another", key="key-another")).status_code == 429

    # A rule leaves one person without a limit while everyone else keeps it.
    await _administrator_writes(
        pool,
        ACTIVE_ANSWER_RUNS,
        {"default": 1, "rules": [{"claim": "email", "value": "intern@example.com", "then": None}]},
    )
    assert (await _submit(client, "free", key="key-free")).status_code == 202


async def test_a_run_is_pinned_to_the_ceiling_it_was_accepted_under_and_a_replay_is_no_conflict(
    app: FastAPI, client: AsyncClient, store: PGRunStore, pool: Any
) -> None:
    await _administrator_writes(pool, AGENT_EFFORT_CEILING, _LOW_FOR_THE_INTERN)
    _as_user(app, _INTERN)
    owner = owner_id_from_user(_INTERN)

    async def stored(run_id: str) -> tuple[Any, Any]:
        record = await store.get_run(owner_id=owner, run_id=run_id)
        assert record is not None and record.prepared_input is not None
        return record.prepared_input["effort"], record.prepared_input["effort_ceiling"]

    first = await _submit(client, "q", key="key", effort="max")
    run_id = first.json()["run_id"]
    assert await stored(run_id) == ("max", "low"), "the choice is kept; the ceiling is pinned"

    # The administrator lifts the ceiling. The same submission, retried, is the same Run - no
    # conflict - and the Run keeps the ceiling it was accepted under.
    await _administrator_writes(pool, AGENT_EFFORT_CEILING, {"default": "max"})
    replay = await _submit(client, "q", key="key", effort="max")
    assert (replay.status_code, replay.json()["run_id"]) == (202, run_id)
    assert await stored(run_id) == ("max", "low")

    # A submission after the change is accepted under the new ceiling.
    later = await _submit(client, "q", key="another-key", effort="max")
    assert await stored(later.json()["run_id"]) == ("max", "max")


@pytest.mark.parametrize("continuation", ["follow-up", "fork"])
async def test_a_continuation_is_pinned_to_the_ceiling_of_whoever_continues(
    app: FastAPI, client: AsyncClient, store: PGRunStore, pool: Any, continuation: str
) -> None:
    _as_user(app, _INTERN)
    owner = owner_id_from_user(_INTERN)
    parent = (await _submit(client, "first", key="parent")).json()["run_id"]
    claimed = await _claim(store, owner, parent)
    await store.finish_success(
        owner_id=owner,
        run_id=parent,
        worker_id=str(claimed.run.lease_owner),
        fencing_epoch=int(claimed.run.fencing_epoch),
        result={"answer": "done", "contexts": {"chunks": []}, "sources": []},
    )
    await _administrator_writes(pool, AGENT_EFFORT_CEILING, _LOW_FOR_THE_INTERN)

    response = await client.post(f"/answer/{parent}/{continuation}", json={"content": "and then?"})

    assert response.status_code == 202
    record = await store.get_run(owner_id=owner, run_id=response.json()["run_id"])
    assert record is not None and record.prepared_input is not None
    assert record.prepared_input["effort_ceiling"] == "low"


@pytest.mark.parametrize("key", [ACTIVE_ANSWER_RUNS, AGENT_EFFORT_CEILING])
async def test_a_policy_that_cannot_be_read_refuses_a_new_submission_and_never_a_replay(
    app: FastAPI, client: AsyncClient, pool: Any, key: str
) -> None:
    _as_user(app, _INTERN)
    accepted = await _submit(client, "q", key="key")
    await pool.execute(
        "INSERT INTO dlightrag_runtime_policies (key, value, updated_by)"
        " VALUES ($1, $2::jsonb, 'admin')",
        key,
        json.dumps({"default": "everything"}),
    )

    refused = await _submit(client, "another", key="another-key")
    replay = await _submit(client, "q", key="key")

    assert refused.status_code == 503
    assert refused.json()["error_type"] == "unavailable"
    assert "everything" not in refused.text
    assert (replay.status_code, replay.json()["run_id"]) == (202, accepted.json()["run_id"])
    assert await pool.fetchval("SELECT count(*) FROM dlightrag_runs") == 1


async def test_a_browser_submission_is_accepted_under_the_readers_policies_in_its_own_transaction(
    app: FastAPI, store: PGRunStore, pool: Any
) -> None:
    """The conversation, the Run and the turn are one transaction, and so is the count: a
    submission refused at the limit leaves no conversation behind, and a retry at the limit is
    answered with the turn it already made."""
    from uuid import uuid7

    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
    from dlightrag.application.answer_runs.service import ActiveRunLimitError
    from dlightrag.application.web_conversations.service import WebConversationService

    web = WebConversationService(
        store=PGWebConversationStore(pool=pool, run_store=store),
        answers=app.state.application.answers,
        max_attachments=8,
        cursor_secret=b"browser-policies-test",
    )
    await _administrator_writes(pool, AGENT_EFFORT_CEILING, _LOW_FOR_THE_INTERN)
    await _administrator_writes(pool, ACTIVE_ANSWER_RUNS, {"default": 1})
    submission_id = str(uuid7())

    async def submit(submission: str, **extra: Any) -> Any:
        return await web.start_answer(
            _INTERN,
            conversation_id=None,
            submission_id=submission,
            query="q",
            workspaces=("default",),
            **extra,
        )

    first = await submit(submission_id, effort="max")

    assert first is not None
    record = await store.get_run(owner_id=owner_id_from_user(_INTERN), run_id=first.run.run_id)
    assert record is not None and record.prepared_input is not None
    assert (record.prepared_input["effort"], record.prepared_input["effort_ceiling"]) == (
        "max",
        "low",
    )
    with pytest.raises(ActiveRunLimitError):
        await submit(str(uuid7()))
    assert await pool.fetchval("SELECT count(*) FROM dlightrag_runs") == 1
    assert await pool.fetchval("SELECT count(*) FROM web_conversations") == 1
    assert await pool.fetchval("SELECT count(*) FROM web_conversation_turns") == 1
    again = await submit(submission_id, effort="max")
    assert again is not None and again.run.run_id == first.run.run_id
