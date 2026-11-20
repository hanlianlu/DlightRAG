# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The call behind each step a Run writes, as PostgreSQL keeps it and as clients and exports see it
(ADR 0043).

A Run is driven through the real coordinator and executor over a scratch database. Its models are
real ``ToolModel`` and ``CompletionModel`` wrappers over scripted providers that record what they
are sent: a query model for the main agent and a different extract model for the Child it spawns.
Everything a test reads back is read where a consumer would meet it: the entries row in
PostgreSQL, the operator's trace export, and the REST, Web and MCP transcript routes.
"""

from __future__ import annotations

import asyncio
import collections
import datetime
import json
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import asyncpg
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository
from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.config.loading import set_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.web_conversations.service import WebConversationService
from dlightrag.engine.agent.environment.toolchain import (
    SearchToolchain,
)
from dlightrag.engine.agent.session.entries import AssistantMessageEntry
from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.reasoning import best_effort_reasoning_profile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.orchestration.orchestrator import AnswerOrchestrator
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools.subagents import SubagentHost
from dlightrag.engine.rag.retrieval.results import (
    RetrievalResult,
)
from dlightrag.engine.runtime.records import run_request_fingerprint
from scripts.export_traces import as_jsonl, export
from tests.conftest import FingerprintingRunStore
from tests.integration.test_answer_run_api_pg import _ANON, _store_backed_application
from tests.integration.test_answer_run_coordinator_pg import (
    _answer_run_request,
    _answer_runtime,
    _wait_for_status,
)
from tests.support.model_provider import (
    ProviderRequest,
    RecordingProvider,
    expected_call,
    tool_model_over,
    untimed,
)
from tests.support.pg import PG_CONN_KWARGS, drop_database, notification_hub, skip_without_postgres
from tests.unit.conftest import answer_image_policy, answer_model_profile

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_OWNER = owner_id_from_user(_ANON)
_QUERY_ENDPOINT = "https://gateway.example/v1"
_EXTRACT_ENDPOINT = "https://extract.example/v1"
#: Every credential and locator the Run is configured with. None of them may be stored.
_SECRETS = (
    "sk-live-QUERY-KEY",
    "QUERY-QUERY-STRING",
    "QUERY-BEARER",
    "QUERY-CUSTOM",
    "QUERY-TOKEN",
    "QUERY-KWARG-KEY",
    "sk-live-EXTRACT-KEY",
    "EXTRACT-BEARER",
    "gateway.example",
    "extract.example",
    "signature=",
)
_PROFILE = ModelProfile(
    context_window_tokens=1_000_000,
    max_output_tokens=64_000,
    reasoning=best_effort_reasoning_profile("openai"),
)
#: What the query model's options digest covers: its header container is not among them, and
#: the key a request carried is masked rather than hashed.
_QUERY_OPTIONS = {"top_p": 0.9, "api_key": "[redacted]", "reasoning_effort": "high"}
#: What each scripted request takes, which the step it made states as its duration.
_REQUEST_SECONDS = 0.05


def _query_settings(**overrides: Any) -> ModelSettings:
    """The main agent's model, configured with a key, a signed URL and request headers."""
    values: dict[str, Any] = {
        "provider": "openai",
        "model": "query-model",
        "api_key": "sk-live-QUERY-KEY",
        "base_url": f"{_QUERY_ENDPOINT}?signature=QUERY-QUERY-STRING",
        "temperature": 0.3,
        "reasoning": "high",
        "model_kwargs": {
            "top_p": 0.9,
            "extra_headers": {
                "Authorization": "Bearer QUERY-BEARER",
                "X-Custom": "QUERY-CUSTOM",
                "X-Custom-Token": "QUERY-TOKEN",
            },
            "api_key": "QUERY-KWARG-KEY",
        },
    }
    return ModelSettings(**{**values, **overrides})


def _extract_settings() -> ModelSettings:
    return ModelSettings(
        provider="openai",
        model="extract-model",
        api_key="sk-live-EXTRACT-KEY",
        base_url=_EXTRACT_ENDPOINT,
        temperature=0.0,
        reasoning="low",
        model_kwargs={"extra_headers": {"Authorization": "Bearer EXTRACT-BEARER"}},
    )


@dataclass
class Script:
    """A provider's scripted turns, and every reply it has given."""

    turns: collections.deque[AssistantTurn]
    final: AssistantTurn
    replies: list[AssistantTurn] = field(default_factory=list)

    def __call__(self, request: ProviderRequest) -> AssistantTurn:
        del request
        turn = self.turns.popleft() if self.turns else self.final
        self.replies.append(turn)
        return turn

    def usage(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for turn in self.replies:
            for key, value in (turn.usage_details or {}).items():
                total[key] = total.get(key, 0) + value
        return total


def _turn(text: str = "", *calls: ToolCall, inputs: int = 1, outputs: int = 1) -> AssistantTurn:
    return AssistantTurn(
        text=text,
        tool_calls=calls,
        stop_reason="tool_use" if calls else "stop",
        usage_details={"input_tokens": inputs, "output_tokens": outputs},
    )


_SEARCH = ToolCall("search-1", "search_knowledge_base", {"query": "figure"})
_SPAWN = ToolCall(
    "spawn-1",
    "spawn_agent",
    {"children": [{"objective": "verify the figure", "model_role": "extract"}]},
)


async def _retrieve(query: str) -> RetrievalResult:
    del query
    return RetrievalResult(
        contexts={
            "chunks": [
                {
                    "chunk_id": "c1",
                    "content": "one grounded fact",
                    "reference_id": "1",
                    "file_path": "doc.txt",
                    "_workspace": "default",
                    "_evidence_key": "search_knowledge_base:c1",
                    "metadata": {
                        "source_type": "corpus",
                        "source_uri": "corpus://doc.txt",
                        "source_download_locator": "/srv/private/doc.txt",
                        "title": "doc.txt",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        },
        trace={"retrieved": 1},
    )


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


@dataclass
class Scratch:
    """A scratch deployment: the stores over a database this test owns."""

    store: FingerprintingRunStore
    web_store: PGWebConversationStore
    pool: Any
    database: str

    @property
    def settings(self) -> dict[str, Any]:
        return {**PG_CONN_KWARGS, "database": self.database}

    async def entries(self, session_id: str) -> list[tuple[str, dict[str, Any]]]:
        """One Session's entries as PostgreSQL keeps them, oldest first."""
        rows = await self.pool.fetch(
            "SELECT entry_type, payload_json FROM dlightrag_agent_session_entries"
            " WHERE owner_id = $1 AND session_id = $2 ORDER BY sequence",
            _OWNER,
            uuid.UUID(session_id),
        )
        return [(row["entry_type"], _json(row["payload_json"])) for row in rows]

    async def assistant_payloads(self, session_id: str) -> list[dict[str, Any]]:
        return [
            payload
            for entry_type, payload in await self.entries(session_id)
            if entry_type == "assistant_message"
        ]

    async def every_entry_text(self) -> str:
        """Every entry of every Session in the database, as the text of its payload."""
        rows = await self.pool.fetch(
            "SELECT payload_json::text AS text FROM dlightrag_agent_session_entries"
        )
        return "\n".join(str(row["text"]) for row in rows)

    async def calls_stored(self) -> int:
        return int(
            await self.pool.fetchval(
                "SELECT count(*) FROM dlightrag_agent_session_entries WHERE payload_json ? 'call'"
            )
        )

    async def exported(self) -> list[dict[str, Any]]:
        records, _ = await export(
            self.settings,
            owner=None,
            since=None,
            until=None,
            run_ids=None,
            limit=1000,
            dedupe=False,
        )
        return records

    async def link_turn(self, session_id: str, run_id: str) -> None:
        """Place the Run in the Web conversation that is its Session, as a submission does."""
        await self.pool.execute(
            "INSERT INTO web_conversation_turns"
            " (turn_id, principal_id, conversation_id, turn_number, submission_id, answer_run_id)"
            " VALUES ($1, $2, $3::text::uuid, 1, $4, $5::text::uuid)",
            uuid.uuid4(),
            _OWNER,
            session_id,
            uuid.uuid4(),
            run_id,
        )


@asynccontextmanager
async def scratch_deployment() -> AsyncIterator[Scratch]:
    await skip_without_postgres()
    database = f"dlightrag_callprov_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": database}, min_size=1, max_size=8
    )
    assert pool is not None
    try:
        async with notification_hub(pool) as hub:
            store = FingerprintingRunStore(pool=pool, notifications=hub)
            await store.initialize()
            web_store = PGWebConversationStore(pool=pool, run_store=store)
            await web_store.initialize()
            yield Scratch(store, web_store, pool, database)
    finally:
        await pool.close()
        await drop_database(database)


@dataclass
class Finished:
    """A Research Run that spawned a Child on another model, run to its end."""

    db: Scratch
    run_id: str
    session_id: str
    plan: AgentRunPlan
    request: dict[str, Any]
    query: RecordingProvider
    extract: RecordingProvider
    query_script: Script
    extract_script: Script
    coordinator: Any
    config: DlightragConfig

    @property
    def store(self) -> FingerprintingRunStore:
        return self.db.store

    async def child_rows(self) -> list[dict[str, Any]]:
        return list(await self.store.list_child_sessions(owner_id=_OWNER, run_id=self.run_id))


def _plan_digest_of(row: dict[str, Any]) -> str:
    return AgentRunPlan.from_payload(row["plan"]).digest


def _web_config(tmp_path: Path) -> DlightragConfig:
    return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        deployment={"working_dir": str(tmp_path / "dlightrag_storage")},
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


@pytest.fixture
async def finished(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[Finished]:
    async with scratch_deployment() as db:
        # Consent is dated by the moment it turns on, and an export covers only conversations that
        # began under it. Write it before the conversation exists, not after its Run finished.
        await db.store.write_privacy_settings(owner_id=_OWNER, contribute=True)
        conversation = await db.web_store.create_conversation(_OWNER)
        session_id = str(conversation["conversation_id"])
        query_script = Script(
            turns=collections.deque(
                [
                    _turn("", _SEARCH, inputs=3),
                    _turn("", _SPAWN, inputs=4),
                    _turn("independent analysis", inputs=5, outputs=2),
                ]
            ),
            final=_turn("final synthesis after the child", inputs=6, outputs=2),
        )
        extract_script = Script(
            turns=collections.deque(), final=_turn("child verified the figure", inputs=2)
        )
        # Every request takes 50 ms of the provider's, which each step must say it took.
        query = RecordingProvider(
            reply=query_script, served_model="query-model-2026-10", delay=_REQUEST_SECONDS
        )
        extract = RecordingProvider(
            reply=extract_script, served_model="extract-model-9", delay=_REQUEST_SECONDS
        )
        query_model = tool_model_over(monkeypatch, query, _query_settings())
        extract_model = tool_model_over(monkeypatch, extract, _extract_settings())

        def resolve_child(role: str) -> Any:
            if role != "extract":
                raise ValueError(f"unexpected child role {role!r}")
            return extract_model, extract_model.stream_text, _PROFILE

        orchestrator = AnswerOrchestrator(
            synthesizer=cast(AnswerSynthesizer, SimpleNamespace()),
            retrieve_knowledge_base=_retrieve,
            model_func=query_model,
            stream_model_func=query_model.stream_text,
            model_profile=_PROFILE,
            telemetry=NOOP_TELEMETRY,
            resolved_mode="research",
            subagent_host=SubagentHost(),
            search_toolchain=SearchToolchain(),
            child_model_resolver=resolve_child,
        )
        plan = AgentRunPlan.from_tools(
            orchestrator.prepare_run("why").tools,
            model_role="query",
            context_policy_revision=CONTEXT_POLICY_REVISION,
        )
        request = _answer_run_request(mode="research", agent_run_plan=plan)
        request["agent_session_id"] = session_id
        application, coordinator = _answer_runtime(db.store, orchestrator=orchestrator)
        creation = await db.store.create_run(owner_id=_OWNER, request=request)
        await db.link_turn(session_id, creation.run.run_id)
        try:
            await coordinator.start()
            coordinator.wake()
            await _wait_for_status(
                db.store,
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                status="succeeded",
                timeout=30.0,
            )
            yield Finished(
                db=db,
                run_id=creation.run.run_id,
                session_id=session_id,
                plan=plan,
                request=request,
                query=query,
                extract=extract,
                query_script=query_script,
                extract_script=extract_script,
                coordinator=coordinator,
                config=_web_config(tmp_path),
            )
        finally:
            await coordinator.aclose()
            await application.aclose()
            await query_model.aclose()
            await extract_model.aclose()


# ── what a Run stores ────────────────────────────────────────────────────────────────────


async def test_each_step_a_run_stores_states_what_its_provider_was_sent(finished: Finished) -> None:
    (child,) = await finished.child_rows()
    main = await finished.db.assistant_payloads(finished.session_id)
    nested = await finished.db.assistant_payloads(child["child_session_id"])

    # Every turn the main agent took, and every turn its Child took, is a step with a call.
    assert len(main) == len(finished.query.requests) >= 4
    assert len(nested) == len(finished.extract.requests) >= 1
    for payload, request in zip(main, finished.query.requests, strict=True):
        assert untimed(payload["call"]) == expected_call(
            request,
            role="query",
            endpoint=_QUERY_ENDPOINT,
            plan_digest=finished.plan.digest,
            reasoning={"requested": "high", "effective": "high"},
            served_model="query-model-2026-10",
            options=_QUERY_OPTIONS,
        )
    # The Child ran on the model of the role it was given, with that model's own parameters.
    for payload, request in zip(nested, finished.extract.requests, strict=True):
        assert untimed(payload["call"]) == expected_call(
            request,
            role="extract",
            endpoint=_EXTRACT_ENDPOINT,
            plan_digest=_plan_digest_of(child),
            reasoning={"requested": "low", "effective": "low"},
            served_model="extract-model-9",
            options={"reasoning_effort": "low"},
        )
    # Each step also stores how long its request took and how long it waited for a slot, as the
    # whole milliseconds PostgreSQL holds them in: the provider took 50 ms over every request.
    for payload in [*main, *nested]:
        assert payload["call"]["duration_ms"] >= 45
        assert payload["call"]["queued_ms"] >= 0
    assert main[0]["call"]["model"]["model"] == "query-model"
    assert nested[0]["call"]["model"]["model"] == "extract-model"
    assert main[0]["call"]["params"]["temperature"] == 0.3
    assert nested[0]["call"]["params"]["temperature"] == 0.0
    # The Child asked something else of another model, so its request is a request of its own.
    assert nested[0]["call"]["prompt"] != main[0]["call"]["prompt"]


async def test_the_totals_a_run_reports_are_the_providers_alone(finished: Finished) -> None:
    run = await finished.store.get_run(owner_id=_OWNER, run_id=finished.run_id)
    assert run is not None and run.result is not None
    (child,) = await finished.child_rows()

    usage = run.result["trace"]["usage"]
    own = finished.query_script.usage()
    nested = finished.extract_script.usage()
    assert usage["usage_details"] == own
    assert usage["child_usage_details"] == nested
    assert usage["inclusive_usage_details"] == {
        key: own.get(key, 0) + nested.get(key, 0) for key in {*own, *nested}
    }
    # A step keeps the counters its provider stated and nothing else among them, and a call holds
    # none of them: the exact `expected_call` of the steps above has no key to hold one.
    stored = await finished.db.assistant_payloads(finished.session_id)
    assert [payload["usage"] for payload in stored] == [
        turn.usage_details for turn in finished.query_script.replies
    ]
    for payload in stored + await finished.db.assistant_payloads(child["child_session_id"]):
        assert "cost" not in payload


# ── what a reader meets ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def _clients(
    finished: Finished, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AsyncClient]:
    """The real REST and Web routes, and the MCP tool, over the Run's own stores."""
    set_config(finished.config)
    application = _store_backed_application(finished.store, finished.config)
    application.web_conversations = WebConversationService(
        store=finished.db.web_store,
        answers=application.answers,
        max_attachments=6,
        cursor_secret=b"call-provenance-web",
    )
    app: FastAPI = create_app()
    app.state.application = application
    app.dependency_overrides[get_current_user] = lambda: _ANON
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            yield client
    finally:
        app.dependency_overrides.clear()


async def _client_views(client: AsyncClient, finished: Finished) -> list[tuple[str, str]]:
    """Every transcript a client can read of the Run: path and the text the server answered."""
    (child,) = await finished.child_rows()
    run, child_id = finished.run_id, child["child_session_id"]
    paths = [
        f"/answer/{run}/transcript",
        f"/answer/{run}/children",
        f"/answer/{run}/children/{child_id}",
        f"/web/api/answer/{run}/transcript",
        f"/web/api/answer/{run}/transcript?child={child_id}",
        f"/web/api/answer/{run}/children",
        f"/web/api/answer/{run}/children/{child_id}",
        f"/web/api/answer/{run}/transcript/search?q=grounded",
    ]
    views: list[tuple[str, str]] = []
    for path in paths:
        response = await client.get(path)
        assert response.status_code == 200, (path, response.text)
        views.append((path, response.text))
    over_mcp: Any = await mcp_server.mcp_app.call_tool("get_answer_transcript", {"run_id": run})
    assert not over_mcp.is_error
    views.append(("mcp:get_answer_transcript", over_mcp.content[0].text))
    return views


def _leaves(value: Any) -> Iterator[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


async def test_the_export_carries_each_steps_call_beside_its_content(finished: Finished) -> None:
    (child,) = await finished.child_rows()

    [record] = await finished.db.exported()

    assert record["run"]["run_id"] == finished.run_id
    main = [step["payload"] for step in record["steps"] if step["type"] == "assistant_message"]
    nested = [
        step["payload"]
        for step in record["children"][0]["steps"]
        if step["type"] == "assistant_message"
    ]
    assert main == await finished.db.assistant_payloads(finished.session_id)
    assert nested == await finished.db.assistant_payloads(child["child_session_id"])
    assert all("call" in payload for payload in main + nested)


async def test_no_credential_leaves_the_store_and_no_client_is_shown_the_call(
    finished: Finished, monkeypatch: pytest.MonkeyPatch
) -> None:
    (child,) = await finished.child_rows()
    stored = [
        *await finished.db.assistant_payloads(finished.session_id),
        *await finished.db.assistant_payloads(child["child_session_id"]),
    ]
    rows = await finished.db.every_entry_text()
    exported = as_jsonl(await finished.db.exported())
    # Whatever a call states: every value long enough to be a model name or a digest.
    stated = {str(leaf) for payload in stored for leaf in _leaves(payload["call"])}
    stated = {text for text in stated if len(text) >= 8}
    assert {"query-model", "extract-model", "query-model-2026-10"} <= stated

    async with _clients(finished, monkeypatch) as client:
        views = await _client_views(client, finished)

    # The text is really there to be searched: the steps and the export hold their calls.
    assert '"call"' in rows and '"call"' in exported
    everything = {"the entries rows": rows, "the export": exported, **dict(views)}
    for where, text in everything.items():
        for secret in _SECRETS:
            assert secret not in text, (where, secret)
    forbidden = {
        "call",
        "prompt",
        "options_digest",
        "plan_digest",
        "served_model",
        "queued_ms",
        "duration_ms",
    }
    for path, text in views:
        assert not forbidden & set(_keys(json.loads(text))), path
        for value in stated:
            assert value not in text, (path, value)
    # The views are not empty: they show the Run's own words, which is what a reader is for.
    assert "final synthesis" in views[0][1] and "grounded" in views[3][1]


async def test_deleting_the_conversation_leaves_no_call_behind(finished: Finished) -> None:
    assert await finished.db.calls_stored() >= 5

    deleted = await finished.db.web_store.delete_conversation(_OWNER, finished.session_id)

    assert deleted is True
    assert await finished.db.calls_stored() == 0
    for table in ("dlightrag_agent_session_entries", "dlightrag_agent_sessions"):
        assert await finished.db.pool.fetchval(f"SELECT count(*) FROM {table}") == 0, table


async def test_steps_written_before_the_call_existed_are_read_everywhere_and_continued(
    finished: Finished, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row an older build wrote has no `call` key; every reader takes it as such."""
    async with _clients(finished, monkeypatch) as client:
        before = await _client_views(client, finished)
    await finished.db.pool.execute(
        "UPDATE dlightrag_agent_session_entries SET payload_json = payload_json - 'call'"
    )
    assert await finished.db.calls_stored() == 0

    # The repository decodes every entry of the Session, assistant messages included.
    repository = PGAgentSessionRepository(
        pool=finished.db.pool,
        owner_id=_OWNER,
        run_id=uuid.UUID(finished.run_id),
        worker_id="reader",
        lease_owner="reader",
        fencing_epoch=1,
    )
    snapshot = await repository.load(SessionId(finished.session_id))
    assistants = [e for e in snapshot.entries if isinstance(e, AssistantMessageEntry)]
    assert assistants and all(entry.call is None for entry in assistants)
    # Clients read the same words, and the operator's export still writes the Run.
    async with _clients(finished, monkeypatch) as client:
        assert await _client_views(client, finished) == before
    [record] = await finished.db.exported()
    assert all("call" not in step["payload"] for step in record["steps"])

    # The conversation goes on over them: the follow-up reads the old turns and writes its own.
    finished.query_script.turns.append(_turn("and next year, more", inputs=7, outputs=3))
    follow_up = {
        **finished.request,
        "query": "and next year?",
        "agent_session_id": finished.session_id,
        "agent_lane_id": "main",
        "parent_run_id": finished.run_id,
        "continuation_kind": "follow_up",
    }
    creation = await finished.store.create_run(
        owner_id=_OWNER,
        request=follow_up,
        idempotency_fingerprint=run_request_fingerprint(follow_up),
    )
    finished.coordinator.wake()
    await _wait_for_status(
        finished.store,
        owner_id=_OWNER,
        run_id=creation.run.run_id,
        status="succeeded",
        timeout=30.0,
    )
    payloads = await finished.db.assistant_payloads(finished.session_id)
    assert payloads[-1]["content"] == "and next year, more"
    assert "call" in payloads[-1]
    assert all("call" not in payload for payload in payloads[:-1])


# ── Fast ─────────────────────────────────────────────────────────────────────────────────


class _Clock:
    """``datetime`` as the synthesizer reads it, standing still at one moment."""

    def __init__(self, moment: datetime.datetime) -> None:
        self._moment = moment

    def now(self, tz: datetime.tzinfo | None = None) -> datetime.datetime:
        return self._moment if tz is None else self._moment.astimezone(tz)


async def _fast_run(
    db: Scratch,
    monkeypatch: pytest.MonkeyPatch,
    provider: RecordingProvider,
    *,
    settings: ModelSettings,
    clock: datetime.datetime,
) -> tuple[str, str]:
    """One Fast Run to its end over a real synthesizer and ``CompletionModel``."""
    monkeypatch.setattr(
        "dlightrag.engine.ai.providers.get_provider", lambda *_args, **_kwargs: provider
    )
    monkeypatch.setattr("dlightrag.engine.answer.synthesizer.datetime", _Clock(clock))
    model = CompletionModel(
        settings, scheduler=ModelScheduler(max_concurrency=2), telemetry=NOOP_TELEMETRY
    )
    orchestrator = AnswerOrchestrator(
        synthesizer=AnswerSynthesizer(
            image_policy=answer_image_policy(),
            model_profile=answer_model_profile(),
            model_func=model,
        ),
        retrieve_knowledge_base=_retrieve,
        model_profile=answer_model_profile(),
        telemetry=NOOP_TELEMETRY,
        resolved_mode="fast",
        search_toolchain=SearchToolchain(),
    )
    session_id = str(uuid.uuid7())
    request = _answer_run_request(mode="fast")
    request["agent_session_id"] = session_id
    application, coordinator = _answer_runtime(db.store, orchestrator=orchestrator)
    creation = await db.store.create_run(owner_id=_OWNER, request=request)
    try:
        await coordinator.start()
        coordinator.wake()
        await _wait_for_status(
            db.store, owner_id=_OWNER, run_id=creation.run.run_id, status="succeeded", timeout=30.0
        )
    finally:
        await coordinator.aclose()
        await application.aclose()
        await model.aclose()
    return creation.run.run_id, session_id


async def test_a_fast_answer_states_the_call_that_wrote_it(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = RecordingProvider(
        text="The figure was 12 [1].", served_model="fast-model-3", delay=_REQUEST_SECONDS
    )
    settings = _query_settings(model="fast-model", reasoning=None)
    async with scratch_deployment() as db:
        run_id, session_id = await _fast_run(
            db,
            monkeypatch,
            provider,
            settings=settings,
            clock=datetime.datetime(2026, 10, 10, 9, 30, tzinfo=datetime.UTC),
        )
        [turn] = await db.assistant_payloads(session_id)
        run = await db.store.get_run(owner_id=_OWNER, run_id=run_id)

    [request] = provider.requests
    assert request.max_tokens is not None
    # No plan: a Fast answer runs no Agent Operation. No tools either, so `tools` digests the
    # empty list and no tool choice is stated. It answers on the query role.
    assert untimed(turn["call"]) == expected_call(
        request,
        role="query",
        endpoint=_QUERY_ENDPOINT,
        plan_digest=None,
        reasoning=None,
        served_model="fast-model-3",
        options={"top_p": 0.9, "api_key": "[redacted]"},
    )
    # The stream's own timings reach the entry too: it took the provider's 50 ms.
    assert turn["call"]["duration_ms"] >= 45 and turn["call"]["queued_ms"] >= 0
    # The usage is what the provider stated, and the Run's totals sum nothing else.
    assert turn["usage"] == {"prompt_tokens": 7, "completion_tokens": 3}
    assert run is not None and run.result is not None
    assert run.result["trace"]["usage"] == {
        "usage_details": {"prompt_tokens": 7, "completion_tokens": 3}
    }
    for secret in _SECRETS:
        assert secret not in json.dumps(turn)
        assert secret not in json.dumps(run.result)
    assert "fast-model-3" not in json.dumps(run.result)


async def test_a_fast_answers_system_and_tools_do_not_move_with_the_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _query_settings(model="fast-model", reasoning=None)
    async with scratch_deployment() as db:
        calls = []
        for moment in (
            datetime.datetime(2026, 10, 10, 9, 30, tzinfo=datetime.UTC),
            datetime.datetime(2027, 3, 1, 21, 5, tzinfo=datetime.UTC),
        ):
            provider = RecordingProvider(text="The figure was 12 [1].")
            _, session_id = await _fast_run(
                db, monkeypatch, provider, settings=settings, clock=moment
            )
            [turn] = await db.assistant_payloads(session_id)
            calls.append(turn["call"])

    early, late = calls
    assert early["prompt"]["system"] == late["prompt"]["system"]
    assert early["prompt"]["tools"] == late["prompt"]["tools"]
    # The question states the time, so the whole request is not the same request.
    assert early["prompt"]["request"] != late["prompt"]["request"]


# ── a Run reclaimed after its configuration moved ───────────────────────────────────────


async def test_a_reclaimed_run_records_the_temperature_it_was_reclaimed_under(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted at 0.3 and detached after its first turn, the Run resumes at 0.9."""
    in_flight = asyncio.Event()

    async def first_worker(request: ProviderRequest) -> AssistantTurn:
        if any(message["role"] == "tool" for message in request.messages):
            in_flight.set()
            await asyncio.Event().wait()
        return _turn("", _SEARCH)

    def orchestrator_for(model: Any) -> AnswerOrchestrator:
        return AnswerOrchestrator(
            synthesizer=cast(AnswerSynthesizer, SimpleNamespace()),
            retrieve_knowledge_base=_retrieve,
            model_func=model,
            stream_model_func=model.stream_text,
            model_profile=_PROFILE,
            telemetry=NOOP_TELEMETRY,
            resolved_mode="research",
            subagent_host=SubagentHost(),
            search_toolchain=SearchToolchain(),
        )

    async with scratch_deployment() as db:
        first_provider = RecordingProvider(reply=first_worker)
        first_model = tool_model_over(monkeypatch, first_provider, _query_settings())
        first = orchestrator_for(first_model)
        plan = AgentRunPlan.from_tools(
            first.prepare_run("why").tools,
            model_role="query",
            context_policy_revision=CONTEXT_POLICY_REVISION,
        )
        request = _answer_run_request(mode="research", agent_run_plan=plan)
        session_id = request["agent_session_id"]
        first_app, first_coordinator = _answer_runtime(db.store, orchestrator=first)
        creation = await db.store.create_run(owner_id=_OWNER, request=request)
        await first_coordinator.start()
        first_coordinator.wake()
        try:
            await asyncio.wait_for(in_flight.wait(), timeout=30.0)
        finally:
            await first_coordinator.aclose()
            await first_app.aclose()
            await first_model.aclose()

        # Configuration moved while the Run waited: the same model at another temperature.
        second_provider = RecordingProvider(reply=lambda _request: _turn("resumed answer"))
        second_model = tool_model_over(
            monkeypatch, second_provider, _query_settings(temperature=0.9)
        )
        second_app, second_coordinator = _answer_runtime(
            db.store, orchestrator=orchestrator_for(second_model)
        )
        try:
            await second_coordinator.start()
            second_coordinator.wake()
            await _wait_for_status(
                db.store,
                owner_id=_OWNER,
                run_id=creation.run.run_id,
                status="succeeded",
                timeout=30.0,
            )
        finally:
            await second_coordinator.aclose()
            await second_app.aclose()
            await second_model.aclose()
        steps = await db.assistant_payloads(session_id)

    assert [step["call"]["params"]["temperature"] for step in steps] == [0.3, 0.9]
    assert steps[0]["call"]["plan_digest"] == steps[1]["call"]["plan_digest"] == plan.digest
    assert steps[1]["call"]["params"]["temperature"] == second_provider.requests[-1].temperature
