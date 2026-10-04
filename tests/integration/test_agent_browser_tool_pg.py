# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The ``browser`` tool through real Run settlement, Child Sessions, and recovery.

A Research Run drives the tool in a real Chromium behind a real Playwright run-server, leased
through PostgreSQL as production leases it. What the tool admits settles with the call that made
it, in a disposable database; the web is a loopback proxy serving ``http://*.example`` pages.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, cast

import pytest

from dlightrag.adapters.agent_browser import PooledBrowserProvider
from dlightrag.adapters.postgres.runtime.browser_leases import PGAgentBrowserLeaseStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.operation import OperationCompleted
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.runtime import AgentSessionRuntime
from dlightrag.engine.agent.tool_content import tool_content_text
from dlightrag.engine.agent.tools.files import ResourceReadRequest
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.agent_browser import (
    AgentBrowserSettings,
    BrowserHolder,
    RunAgentBrowser,
)
from dlightrag.engine.answer.research.runtime import (
    FetchedResourceBuffer,
    ResearchRuntimeEffects,
    _bound_child_dispatch_preparer,
    _bound_child_runner,
    _check_child_write,
)
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.browser import BrowserToolHost
from dlightrag.engine.answer.tools.resources import make_resource_reader
from dlightrag.engine.answer.tools.subagents import SubagentHost
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_attachment_replay_pg import (
    OWNER,
    drive,
    executor,
    new_run,
    orchestrator,
)
from tests.support.agent_browser import (
    Served,
    WebProxy,
    browser_settings,
    idle_accounts,
    run_server,
    web_proxy,
)
from tests.support.dns import public_dns
from tests.support.pg import skip_without_postgres
from tests.tool_helpers import tool_runtime
from tests.unit.conftest import answer_model_profile
from tests.unit.test_research_runtime_migration import _Session

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

SHOP = "http://shop.example"
HOME = """<html><head><title>Shop</title></head><body><h1>Shop</h1>
<form action="/results" method="get"><input aria-label="Search" name="q">
<button type="submit">Go</button></form></body></html>"""
RESULTS = """<html><head><title>Results</title></head><body><h1>Results</h1>
<p>Results for lamp: Albert Einstein, 1921.</p><a href="/data.csv">Download CSV</a></body></html>"""
PAGES = {
    f"{SHOP}/": Served(HOME),
    f"{SHOP}/results?q=lamp": Served(RESULTS),
    f"{SHOP}/data.csv": Served(
        "name,year\nEinstein,1921\n",
        headers={
            "content-type": "text/csv",
            "content-disposition": "attachment; filename=data.csv",
        },
    ),
}


@pytest.fixture(autouse=True)
async def _postgres(monkeypatch: pytest.MonkeyPatch) -> None:
    await skip_without_postgres()
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)


@pytest.fixture
async def pg() -> AsyncIterator[tuple[PGRunStore, Any]]:
    async with isolated_run_runtime("agent_browser_tool") as pair:
        yield pair


def ref_of(text: str, name: str) -> str:
    for line in text.splitlines():
        if name in line and (marker := re.search(r"\[ref=([^\]]+)\]", line)):
            return marker.group(1)
    raise AssertionError(f"no ref for {name!r} in:\n{text}")


def handle_in(text: str) -> str:
    found = re.search(r"\b(res-[0-9a-f]{24})\b", text)
    assert found is not None, text
    return found.group(1)


Step = Callable[[str], ToolCall | str | Awaitable[ToolCall | str]]


class Script:
    """A model that takes its steps in turn, each seeing the text of the last tool result.

    A step returns the call the model makes, or the text it answers with, as a value or
    awaited. ``results`` holds every tool result the model has been shown, by call id.
    """

    def __init__(self, *steps: Step) -> None:
        self._steps = iter(steps)
        self.results: dict[str, str] = {}

    async def __call__(self, **kwargs: Any) -> AssistantTurn:
        shown = [m for m in kwargs["messages"] if m.get("role") == "tool"]
        self.results.update({m["tool_call_id"]: str(m["content"]) for m in shown})
        outcome = next(self._steps)(str(shown[-1]["content"]) if shown else "")
        if isinstance(outcome, Awaitable):
            outcome = await outcome
        if isinstance(outcome, str):
            return AssistantTurn(text=outcome, tool_calls=(), stop_reason="stop")
        return AssistantTurn(text="", tool_calls=(outcome,), stop_reason="tool_use")


def call(call_id: str, action: str, **arguments: Any) -> Step:
    return lambda _last: ToolCall(call_id, "browser", {"action": action, **arguments})


class CountingLeases(PGAgentBrowserLeaseStore):
    """The lease store, keeping every endpoint it granted."""

    def __init__(self, *, pool: Any) -> None:
        super().__init__(pool=pool)
        self.granted: list[str] = []

    async def claim(
        self, holder: BrowserHolder, endpoints: Sequence[str], exclude: Sequence[str] = ()
    ) -> str | None:
        endpoint = await super().claim(holder, endpoints, exclude)
        if endpoint is not None:
            self.granted.append(endpoint)
        return endpoint


@dataclass
class World:
    """A pool of one real run-server, leased through the disposable database."""

    pool: Any
    server: Any
    proxy: WebProxy
    provider: PooledBrowserProvider
    leases: CountingLeases

    def browser(self, session: Any, **bounds: Any) -> RunAgentBrowser:
        """The browser the worker holding ``session`` gives its Run."""
        holder = BrowserHolder(
            session.owner_id, session.run_id, session.worker_id, session.fencing_epoch
        )
        settings: AgentBrowserSettings = browser_settings(
            **{"navigation": 5, "settle": 0.3, "action": 3, **bounds}
        )
        return RunAgentBrowser(self.provider, holder, settings)

    async def holder(self) -> str | None:
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                "SELECT run_id::text FROM dlightrag_agent_browser_leases WHERE endpoint = $1",
                self.server.endpoint,
            )

    async def free(self, *, seconds: float = 10.0) -> None:
        deadline = time.monotonic() + seconds
        while await self.holder() is not None:
            assert time.monotonic() < deadline, "the endpoint is still held"
            await asyncio.sleep(0.05)


@asynccontextmanager
async def world(
    pg: tuple[PGRunStore, Any], pages: dict[str, Served] | None = None
) -> AsyncIterator[World]:
    # The slow page never answers, and is let go when the test is over.
    slow = Served("never", hold=asyncio.Event())
    async with AsyncExitStack() as stack:
        server = await stack.enter_async_context(run_server())
        proxy = await stack.enter_async_context(
            web_proxy({**(PAGES if pages is None else pages), f"{SHOP}/slow": slow})
        )
        leases = CountingLeases(pool=pg[1])
        provider = PooledBrowserProvider(
            endpoints=(server.endpoint,),
            egress_proxy=proxy.url,
            chromium_sandbox=True,
            connect_timeout_seconds=20,
            leases=leases,
        )
        stack.push_async_callback(provider.aclose)
        yield World(pg[1], server, proxy, provider, leases)


def attempt_of(
    model: Callable[..., Any], web: World, session: Any, **bounds: Any
) -> tuple[RunAgentBrowser, ResourceRegistry, FetchedResourceBuffer, Any]:
    """One worker's attempt: its browser, its Run's registry, and the orchestrator over them."""
    run = web.browser(session, **bounds)
    buffer = FetchedResourceBuffer()

    async def sink(fetched: Any, owner: Any) -> None:
        buffer.append(fetched, owner)

    registry = ResourceRegistry(
        resource_secret=b"browser-run", cursor_secret=b"browser-cursor", fetched_bytes_sink=sink
    )
    host = orchestrator(
        model,
        registry=registry,
        browser=BrowserToolHost(
            run, registry, make_resource_reader(registry, 4000), idle_accounts()
        ),
    )
    return run, registry, buffer, host


async def stored_rows(pool: Any, run_id: str) -> list[dict[str, Any]]:
    """The Run's settled Resource rows as recovery reads them."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT resource_id, capabilities, source_locator, session_id::text AS session_id,"
            " intent_id::text AS intent_id FROM dlightrag_answer_resources WHERE run_id = $1"
            " ORDER BY resource_id",
            uuid.UUID(run_id),
        )
    return [
        {
            **json.loads(row["capabilities"]),
            "resource_id": row["resource_id"],
            "locator": bytes(row["source_locator"]).decode(),
            "session_id": row["session_id"],
            "intent_id": row["intent_id"],
        }
        for row in rows
    ]


def intents_of(snapshot: Any) -> dict[str, str]:
    """The intent each tool call of the Session was settled under, by call id."""
    return {
        entry.result.call_id: entry.intent_id.value
        for entry in snapshot.entries
        if isinstance(entry, ToolResultMessageEntry) and entry.intent_id is not None
    }


async def test_captures_and_downloads_settle_with_their_call_and_restore_without_the_browser(
    pg, monkeypatch: pytest.MonkeyPatch
) -> None:
    session, session_id = await new_run(pg[0])
    model = Script(
        call("open", "navigate", url=f"{SHOP}/"),
        lambda last: ToolCall(
            "search",
            "browser",
            {"action": "type", "ref": ref_of(last, "Search"), "text": "lamp", "submit": True},
        ),
        lambda last: ToolCall(
            "download", "browser", {"action": "click", "ref": ref_of(last, "Download CSV")}
        ),
        lambda last: ToolCall("read-download", "read", {"resource_id": handle_in(last)}),
        call("capture", "capture"),
        lambda last: ToolCall("read-capture", "read", {"resource_id": handle_in(last)}),
        lambda _last: "done",
    )
    async with world(pg) as web:
        run, registry, buffer, host = attempt_of(model, web, session)
        try:
            snapshot = await drive(
                session,
                session_id,
                host,
                host.prepare_run("search", registry=registry),
                fetched_buffer=buffer,
            )
        finally:
            await run.aclose()
            await registry.aclose()

        assert [request.target for request in web.proxy.requests] == [
            f"{SHOP}/",
            f"{SHOP}/results?q=lamp",
            f"{SHOP}/data.csv",
        ]
    download_id, capture_id = (
        handle_in(model.results["download"]),
        handle_in(model.results["capture"]),
    )
    intents = intents_of(snapshot)
    rows = {row["resource_id"]: row for row in await stored_rows(pg[1], session.run_id)}
    download, capture = rows[download_id], rows[capture_id]
    assert (download["resource_kind"], download["acquisition"]) == ("web", "browser_download")
    assert (capture["resource_kind"], capture["acquisition"]) == ("web", "browser_capture")
    assert download["admission_origin"] == capture["admission_origin"] == "agent"
    assert download["locator"] == f"{SHOP}/data.csv"
    assert capture["locator"] == f"{SHOP}/results?q=lamp"
    # Each settled with the call that made it, in the Session that made it.
    assert (download["intent_id"], capture["intent_id"]) == (
        intents["download"],
        intents["capture"],
    )
    assert {download["session_id"], capture["session_id"]} == {session_id.value}
    views = {
        row["locator"] for row in rows.values() if row["resource_kind"] == "conversion_snapshot"
    }
    assert views == {download_id, capture_id}

    # A resumed Run reads both as it did, with no browser, no fetch and no conversion.
    async def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a resumed Run neither converts nor browses")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.fetch_public_http", forbidden)
    async with ResourceRegistry(
        resource_secret=b"browser-run", cursor_secret=b"browser-cursor", page_renderer=forbidden
    ) as resumed:
        await executor(pg)._restore_registry_fetches(resumed, owner_id=OWNER, run_id=session.run_id)
        read = make_resource_reader(resumed, 4000)
        for resource_id, original in ((download_id, "read-download"), (capture_id, "read-capture")):
            again = await read(
                ResourceReadRequest(resource_id=resource_id, url=None, focus=None, cursor=None),
                tool_runtime(tool_name="read"),
            )
            # The model saw the read under the citation label its evidence was given.
            assert again.text_content in model.results[original]
        source = resumed.evidence_source(capture_id, text=True)
        assert (source["source_type"], source["source_uri"]) == (
            "web_search",
            f"{SHOP}/results?q=lamp",
        )


async def test_the_lease_is_held_while_a_page_is_open_and_the_idle_clock_starts_when_none_is(
    pg,
) -> None:
    session, _ = await new_run(pg[0])
    async with world(pg) as web:
        run = web.browser(session, idle=0.2)
        navigate = lambda s: s.navigate(f"{SHOP}/")  # noqa: E731
        try:
            await run.with_page("parent", navigate, open_page=True)
            await asyncio.sleep(0.6)
            assert await web.holder() == session.run_id

            await run.close_page("parent")
            await asyncio.sleep(0.6)
            assert await web.holder() is None

            await run.with_page("parent", navigate, open_page=True)
            await asyncio.sleep(0.6)
            assert await web.holder() == session.run_id
        finally:
            await run.aclose()
        assert await web.holder() is None


def children_of(
    host: Any,
    session: Any,
    session_id: SessionId,
    run: RunAgentBrowser,
    buffer: FetchedResourceBuffer,
    pg: tuple[PGRunStore, Any],
) -> SubagentHost:
    """The Child Session machinery of a Run, bound as the executor binds it."""
    store = pg[0]

    def fenced(write: Any, **extra: Any) -> Any:
        return _check_child_write(
            partial(
                write, worker_id=session.worker_id, fencing_epoch=session.fencing_epoch, **extra
            )
        )

    persist = fenced(store.upsert_child_session)
    subagents = SubagentHost(
        parent_session_id=session_id,
        owner_id=OWNER,
        run_id=session.run_id,
        persist=persist,
        load_child=store.load_child_session,
        list_children=store.list_child_sessions,
        finish_child=fenced(store.finish_child_session),
        request_cancel=fenced(store.request_child_cancellation),
        release_children=fenced(store.release_child_sessions),
        prepare_dispatch=_bound_child_dispatch_preparer(host),
        run_child=_bound_child_runner(
            close_agent_page=run.close_page,
            telemetry=NOOP_TELEMETRY,
            orchestrator=host,
            repository=session.execution.session_repository,
            session=session,
            fetched_buffer=buffer,
            parent_session_id=session_id,
            persist_child_runtime=persist,
            claim_child=fenced(store.claim_child_session),
            renew_child=fenced(store.heartbeat_child_session),
            load_child=store.load_child_session,
            restore_child_attachments=lambda child, context: executor(
                pg
            )._restore_child_attachments(session, child, context),
        ),
    )
    host._subagent_host = subagents
    return subagents


def spawn(call_id: str, *objectives: str) -> ToolCall:
    return ToolCall(
        call_id,
        "spawn_agent",
        {
            "children": [
                {"objective": objective, "context": "isolated", "tools": ["browser"]}
                for objective in objectives
            ]
        },
    )


def by_site(**models: Callable[..., Any]) -> Callable[..., Any]:
    """A Child model that follows the script written for the site its objective names."""

    async def model(**kwargs: Any) -> AssistantTurn:
        objective = " ".join(str(m["content"]) for m in kwargs["messages"] if m["role"] == "user")
        site = next(name for name in models if name in objective)
        return await models[site](**kwargs)

    return model


def site_script(site: str) -> Script:
    """One Child's work: open a site's page, capture it, and answer."""
    return Script(
        call(f"{site}-open", "navigate", url=f"http://{site}.example/"),
        call(f"{site}-capture", "capture"),
        lambda _last: f"{site} done",
    )


def parent_spawning(*objectives: str) -> Callable[..., Any]:
    return Script(lambda _last: spawn("spawn", *objectives), lambda _last: "dispatched")


async def eventually(condition: Callable[[], bool], *, seconds: float = 15.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "the condition never held"
        await asyncio.sleep(0.02)


async def test_two_children_browse_in_parallel_without_interfering(pg) -> None:
    session, session_id = await new_run(pg[0])
    alpha_gate = asyncio.Event()
    pages = {
        **PAGES,
        "http://alpha.example/": Served(
            "<html><head><title>Alpha</title></head><body><p>Alpha stock</p></body></html>",
            hold=alpha_gate,
        ),
        "http://beta.example/": Served(
            "<html><head><title>Beta</title></head><body><p>Beta stock</p></body></html>"
        ),
    }
    alpha, beta = site_script("alpha"), site_script("beta")
    async with world(pg, pages) as web:
        run, registry, buffer, host = attempt_of(
            parent_spawning("Capture the alpha stock page.", "Capture the beta stock page."),
            web,
            session,
            idle=0.3,
        )
        host._child_model_resolver = cast(
            Any,
            lambda role: (
                by_site(alpha=alpha, beta=beta),
                None,
                answer_model_profile(supports_images=True),
            ),
        )
        subagents = children_of(host, session, session_id, run, buffer, pg)

        async def release_alpha_once_beta_is_asked() -> None:
            await eventually(lambda: bool(web.proxy.fetched("http://beta.example/")))
            alpha_gate.set()

        releasing = asyncio.create_task(release_alpha_once_beta_is_asked())
        try:
            await drive(
                session,
                session_id,
                host,
                host.prepare_run("browse two sites", registry=registry),
                fetched_buffer=buffer,
            )
            outcomes = await asyncio.wait_for(asyncio.gather(*subagents.tasks.values()), 30)
            await releasing
            children = {outcome.summary: outcome.child_session_id for outcome in outcomes}

            # Alpha's page was held until beta's was asked for, so the two pages were open at
            # once. Each Child saw only its own site, and one browser served them both.
            assert [outcome.status for outcome in outcomes] == ["succeeded", "succeeded"]
            seen = {
                name: " ".join(model.results.values())
                for name, model in (("alpha", alpha), ("beta", beta))
            }
            assert "Alpha stock" in seen["alpha"] and "beta" not in seen["alpha"].lower()
            assert "Beta stock" in seen["beta"] and "alpha" not in seen["beta"].lower()
            assert web.leases.granted == [web.server.endpoint]
            captures = {
                row["locator"]: row["session_id"]
                for row in await stored_rows(pg[1], session.run_id)
                if row.get("acquisition") == "browser_capture"
            }
            assert captures == {
                "http://alpha.example/": children["alpha done"],
                "http://beta.example/": children["beta done"],
            }
            # Their pages closed with their drives, so the Run's browser rests and goes back.
            assert all(run.current_url(child) is None for child in children.values())
            assert await web.holder() == session.run_id
            await web.free()
        finally:
            releasing.cancel()
            await run.aclose()
            await registry.aclose()
        assert await web.holder() is None


async def test_a_childs_page_closes_when_its_drive_is_cancelled(pg) -> None:
    session, session_id = await new_run(pg[0])
    async with world(pg) as web:
        cancelled: dict[str, Any] = {}

        async def cancel_the_child(_last: str) -> ToolCall:
            await eventually(lambda: bool(web.proxy.fetched(f"{SHOP}/slow")))
            # The Child's page is open in the Run's browser while it waits for the slow page.
            cancelled["held"] = await web.holder()
            (child,) = await pg[0].list_child_sessions(owner_id=OWNER, run_id=session.run_id)
            cancelled["id"] = child["child_session_id"]
            return ToolCall(
                "cancel", "cancel_subagent", {"child_session_id": child["child_session_id"]}
            )

        parent = Script(
            lambda _last: spawn("spawn", "Open the slow page on alpha."),
            cancel_the_child,
            lambda _last: "cancelled the child",
        )
        run, registry, buffer, host = attempt_of(parent, web, session, idle=0.3, navigation=30)
        host._child_model_resolver = cast(
            Any,
            lambda role: (
                Script(call("slow", "navigate", url=f"{SHOP}/slow"), lambda _last: "never"),
                None,
                answer_model_profile(supports_images=True),
            ),
        )
        subagents = children_of(host, session, session_id, run, buffer, pg)
        try:
            await drive(
                session,
                session_id,
                host,
                host.prepare_run("cancel the browsing child", registry=registry),
                fetched_buffer=buffer,
            )
            (outcome,) = await asyncio.wait_for(asyncio.gather(*subagents.tasks.values()), 30)

            assert outcome.status == "cancelled"
            assert cancelled["held"] == session.run_id
            assert run.current_url(cancelled["id"]) is None
            await web.free()
        finally:
            await run.aclose()
            await registry.aclose()


def runtime_of(
    session: Any, session_id: SessionId, host: Any, prepared: Any, buffer: FetchedResourceBuffer
) -> AgentSessionRuntime[Any]:
    return AgentSessionRuntime(
        repository=session.execution.session_repository,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=host,
            prepared=prepared,
            session=session,
            session_id=session_id,
            fetched_buffer=buffer,
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=session.fencing_epoch,
        holder="run",
    )


def plan_of(prepared: Any) -> AgentRunPlan:
    return AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="test",
        model_identity={"role": "query"},
        model_profile=asdict(prepared.model_profile),
    )


async def test_a_pending_call_settles_unknown_and_resumes_with_no_page(pg) -> None:
    store, pool = pg
    session, session_id = await new_run(store)
    async with world(pg) as web:
        first = Script(
            call("open", "navigate", url=f"{SHOP}/"), call("stall", "navigate", url=f"{SHOP}/slow")
        )
        run, registry, buffer, host = attempt_of(first, web, session, navigation=30)
        prepared = host.prepare_run("browse", registry=registry)
        runtime = runtime_of(session, session_id, host, prepared, buffer)
        accepted = await runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key=session.run_id,
            content="browse",
            plan=plan_of(prepared),
        )
        old_holder = BrowserHolder(
            session.owner_id, session.run_id, session.worker_id, session.fencing_epoch
        )
        driving = asyncio.create_task(
            runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
        )
        await eventually(lambda: bool(web.proxy.fetched(f"{SHOP}/slow")))
        assert await web.holder() == session.run_id

        # The worker is detached from its Run mid-call: no cancellation was requested.
        driving.cancel()
        with pytest.raises(asyncio.CancelledError):
            await driving
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs SET lease_expires_at = NOW() - INTERVAL '1 second'"
                " WHERE run_id = $1",
                uuid.UUID(session.run_id),
            )
        claim = await store.claim_next(worker_id="worker-2")
        assert claim is not None and claim.run.fencing_epoch == session.fencing_epoch + 1
        resumed: Any = _Session()
        resumed.owner_id, resumed.run_id, resumed.worker_id = OWNER, session.run_id, "worker-2"
        resumed.fencing_epoch, resumed.execution = claim.run.fencing_epoch, claim.execution

        # The dead attempt's browser is not the new attempt's: it can lease nothing.
        assert await web.leases.claim(old_holder, (web.server.endpoint,)) is None
        second = Script(
            call("looks", "snapshot"),
            call("start-over", "navigate", url=f"{SHOP}/"),
            lambda _last: "done",
        )
        run2, registry2, buffer2, host2 = attempt_of(second, web, resumed)
        prepared2 = host2.prepare_run("browse", registry=registry2)
        again = runtime_of(resumed, session_id, host2, prepared2, buffer2)
        accepted2 = await again.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key=session.run_id,
            content="browse",
            plan=plan_of(prepared2),
        )
        try:
            assert accepted2.operation_id == accepted.operation_id
            final = await again.drive(session_id=session_id, operation_id=accepted.operation_id)
            snapshot = await resumed.execution.session_repository.load(session_id)
            results = {
                entry.result.call_id: entry.result
                for entry in snapshot.entries
                if isinstance(entry, ToolResultMessageEntry)
            }
            assert isinstance(final.state, OperationCompleted)
            # The call that was in flight is not replayed, and nothing opens a page for it.
            assert results["stall"].outcome == "outcome_unknown"
            assert results["looks"].outcome == "failed"
            looks = tool_content_text(results["looks"].parts)
            assert "No page is open in this Agent Session" in looks
            assert "a Run that resumed after an interruption starts with no open page" in looks
            assert web.proxy.fetched(f"{SHOP}/slow") != [] and len(web.proxy.requests) == 3
            assert results["start-over"].outcome == "succeeded"
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT lease_owner, fencing_epoch FROM dlightrag_agent_browser_leases"
                    " WHERE endpoint = $1",
                    web.server.endpoint,
                )
            assert (row["lease_owner"], row["fencing_epoch"]) == ("worker-2", resumed.fencing_epoch)
        finally:
            await run2.aclose()
            await registry2.aclose()
            await run.aclose()
            await registry.aclose()
