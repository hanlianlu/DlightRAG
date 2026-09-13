# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Actual PDF/XLSX adapters through Host settlement and isolated PG recovery."""

import hashlib
import re

import pytest

from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore
from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
from dlightrag.engine.agent.tool_content import tool_content_attachments
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.answer.execution.executor import AnswerExecutor
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.resources.snapshots import ConversionSnapshot
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_attachment_replay_pg import (  # noqa: F401
    OWNER,
    drive,
    new_run,
    orchestrator,
)
from tests.unit.test_format_routes import assert_route_gold, route_source

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@pytest.fixture
async def pg():
    async with isolated_run_runtime("format_routes") as pair:
        async with pair[1].acquire() as conn:
            assert int(await conn.fetchval("SHOW server_version_num")) >= 180000
        yield pair


@pytest.mark.parametrize("kind", ["multi", "scan", "mixed", "nonlatin", "xlsx"])
async def test_format_route_host_settlement_and_recovery(pg, monkeypatch, kind):
    store, pool = pg
    session, session_id = await new_run(store)
    source = route_source(kind)
    async with ResourceRegistry(resource_secret=b"format", cursor_secret=b"cursor") as registry:
        resource = registry.register(source)
        calls = 0

        async def model(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AssistantTurn(
                    text="",
                    tool_calls=(ToolCall("read", "read", {"resource_id": resource}),),
                    stop_reason="tool_use",
                )
            view_count = 3 if kind == "xlsx" else 1
            if calls <= 1 + view_count:
                args = {"resource_id": resource}
                if kind == "xlsx":
                    handles = list(
                        dict.fromkeys(re.findall(r"vis-[a-f0-9]{24}", str(kwargs["messages"])))
                    )
                    args["locator"] = handles[calls - 2]
                return AssistantTurn(
                    text="",
                    tool_calls=(ToolCall(str(calls), "view", args),),
                    stop_reason="tool_use",
                )
            return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

        host = orchestrator(model, registry=registry)
        settled = await drive(
            session, session_id, host, host.prepare_run("read then view", registry=registry)
        )
        first = await registry.read(resource, max_window_tokens=1000)
        effects = registry.conversion_effects(resource)
        snapshot = ConversionSnapshot.restore(
            effects[-1].content, {e.resource_id: e.content for e in effects[:-1]}
        )
        assert_route_gold(kind, snapshot)
        entries = [e for e in settled.entries if isinstance(e, ToolResultMessageEntry)]
        assert not tool_content_attachments(entries[0].result.parts)
        attachments = [p for e in entries[1:] for p in tool_content_attachments(e.result.parts)]
        assert (
            len(attachments) == {"multi": 3, "scan": 2, "mixed": 1, "nonlatin": 2, "xlsx": 3}[kind]
        )
        assert len({a.source for a in attachments}) == len(attachments)
        for a in attachments:
            assert a.source is not None
            assert a.source.resource_id == resource
            if kind == "xlsx":
                visual = next(v for v in snapshot.visuals if v.handle_id == a.source.handle_id)
                assert a.source.anchor == visual.anchor and a.source.page is None
            else:
                assert a.source.overview and a.source.page is not None

    async def forbidden(*args, **kwargs):
        raise AssertionError("settled conversion cannot be rerun")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    executor = object.__new__(AnswerExecutor)
    executor._store, executor._blob_store = store, PGRunBlobStore(pool=pool)
    async with ResourceRegistry(resource_secret=b"format", cursor_secret=b"cursor") as restored:
        assert restored.register(source) == resource
        returned = await executor._restore_registry_fetches(
            restored, owner_id=OWNER, run_id=session.run_id
        )
        assert await restored.read(resource, max_window_tokens=1000) == first
        assert restored.conversion_effects(resource) == effects
        for a in attachments:
            # Returned derivatives, not regenerated pixels; digest + source live
            # in the durable tool result and are bound to these retained bytes.
            assert hashlib.sha256(returned[a.resource_id]).hexdigest() == a.content_digest
        for v in snapshot.visuals:
            asset = await restored.visual_asset(resource, v.handle_id)
            assert asset.data == v.data and asset.anchor == v.anchor
