# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Real Answer Host/Agent settlements restore adopted views and located pixels."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.operation import OperationCompleted
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.runtime import AgentSessionRuntime
from dlightrag.engine.agent.tool_content import (
    decode_tool_content,
    encode_tool_content,
    tool_content_attachments,
    tool_content_message_fields,
)
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.execution.executor import AnswerExecutor
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.orchestration.orchestrator import (
    _admit_durable_attachment_messages,
    _hydrate_attachment_messages,
)
from dlightrag.engine.answer.research.runtime import FetchedResourceBuffer, ResearchRuntimeEffects
from dlightrag.engine.answer.resources.models import ResourceInput, TextWindowBudget
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
from dlightrag.engine.runtime.records import RunFetchedResource
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.unit.conftest import answer_image_policy, answer_model_profile
from tests.unit.test_research_runtime_migration import _Session
from tests.unit.test_resource_tools import call, docx_images, tools
from tests.unit.test_resource_visual import pdf_bytes


class _SettledBlobs:
    """Isolated owner-scoped adapter populated ONLY from committed Host deltas."""

    def __init__(self, deltas):
        self.blobs = {}
        self.resources = {}
        for _, delta in deltas:
            for fetched in delta.fetched:
                blob, resource = fetched.complete_blob, fetched.resource
                self.blobs[blob.digest] = b"".join(blob.chunks)
                self.resources[resource.resource_id] = RunFetchedResource(
                    resource_id=resource.resource_id,
                    ordinal=resource.ordinal,
                    digest=resource.blob_digest,
                    filename=resource.safe_name,
                    mime_type=resource.media_type,
                    source_locator=resource.source_locator,
                    capabilities=resource.capabilities,
                )

    async def list_fetched_resources(self, *, owner_id, run_id):
        assert (owner_id, run_id) == ("owner", "run")
        return tuple(self.resources.values())

    async def stream(self, *, owner_id, digest):
        assert owner_id == "owner"
        if digest in self.blobs:
            yield self.blobs[digest]


@pytest.mark.parametrize("format", ["docx", "pdf"])
async def test_host_settlement_restores_conversion_and_derivative_without_reparsing(
    monkeypatch, format
):
    data = docx_images(2) if format == "docx" else pdf_bytes(3)
    source = ResourceInput(filename=f"source.{format}", content=data)
    profile = answer_model_profile(supports_images=True)
    budget = TextWindowBudget(4000)
    repository = MemoryAgentSessionRepository[EffectHostUpdate]()
    session_id = SessionId.new()
    async with ResourceRegistry(resource_secret=b"identity", cursor_secret=b"cursor") as registry:
        resource_id = registry.register(source)
        turn = 0

        async def model(**kwargs):
            nonlocal turn
            turn += 1
            if turn == 1:
                return AssistantTurn(
                    text="",
                    tool_calls=(ToolCall("read", "read", {"resource_id": resource_id}),),
                    stop_reason="tool_use",
                )
            if turn == 2:
                import re

                match = re.search(r"vis-[a-f0-9]{24}", str(kwargs["messages"]))
                assert format == "pdf" or match is not None
                locator = match.group() if match is not None else "2"
                return AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall("view", "view", {"resource_id": resource_id, "locator": locator}),
                    ),
                    stop_reason="tool_use",
                )
            return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

        async def retrieve(*args, **kwargs):
            raise AssertionError("no corpus access")

        orchestrator = AnswerOrchestrator(
            synthesizer=cast(Any, SimpleNamespace()),
            retrieve_knowledge_base=retrieve,
            model_func=model,
            text_window_budget=budget,
            model_profile=profile,
            image_budget=answer_image_policy(max_images=8).new_budget(),
            telemetry=NOOP_TELEMETRY,
            resource_reader=make_resource_reader(registry, budget),
            resource_viewer=make_resource_viewer(registry),
            resolved_mode="research",
        )
        prepared = orchestrator.prepare_run("read then view", registry=registry)
        plan = AgentRunPlan.from_tools(
            prepared.tools,
            model_role="query",
            context_policy_revision="test",
            model_identity={"role": "query"},
            model_profile=asdict(profile),
        )
        runtime = AgentSessionRuntime(
            repository=repository,
            effects=ResearchRuntimeEffects(
                orchestrator=orchestrator,
                prepared=prepared,
                session=cast(Any, _Session()),
                session_id=session_id,
                fetched_buffer=FetchedResourceBuffer(),
                persist_child_intent=None,
            ),
            tools=prepared.tools,
            fencing_epoch=1,
        )
        accepted = await runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="run",
            content="read then view",
            plan=plan,
        )
        final = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
        assert isinstance(final.state, OperationCompleted)
        snapshot = await repository.load(session_id)
        entries = [entry for entry in snapshot.entries if isinstance(entry, ToolResultMessageEntry)]
        assert len(entries) == 2
        # Simulate the durable codec boundary; no raw pixels in text/entry encoding.
        encoded = encode_tool_content(entries[1].result.parts)
        assert all("data" not in part for part in encoded)
        parts = decode_tool_content(encoded)
        (attachment,) = tool_content_attachments(parts)
        assert attachment.source is not None
        assert attachment.source.resource_id == resource_id
        assert attachment.source.page == (2 if format == "pdf" else None)
        deltas = repository.applied_host_deltas(session_id)
        stored = _SettledBlobs(deltas)
        assert (
            stored.resources[f"{resource_id}-conversion"].capabilities["resource_kind"]
            == "conversion_snapshot"
        )
        assert stored.resources[attachment.resource_id].capabilities["visual_source"] == asdict(
            attachment.source
        )
        assert any(delta.evidence for _, delta in deltas)
        if format == "docx":
            import json

            conversion = stored.resources[f"{resource_id}-conversion"]
            payload = json.loads(stored.blobs[conversion.digest])
            assert payload["converter"] == "firecrawl-anydoc"
            assert payload["converter_version"] == "0.2.4"
            assert len(payload["assets"]) == 2
            assert attachment.source.origin_part == "word/media/image1.png"
            assert attachment.source.anchor is None

    async def forbidden(*args, **kwargs):
        raise AssertionError("recovery cannot reselect or rerun a converter")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    executor = object.__new__(AnswerExecutor)
    executor._store = cast(Any, stored)
    executor._blob_store = cast(Any, stored)
    async with ResourceRegistry(resource_secret=b"identity", cursor_secret=b"cursor") as registry:
        assert registry.register(source) == resource_id
        restored = await executor._restore_registry_fetches(
            registry, owner_id="owner", run_id="run"
        )
        read, _ = tools(registry)
        result = await call(read, resource_id=resource_id)
        assert "extraction_status=" in result.text_content
        messages = [{"role": "tool", **tool_content_message_fields(parts)}]
        replay_budget = answer_image_policy(max_images=1).new_budget()
        admissions = _admit_durable_attachment_messages(messages, restored, replay_budget)
        _hydrate_attachment_messages(messages, restored, admissions=admissions)
        assert messages[0]["attachments"][0]["data_url"].startswith("data:image/")
        assert replay_budget.count == 1
        assert messages[0]["attachments"][0]["source"] == asdict(attachment.source)


def test_postgres_resource_recovery_query_includes_adopted_conversion_records():
    from dlightrag.adapters.postgres.runtime.run_store import _SELECT_RUN_FETCHED_RESOURCES

    # The real Host proof above uses an isolated adapter; separately guard the
    # production query's closed representation filter (no PostgreSQL is contacted).
    assert "owner_id = $1 AND run_id = $2" in _SELECT_RUN_FETCHED_RESOURCES
    assert "'conversion_snapshot'" in _SELECT_RUN_FETCHED_RESOURCES
    assert "'conversion_asset'" in _SELECT_RUN_FETCHED_RESOURCES
