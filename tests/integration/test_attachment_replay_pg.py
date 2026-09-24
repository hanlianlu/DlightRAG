# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Generated visuals through real Host settlements and isolated PG retention races."""

import asyncio
import uuid
from dataclasses import asdict, replace
from functools import partial
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.adapters.postgres.answer.attachment_replay import retain_attachment_occurrences
from dlightrag.adapters.postgres.runtime.run_blob_store import PGRunBlobStore
from dlightrag.engine.agent.session.fold import project_session_messages
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.operation import OperationCompleted
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
from dlightrag.engine.agent.session.runtime import AgentSessionRuntime
from dlightrag.engine.agent.session.transactions import (
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
)
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.attachment_replay import AttachmentReplaySelection
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.execution.executor import AnswerExecutor
from dlightrag.engine.answer.fast import ensure_session_lane
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.research.persistence import ResearchRunStore
from dlightrag.engine.answer.research.runtime import FetchedResourceBuffer, ResearchRuntimeEffects
from dlightrag.engine.answer.resources.models import ResourceInput, TextWindowBudget
from dlightrag.engine.answer.resources.registry import ResourceRegistry
from dlightrag.engine.answer.tools.resources import make_resource_reader, make_resource_viewer
from dlightrag.engine.runtime.coordinator import LeaseLostError
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope
from tests.unit.conftest import answer_image_policy, answer_model_profile
from tests.unit.test_research_runtime_migration import _Session
from tests.unit.test_resource_visual import pdf_bytes

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]
OWNER = "generated-replay-owner"
WORKER = "generated-worker"


@pytest.fixture
async def pg():
    # No skip: a named validation command must prove transactions, not turn green
    # when the explicitly configured disposable PostgreSQL fixture is unavailable.
    async with isolated_run_runtime("attachment_replay") as pair:
        async with pair[1].acquire() as conn:
            assert int(await conn.fetchval("SHOW server_version_num")) >= 180000
        yield pair


async def new_run(
    store, *, session_id=None, lane="main", source_lane=None, owner=OWNER, mode="research"
):
    session_id = session_id or SessionId.new()
    run_id = str(uuid.uuid7())
    envelope = run_envelope("answer", key=run_id, owner=owner, mode=mode)
    envelope = replace(
        envelope,
        payload={
            **envelope.payload,
            "agent_session_id": session_id.value,
            "agent_lane_id": lane,
            "source_lane_id": source_lane,
        },
    )
    await store.create_run(envelope=envelope, run_id=run_id)
    claim = await store.claim_next(worker_id=WORKER)
    assert claim is not None and claim.run.run_id == run_id
    session: Any = _Session()
    session.owner_id, session.run_id, session.worker_id = owner, run_id, WORKER
    session.fencing_epoch, session.execution = claim.run.fencing_epoch, claim.execution
    return session, session_id


async def record_fork_point(store: Any, session: Any) -> None:
    """Settle the way an executor does: the settled state is recorded while the claim is live.

    A fixture that skips this produces a Run the product would never produce — one
    with no Fork Point — and every Fork from it would refuse.
    """
    from dlightrag.engine.answer.execution.executor import _lane_projection

    routing = await store.load_routing(owner_id=session.owner_id, run_id=session.run_id)
    assert routing is not None
    snapshot = await session.execution.session_repository.load(SessionId(routing.agent_session_id))
    lane_id = LaneId(routing.agent_lane_id)
    head = snapshot.tree.lane(lane_id).head_entry_id
    projection = _lane_projection(snapshot, lane_id)
    await store.record_fork_point(
        owner_id=session.owner_id,
        run_id=session.run_id,
        worker_id=WORKER,
        fencing_epoch=session.fencing_epoch,
        entry_id=head.value if head is not None else None,
        projection_id=(projection.projection_id.value if projection is not None else None),
    )


async def finish(store, session):
    await record_fork_point(store, session)
    result = await store.finish_success(
        owner_id=session.owner_id,
        run_id=session.run_id,
        worker_id=WORKER,
        fencing_epoch=session.fencing_epoch,
        result={"answer": "generated"},
    )
    assert result.committed


def orchestrator(model, *, registry=None, max_images=8):
    async def retrieve(*args, **kwargs):
        raise AssertionError("no corpus or provider access")

    budget = TextWindowBudget(4000)
    return AnswerOrchestrator(
        synthesizer=cast(Any, SimpleNamespace()),
        retrieve_knowledge_base=retrieve,
        model_func=model,
        model_profile=answer_model_profile(supports_images=True),
        text_window_budget=budget,
        image_budget=answer_image_policy(max_images=max_images).new_budget(),
        telemetry=NOOP_TELEMETRY,
        resolved_mode="research",
        resource_reader=make_resource_reader(registry, budget) if registry else None,
        resource_viewer=make_resource_viewer(registry) if registry else None,
    )


async def drive(session, session_id, host, prepared, *, lane="main", fetched_buffer=None):
    plan = AgentRunPlan.from_tools(
        prepared.tools,
        model_role="query",
        context_policy_revision="test",
        model_identity={"role": "query"},
        model_profile=asdict(prepared.model_profile),
    )
    runtime = AgentSessionRuntime(
        repository=session.execution.session_repository,
        effects=ResearchRuntimeEffects(
            telemetry=NOOP_TELEMETRY,
            orchestrator=host,
            prepared=prepared,
            session=session,
            session_id=session_id,
            fetched_buffer=fetched_buffer or FetchedResourceBuffer(),
            persist_child_intent=None,
        ),
        tools=prepared.tools,
        fencing_epoch=session.fencing_epoch,
    )
    accepted = await runtime.accept(
        session_id=session_id,
        lane_id=LaneId(lane),
        idempotency_key=session.run_id,
        content="generated read/view",
        plan=plan,
    )
    result = await runtime.drive(session_id=session_id, operation_id=accepted.operation_id)
    assert isinstance(result.state, OperationCompleted)
    return await session.execution.session_repository.load(session_id)


async def origin(pg, *, finish_run=True):
    store, pool = pg
    session, session_id = await new_run(store)
    calls = 0
    async with ResourceRegistry(
        resource_secret=b"origin", cursor_secret=b"origin-cursor"
    ) as registry:
        resource_id = registry.register(
            ResourceInput(filename="generated.pdf", content=pdf_bytes(2))
        )

        async def model(**kwargs):
            nonlocal calls
            calls += 1
            if calls <= 3:
                name = "read" if calls == 1 else "view"
                args = {"resource_id": resource_id}
                if name == "view":
                    args["locator"] = "1"
                return AssistantTurn(
                    text="", tool_calls=(ToolCall(str(calls), name, args),), stop_reason="tool_use"
                )
            return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

        host = orchestrator(model, registry=registry)
        snapshot = await drive(
            session, session_id, host, host.prepare_run("source", registry=registry)
        )
        cursor = registry.visual_cursor(resource_id, 1, "overview")
    if finish_run:
        await finish(store, session)
    selection = AttachmentReplaySelection.from_snapshot(snapshot)
    assert len(selection.occurrences) == 2
    assert (
        selection.occurrences[0].attachment.resource_id
        == selection.occurrences[1].attachment.resource_id
    )
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_answer_resources WHERE capabilities->>'resource_kind'='attachment_occurrence'"
        )
        assert count == 2  # Deduplicated pixels, two independently settled occurrences.
    return session, session_id, snapshot, selection, resource_id, cursor


def executor(pg):
    value = object.__new__(AnswerExecutor)
    value._store = pg[0]
    value._blob_store = PGRunBlobStore(pool=pg[1])
    return value


async def retain(store, session, selection):
    return await store.retain_attachment_occurrences(
        owner_id=session.owner_id,
        run_id=session.run_id,
        worker_id=WORKER,
        fencing_epoch=session.fencing_epoch,
        selection=selection,
    )


async def delete_origin(pg, session):
    async with pg[1].acquire() as conn, conn.transaction():
        result = await pg[0].delete_runs_in(
            conn, owner_id=session.owner_id, run_ids=[session.run_id]
        )
        assert result.runs == 1


@pytest.mark.parametrize("kind", ["follow_up", "fork"])
async def test_follow_up_fork_real_host_replay_and_multigeneration_retention(pg, monkeypatch, kind):
    old, session_id, snapshot, selection, resource_id, cursor = await origin(pg)
    lane = "main" if kind == "follow_up" else LaneId.new().value
    current, _ = await new_run(
        pg[0], session_id=session_id, lane=lane, source_lane="main" if kind == "fork" else None
    )

    async def forbidden(*args, **kwargs):
        raise AssertionError("historical hydration cannot reparse/refetch")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    snapshots = await executor(pg)._restore_selected_attachments(current, snapshot)
    assert len(await retain(pg[0], current, selection)) == 2  # Exact retry is idempotent.
    await delete_origin(pg, old)
    assert snapshots == await executor(pg)._restore_selected_attachments(current, snapshot)
    seen = []

    async def model(**kwargs):
        attachments = [a for m in kwargs["messages"] for a in m.get("attachments", [])]
        assert len(attachments) == 2
        assert all(a["data_url"].startswith("data:image/") for a in attachments)
        assert all(a["source"]["page"] == 1 for a in attachments)
        seen.extend(attachments)
        return AssistantTurn(text="replayed", tool_calls=(), stop_reason="stop")

    async with ResourceRegistry(
        resource_secret=b"new-run", cursor_secret=b"new-cursor"
    ) as registry:
        assert (
            await executor(pg)._restore_registry_fetches(
                registry, owner_id=OWNER, run_id=current.run_id
            )
            == {}
        )
        with pytest.raises(Exception, match="resource|cursor"):
            registry.resolve_visual_cursor(cursor, resource_id, "overview")
        with pytest.raises(Exception, match="resource"):
            await registry.materialize(resource_id)
        host = orchestrator(model)
        messages = project_session_messages(
            snapshot.tree.ancestry(LaneId.main()), snapshot.active_projection
        )
        admissions = host.admit_durable_attachments(messages, snapshots)
        assert host._image_budget is not None
        assert host._image_budget.count == 2
        prepared = host.prepare_run(
            "follow", attachment_snapshots=snapshots, attachment_admissions=admissions
        )
        await ensure_session_lane(
            repository=current.execution.session_repository,
            snapshot=snapshot,
            fencing_epoch=current.fencing_epoch,
            session_id=session_id,
            lane_id=LaneId(lane),
            source_lane_id=LaneId.main() if kind == "fork" else None,
        )
        next_snapshot = await drive(current, session_id, host, prepared, lane=lane)
        assert seen
    await finish(pg[0], current)
    descendant, _ = await new_run(pg[0], session_id=session_id, lane=lane)
    next_snapshot = replace(next_snapshot, selected_lane_id=LaneId(lane))
    inherited = await executor(pg)._restore_selected_attachments(descendant, next_snapshot)
    assert inherited == snapshots
    await delete_origin(pg, current)
    assert inherited == await executor(pg)._restore_selected_attachments(descendant, next_snapshot)
    async with pg[1].acquire() as conn:
        origins = await conn.fetch(
            "SELECT capabilities->>'origin_run_id' AS origin FROM dlightrag_answer_resources WHERE run_id=$1",
            uuid.UUID(descendant.run_id),
        )
        assert {row["origin"] for row in origins} == {old.run_id}


async def test_wrong_owner_unrelated_session_and_unselected_sibling_are_rejected(pg):
    _, session_id, snapshot, selection, _, _ = await origin(pg)
    unrelated, _ = await new_run(pg[0])
    with pytest.raises(ValueError, match="Session"):
        await retain(pg[0], unrelated, selection)
    other, _ = await new_run(pg[0], owner="other-generated-owner")
    with pytest.raises(ValueError, match="Session"):
        await retain(pg[0], other, selection)
    sibling = LaneId.new()
    current, _ = await new_run(pg[0], session_id=session_id, lane=sibling.value, source_lane="main")
    # A sibling pinned before either tool view has no authority for later main entries.
    root = snapshot.entries[0].entry_id
    committed = await current.execution.session_repository.transact(
        session_id=session_id,
        fencing_epoch=current.fencing_epoch,
        transaction=SessionTransaction.from_parts(
            register_writes=[
                SetRegister(LaneHead(sibling, root)),
                SetRegister(LaneState(sibling)),
            ],
            expectations=[
                RegisterExpectation(LaneHead(sibling, root).ref, None),
                RegisterExpectation(LaneState(sibling).ref, None),
            ],
        ),
    )
    assert isinstance(committed, TransactionCommit)
    with pytest.raises(ValueError, match="Lane Head"):
        await retain(pg[0], current, selection)
    forged = replace(selection, lane_id=sibling.value, head_entry_id=root.value)
    with pytest.raises(ValueError, match="ancestry"):
        await retain(pg[0], current, forged)


@pytest.mark.parametrize(
    "fault",
    [
        "digest",
        "source",
        "part",
        "missing_reference",
        "missing_bytes",
        "corrupt_bytes",
        "fence",
        "owner",
    ],
)
async def test_replay_integrity_and_fence_fail_explicitly(pg, fault):
    _, session_id, snapshot, selection, _, _ = await origin(pg)
    current, _ = await new_run(pg[0], session_id=session_id)
    part = selection.occurrences[0].attachment
    async with pg[1].acquire() as conn:
        if fault in {"digest", "source", "part"}:
            first = selection.occurrences[0]
            if fault == "digest":
                first = replace(first, attachment=replace(part, content_digest="0" * 64))
            elif fault == "source":
                assert part.source is not None
                first = replace(
                    first, attachment=replace(part, source=replace(part.source, page=2))
                )
            else:
                first = replace(first, part_index=0)
            selection = replace(selection, occurrences=(first, *selection.occurrences[1:]))
        elif fault == "missing_reference":
            await conn.execute(
                "DELETE FROM dlightrag_answer_resources WHERE resource_id=$1",
                selection.occurrences[0].reference_id,
            )
        elif fault == "missing_bytes":
            await conn.execute(
                "DELETE FROM dlightrag_blob_chunks WHERE digest=$1", part.content_digest
            )
        elif fault == "corrupt_bytes":
            await conn.execute(
                "UPDATE dlightrag_blob_chunks SET content=$2 WHERE digest=$1",
                part.content_digest,
                b"bad",
            )
        elif fault == "fence":
            current.fencing_epoch += 1
        else:
            current.owner_id = "wrong-owner"
    with pytest.raises(LeaseLostError if fault in {"fence", "owner"} else ValueError):
        if fault in {"missing_bytes", "corrupt_bytes"}:
            await executor(pg)._restore_selected_attachments(current, snapshot)
        else:
            await retain(pg[0], current, selection)


async def test_replayed_occurrences_share_consuming_image_budget(pg):
    _, session_id, snapshot, _, _, _ = await origin(pg)
    current, _ = await new_run(pg[0], session_id=session_id)
    snapshots = await executor(pg)._restore_selected_attachments(current, snapshot)
    messages = project_session_messages(snapshot.entries, snapshot.active_projection)
    host = orchestrator(None, max_images=1)
    with pytest.raises(AnswerInputOverflowError):
        host.admit_durable_attachments(messages, snapshots)


@pytest.mark.parametrize("winner", ["adoption", "cleanup", "rollback"])
async def test_adoption_cleanup_lock_order_and_rollback(pg, winner):
    old, session_id, _, selection, _, _ = await origin(pg)
    current, _ = await new_run(pg[0], session_id=session_id)
    pool = pg[1]
    async with pool.acquire() as held:
        transaction = held.transaction()
        await transaction.start()
        if winner == "cleanup":
            await pg[0].delete_runs_in(held, owner_id=OWNER, run_ids=[old.run_id])
            waiting = asyncio.create_task(retain(pg[0], current, selection))
        else:
            await retain_attachment_occurrences(
                held,
                owner_id=OWNER,
                run_id=uuid.UUID(current.run_id),
                worker_id=WORKER,
                fencing_epoch=current.fencing_epoch,
                selection=selection,
            )
            waiting = asyncio.create_task(delete_origin(pg, old))
        # Observe a real PostgreSQL lock wait rather than assuming a sleep proves overlap.
        async with pool.acquire() as monitor:
            blocked = 0
            for _ in range(100):
                blocked = await monitor.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND wait_event_type='Lock'"
                )
                if blocked:
                    break
                await asyncio.sleep(0.01)
            assert blocked and not waiting.done()
        if winner == "rollback":
            await transaction.rollback()
        else:
            await transaction.commit()
        if winner == "cleanup":
            with pytest.raises(ValueError, match="missing"):
                await asyncio.wait_for(waiting, 10)
        else:
            await asyncio.wait_for(waiting, 10)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_answer_resources WHERE run_id=$1",
            uuid.UUID(current.run_id),
        )
        assert count == (2 if winner == "adoption" else 0)
        blob_count = await conn.fetchval(
            "SELECT count(*) FROM dlightrag_blobs WHERE digest=$1",
            selection.occurrences[0].attachment.content_digest,
        )
        assert blob_count == (1 if winner == "adoption" else 0)


async def pin_child(pg, *, adopt=True, same_run=False):
    from dlightrag.engine.answer.research.runtime import _bound_child_dispatch_preparer
    from dlightrag.engine.answer.tools.subagents import ChildRequest, SubagentHost

    old, session_id, snapshot, selection, _, _ = await origin(pg, finish_run=not same_run)
    current = old
    if not same_run:
        current, _ = await new_run(pg[0], session_id=session_id)
    snapshots = await executor(pg)._restore_selected_attachments(current, snapshot) if adopt else {}

    async def unused_model(**kwargs):
        raise AssertionError("pinning cannot call a model")

    host = orchestrator(unused_model)
    host._subagent_host = SubagentHost(parent_session_id=session_id)
    prepared = host.prepare_run("parent", attachment_snapshots=snapshots)
    host.bind_child_context(
        prepared,
        cast(Any, SimpleNamespace(session_id=session_id, lane_id=LaneId.main(), snapshot=snapshot)),
    )
    context = host._subagent_host.context_snapshot
    assert context is not None
    assert context.attachment_occurrences == selection.occurrences
    child_id = SessionId.new()
    request = ChildRequest(
        objective="Describe the exact inherited page", context="parent", tools=()
    )
    # Freeze the same child plan without charging a second process's recovery budget.
    if not adopt:
        host.restore_child_attachment_snapshots(
            await executor(pg)._restore_selected_attachments(current, snapshot)
        )
    envelope = _bound_child_dispatch_preparer(host)(child_id, request, context)
    assert await pg[0].upsert_child_session(
        owner_id=OWNER,
        run_id=current.run_id,
        worker_id=WORKER,
        fencing_epoch=current.fencing_epoch,
        child_session_id=child_id.value,
        parent_session_id=session_id.value,
        parent_call_id="generated-child",
        objective=request.objective,
        context_mode=request.context,
        model_role=request.model_role,
        tools=request.tools,
        depth=1,
        context_snapshot=context.canonical_payload(),
        **envelope,
    )
    if not adopt:
        async with pg[1].acquire() as conn:
            await conn.execute(
                "DELETE FROM dlightrag_answer_resources WHERE owner_id=$1 AND run_id=$2",
                OWNER,
                uuid.UUID(current.run_id),
            )
    return old, current, snapshot, child_id, request, context


async def compact_parent(current, snapshot):
    from dlightrag.engine.agent.session.ids import ProjectionId
    from dlightrag.engine.agent.session.projection import (
        CompactionSummary,
        ContextProjection,
        projection_source_digest,
    )
    from dlightrag.engine.agent.session.registers import ContextProjectionRegister

    entries = snapshot.graph.ancestry()
    last = entries[-1]
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=last.sequence + 1,
        covered_through_sequence=last.sequence,
        summary=CompactionSummary(goal="Parent completed the page review").canonical_json(),
        covered_through_entry_id=last.entry_id,
        source_digest=projection_source_digest([entry.entry_id for entry in entries]),
    )
    register = ContextProjectionRegister(LaneId.main(), projection)
    result = await current.execution.session_repository.transact(
        session_id=snapshot.session_id,
        fencing_epoch=current.fencing_epoch,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(register)],
            expectations=[RegisterExpectation(register.ref, None)],
        ),
    )
    assert isinstance(result, TransactionCommit)
    return await current.execution.session_repository.load(snapshot.session_id)


@pytest.mark.parametrize("same_run", [True, False])
async def test_child_pinned_pixels_survive_parent_compaction_and_origin_cleanup(
    pg, monkeypatch, same_run
):
    import base64
    import json

    from dlightrag.engine.answer.research.runtime import _check_child_write, run_child_session
    from dlightrag.engine.answer.tools.subagents import SubagentHost, _dispatch_from_row

    child_store: ResearchRunStore = pg[0]
    old, current, snapshot, child_id, request, context = await pin_child(pg, same_run=same_run)
    compacted = await compact_parent(current, snapshot)
    assert AttachmentReplaySelection.from_snapshot(compacted).occurrences == ()
    assert await executor(pg)._restore_selected_attachments(current, compacted) == {}
    if not same_run:
        await delete_origin(pg, old)
    rows = await pg[0].list_child_sessions(owner_id=OWNER, run_id=current.run_id)
    row = next(item for item in rows if item["child_session_id"] == child_id.value)
    encoded = json.dumps(row["context_snapshot"])
    assert "data_url" not in encoded and "base64" not in encoded
    decoded = _dispatch_from_row(SubagentHost(parent_session_id=snapshot.session_id), row)
    assert decoded is not None and decoded[3] == context
    restored = await executor(pg)._restore_child_attachments(current, child_id, decoded[3])
    assert len(restored) == 1  # Two occurrences; one shared pixel Blob.
    seen = []

    async def model(**kwargs):
        attachments = [
            part for message in kwargs["messages"] for part in message.get("attachments", [])
        ]
        assert len(attachments) == 2
        for part, occurrence in zip(attachments, context.attachment_occurrences, strict=True):
            assert (
                base64.b64decode(part["data_url"].partition(",")[2])
                == restored[part["resource_id"]]
            )
            assert occurrence.attachment.source is not None
            assert part["source"] == asdict(occurrence.attachment.source)
            assert part["source"]["page"] == 1
            assert part["content_digest"] == occurrence.attachment.content_digest
        seen.extend(attachments)
        return AssistantTurn(text="Exact pinned page", tool_calls=(), stop_reason="stop")

    async def forbidden(*args, **kwargs):
        raise AssertionError("pinned hydration cannot acquire or reparse")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    host = orchestrator(model, max_images=2)
    host._subagent_host = SubagentHost(parent_session_id=snapshot.session_id)
    host.prepare_run("compacted parent")  # Fresh process has no projected tool pixels.
    persist = _check_child_write(
        partial(
            child_store.upsert_child_session,
            worker_id=current.worker_id,
            fencing_epoch=current.fencing_epoch,
        )
    )
    claim = _check_child_write(
        partial(
            child_store.claim_child_session,
            worker_id=current.worker_id,
            fencing_epoch=current.fencing_epoch,
        )
    )
    assert persist is not None and claim is not None
    outcome = await run_child_session(
        telemetry=NOOP_TELEMETRY,
        orchestrator=host,
        repository=current.execution.session_repository,
        session=current,
        fetched_buffer=FetchedResourceBuffer(),
        child_id=child_id,
        request=request,
        parent_call_id="generated-child",
        parent_session_id=snapshot.session_id,
        context_snapshot=decoded[3],
        persist_child_runtime=persist,
        claim_child=claim,
        load_child=pg[0].load_child_session,
        restore_child_attachments=lambda child, pin: executor(pg)._restore_child_attachments(
            current, child, pin
        ),
    )
    assert outcome.status == "succeeded" and len(seen) == 2
    assert host._image_budget is not None
    assert host._image_budget.count == 2
    assert host._image_budget.used_bytes == 2 * len(next(iter(restored.values())))
    assert not host._image_budget.reserve_prepared(
        next(iter(restored.values())), label="no budget reset"
    )
    stricter = orchestrator(model, max_images=1)
    stricter.restore_child_attachment_snapshots(restored)
    with pytest.raises(AnswerInputOverflowError, match="image budget"):
        stricter.prepare_child_session(request, context_snapshot=context)
    nonvisual = orchestrator(model)
    nonvisual._child_model_resolver = lambda role: (
        model,
        cast(Any, None),  # Capability refusal must precede either provider transport.
        answer_model_profile(supports_images=False),
    )
    nonvisual.restore_child_attachment_snapshots(restored)
    with pytest.raises(ValueError, match="does not support inherited images"):
        nonvisual.prepare_child_session(request, context_snapshot=context)
    assert len(seen) == 2  # Both refusals precede any further provider work.


@pytest.mark.parametrize(
    "fault",
    [
        "unpinned_child",
        "forged_snapshot",
        "unadopted",
        "missing_reference",
        "binding",
        "missing_bytes",
        "fence",
        "owner",
    ],
)
async def test_child_pinned_hydration_rejects_untrusted_or_missing_bindings(pg, fault):
    old, current, _, child_id, _, context = await pin_child(pg, adopt=fault != "unadopted")
    if fault == "unpinned_child":
        child_id = SessionId.new()
    elif fault == "forged_snapshot":
        from dlightrag.engine.agent.session.ids import EntryId

        context = replace(
            context, parent_entry_id=EntryId(context.attachment_occurrences[0].entry_id)
        )
    elif fault == "fence":
        current.fencing_epoch += 1
    elif fault == "owner":
        current.owner_id = "wrong-owner"
    elif fault in {"missing_reference", "binding", "missing_bytes"}:
        async with pg[1].acquire() as conn:
            occurrence = context.attachment_occurrences[0]
            if fault == "missing_reference":
                await conn.execute(
                    "DELETE FROM dlightrag_answer_resources WHERE run_id=$1 AND resource_id=$2",
                    uuid.UUID(current.run_id),
                    occurrence.reference_id,
                )
            elif fault == "binding":
                await conn.execute(
                    "UPDATE dlightrag_answer_resources SET safe_name='forged' WHERE run_id=$1 AND resource_id=$2",
                    uuid.UUID(current.run_id),
                    occurrence.reference_id,
                )
            else:
                await conn.execute(
                    "DELETE FROM dlightrag_blob_chunks WHERE digest=$1",
                    occurrence.attachment.content_digest,
                )
    # The origin still exists: none of these failures may fall back to its row.
    assert old.run_id != current.run_id
    with pytest.raises(LeaseLostError if fault in {"fence", "owner"} else ValueError):
        await executor(pg)._restore_child_attachments(current, child_id, context)


async def test_unified_docx_host_settlement_restores_exact_text_assets_and_cursors(pg, monkeypatch):
    import io
    import json
    import re

    from docx import Document

    from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
    from dlightrag.engine.agent.tool_content import tool_content_attachments
    from tests.unit.test_resource_tools import docx_images

    store, pool = pg
    session, session_id = await new_run(store)
    document = Document(io.BytesIO(docx_images(2)))
    document.add_paragraph("Stable continuation fact 718.40. " * 300)
    stream = io.BytesIO()
    document.save(stream)
    source = ResourceInput(filename="synthetic.docx", content=stream.getvalue())
    async with ResourceRegistry(resource_secret=b"docx", cursor_secret=b"docx-cursor") as registry:
        resource_id = registry.register(source)
        calls = 0

        async def model(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return AssistantTurn(
                    text="",
                    tool_calls=(ToolCall("read", "read", {"resource_id": resource_id}),),
                    stop_reason="tool_use",
                )
            if calls in {2, 3}:
                handles = list(
                    dict.fromkeys(re.findall(r"vis-[a-f0-9]{24}", str(kwargs["messages"])))
                )
                return AssistantTurn(
                    text="",
                    tool_calls=(
                        ToolCall(
                            str(calls),
                            "view",
                            {"resource_id": resource_id, "locator": handles[calls - 2]},
                        ),
                    ),
                    stop_reason="tool_use",
                )
            return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

        host = orchestrator(model, registry=registry)
        settled = await drive(
            session, session_id, host, host.prepare_run("read and view", registry=registry)
        )
        first = await registry.read(resource_id, max_window_tokens=700)
        assert first.next_cursor
        continuation = await registry.read(
            resource_id, max_window_tokens=700, cursor=first.next_cursor
        )
        effects = registry.conversion_effects(resource_id)
        raw = json.loads(effects[-1].content)
        assert raw["converter"] == "firecrawl-anydoc" and raw["converter_version"] == "0.2.4"
        assert raw["extraction_status"] == "usable_text_unverified_coverage"
        assert len(raw["assets"]) == 2
        assert len({a["resource_id"] for a in raw["assets"]}) == 2
        assert len({a["digest"] for a in raw["assets"]}) == 1
        entries = [e for e in settled.entries if isinstance(e, ToolResultMessageEntry)]
        assert not tool_content_attachments(entries[0].result.parts)
        attachments = [tool_content_attachments(e.result.parts)[0] for e in entries[1:]]
        assert len(attachments) == 2
        for attachment in attachments:
            assert attachment.source is not None
            assert attachment.source.origin_part == "word/media/image1.png"
            assert attachment.source.anchor is None and attachment.source.page is None

    async def forbidden(*args, **kwargs):
        raise AssertionError("settled conversion cannot reparse or reselect")

    monkeypatch.setattr("dlightrag.engine.answer.resources.registry.convert_resource", forbidden)
    executor = object.__new__(AnswerExecutor)
    executor._store, executor._blob_store = store, PGRunBlobStore(pool=pool)
    async with ResourceRegistry(resource_secret=b"docx", cursor_secret=b"docx-cursor") as restored:
        assert restored.register(source) == resource_id
        returned = await executor._restore_registry_fetches(
            restored, owner_id=OWNER, run_id=session.run_id
        )
        assert (await restored.read(resource_id, max_window_tokens=700)) == first
        assert (
            await restored.read(resource_id, max_window_tokens=700, cursor=first.next_cursor)
        ) == continuation
        assert restored.conversion_effects(resource_id) == effects
        for attachment in attachments:
            assert attachment.source is not None
            assert attachment.source.handle_id is not None
            asset = await restored.visual_asset(resource_id, attachment.source.handle_id)
            assert asset.origin_part == attachment.source.origin_part
            assert asset.data
            assert attachment.resource_id in returned
