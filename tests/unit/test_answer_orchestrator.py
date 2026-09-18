# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Answer Host coordination around the deep AgentSessionRuntime."""

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.session.entries import AssistantMessageEntry
from dlightrag.engine.agent.session.ids import EntryId, ProjectionId, SessionId
from dlightrag.engine.agent.session.projection import ContextProjection
from dlightrag.engine.agent.skills import SkillsBundle
from dlightrag.engine.agent.tools.files import ResourceReadRequest
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.continuation_handles import MAX_SPILL_HANDLES
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.orchestration import AnswerOrchestrator
from dlightrag.engine.answer.orchestration.orchestrator import _last_provider_input_tokens
from dlightrag.engine.answer.research.runtime import (
    _record_prompt_cache,
    _usage_from_snapshot_entries,
)
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer
from dlightrag.engine.answer.tools.composition import _resource_rows
from dlightrag.engine.answer.workspace import RunWorkspace
from dlightrag.engine.runtime.settlements import InventoryPathRecord
from dlightrag.engine.runtime.workspace import CommittedSpillRecord, InMemoryWorkspaceStore
from tests.tool_helpers import tool_runtime
from tests.unit.conftest import answer_image_policy, answer_model_profile


def _orchestrator(*, mode: str, model=None, retrieve=None, synthesizer=None, environment=None):

    profile = answer_model_profile()

    async def default_retrieve(_query: str):
        return MagicMock(contexts={"chunks": [], "entities": [], "relationships": []}, trace={})

    return AnswerOrchestrator(
        synthesizer=synthesizer or MagicMock(spec=AnswerSynthesizer),
        retrieve_knowledge_base=retrieve or default_retrieve,
        model_func=model,
        stream_model_func=None,
        text_window_budget=TextWindowBudget(profile.context_window_tokens),
        model_profile=profile,
        telemetry=NOOP_TELEMETRY,
        environment=environment,
        resolved_mode=mode,  # type: ignore[arg-type]
    )


def test_requested_skill_contribution_precedes_skill_metadata(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    (global_root / "review").mkdir(parents=True)
    (global_root / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review plans.\n---\nbody",
        encoding="utf-8",
    )

    bundle = SkillsBundle(global_root=global_root, requested_skill="review")
    contributions = bundle.context_contributions()

    assert [item.source for item in contributions] == ["agent.skills.requested", "agent.skills"]
    assert contributions[0].authority == "user"
    assert contributions[1].authority == "reference"
    assert "load_skill(name='review')" in str(contributions[0].messages[0]["content"])


def test_context_contributions_without_requested_skill_keep_metadata_only(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    (global_root / "review").mkdir(parents=True)
    (global_root / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review plans.\n---\nbody",
        encoding="utf-8",
    )

    bundle = SkillsBundle(global_root=global_root)
    contributions = bundle.context_contributions()

    assert [item.source for item in contributions] == ["agent.skills"]
    assert contributions[0].authority == "reference"


def test_skills_bundle_tool_membership_differs_between_parent_and_child(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    bundle = SkillsBundle(owner_root=owner_root)

    parent = {tool.name for tool in bundle.tools(child=False)}
    child = {tool.name for tool in bundle.tools(child=True)}

    assert {"load_skill", "publish_skill", "delete_skill"} <= parent
    assert "load_skill" in child
    assert "publish_skill" not in child
    assert "delete_skill" not in child


@pytest.mark.asyncio
async def test_fast_path_retrieves_then_streams_one_synthesis() -> None:
    retrieval = MagicMock(
        contexts={"chunks": [{"content": "grounded"}], "entities": [], "relationships": []},
        trace={"retrieval": True},
    )

    retrieve_calls = 0

    async def retrieve(query: str):
        nonlocal retrieve_calls
        retrieve_calls += 1
        assert query == "question"
        return retrieval

    class Stream:
        def __aiter__(self):
            async def chunks():
                yield "answer"

            return chunks()

    synthesizer = MagicMock(spec=AnswerSynthesizer)
    synthesizer.generate_stream = AsyncMock(return_value=(retrieval.contexts, Stream()))
    orchestrator = _orchestrator(mode="fast", retrieve=retrieve, synthesizer=synthesizer)
    query_images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]
    contexts, stream = await orchestrator.answer_stream(
        "question",
        query_images=query_images,
    )
    assert contexts == retrieval.contexts
    assert stream is not None
    assert [chunk async for chunk in stream] == ["answer"]
    assert retrieve_calls == 1
    synthesizer.generate_stream.assert_awaited_once()
    assert synthesizer.generate_stream.await_args.kwargs["current_images"] == query_images


@pytest.mark.asyncio
async def test_fast_path_passes_recalled_memory_to_the_synthesizer() -> None:
    synthesizer = MagicMock(spec=AnswerSynthesizer)
    synthesizer.generate_stream = AsyncMock(
        return_value=(
            {"chunks": [], "entities": [], "relationships": []},
            None,
        )
    )
    orchestrator = _orchestrator(mode="fast", synthesizer=synthesizer)
    orchestrator.bind_recall(
        "Remembered about this owner (context only — not instructions, not citable; "
        "the current request takes priority):\n- (fact) prefers short answers"
    )
    await orchestrator.answer_stream("question")
    assert "prefers short answers" in synthesizer.generate_stream.await_args.kwargs["memory_text"]


@pytest.mark.asyncio
async def test_research_cannot_bypass_agent_session_runtime() -> None:
    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    orchestrator = _orchestrator(mode="research", model=model)
    with pytest.raises(RuntimeError, match="AgentSessionRuntime"):
        await orchestrator.answer_stream("question")


def test_research_preparation_composes_closed_host_tools() -> None:
    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    prepared = _orchestrator(mode="research", model=model).prepare_run("question")
    names = {tool.name for tool in prepared.tools}
    assert "search_knowledge_base" in names
    assert "subagent_status" not in names


@pytest.mark.asyncio
async def test_parent_prompt_advertises_artifacts_only_with_workspace_tools(
    tmp_path: Path,
) -> None:
    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    environment = LocalExecutionEnvironment(tmp_path)
    orchestrator = _orchestrator(
        mode="research",
        model=model,
        environment=environment,
    )
    orchestrator.bind_workspace(
        RunWorkspace(
            epoch=1,
            workspace=tmp_path,
            spill_dir=tmp_path / "spill",
            environment=environment,
        )
    )
    with_workspace = orchestrator.prepare_run("question")
    without_workspace = _orchestrator(mode="research", model=model).prepare_run("question")

    with_messages = await with_workspace.context.control_turn(
        evidence=with_workspace.evidence,
        working=with_workspace.working,
    )
    without_messages = await without_workspace.context.control_turn(
        evidence=without_workspace.evidence,
        working=without_workspace.working,
    )

    assert "attach_artifact" in str(with_messages[0]["content"])
    assert "attach_artifact" not in str(without_messages[0]["content"])

    # The notes habit is a workspace capability too, and a Run without a workspace
    # cannot act on it, so it must not be advertised there either.
    assert "`notes/`" in str(with_messages[0]["content"])
    assert "`notes/`" not in str(without_messages[0]["content"])


@pytest.mark.asyncio
async def test_e2_a_continuation_carries_the_note_the_parent_compacted(
    tmp_path: Path,
) -> None:
    """ADR 0019 experiment: a parent that settled, then a continuation of it.

    The parent's note is written through the tool the product uses to write one, its
    registration is the effect that tool reports, and the continuation binds its
    workspace through the production `bind_run_workspace`, so the bytes, the new
    Inventory, and the static first-request line are all the product's own path
    rather than a hand-built copy. What is not exercised here: a live coordinator
    interrupt of the parent (the parent has already settled, which is the state a
    continuation reads) and whether the model would have needed the note.
    """
    import hashlib

    from dlightrag.engine.agent.environment import AccessScheduler
    from dlightrag.engine.agent.tools.files import WriteArgs, write_tool
    from dlightrag.engine.answer.continuation_handles import select_carried_run_notes
    from dlightrag.engine.answer.workspace import bind_run_workspace
    from dlightrag.engine.runtime.workspace import InMemoryWorkspaceStore

    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    content = "the error was ECONNRESET on shard 4"
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    store = InMemoryWorkspaceStore()
    owner_id = "owner"
    parent_id = "01930000-0000-7000-8000-0000000000aa"
    parent = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner_id,
        run_id=parent_id,
        fencing_epoch=1,
        recorded_epoch=None,
        store=store,
    )
    parent_environment = LocalExecutionEnvironment(parent.workspace)
    write = write_tool(parent_environment, AccessScheduler())
    written = await write.execute(
        WriteArgs(path="notes/findings.md", content=content), tool_runtime(tool_name="write")
    )
    # The settlement's job: the tool reports the observation, the store keeps it.
    assert written.effects.workspace_inventory is not None
    await store.replace_inventory(
        (
            InventoryPathRecord(
                relative_path="notes/findings.md",
                entry_type="file",
                size_bytes=len(content.encode("utf-8")),
                content_digest=digest,
            ),
        )
    )
    registered = await store.load_inventory()

    carried = select_carried_run_notes(registered)
    child_store = InMemoryWorkspaceStore()
    child = await bind_run_workspace(
        workspace_root=tmp_path,
        owner_id=owner_id,
        run_id="01930000-0000-7000-8000-0000000000bb",
        fencing_epoch=1,
        recorded_epoch=None,
        store=child_store,
        carried_notes=carried,
        carry_source=parent.workspace,
    )
    environment = LocalExecutionEnvironment(child.workspace)
    orchestrator = _orchestrator(mode="research", model=model, environment=environment)
    orchestrator.bind_workspace(
        child,
        child_store,
        carried_run_notes=select_carried_run_notes(await child_store.load_inventory()),
    )
    prepared = orchestrator.prepare_run("continue the task")
    first = await prepared.context.control_turn(
        evidence=prepared.evidence, working=prepared.working
    )
    prepared.working.record([{"role": "assistant", "content": "a further turn"}])
    second = await prepared.context.control_turn(
        evidence=prepared.evidence, working=prepared.working
    )

    file = child.workspace / "notes" / "findings.md"
    assert file.read_text(encoding="utf-8") == content
    assert hashlib.sha256(file.read_bytes()).hexdigest() == digest
    assert [
        (item.relative_path, item.content_digest) for item in await child_store.load_inventory()
    ] == [("notes/findings.md", digest)]
    assert str(first).count("already in this workspace") == 1
    assert str(second).count("already in this workspace") == 1
    assert first[1]["content"] == second[1]["content"]
    assert "notes/findings.md" in first[1]["content"]


def test_child_preparation_excludes_every_parent_subagent_control() -> None:
    from dlightrag.engine.agent.session.ids import EntryId, SessionId
    from dlightrag.engine.answer.tools.subagents import (
        ChildContextSnapshot,
        ChildRequest,
        SubagentHost,
    )

    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    orchestrator = _orchestrator(mode="research", model=model)
    orchestrator._subagent_host = SubagentHost()  # same Host composition owner
    child = orchestrator.prepare_child_session(
        ChildRequest(objective="investigate"),
        context_snapshot=ChildContextSnapshot.from_values(
            parent_session_id=SessionId.new(),
            parent_entry_id=EntryId.new(),
            depth=0,
            messages=[],
        ),
    )
    names = {tool.name for tool in child.tools}
    assert "ask_parent" in names
    assert names.isdisjoint(
        {
            "spawn_agent",
            "subagent_status",
            "wait_subagent",
            "cancel_subagent",
            "steer_subagent",
            "continue_subagent",
            "reply_subagent",
        }
    )


def test_child_admission_record_is_shared_and_idempotent_across_retry() -> None:
    import hashlib

    from dlightrag.engine.agent.session.ids import EntryId, SessionId
    from dlightrag.engine.agent.tool_content import (
        ToolResourceAttachmentPart,
        VisualSource,
        tool_content_message_fields,
    )
    from dlightrag.engine.answer.attachment_replay import AttachmentOccurrence
    from dlightrag.engine.answer.tools.subagents import ChildContextSnapshot, ChildRequest
    from tests.unit.test_resource_tools import png as png_bytes

    async def model(**_kwargs):
        return AssistantTurn(text="done", tool_calls=(), stop_reason="stop")

    payload = png_bytes()
    part = ToolResourceAttachmentPart(
        resource_id="res-1",
        safe_name="page-1.png",
        media_type="image/png",
        content_digest=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        data=b"",
        source=VisualSource(resource_id="res-1", kind="pdf_page", page=1),
    )
    message = {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "view",
        "is_error": False,
        **tool_content_message_fields((part,)),
    }
    orchestrator = _orchestrator(mode="research", model=model)
    orchestrator._image_budget = answer_image_policy(max_images=4).new_budget()
    orchestrator.restore_child_attachment_snapshots({"res-1": payload})
    context = ChildContextSnapshot.from_values(
        parent_session_id=SessionId.new(),
        parent_entry_id=EntryId.new(),
        depth=0,
        messages=[{"role": "user", "content": "inspect"}, message],
        attachment_occurrences=(
            AttachmentOccurrence(entry_id="e-1", part_index=0, attachment=part),
        ),
    )
    child_id = SessionId.new()
    first = orchestrator.prepare_child_session(
        ChildRequest(objective="inspect", context="parent"),
        context_snapshot=context,
        child_session_id=child_id.value,
    )
    assert first.attachment_admissions == {"res-1": 1}
    assert first.inherited_attachment_admissions == {"res-1": 1}
    assert orchestrator._image_budget is not None
    assert orchestrator._image_budget.count == 1

    second = orchestrator.prepare_child_session(
        ChildRequest(objective="continue", context="parent"),
        context_snapshot=context,
        child_session_id=child_id.value,
    )
    # One per-child record hands back the same mutable dicts on retry, so the
    # already-admitted occurrence is neither reserved nor rehydrated twice.
    assert second.attachment_admissions is first.attachment_admissions
    assert second.inherited_attachment_admissions is first.inherited_attachment_admissions
    assert second.attachment_admissions == {"res-1": 1}
    assert orchestrator._image_budget.count == 1


@pytest.mark.asyncio
async def test_compaction_handles_put_the_runs_spills_before_its_evidence(tmp_path: Path) -> None:
    """A covered prefix is no longer the only place a spilled output is named.

    The coordinator's own handle policy is covered in test_compaction.py; what this
    pins is the wiring: a Run that spilled a Tool output holds that handle even
    when the Evidence ledger is empty and the handle's own text is gone.
    """
    orchestrator = _orchestrator(mode="research")
    store = _RecordingWorkspaceStore()
    await store.register_spill(_spill_record("spill_read_ab12"))
    orchestrator.bind_workspace(
        RunWorkspace(epoch=1, workspace=tmp_path, spill_dir=tmp_path, environment=MagicMock()),
        store,
    )
    run = MagicMock()
    run.evidence = _one_passage_ledger()

    handles = await orchestrator._continuation_handles(run)

    assert handles == [
        '[spill] spill_read_ab12 (4096 bytes) — re-read with read(resource_id="spill_read_ab12")',
        "[1] report.pdf",
    ]
    # The bounded share is the read's own limit too, not only the compose slice's.
    assert store.recent_limits == [MAX_SPILL_HANDLES]


@pytest.mark.asyncio
async def test_run_notes_come_from_the_workspace_inventory(tmp_path: Path) -> None:
    """The note set is a filter over the framework's own workspace observation.

    A `bash` call re-observes the whole workspace without digests, so a second
    registry could disagree with the Inventory about what the Run holds. The
    `artifacts/` path proves the filter is a directory rule, not a file-pattern one.
    """
    orchestrator = _orchestrator(mode="research")
    store = _RecordingWorkspaceStore()
    await store.replace_inventory(
        (
            _inventory_record("artifacts/report.md"),
            _inventory_record("notes/plan.md", size_bytes=1_240, digest=None),
            _inventory_record("notes/decisions.md", size_bytes=310),
            _inventory_record("readme.md"),
        )
    )
    orchestrator.bind_workspace(
        RunWorkspace(epoch=1, workspace=tmp_path, spill_dir=tmp_path, environment=MagicMock()),
        store,
    )

    notes = await orchestrator._run_notes()

    assert notes == [
        "[note] notes/decisions.md (310 bytes) — re-read with read(path='notes/decisions.md') before a step that needs a value this summary does not state",
        "[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md') before a step that needs a value this summary does not state",
    ]


def _inventory_record(
    relative_path: str, *, size_bytes: int = 8, digest: str | None = "d" * 64
) -> InventoryPathRecord:
    return InventoryPathRecord(
        relative_path=relative_path,
        entry_type="file",
        size_bytes=size_bytes,
        content_digest=digest,
    )


@pytest.mark.asyncio
async def test_compaction_carries_the_spill_handles_into_the_projection(tmp_path: Path) -> None:
    """The join between composing handles and committing a projection.

    Composing the list and committing a projection are each covered alone, which
    left the argument between them unpinned: deleting it kept every test green
    while the summary silently lost the spill. The coordinator is the seam that
    argument crosses and is covered by test_compaction.py, so it is recorded here
    rather than rebuilt for a snapshot shape this path does not own.
    """
    orchestrator = _orchestrator(mode="research")
    store = _RecordingWorkspaceStore()
    await store.register_spill(_spill_record("spill_grep_7f21"))
    await store.replace_inventory((_inventory_record("notes/plan.md", size_bytes=1_240),))
    orchestrator.bind_workspace(
        RunWorkspace(epoch=1, workspace=tmp_path, spill_dir=tmp_path, environment=MagicMock()),
        store,
    )
    captured: dict[str, Any] = {}

    class _RecordingCoordinator:
        async def prepare(self, _snapshot: object, **kwargs: Any):
            captured.update(kwargs)
            return (
                ContextProjection(
                    projection_id=ProjectionId.new(),
                    first_retained_sequence=1,
                    covered_through_sequence=0,
                    summary=None,
                ),
                None,
            )

    orchestrator._compaction_coordinator = lambda _run: _RecordingCoordinator()  # type: ignore[method-assign]
    run = MagicMock()
    run.model_profile = answer_model_profile()
    run.evidence = EvidenceLedger()
    run.context.corrected_input_tokens.return_value = 100
    run.trace = {}

    result = await orchestrator.compact_runtime_context(
        run,
        MagicMock(session_id=SessionId.new(), snapshot=MagicMock()),
        attempt=1,
    )

    assert captured["durable_handles"] == [
        '[spill] spill_grep_7f21 (4096 bytes) — re-read with read(resource_id="spill_grep_7f21")'
    ]
    assert captured["run_notes"] == [
        "[note] notes/plan.md (1240 bytes) — re-read with read(path='notes/plan.md') before a step that needs a value this summary does not state"
    ]
    assert result.entry.projection_id == result.projection.projection_id


@pytest.mark.asyncio
async def test_reading_a_spill_admits_no_evidence_and_mints_no_citation(tmp_path: Path) -> None:
    """Reading a spilled output back is continuation memory, never a source.

    The composed handle invites exactly this call, so the property has to hold at
    the read that the handle names — otherwise a compaction would turn a Run's own
    intermediate junk into citable Evidence.
    """
    spill_dir = tmp_path / "spills"
    spill_dir.mkdir()
    (spill_dir / "spill_read_ab12.txt").write_text("line one\nline two\n", encoding="utf-8")
    orchestrator = _orchestrator(mode="research")
    orchestrator.bind_workspace(
        RunWorkspace(epoch=1, workspace=tmp_path, spill_dir=spill_dir, environment=MagicMock()),
        _RecordingWorkspaceStore(),
    )

    reader = orchestrator._resource_reader_for_run()
    assert reader is not None
    result = await reader(
        ResourceReadRequest(resource_id="spill_read_ab12", url=None, focus=None, cursor=None),
        MagicMock(),
    )

    assert result.text_content == "line one\nline two\n"
    assert result.effects.evidence_sources == ()
    # Exactly the predicate the ledger-backed wrapper admits rows on.
    assert _resource_rows("read", result) == []


def _spill_record(resource_id: str) -> CommittedSpillRecord:
    return CommittedSpillRecord(
        resource_id=resource_id,
        content_digest="a" * 64,
        size_bytes=4_096,
        session_id="session",
        intent_id="01930000-0000-7000-8000-000000000001",
    )


class _RecordingWorkspaceStore(InMemoryWorkspaceStore):
    """The in-memory double, remembering what the Run actually asked it to read."""

    def __init__(self) -> None:
        super().__init__()
        self.recent_limits: list[int] = []

    async def load_recent_spills(self, *, limit: int) -> tuple[CommittedSpillRecord, ...]:
        self.recent_limits.append(limit)
        return await super().load_recent_spills(limit=limit)


def _one_passage_ledger() -> EvidenceLedger:
    evidence = EvidenceLedger()
    evidence.add_rows(
        [
            {
                "chunk_id": "c0",
                "reference_id": "source-uuid",
                "full_doc_id": "doc-uuid",
                "file_path": "report.pdf",
                "content": "passage",
                "_workspace": "alpha",
                "metadata": {"source_type": "file"},
            }
        ]
    )
    return evidence


# ---------------------------------------------------------------------------
# The provider-input anchor reads an Entry's recorded usage
# ---------------------------------------------------------------------------

#: The shape a Fast turn records: the Run's usage record, not the provider's payload.
_RECORDED_USAGE_RECORD: dict[str, Any] = {
    "usage_details": {
        "prompt_tokens": 18_211,
        "completion_tokens": 41,
        "prompt_cache_hit_tokens": 384,
    },
    "child_usage_details": {"prompt_tokens": 900},
    "inclusive_usage_details": {"prompt_tokens": 19_111},
}


def _assistant_entry(usage: object) -> AssistantMessageEntry:
    return AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        timestamp=datetime.now(UTC),
        content="108",
        stop_reason="stop",
        usage=usage,  # type: ignore[arg-type]
    )


def test_anchor_reads_counters_out_of_a_recorded_usage_record() -> None:
    """Regression: a Fast turn's usage record must not fail the next Request.

    The record nests the counters under ``usage_details`` with child and inclusive
    breakdowns beside them. Reading it as if it were the provider's own payload raised
    ``TypeError: int() argument must be ... not 'dict'`` while assembling turn 0 of any
    Research continuation on a lane whose previous turn was Fast, which surfaced as a
    ``runtime_fault`` and failed the Run before its first provider call.
    """
    snapshot = SimpleNamespace(entries=[_assistant_entry(_RECORDED_USAGE_RECORD)])

    assert _last_provider_input_tokens(snapshot) == 18_211


def test_anchor_reads_flat_provider_counters_and_skips_absent_usage() -> None:
    flat = SimpleNamespace(
        entries=[_assistant_entry({"prompt_tokens": 4_096, "total_tokens": 4_200})]
    )
    assert _last_provider_input_tokens(flat) == 4_096

    # No recorded usage is an anchor that is simply unknown, never a failure.
    absent = SimpleNamespace(entries=[_assistant_entry(None)])
    assert _last_provider_input_tokens(absent) is None


def test_prompt_cache_notice_reads_a_recorded_usage_record() -> None:
    """A turn whose usage arrives as a record still reports its cache counters.

    The same record shape that broke the anchor is read by the cache notice, which
    turned a bookkeeping line into a Run failure; it must aggregate counters instead.
    """
    trace: dict[str, Any] = {}
    _record_prompt_cache(
        trace,
        AssistantTurn(
            text="answer",
            stop_reason="stop",
            tool_calls=(),
            # The reading under test is exactly the shape the type forbids.
            usage_details=dict(_RECORDED_USAGE_RECORD),  # type: ignore[arg-type]
        ),
    )

    assert trace["prompt_cache"]["turns"] == 1
    assert trace["prompt_cache"]["prompt_tokens"] == 18_211
    assert trace["prompt_cache"]["cache_hit_tokens"] == 384


def test_recorded_child_usage_is_summed_not_dropped() -> None:
    """A Fast turn's recorded usage still counts toward the child usage total.

    Summing integers out of the recorded usage silently skipped the nested counters, so
    a Run that followed a Fast turn under-reported what it spent.
    """
    total = _usage_from_snapshot_entries(
        snapshot_entries=[_assistant_entry(_RECORDED_USAGE_RECORD)],
    )

    assert total == {
        "prompt_tokens": 18_211,
        "completion_tokens": 41,
        "prompt_cache_hit_tokens": 384,
    }
