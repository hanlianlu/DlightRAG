# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable asynchronous child Agent tools owned by one parent Answer Run.

Accepted child envelopes live in the roster before ``spawn_agent`` returns.
Process-local tasks only accelerate that durable work; parent reclaim rebuilds
running children from their stored envelope under the current contract.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.ids import EntryId, IntentId, SessionId
from dlightrag.engine.agent.tool_content import tool_content_message_fields
from dlightrag.engine.agent.tools import AgentTool, ToolResult, ToolRuntime
from dlightrag.engine.answer.attachment_replay import AttachmentOccurrence
from dlightrag.engine.answer.errors import ChildToolNarrowingError
from dlightrag.engine.answer.evidence import EvidenceDelta
from dlightrag.engine.answer.research.persistence import (
    CancelChild,
    ContinueChild,
    CreateChildGuidance,
    ExpireChildGuidance,
    FinishChild,
    ListChildGuidance,
    ListChildren,
    LoadChild,
    LoadChildGuidance,
    PersistChild,
    ReleaseChildren,
    ReplyChildGuidance,
    SteerChild,
    WaitChildGuidance,
)
from dlightrag.engine.runtime.coordinator import RunCancellationObserved
from dlightrag.engine.runtime.errors import RunCancelledError

type ChildStatus = Literal["running", "succeeded", "failed", "cancelled"]
type ChildContextMode = Literal["isolated", "parent"]
type ChildModelRole = Literal["query", "extract", "keyword", "vlm", "default"]

logger = logging.getLogger(__name__)


class _ParentRunCancelled(asyncio.CancelledError):
    """A cooperative parent cancellation crossing the Tool execution seam."""


class ChildRequest(BaseModel):
    """One bounded child invocation selected by the parent model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    objective: str = Field(min_length=1, description="One concrete child objective.")
    context: ChildContextMode = Field(
        default="isolated",
        description="isolated starts from the objective; parent also receives parent context.",
    )
    model_role: ChildModelRole = Field(
        default="query", description="Configured tool-capable model role for the child."
    )
    tools: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Optional narrowing subset for this child. It can never restore what the Run "
            "withholds: roster controls, durable owner memory writes, and publication."
        ),
    )


class SpawnAgentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    children: tuple[ChildRequest, ...] = Field(
        min_length=1,
        max_length=8,
        description="One or more foreground child requests, run in parallel when possible.",
    )

    @model_validator(mode="after")
    def _unique_objectives(self) -> SpawnAgentInput:
        if len({child.objective for child in self.children}) != len(self.children):
            raise ValueError("child objectives must be unique within one spawn call")
        return self


class ChildControlInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    child_session_id: str = Field(min_length=1, description="Child session id from spawn_agent.")


class ChildMessageInput(ChildControlInput):
    content: str = Field(min_length=1, max_length=20_000)


class GuidanceReplyInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    request_id: str = Field(min_length=1, description="Correlated request id from ask_parent.")
    content: str = Field(min_length=1, max_length=20_000)


class AskParentInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)

    question: str = Field(min_length=1, max_length=20_000)
    expires_after_seconds: int | None = Field(default=None, ge=1, le=86_400)


@dataclass(frozen=True, slots=True)
class ChildContextSnapshot:
    """Bounded immutable parent context explicitly handed to one Child Session."""

    parent_session_id: SessionId
    parent_entry_id: EntryId
    depth: int
    messages_json: str
    evidence_state_json: str = "{}"
    attachment_occurrences: tuple[AttachmentOccurrence, ...] = ()

    def __post_init__(self) -> None:
        import json

        if self.depth < 0:
            raise ValueError("Child context depth cannot be negative")
        if not isinstance(json.loads(self.messages_json), list):
            raise ValueError("Child context messages must be an array")
        if not isinstance(json.loads(self.evidence_state_json), dict):
            raise ValueError("Child context evidence state must be an object")
        expected = [
            tool_content_message_fields((occurrence.attachment,))["attachments"][0]
            for occurrence in self.attachment_occurrences
        ]
        actual = [
            attachment for message in self.messages for attachment in message.get("attachments", [])
        ]
        if actual != expected or any(
            occurrence.attachment.data for occurrence in self.attachment_occurrences
        ):
            raise ValueError("Child tool attachments require exact byte-free occurrence pins")
        references = [item.reference_id for item in self.attachment_occurrences]
        if len(set(references)) != len(references):
            raise ValueError("duplicate Child attachment occurrence pin")

    @classmethod
    def from_values(
        cls,
        *,
        parent_session_id: SessionId,
        parent_entry_id: EntryId,
        depth: int,
        messages: list[dict[str, Any]],
        evidence_state: Mapping[str, Any] | None = None,
        attachment_occurrences: tuple[AttachmentOccurrence, ...] = (),
    ) -> ChildContextSnapshot:
        # TOOL pixels are transport-private. User image blocks keep their existing
        # input contract; only tool attachments are replaced by occurrence pins.
        messages = [
            {
                **message,
                "attachments": [
                    {key: value for key, value in attachment.items() if key != "data_url"}
                    for attachment in message["attachments"]
                ],
            }
            if message.get("role") == "tool" and "attachments" in message
            else message
            for message in messages
        ]
        return cls(
            parent_session_id=parent_session_id,
            parent_entry_id=parent_entry_id,
            depth=depth,
            messages_json=canonical_json(messages),
            evidence_state_json=canonical_json(dict(evidence_state or {})),
            attachment_occurrences=attachment_occurrences,
        )

    @property
    def messages(self) -> list[dict[str, Any]]:
        import json

        return json.loads(self.messages_json)

    @property
    def evidence_state(self) -> dict[str, Any]:
        import json

        return json.loads(self.evidence_state_json)

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "parent_session_id": self.parent_session_id.value,
            "parent_entry_id": self.parent_entry_id.value,
            "depth": self.depth,
            "messages": self.messages,
            "evidence_state": self.evidence_state,
            "attachment_occurrences": [
                item.canonical_payload() for item in self.attachment_occurrences
            ],
        }


@dataclass(frozen=True, slots=True)
class ChildOutcome:
    """Distilled Child result returned to the parent ToolResult."""

    status: ChildStatus
    summary: str
    handles: tuple[str, ...] = ()
    usage: Mapping[str, int] | None = None
    delta: EvidenceDelta | None = None
    child_session_id: str = ""
    evidence_state: Mapping[str, Any] | None = None
    operation_id: str = ""
    fencing_epoch: int | None = None

    def durable_payload(self) -> dict[str, Any]:
        """Return the exact parent-visible outcome needed for replay."""
        return {
            "status": self.status,
            "summary": self.summary,
            "handles": list(self.handles),
            "usage": dict(self.usage or {}),
            "child_session_id": self.child_session_id,
            "operation_id": self.operation_id,
            "evidence_state": (
                dict(self.evidence_state) if self.evidence_state is not None else None
            ),
        }

    @classmethod
    def from_durable_payload(cls, payload: Mapping[str, Any]) -> ChildOutcome:
        status = payload.get("status")
        if status not in {"running", "succeeded", "failed", "cancelled"}:
            raise ValueError("persisted Child outcome has an invalid status")
        handles = payload.get("handles")
        usage = payload.get("usage")
        evidence_state = payload.get("evidence_state")
        if not isinstance(handles, list) or not all(isinstance(item, str) for item in handles):
            raise ValueError("persisted Child outcome has invalid handles")
        if not isinstance(usage, Mapping) or not all(
            isinstance(key, str) and isinstance(value, int) for key, value in usage.items()
        ):
            raise ValueError("persisted Child outcome has invalid usage")
        if "evidence_state" not in payload or (
            evidence_state is not None and not isinstance(evidence_state, Mapping)
        ):
            raise ValueError("persisted Child outcome has invalid evidence state")
        return cls(
            status=cast(ChildStatus, status),
            summary=str(payload.get("summary") or ""),
            handles=tuple(handles),
            usage={str(key): int(value) for key, value in usage.items()},
            child_session_id=str(payload.get("child_session_id") or ""),
            evidence_state=(dict(evidence_state) if isinstance(evidence_state, Mapping) else None),
            operation_id=str(payload.get("operation_id") or ""),
        )


@dataclass
class SubagentHost:
    """Late-bound durable child scheduler for one parent execution owner."""

    parent_session_id: SessionId | None = None
    run_id: str = ""
    owner_id: str = ""
    max_concurrency: int = 4
    async_lifecycle: bool = True
    interactive_controls: bool = True
    check_cancelled: Callable[[], Awaitable[None]] | None = None
    persist: PersistChild | None = None
    load_child: LoadChild | None = None
    list_children: ListChildren | None = None
    finish_child: FinishChild | None = None
    request_cancel: CancelChild | None = None
    release_children: ReleaseChildren | None = None
    steer_child: SteerChild | None = None
    continue_child: ContinueChild | None = None
    reply_guidance: ReplyChildGuidance | None = None
    create_guidance: CreateChildGuidance | None = None
    load_guidance: LoadChildGuidance | None = None
    wait_guidance: WaitChildGuidance | None = None
    expire_guidance: ExpireChildGuidance | None = None
    list_guidance: ListChildGuidance | None = None
    guidance_timeout_seconds: int = 300
    model_guidance: str | None = None
    prepare_dispatch: (
        Callable[[SessionId, ChildRequest, ChildContextSnapshot], Mapping[str, Any]] | None
    ) = None
    run_child: (
        Callable[[SessionId, ChildRequest, str, ChildContextSnapshot], Awaitable[ChildOutcome]]
        | None
    ) = None
    context_snapshot: ChildContextSnapshot | None = None
    depth: int = 0
    merge_evidence: Callable[[Mapping[str, Any], str, str], tuple[str, ...]] | None = None
    record_usage: Callable[[Mapping[str, int]], None] | None = None
    tasks: dict[str, asyncio.Task[ChildOutcome]] = field(default_factory=dict)
    outcomes: dict[str, ChildOutcome] = field(default_factory=dict)
    _semaphore: asyncio.Semaphore | None = field(default=None, init=False, repr=False)
    _detaching: bool = field(default=False, init=False, repr=False)
    _cancel_requested: set[str] = field(default_factory=set, init=False, repr=False)
    _parent_wake: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    @property
    def detaching(self) -> bool:
        return self._detaching

    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.max_concurrency))
        return self._semaphore

    async def restore_pending(self) -> None:
        """Rebuild runnable process tasks from durable accepted envelopes."""
        if not self.async_lifecycle or self.list_children is None:
            return
        rows = await self.list_children(owner_id=self.owner_id, run_id=self.run_id)
        for row in rows or ():
            if not isinstance(row, Mapping) or str(row.get("status")) != "running":
                continue
            dispatch = _dispatch_from_row(self, row)
            if dispatch is not None:
                _start_child_task(self, *dispatch)

    async def completed_dispatch_notifications(
        self,
        *,
        seen: set[str],
    ) -> tuple[tuple[str, str], ...]:
        """Return complete durable dispatch outcomes not driven in this execution.

        Notification identity includes each Child Operation identity. Slice 2 can
        therefore expose later Operations in the same Child Session without
        deduplicating the Session forever.
        """
        if not self.async_lifecycle or self.list_children is None:
            return ()
        if self.list_guidance is not None and self.parent_session_id is not None:
            questions = await self.list_guidance(
                owner_id=self.owner_id,
                run_id=self.run_id,
                parent_session_id=self.parent_session_id.value,
            )
            for question in questions or ():
                request_id = str(question.get("request_id") or "")
                notification_id = f"child-guidance:{request_id}"
                if request_id and notification_id not in seen:
                    child_id = str(question.get("child_session_id") or "")
                    content = (
                        f"Child session {child_id} asks for guidance "
                        f"(request_id={request_id}): {question.get('question') or ''}. "
                        "Reply with reply_subagent using exactly this request_id."
                    )
                    return ((notification_id, content),)
        rows = await self.list_children(owner_id=self.owner_id, run_id=self.run_id)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows or ():
            if not isinstance(row, Mapping):
                continue
            parent_intent_id = str(row.get("parent_intent_id") or "")
            if parent_intent_id:
                grouped.setdefault(parent_intent_id, []).append(row)
        notifications: list[tuple[str, str]] = []
        for parent_intent_id, dispatch_rows in grouped.items():
            if any(str(row.get("status")) == "running" for row in dispatch_rows):
                continue
            outcomes = tuple(
                _adopt_outcome(
                    self,
                    _terminal_outcome_from_row(row, str(row.get("child_session_id") or "")),
                    parent_call_id=str(row.get("parent_call_id") or ""),
                )
                for row in dispatch_rows
            )
            identities = sorted(
                f"{outcome.child_session_id}:{outcome.operation_id or 'initial'}"
                for outcome in outcomes
            )
            digest = hashlib.sha256("\0".join(identities).encode("utf-8")).hexdigest()[:24]
            notification_id = f"child-results:{parent_intent_id}:{digest}"
            if notification_id in seen:
                continue
            notifications.append((notification_id, _many_result(outcomes).text_content))
        return tuple(notifications)

    def notify_parent(self) -> None:
        """Wake the parent barrier after a durable result or future question write."""
        self._parent_wake.set()

    async def wait_for_activity(self, *, child_id: str | None = None) -> None:
        """Park until durable child work changes; never poll a provider in a loop.

        This is intentionally an activity barrier rather than ``gather(all)``.
        Slice-2 question delivery can become another durable parent notification
        and wake the same loop while a Child is parked awaiting its parent.
        """
        await self.restore_pending()
        selected = (
            [self.tasks[child_id]]
            if child_id is not None and child_id in self.tasks
            else list(self.tasks.values())
        )
        finished = [task for task in selected if task.done()]
        for task in finished:
            await task
        active = [task for task in selected if not task.done()]
        if self._parent_wake.is_set():
            self._parent_wake.clear()
            return
        if active:
            wake = asyncio.create_task(self._parent_wake.wait())
            done, _pending = await asyncio.wait(
                (*active, wake),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if wake in done:
                self._parent_wake.clear()
            else:
                wake.cancel()
                await asyncio.gather(wake, return_exceptions=True)
            for task in done:
                if task is not wake:
                    await task
            return
        if self.list_children is None:
            return
        rows = await self.list_children(owner_id=self.owner_id, run_id=self.run_id)
        if any(_is_running_work(self, row) for row in rows or () if isinstance(row, Mapping)):
            raise RuntimeError("accepted Child dispatch lost its reconstructible envelope")

    async def has_running_children(self) -> bool:
        if self.list_children is None:
            return any(not task.done() for task in self.tasks.values())
        rows = await self.list_children(owner_id=self.owner_id, run_id=self.run_id)
        return any(_is_running_work(self, row) for row in rows or () if isinstance(row, Mapping))

    async def stop(self, *, cancel: bool) -> None:
        """Join local tasks, preserving durable work on process detach."""
        child_ids = set(self.tasks)
        if self.list_children is not None:
            rows = await self.list_children(owner_id=self.owner_id, run_id=self.run_id)
            child_ids.update(
                str(row.get("child_session_id") or "")
                for row in rows or ()
                if isinstance(row, Mapping) and str(row.get("status")) == "running"
            )
        if cancel:
            for child_id in sorted(child_ids - {""}):
                await _request_child_cancel(self, child_id)
        else:
            self._detaching = True
        tasks = tuple(task for task in self.tasks.values() if not task.done())
        for task in tasks:
            task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    raise result
        if cancel and await self.has_running_children():
            # A task cancelled before entering its coroutine did not get a
            # chance to close the durable Agent Operation. Reconstruct it with
            # the persisted cancellation request and drive that closure now.
            self.tasks.clear()
            await self.restore_pending()
            closing = tuple(task for task in self.tasks.values() if not task.done())
            if closing:
                await asyncio.gather(*closing)
            if await self.has_running_children():
                raise RuntimeError("cancelled Child Session did not settle")
        if not cancel and self.release_children is not None:
            await self.release_children(owner_id=self.owner_id, run_id=self.run_id)
        self.tasks.clear()


_SPAWN_DESCRIPTION = (
    "Accept one or many asynchronous child Agent Sessions and return stable handles "
    "immediately. A child runs with its parent's tools except the ones that spend the "
    "Run's authority — its roster controls, durable owner memory, and publication — so "
    "pass `tools` to narrow a child (a read-only investigator, say) rather than to grant "
    "one."
)


def subagent_tools(*, host: SubagentHost) -> tuple[AgentTool, ...]:
    """Return versioned spawn/status/wait/cancel tools over one durable roster."""

    async def spawn(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(SpawnAgentInput, raw)
        return await _spawn(host, args, runtime)

    async def status(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildControlInput, raw)
        await _check_cancelled(host)
        return _result_with_guidance(
            host,
            await _status(host, args.child_session_id),
            await _pending_guidance_for_child(host, args.child_session_id),
        )

    async def wait(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildControlInput, raw)
        await _check_cancelled(host)
        current = await _status(host, args.child_session_id)
        if current.status != "running":
            return _result_with_guidance(
                host, current, await _pending_guidance_for_child(host, args.child_session_id)
            )
        await host.restore_pending()
        pending = await _pending_guidance_for_child(host, args.child_session_id)
        if not pending:
            task = host.tasks.get(args.child_session_id)
            if task is not None and not task.done():
                await host.wait_for_activity(child_id=args.child_session_id)
            pending = await _pending_guidance_for_child(host, args.child_session_id)
        return _result_with_guidance(
            host,
            await _status(host, args.child_session_id),
            pending,
        )

    async def cancel(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildControlInput, raw)
        await _check_cancelled(host)
        current = await _status(host, args.child_session_id)
        if current.status != "running":
            return _single_result(_adopt_outcome(host, current))
        outcome = await _cancel_child(host, args.child_session_id)
        return _single_result(_adopt_outcome(host, outcome))

    async def steer(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildMessageInput, raw)
        if host.steer_child is None or host.parent_session_id is None:
            return ToolResult.text("Child steer is unavailable.", is_error=True)
        receipt = await host.steer_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=args.child_session_id,
            parent_session_id=host.parent_session_id.value,
            content=args.content,
            submission_key=f"parent-steer:{runtime.intent_id.value}",
            origin="parent",
        )
        return ToolResult.text(canonical_json(dict(receipt)))

    async def continue_child(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(ChildMessageInput, raw)
        if host.continue_child is None or host.parent_session_id is None:
            return ToolResult.text("Child continuation is unavailable.", is_error=True)
        receipt = await host.continue_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=args.child_session_id,
            parent_session_id=host.parent_session_id.value,
            content=args.content,
            submission_key=f"parent-continuation:{runtime.intent_id.value}",
            origin="parent",
            reauthorize_user_cancelled=False,
        )
        if receipt.get("outcome") == "accepted":
            await host.restore_pending()
        return ToolResult.text(canonical_json(dict(receipt)))

    async def reply(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(GuidanceReplyInput, raw)
        if host.reply_guidance is None or host.parent_session_id is None:
            return ToolResult.text("Child guidance reply is unavailable.", is_error=True)
        receipt = await host.reply_guidance(
            owner_id=host.owner_id,
            run_id=host.run_id,
            request_id=args.request_id,
            parent_session_id=host.parent_session_id.value,
            content=args.content,
            submission_key=f"parent-reply:{runtime.intent_id.value}",
            origin="parent",
        )
        return ToolResult.text(canonical_json(dict(receipt)))

    if not host.async_lifecycle:
        descriptions = (
            "Run one or many foreground child Agent Sessions and wait for all results.",
            "Read one foreground or completed child session status.",
            "Wait for one known foreground child session.",
            "Cancel one known foreground child session.",
        )
    elif host.interactive_controls:
        descriptions = (
            _SPAWN_DESCRIPTION,
            "Read one accepted asynchronous or completed child session status.",
            "Wait for one known asynchronous child session to settle.",
            "Durably cancel one known child session without cancelling its siblings.",
        )
    else:
        descriptions = (
            _SPAWN_DESCRIPTION,
            "Read one accepted asynchronous or completed child session status.",
            "Wait for one known asynchronous child session to settle.",
            "Durably cancel one known child session without cancelling its siblings.",
        )

    five_models = host.model_guidance is not None
    version = 5
    return (
        AgentTool(
            "spawn_agent",
            descriptions[0] + ("\n" + (host.model_guidance or "") if five_models else ""),
            SpawnAgentInput,
            spawn,
            replay_policy="replayable",
            contract_version=version,
        ),
        AgentTool(
            "subagent_status",
            descriptions[1],
            ChildControlInput,
            status,
            replay_policy="replayable",
            contract_version=version,
        ),
        AgentTool(
            "wait_subagent",
            descriptions[2],
            ChildControlInput,
            wait,
            replay_policy="replayable",
            contract_version=version,
        ),
        AgentTool(
            "cancel_subagent",
            descriptions[3],
            ChildControlInput,
            cancel,
            replay_policy="never",
            contract_version=version,
        ),
        *(
            (
                AgentTool(
                    "steer_subagent",
                    "Queue guidance for only the current Operation of a running child.",
                    ChildMessageInput,
                    steer,
                    replay_policy="replayable",
                    contract_version=version,
                ),
                AgentTool(
                    "continue_subagent",
                    "Start an explicit new Operation in a settled child Session with its pinned model and tools.",
                    ChildMessageInput,
                    continue_child,
                    replay_policy="replayable",
                    contract_version=version,
                ),
                AgentTool(
                    "reply_subagent",
                    "Reply to one correlated ask_parent request from a child.",
                    GuidanceReplyInput,
                    reply,
                    replay_policy="replayable",
                    contract_version=version,
                ),
            )
            if host.async_lifecycle and host.interactive_controls
            else ()
        ),
    )


def child_guidance_tools(*, host: SubagentHost) -> tuple[AgentTool, ...]:
    """Return the durable child-to-parent question tool."""

    async def ask(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = cast(AskParentInput, raw)
        if (
            host.create_guidance is None
            or host.load_guidance is None
            or host.wait_guidance is None
            or host.expire_guidance is None
            or host.load_child is None
            or host.parent_session_id is None
        ):
            return ToolResult.text("Parent guidance is unavailable.", is_error=True)
        child_id = runtime.execution_scope
        child = await host.load_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
        )
        if not isinstance(child, Mapping) or not child.get("operation_id"):
            return ToolResult.text("Current Child Operation is unavailable.", is_error=True)
        operation_id = str(child["operation_id"])
        child_epoch = int(child.get("fencing_epoch") or 0)
        request_id = runtime.intent_id.value
        timeout = args.expires_after_seconds or host.guidance_timeout_seconds
        guidance = await host.create_guidance(
            owner_id=host.owner_id,
            run_id=host.run_id,
            request_id=request_id,
            child_session_id=child_id,
            child_operation_id=operation_id,
            parent_session_id=host.parent_session_id.value,
            question=args.question,
            expires_after_seconds=timeout,
            child_fencing_epoch=child_epoch,
        )
        if guidance is None:
            return ToolResult.text("Child guidance request lost its lease.", is_error=True)
        if guidance.get("status") == "queue_full":
            return ToolResult.text("Too many pending parent guidance requests.", is_error=True)
        host.notify_parent()
        await runtime.emit_update(
            ToolResult.text(f"Waiting for parent reply (request_id={request_id}).")
        )
        while True:
            status = str(guidance.get("status") or "")
            if status == "replied":
                origin = str(guidance.get("reply_origin") or "parent").capitalize()
                return ToolResult.text(
                    f"{origin} reply (request_id={request_id}): {guidance.get('reply') or ''}"
                )
            if status == "cancelled":
                return ToolResult.text("Parent guidance request was cancelled.", is_error=True)
            if status == "expired":
                return ToolResult.text("Parent guidance request expired.", is_error=True)
            expires_at = guidance.get("expires_at")
            if expires_at is None:
                return ToolResult.text("Parent guidance request is corrupt.", is_error=True)
            remaining = max(0.0, (expires_at - datetime.now(UTC)).total_seconds())
            if remaining <= 0:
                await host.expire_guidance(
                    owner_id=host.owner_id,
                    run_id=host.run_id,
                    request_id=request_id,
                    child_session_id=child_id,
                    child_operation_id=operation_id,
                    child_fencing_epoch=child_epoch,
                )
                guidance = await host.load_guidance(
                    owner_id=host.owner_id, run_id=host.run_id, request_id=request_id
                )
                if not isinstance(guidance, Mapping):
                    return ToolResult.text("Parent guidance request disappeared.", is_error=True)
                if str(guidance.get("status") or "") == "pending":
                    from dlightrag.engine.runtime.coordinator import LeaseLostError

                    raise LeaseLostError
                continue
            guidance = await host.wait_guidance(
                owner_id=host.owner_id,
                run_id=host.run_id,
                request_id=request_id,
                timeout_seconds=remaining,
            )
            if not isinstance(guidance, Mapping):
                return ToolResult.text("Parent guidance request disappeared.", is_error=True)

    return (
        AgentTool(
            "ask_parent",
            "Ask the parent one correlated question and wait durably for its reply.",
            AskParentInput,
            ask,
            replay_policy="replayable",
            contract_version=4,
        ),
    )


async def _spawn(
    host: SubagentHost,
    args: SpawnAgentInput,
    runtime: ToolRuntime,
) -> ToolResult:
    if host.parent_session_id is None or not host.run_id:
        raise RuntimeError("spawn_agent is not bound to a parent session")
    if host.run_child is None:
        raise RuntimeError("spawn_agent has no child runner")
    await _check_cancelled(host)
    context_snapshot = host.context_snapshot
    if context_snapshot is None:
        raise RuntimeError("spawn_agent has no explicit parent ContextSnapshot")
    child_ids = tuple(
        child_session_id(
            run_id=host.run_id,
            parent_session_id=host.parent_session_id,
            parent_intent_id=runtime.intent_id,
            position=position,
        )
        for position in range(len(args.children))
    )

    # Every reconstructible async envelope commits before any handle becomes
    # visible. Foreground replay can return an already settled outcome.
    for child_id, request in zip(child_ids, args.children, strict=True):
        terminal = (
            await _load_terminal_child(host, child_id.value) if not host.async_lifecycle else None
        )
        envelope: Mapping[str, Any] = {}
        if host.async_lifecycle and host.persist is not None:
            if host.prepare_dispatch is None:
                raise RuntimeError("spawn_agent has no durable dispatch envelope builder")
            envelope = host.prepare_dispatch(child_id, request, context_snapshot)
        if terminal is None and host.persist is not None:
            await host.persist(
                owner_id=host.owner_id,
                run_id=host.run_id,
                child_session_id=child_id.value,
                parent_session_id=host.parent_session_id.value,
                parent_call_id=runtime.call_id,
                parent_intent_id=runtime.intent_id.value,
                objective=request.objective,
                context_mode=request.context,
                model_role=request.model_role,
                tools=request.tools,
                depth=context_snapshot.depth + 1,
                context_snapshot=context_snapshot.canonical_payload(),
                **envelope,
            )

    tasks = tuple(
        _start_child_task(host, child_id, request, runtime.call_id, context_snapshot)
        for child_id, request in zip(child_ids, args.children, strict=True)
    )
    if host.async_lifecycle:
        return _many_result(
            tuple(
                ChildOutcome(
                    status="running",
                    summary="Child session accepted; use status, wait, or cancel with this handle.",
                    child_session_id=child_id.value,
                )
                for child_id in child_ids
            )
        )
    try:
        outcomes = await asyncio.gather(*tasks)
        return _many_result(tuple(outcomes))
    finally:
        for child_id in child_ids:
            task = host.tasks.pop(child_id.value, None)
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _start_child_task(
    host: SubagentHost,
    child_id: SessionId,
    request: ChildRequest,
    parent_call_id: str,
    context_snapshot: ChildContextSnapshot,
) -> asyncio.Task[ChildOutcome]:
    existing = host.tasks.get(child_id.value)
    if existing is not None and not existing.done():
        return existing
    if existing is not None:
        host.tasks.pop(child_id.value, None)
    task = asyncio.create_task(
        _run_one(host, child_id, request, parent_call_id, context_snapshot),
        name=f"agent-child:{child_id.value}",
    )
    host.tasks[child_id.value] = task
    task.add_done_callback(lambda _task: host.notify_parent())
    return task


async def _run_one(
    host: SubagentHost,
    child_id: SessionId,
    request: ChildRequest,
    parent_call_id: str,
    context_snapshot: ChildContextSnapshot,
) -> ChildOutcome:
    persisted = await _load_terminal_child(host, child_id.value)
    if persisted is not None:
        if not host.async_lifecycle:
            if persisted.evidence_state is not None and host.merge_evidence is not None:
                host.merge_evidence(persisted.evidence_state, child_id.value, parent_call_id)
            if persisted.usage is not None and host.record_usage is not None:
                host.record_usage(persisted.usage)
        host.outcomes[child_id.value] = persisted
        return persisted
    if host.run_child is None:
        raise RuntimeError("spawn_agent has no child runner")
    try:
        async with host.semaphore():
            await _check_cancelled(host)
            outcome = await host.run_child(
                child_id,
                request,
                parent_call_id,
                context_snapshot,
            )
    except (RunCancellationObserved, RunCancelledError) as exc:
        await _finish_cancelled_child(host, child_id.value)
        raise _ParentRunCancelled from exc
    except _ParentRunCancelled:
        await _finish_cancelled_child(host, child_id.value)
        raise
    except asyncio.CancelledError:
        if host._detaching:
            raise
        return await _finish_cancelled_child(host, child_id.value)
    except ChildToolNarrowingError as exc:
        # Caller input: the parent model named these Tools, so the parent is told which
        # ones its Run does not offer instead of receiving a generic child failure.
        logger.warning("Child Session %s named impossible Tools: %s", child_id.value, exc)
        outcome = ChildOutcome(
            status="failed",
            summary=f"Child session was not started: {exc}.",
            child_session_id=child_id.value,
        )
    except Exception as exc:
        # Lease/fencing failures must trigger parent reclaim, never false failure.
        from dlightrag.engine.runtime.coordinator import LeaseLostError
        from dlightrag.engine.runtime.errors import IncompatibleActiveRunError

        if isinstance(exc, (LeaseLostError, IncompatibleActiveRunError)):
            raise
        logger.warning(
            "Child Session %s failed before returning an outcome",
            child_id.value,
            exc_info=True,
        )
        outcome = ChildOutcome(
            status="failed",
            summary="Child session failed before producing a result.",
            child_session_id=child_id.value,
        )
    if not host.async_lifecycle:
        outcome = _adopt_outcome(host, outcome, parent_call_id=parent_call_id)
        if outcome.usage is not None and host.record_usage is not None:
            host.record_usage(outcome.usage)
    return await _finish_outcome(host, child_id.value, outcome)


async def _finish_outcome(
    host: SubagentHost,
    child_id: str,
    outcome: ChildOutcome,
) -> ChildOutcome:
    if host.finish_child is not None:
        committed = await host.finish_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
            status=outcome.status,
            summary=outcome.summary,
            usage=outcome.usage,
            outcome=outcome.durable_payload(),
            child_fencing_epoch=outcome.fencing_epoch,
        )
        if committed is False:
            persisted = await _load_terminal_child(host, child_id)
            if persisted is None:
                from dlightrag.engine.runtime.coordinator import LeaseLostError

                raise LeaseLostError
            outcome = persisted
    host.outcomes[child_id] = outcome
    return outcome


def _dispatch_from_row(
    host: SubagentHost,
    row: Mapping[str, Any],
) -> tuple[SessionId, ChildRequest, str, ChildContextSnapshot] | None:
    """Decode only complete accepted envelopes; sparse precreation is not work."""
    if host.parent_session_id is None:
        return None
    raw_snapshot = row.get("context_snapshot")
    if not isinstance(raw_snapshot, Mapping) or not raw_snapshot.get("parent_entry_id"):
        return None
    try:
        snapshot = ChildContextSnapshot.from_values(
            parent_session_id=SessionId(str(raw_snapshot["parent_session_id"])),
            parent_entry_id=EntryId(str(raw_snapshot["parent_entry_id"])),
            depth=int(raw_snapshot.get("depth") or 0),
            messages=list(raw_snapshot.get("messages") or ()),
            attachment_occurrences=tuple(
                AttachmentOccurrence.from_payload(item)
                for item in raw_snapshot.get("attachment_occurrences", [])
            ),
            evidence_state=(
                dict(raw_snapshot.get("evidence_state") or {})
                if isinstance(raw_snapshot.get("evidence_state"), Mapping)
                else {}
            ),
        )
        if snapshot.parent_session_id != host.parent_session_id:
            raise ValueError("Child envelope parent identity changed")
        raw_tools = row.get("tools")
        request = ChildRequest(
            objective=str(row.get("operation_input") or row["objective"]),
            context=str(row["context"]),  # type: ignore[arg-type]
            model_role=str(row["model_role"]),  # type: ignore[arg-type]
            tools=(tuple(str(item) for item in raw_tools) if isinstance(raw_tools, list) else None),
        )
        return (
            SessionId(str(row["child_session_id"])),
            request,
            str(row["parent_call_id"]),
            snapshot,
        )
    except KeyError, TypeError, ValueError:
        logger.warning("Accepted Child Session envelope is not reconstructible", exc_info=True)
        return None


def _is_running_work(host: SubagentHost, row: Mapping[str, Any]) -> bool:
    """Return whether one roster row is reconstructible running Child work."""
    return str(row.get("status")) == "running" and _dispatch_from_row(host, row) is not None


async def _check_cancelled(host: SubagentHost) -> None:
    if host.check_cancelled is None:
        return
    try:
        await host.check_cancelled()
    except (RunCancellationObserved, RunCancelledError) as exc:
        raise _ParentRunCancelled from exc


async def _request_child_cancel(host: SubagentHost, child_id: str) -> None:
    """Persist cancellation intent without revoking the active Child writer."""
    host._cancel_requested.add(child_id)
    if host.request_cancel is None:
        return
    requested = await host.request_cancel(
        owner_id=host.owner_id,
        run_id=host.run_id,
        child_session_id=child_id,
    )
    if requested is False:
        persisted = await _load_terminal_child(host, child_id)
        if persisted is None:
            from dlightrag.engine.runtime.coordinator import LeaseLostError

            raise LeaseLostError


async def _cancel_child(host: SubagentHost, child_id: str) -> ChildOutcome:
    await _request_child_cancel(host, child_id)
    task = host.tasks.get(child_id)
    if task is not None and not task.done():
        task.cancel()
        try:
            outcome = await task
            if host.load_child is None:
                return outcome
            current = await _status(host, child_id)
            if current.status != "running":
                return current
        except asyncio.CancelledError:
            # Cancellation before coroutine entry leaves the durable request for
            # the reconstructed closure path below.
            pass
    if host.request_cancel is None:
        return await _finish_cancelled_child(host, child_id)
    host.tasks.pop(child_id, None)
    await host.restore_pending()
    task = host.tasks.get(child_id)
    if task is not None and not task.done():
        await task
    outcome = await _status(host, child_id)
    if outcome.status == "running":
        raise RuntimeError("cancelled Child Session did not close its Agent Operation")
    return outcome


async def _finish_cancelled_child(host: SubagentHost, child_id: str) -> ChildOutcome:
    host._cancel_requested.add(child_id)
    cancelled = ChildOutcome(
        status="cancelled",
        summary="Child session cancelled.",
        child_session_id=child_id,
    )
    if host.finish_child is not None:
        committed = await host.finish_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
            status="cancelled",
            summary=cancelled.summary,
            usage=None,
            outcome=cancelled.durable_payload(),
        )
        if committed is False:
            persisted = await _load_terminal_child(host, child_id)
            if persisted is not None:
                host.outcomes[child_id] = persisted
                return persisted
            from dlightrag.engine.runtime.coordinator import LeaseLostError

            raise LeaseLostError
    host.outcomes[child_id] = cancelled
    return cancelled


async def _load_terminal_child(host: SubagentHost, child_id: str) -> ChildOutcome | None:
    if host.load_child is None:
        return None
    row = await host.load_child(
        owner_id=host.owner_id,
        run_id=host.run_id,
        child_session_id=child_id,
    )
    if row is None or str(row.get("status") or "running") == "running":
        return None
    return _terminal_outcome_from_row(row, child_id)


async def _status(host: SubagentHost, child_id: str) -> ChildOutcome:
    if host.load_child is not None:
        row = await host.load_child(
            owner_id=host.owner_id,
            run_id=host.run_id,
            child_session_id=child_id,
        )
        if row is not None:
            if str(row.get("status") or "failed") != "running":
                return _terminal_outcome_from_row(row, child_id)
            return ChildOutcome(
                status="running",
                summary="Child session is running.",
                child_session_id=child_id,
                operation_id=str(row.get("operation_id") or ""),
            )
    else:
        task = host.tasks.get(child_id)
        if task is not None:
            if task.done():
                return task.result()
            return ChildOutcome(
                status="running", summary="Child session is running.", child_session_id=child_id
            )
        if child_id in host.outcomes:
            return host.outcomes[child_id]
    return ChildOutcome(
        status="failed", summary="Unknown child session.", child_session_id=child_id
    )


def _terminal_outcome_from_row(row: Mapping[str, Any], child_id: str) -> ChildOutcome:
    host_state = row.get("host_state")
    payload = host_state.get("terminal_outcome") if isinstance(host_state, Mapping) else None
    if not isinstance(payload, Mapping):
        raise RuntimeError("terminal Child session lost its durable outcome")
    outcome = ChildOutcome.from_durable_payload(payload)
    if outcome.child_session_id != child_id or outcome.status != str(row.get("status")):
        raise RuntimeError("terminal Child session outcome identity changed")
    return outcome


def _adopt_outcome(
    host: SubagentHost,
    outcome: ChildOutcome,
    *,
    parent_call_id: str = "",
) -> ChildOutcome:
    """Idempotently admit one durable outcome into the live parent materializer."""
    if outcome.evidence_state is None or host.merge_evidence is None:
        return outcome
    return replace(
        outcome,
        handles=host.merge_evidence(
            outcome.evidence_state,
            outcome.child_session_id,
            parent_call_id,
        ),
    )


def child_session_id(
    *,
    run_id: str,
    parent_session_id: SessionId,
    parent_intent_id: IntentId,
    position: int = 0,
) -> SessionId:
    """Deterministic child identity owned by a durable parent Effect intent."""
    suffix = f":{position}" if position else ""
    return SessionId.deterministic(
        run_id=run_id,
        name=f"child:{parent_session_id.value}:{parent_intent_id.value}{suffix}",
    )


async def _pending_guidance_for_child(
    host: SubagentHost,
    child_id: str,
) -> tuple[Mapping[str, Any], ...]:
    if host.list_guidance is None or host.parent_session_id is None:
        return ()
    questions = await host.list_guidance(
        owner_id=host.owner_id,
        run_id=host.run_id,
        parent_session_id=host.parent_session_id.value,
    )
    return tuple(
        question
        for question in questions or ()
        if isinstance(question, Mapping) and str(question.get("child_session_id") or "") == child_id
    )


def _guidance_notice(questions: tuple[Mapping[str, Any], ...]) -> str:
    lines: list[str] = []
    for question in questions:
        request_id = str(question.get("request_id") or "")
        if not request_id:
            continue
        lines.append(
            f"Child session {question.get('child_session_id') or ''} asks for guidance "
            f"(request_id={request_id}): {question.get('question') or ''}. "
            "Reply with reply_subagent using exactly this request_id."
        )
    return "\n".join(lines)


def _result_with_guidance(
    host: SubagentHost,
    outcome: ChildOutcome,
    questions: tuple[Mapping[str, Any], ...],
) -> ToolResult:
    result = _single_result(_adopt_outcome(host, outcome))
    notice = _guidance_notice(questions)
    if not notice:
        return result
    return ToolResult.text(f"{result.text_content}\n{notice}", details=result.details)


def _single_result(outcome: ChildOutcome) -> ToolResult:
    return _many_result((outcome,))


def _many_result(outcomes: tuple[ChildOutcome, ...]) -> ToolResult:
    lines: list[str] = []
    children: list[dict[str, Any]] = []
    inclusive_usage: dict[str, int] = {}
    for outcome in outcomes:
        lines.append(
            f"Child {outcome.child_session_id} [{outcome.status}]: "
            f"{outcome.summary.strip() or '(no summary)'}"
        )
        if outcome.handles:
            lines.extend(f"- merged {item}" for item in outcome.handles)
        if outcome.usage:
            for key, value in outcome.usage.items():
                inclusive_usage[key] = inclusive_usage.get(key, 0) + int(value)
        children.append(
            {
                "child_session_id": outcome.child_session_id,
                "status": outcome.status,
                "evidence_handles": list(outcome.handles),
                "usage": dict(outcome.usage or {}),
            }
        )
    return ToolResult.text(
        "\n".join(lines),
        details={"children": children, "inclusive_usage": inclusive_usage},
    )


__all__ = [
    "AskParentInput",
    "ChildContextMode",
    "ChildContextSnapshot",
    "ChildControlInput",
    "ChildModelRole",
    "ChildOutcome",
    "ChildRequest",
    "ChildStatus",
    "SpawnAgentInput",
    "SubagentHost",
    "child_guidance_tools",
    "child_session_id",
    "subagent_tools",
]
