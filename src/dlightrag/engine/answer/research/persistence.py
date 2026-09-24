# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research persistence contracts and the callbacks bound to a parent Run lease.

The PostgreSQL adapter satisfies ResearchRunStore at composition. Tool-definition
Hosts may remain unbound; live Research binds every callback through this contract.
JSON documents remain opaque here; method names, arguments and outcomes do not.
"""

from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, Literal, Protocol


class PersistChild(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        parent_session_id: str,
        parent_call_id: str,
        parent_intent_id: str | None = None,
        objective: str | None = None,
        context_mode: str | None = None,
        model_role: str | None = None,
        tools: Sequence[str] | None = None,
        depth: int = 1,
        context_snapshot: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        host_state: Mapping[str, Any] | None = None,
    ) -> Awaitable[object]: ...


class ClaimChild(Protocol):
    def __call__(
        self, *, owner_id: str, run_id: str, child_session_id: str
    ) -> Awaitable[int | None]: ...


class RenewChild(Protocol):
    def __call__(
        self, *, owner_id: str, run_id: str, child_session_id: str, child_fencing_epoch: int
    ) -> Awaitable[bool]: ...


class LoadChild(Protocol):
    def __call__(
        self, *, owner_id: str, run_id: str, child_session_id: str
    ) -> Awaitable[dict[str, Any] | None]: ...


class ListChildren(Protocol):
    def __call__(self, *, owner_id: str, run_id: str) -> Awaitable[tuple[dict[str, Any], ...]]: ...


class FinishChild(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        status: str,
        summary: str,
        outcome: Mapping[str, Any],
        usage: Mapping[str, int] | None = None,
        child_fencing_epoch: int | None = None,
    ) -> Awaitable[bool]: ...


class CancelChild(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        cancellation_origin: str = "parent",
    ) -> Awaitable[bool]: ...


class ReleaseChildren(Protocol):
    def __call__(self, *, owner_id: str, run_id: str) -> Awaitable[bool]: ...


class SteerChild(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
    ) -> Awaitable[dict[str, Any]]: ...


class ContinueChild(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        reauthorize_user_cancelled: bool = False,
    ) -> Awaitable[dict[str, Any]]: ...


class ReplyChildGuidance(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
    ) -> Awaitable[dict[str, Any]]: ...


class CreateChildGuidance(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        child_session_id: str,
        child_operation_id: str,
        parent_session_id: str,
        question: str,
        expires_after_seconds: int,
        child_fencing_epoch: int,
    ) -> Awaitable[dict[str, Any] | None]: ...


class LoadChildGuidance(Protocol):
    def __call__(
        self, *, owner_id: str, run_id: str, request_id: str
    ) -> Awaitable[dict[str, Any] | None]: ...


class WaitChildGuidance(Protocol):
    def __call__(
        self, *, owner_id: str, run_id: str, request_id: str, timeout_seconds: float
    ) -> Awaitable[dict[str, Any] | None]: ...


class ExpireChildGuidance(Protocol):
    def __call__(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        child_session_id: str,
        child_operation_id: str,
        child_fencing_epoch: int,
    ) -> Awaitable[bool]: ...


class ListChildGuidance(Protocol):
    def __call__(
        self, *, owner_id: str, run_id: str, parent_session_id: str
    ) -> Awaitable[tuple[dict[str, Any], ...]]: ...


class ResearchRunStore(Protocol):
    """Required persistence for Research execution, distinct from generic RunStore."""

    async def upsert_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        parent_session_id: str,
        parent_call_id: str,
        worker_id: str,
        fencing_epoch: int,
        parent_intent_id: str | None = None,
        objective: str | None = None,
        context_mode: str | None = None,
        model_role: str | None = None,
        tools: Sequence[str] | None = None,
        depth: int = 1,
        context_snapshot: Mapping[str, Any] | None = None,
        plan: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        host_state: Mapping[str, Any] | None = None,
    ) -> bool: ...

    async def claim_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
    ) -> int | None: ...

    async def heartbeat_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> bool: ...

    async def load_child_session(
        self, *, owner_id: str, run_id: str, child_session_id: str
    ) -> dict[str, Any] | None: ...

    async def list_child_sessions(
        self, *, owner_id: str, run_id: str
    ) -> tuple[dict[str, Any], ...]: ...

    async def finish_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        status: str,
        summary: str,
        outcome: Mapping[str, Any],
        worker_id: str,
        fencing_epoch: int,
        usage: Mapping[str, int] | None = None,
        child_fencing_epoch: int | None = None,
    ) -> bool: ...

    async def request_child_cancellation(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        worker_id: str,
        fencing_epoch: int,
        cancellation_origin: str = "parent",
    ) -> bool: ...

    async def release_child_sessions(
        self, *, owner_id: str, run_id: str, worker_id: str, fencing_epoch: int
    ) -> bool: ...

    async def enqueue_child_control(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> dict[str, Any] | Literal[False]: ...

    async def continue_child_session(
        self,
        *,
        owner_id: str,
        run_id: str,
        child_session_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        reauthorize_user_cancelled: bool = False,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> dict[str, Any] | Literal[False]: ...

    async def reply_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        content: str,
        submission_key: str,
        origin: str = "user",
        parent_session_id: str | None = None,
        worker_id: str | None = None,
        fencing_epoch: int | None = None,
    ) -> dict[str, Any] | Literal[False]: ...

    async def create_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        child_session_id: str,
        child_operation_id: str,
        parent_session_id: str,
        question: str,
        expires_after_seconds: int,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> dict[str, Any] | None: ...

    async def load_child_guidance(
        self, *, owner_id: str, run_id: str, request_id: str
    ) -> dict[str, Any] | None: ...

    async def wait_for_child_guidance(
        self, *, owner_id: str, run_id: str, request_id: str, timeout_seconds: float
    ) -> dict[str, Any] | None: ...

    async def expire_child_guidance(
        self,
        *,
        owner_id: str,
        run_id: str,
        request_id: str,
        child_session_id: str,
        child_operation_id: str,
        worker_id: str,
        fencing_epoch: int,
        child_fencing_epoch: int,
    ) -> bool: ...

    async def list_pending_child_guidance(
        self, *, owner_id: str, run_id: str, parent_session_id: str
    ) -> tuple[dict[str, Any], ...]: ...

    async def load_pending_agent_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        worker_id: str,
        fencing_epoch: int,
        target_session_id: str | None = None,
        target_operation_id: str | None = None,
        child_fencing_epoch: int | None = None,
    ) -> tuple[dict[str, Any], ...] | None: ...

    async def acknowledge_agent_controls(
        self,
        *,
        owner_id: str,
        run_id: str,
        control_sequences: Sequence[int],
        worker_id: str,
        fencing_epoch: int,
        target_session_id: str | None = None,
        target_operation_id: str | None = None,
        child_fencing_epoch: int | None = None,
    ) -> bool: ...


class ControlReader(Protocol):
    def __call__(
        self,
        *,
        target_session_id: str | None = None,
        target_operation_id: str | None = None,
        child_fencing_epoch: int | None = None,
    ) -> Awaitable[tuple[Mapping[str, Any], ...]]: ...


class ControlAcknowledger(Protocol):
    def __call__(
        self,
        sequences: tuple[int, ...],
        *,
        target_session_id: str | None = None,
        target_operation_id: str | None = None,
        child_fencing_epoch: int | None = None,
    ) -> Awaitable[bool]: ...
