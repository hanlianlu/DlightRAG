# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The async REST client for durable Answer runs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any

import httpx

from dlightrag.runtime.errors import AnswerRunCancelledError, AnswerRunFailedError
from dlightrag.sdk.attachments import AnswerAttachmentUpload

logger = logging.getLogger(__name__)

STATUS_POLL_SECONDS = 1.0
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_SECONDS = 0.5
EVENT_READ_IDLE_SECONDS = 30.0
_EVENT_STREAM_TIMEOUT = httpx.Timeout(
    connect=10.0, read=EVENT_READ_IDLE_SECONDS, write=10.0, pool=10.0
)

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class AnswerRunDescriptor:
    """The accepted run one request created, with its owner-scoped URLs."""

    run_id: str
    status: str
    status_url: str
    events_url: str
    cancel_url: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnswerRunDescriptor:
        run_id = str(payload["run_id"])
        return cls(
            run_id=run_id,
            status=str(payload["status"]),
            status_url=str(payload.get("status_url") or f"/answer/{run_id}"),
            events_url=str(payload.get("events_url") or f"/answer/{run_id}/events"),
            cancel_url=str(payload.get("cancel_url") or f"/answer/{run_id}"),
        )


@dataclass(frozen=True, slots=True)
class AnswerStreamEvent:
    """One durable event replayed from the run's gap-free sequence."""

    sequence: int
    event_type: str
    payload: Mapping[str, Any]


def parse_sse_frames(chunk: str, *, buffer: str = "") -> tuple[list[AnswerStreamEvent], str]:
    """Decode complete SSE frames and return their incomplete tail."""
    text = buffer + chunk
    frames = text.split("\n\n")
    tail = frames.pop()
    events: list[AnswerStreamEvent] = []
    for frame in frames:
        sequence: int | None = None
        event_type = "message"
        data_lines: list[str] = []
        for line in frame.splitlines():
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "id" and value.isdigit():
                sequence = int(value)
            elif field == "event":
                event_type = value
            elif field == "data":
                data_lines.append(value)
        if sequence is None or not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            logger.warning("Discarding an answer event with undecodable data")
            continue
        events.append(
            AnswerStreamEvent(
                sequence=sequence,
                event_type=event_type,
                payload=payload if isinstance(payload, dict) else {"value": payload},
            )
        )
    return events, tail


class AnswerRunClient:
    """Create, follow, read, and cancel durable Answer runs over REST."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        attachments: Sequence[AnswerAttachmentUpload] = (),
        idempotency_key: str | None = None,
    ) -> AnswerRunDescriptor:
        """Submit one Answer request and return its 202 descriptor."""
        headers = dict(self._headers)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if attachments:
            headers.pop("Content-Type", None)
            response = await self._client.post(
                self._url("/answer"),
                data={"request": json.dumps(payload)},
                files=[
                    ("attachments", (item.filename, item.content, item.content_type))
                    for item in attachments
                ],
                headers=headers,
            )
        else:
            response = await self._client.post(
                self._url("/answer"), json=dict(payload), headers=headers
            )
        response.raise_for_status()
        return AnswerRunDescriptor.from_payload(response.json())

    async def status(self, run_id: str) -> dict[str, Any]:
        """Read one run's authoritative status and terminal result."""
        response = await self._client.get(self._url(f"/answer/{run_id}"), headers=self._headers)
        response.raise_for_status()
        return dict(response.json())

    async def cancel(self, run_id: str) -> dict[str, Any]:
        """Request cancellation; repeating it on a terminal run is a no-op."""
        response = await self._client.request(
            "DELETE", self._url(f"/answer/{run_id}"), headers=self._headers
        )
        response.raise_for_status()
        return dict(response.json())

    async def events(self, run_id: str, *, after: int = 0) -> AsyncGenerator[AnswerStreamEvent]:
        """Yield durable events after ``after``, reconnecting by sequence."""
        cursor = max(0, after)
        attempts = 0
        while True:
            try:
                async with aclosing(self._stream_once(run_id, cursor)) as stream:
                    async for event in stream:
                        cursor = event.sequence
                        attempts = 0
                        yield event
                        if event.event_type in {"done", "error"}:
                            return
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError:
                attempts += 1
                if attempts >= MAX_RECONNECT_ATTEMPTS:
                    raise
                logger.info("Answer event stream dropped; resuming after sequence %d", cursor)
                await asyncio.sleep(RECONNECT_BACKOFF_SECONDS * attempts)
                continue
            if (await self.status(run_id))["status"] in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(STATUS_POLL_SECONDS)

    async def answer(
        self,
        payload: Mapping[str, Any],
        *,
        attachments: Sequence[AnswerAttachmentUpload] = (),
        idempotency_key: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Create one run and wait for its canonical result."""
        descriptor = await self.create(
            payload, attachments=attachments, idempotency_key=idempotency_key
        )
        try:
            async with aclosing(self.events(descriptor.run_id)) as events:
                async for event in events:
                    if event.event_type == "token" and on_token is not None:
                        on_token(str(event.payload.get("text") or ""))
                    elif event.event_type == "done":
                        return self._terminal_result(
                            descriptor.run_id,
                            status=str(event.payload.get("status") or ""),
                            result=event.payload.get("result"),
                        )
                    elif event.event_type == "error":
                        raise AnswerRunFailedError(
                            str(event.payload.get("kind") or "answer_stream_failed"),
                            str(event.payload.get("message") or "Answer run failed."),
                        )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 410:
                raise
        return await self._await_terminal_status(descriptor.run_id)

    async def _stream_once(self, run_id: str, cursor: int) -> AsyncGenerator[AnswerStreamEvent]:
        headers = dict(self._headers)
        headers.pop("Content-Type", None)
        if cursor:
            headers["Last-Event-ID"] = str(cursor)
        async with self._client.stream(
            "GET",
            self._url(f"/answer/{run_id}/events"),
            headers=headers,
            timeout=_EVENT_STREAM_TIMEOUT,
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                events, buffer = parse_sse_frames(chunk, buffer=buffer)
                for event in events:
                    yield event

    async def _await_terminal_status(self, run_id: str) -> dict[str, Any]:
        while True:
            status = await self.status(run_id)
            if status["status"] in _TERMINAL_STATUSES:
                return self._terminal_result(
                    run_id,
                    status=str(status["status"]),
                    result=status.get("result"),
                    error_kind=status.get("error_kind"),
                    error_message=status.get("error_message"),
                )
            await asyncio.sleep(STATUS_POLL_SECONDS)

    @staticmethod
    def _terminal_result(
        run_id: str,
        *,
        status: str,
        result: Any,
        error_kind: Any = None,
        error_message: Any = None,
    ) -> dict[str, Any]:
        if status == "succeeded" and isinstance(result, dict):
            return dict(result)
        if status == "cancelled":
            raise AnswerRunCancelledError(run_id)
        raise AnswerRunFailedError(
            str(error_kind or "answer_stream_failed"),
            str(error_message or "Answer run failed."),
        )


__all__ = [
    "EVENT_READ_IDLE_SECONDS",
    "MAX_RECONNECT_ATTEMPTS",
    "STATUS_POLL_SECONDS",
    "AnswerRunClient",
    "AnswerRunDescriptor",
    "AnswerStreamEvent",
    "parse_sse_frames",
]
