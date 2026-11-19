# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The `call` a step records: the model, parameters, prompt digests and timings it ran on
(ADR 0041).

A wrapper states the model, the parameters and the timings of the attempt that produced a result.
Whoever commits the result knows the rest: the role the call was made for, the exact request it
was sent, and the Plan it ran under. This module merges the two into the one object an Assistant
or Compaction Entry stores beside its content, in the same transaction.

The prompt is never stored, only three digests of it, so a later reader can tell that two steps
ran on the same instructions and tools, and prove that a rebuilt request is byte for byte the one
that was sent, without the Session holding a second copy of the text.

The record carries no token count, and nothing here takes part in usage, cost, a Plan digest or
the checks that decide whether an active Run may resume.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.registers import RequestSnapshot
from dlightrag.engine.ai.messages import message_text
from dlightrag.engine.ai.provenance import ModelCall


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PromptDigests:
    """SHA-256 digests, in lowercase hex, of what a call was sent.

    ``system`` covers the text of the system messages alone, joined by a blank line as providers
    join them; ``tools`` the canonical tool definitions alone; ``request`` the messages and the
    tools together, as the canonical JSON object ``{"messages": [...], "tools": [...]}``.
    """

    system: str
    tools: str
    request: str

    def as_json(self) -> dict[str, str]:
        return {"system": self.system, "tools": self.tools, "request": self.request}


def _digests(
    messages_json: str, tools_json: str, messages: Sequence[Mapping[str, Any]]
) -> PromptDigests:
    system = "\n\n".join(
        message_text(message.get("content"))
        for message in messages
        if message.get("role") == "system"
    )
    return PromptDigests(
        system=_sha256(system),
        tools=_sha256(tools_json),
        request=_sha256(f'{{"messages":{messages_json},"tools":{tools_json}}}'),
    )


def prompt_digests(
    messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]] = ()
) -> PromptDigests:
    """Digest a request that has no Request Snapshot, such as a Fast answer or a compaction."""
    return _digests(canonical_json(list(messages)), canonical_json(list(tools)), messages)


def request_prompt_digests(request: RequestSnapshot) -> PromptDigests:
    """Digest the request a Snapshot holds, from the canonical strings it already has."""
    return _digests(request.messages_json, request.tools_json, request.messages)


def call_payload(
    call: ModelCall | None,
    *,
    role: str,
    prompt: PromptDigests,
    plan_digest: str | None = None,
) -> dict[str, Any] | None:
    """The `call` object of one step, or None when no wrapper stated how it was made.

    ``plan_digest`` names the immutable Plan a Research step ran under; a Fast answer has none.
    """
    if call is None:
        return None
    payload = call.as_json()
    payload["model"] = {"role": role, **payload["model"]}
    payload["prompt"] = prompt.as_json()
    if plan_digest is not None:
        payload["plan_digest"] = plan_digest
    return payload


@dataclass(slots=True)
class PendingCall:
    """A call in flight: its prompt, known when it was sent, and its model and parameters, which
    the wrapper states in ``holder`` when the attempt that produced the result ends."""

    prompt: PromptDigests
    holder: dict[str, Any] = field(default_factory=dict)

    def payload(self, *, role: str, plan_digest: str | None = None) -> dict[str, Any] | None:
        """The step's `call` once the attempt has ended, or None if the wrapper stated none."""
        return call_payload(
            self.holder.get("call"), role=role, prompt=self.prompt, plan_digest=plan_digest
        )


__all__ = [
    "PendingCall",
    "PromptDigests",
    "call_payload",
    "prompt_digests",
    "request_prompt_digests",
]
