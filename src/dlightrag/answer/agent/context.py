# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Assemble one research request from the run's memory under one capacity."""

import asyncio
from typing import Any

from dlightrag_agent.session.fold import PriorTurns, SessionEpisode
from dlightrag_ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag_ai.tokens import estimate_messages_tokens
from dlightrag_rag.sourcing.source_contract import safe_source_filename

from dlightrag.answer.citations.indexer import CitationIndexer
from dlightrag.answer.errors import AnswerInputOverflowError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.memory import standing_memory_message
from dlightrag.answer.prompts import (
    CONTROL_TURN_INSTRUCTION,
    FINAL_TURN_INSTRUCTION,
    agent_control_prompt,
    answer_core,
)
from dlightrag.answer.resources.models import ResourceManifestEntry


class ContextAssembler:
    """Build each turn of one request from the stores, never by extending the last turn.

    A control turn replays the episode and packs the ledger; the answer turn
    swaps in the answer prompt and carries one exchange instead of the episode,
    so it packs more evidence into the same window. Both shed the oldest
    conversation turns first: those are the only part a request can drop without
    losing evidence or the question itself.
    """

    def __init__(
        self,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        query: str,
        history: PriorTurns,
        query_images: list[dict[str, Any]] | None,
        resource_manifest: tuple[ResourceManifestEntry, ...],
        memory_text: str = "",
    ) -> None:
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._input_limit = context_policy.hard_input_limit(model_profile)
        self._control_target = context_policy.compaction_trigger(model_profile)
        self._history = history
        self._question = _question_message(query, query_images, resource_manifest)
        self._memory_text = memory_text

    async def control_turn(
        self,
        *,
        evidence: EvidenceLedger,
        episode: SessionEpisode,
        tool_schema_tokens: int,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(
            self._build_control_turn,
            evidence,
            episode,
            tool_schema_tokens,
        )

    async def answer_turn(
        self,
        *,
        evidence: EvidenceLedger,
        episode: SessionEpisode,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        return await asyncio.to_thread(self._build_answer_turn, evidence, episode)

    def measure_control_input(
        self,
        *,
        evidence: EvidenceLedger,
        episode: SessionEpisode,
    ) -> int:
        """Measure the exact control-turn messages without enforcing the limit."""
        return estimate_messages_tokens(self._compose_control_turn(evidence, episode))

    def observation_residual(
        self,
        transcript_with_assistant: list[dict[str, Any]],
        *,
        tool_schema_tokens: int,
    ) -> int:
        """Return capacity left for the next model-visible tool-result batch."""
        fixed = list(transcript_with_assistant)
        if len(fixed) >= 2 and _is_control_evidence_message(fixed[-2]):
            fixed.pop(-2)
        assistant = fixed[-1] if fixed else {}
        tool_messages = [_empty_tool_message(call) for call in assistant.get("tool_calls") or ()]
        instruction = {
            "role": "user",
            "content": [{"type": "text", "text": CONTROL_TURN_INSTRUCTION}],
        }
        used = estimate_messages_tokens([*fixed, *tool_messages, instruction]) + tool_schema_tokens
        return max(0, self._control_target - used)

    def output_allowance(
        self,
        messages: list[dict[str, Any]],
        *,
        additional_input_tokens: int = 0,
    ) -> int | None:
        """Preflight one exact model request and return its provider output cap."""
        if additional_input_tokens < 0:
            raise ValueError("additional_input_tokens cannot be negative")
        input_tokens = estimate_messages_tokens(messages) + additional_input_tokens
        self._check_input_tokens(input_tokens)
        return self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=input_tokens,
        )

    def control_output_allowance(
        self,
        messages: list[dict[str, Any]],
        *,
        tool_schema_tokens: int,
    ) -> int:
        """Bound control output while preserving dynamic space for tool results."""
        provider_allowance = self.output_allowance(
            messages,
            additional_input_tokens=tool_schema_tokens,
        )
        accumulation_gap = self._input_limit - self._control_target
        control_allowance = accumulation_gap - tool_schema_tokens
        if control_allowance <= 0:
            raise AnswerInputOverflowError(
                "Research tool schemas leave no model residual for a control completion"
            )
        return (
            control_allowance
            if provider_allowance is None
            else min(provider_allowance, control_allowance)
        )

    def _build_control_turn(
        self,
        evidence: EvidenceLedger,
        episode: SessionEpisode,
        tool_schema_tokens: int,
    ) -> list[dict[str, Any]]:
        # The orchestrator owns the proactive H trigger and compacts before
        # composing; this assembler enforces only the hard L limit later via
        # ``output_allowance`` / ``control_output_allowance``.
        return self._compose_control_turn(
            evidence,
            episode,
            tool_schema_tokens=tool_schema_tokens,
        )

    def _compose_control_turn(
        self,
        evidence: EvidenceLedger,
        episode: SessionEpisode,
        *,
        tool_schema_tokens: int = 0,
    ) -> list[dict[str, Any]]:
        system = {"role": "system", "content": agent_control_prompt()}
        head = self._head(system, episode.messages())
        messages = list(head)
        if evidence.row_count:
            blocks, _ = self._pack(
                evidence,
                head=head,
                final=False,
                tool_schema_tokens=tool_schema_tokens,
            )
            messages.append({"role": "user", "content": blocks})
        memory_message = standing_memory_message(self._memory_text)
        if memory_message is not None:
            messages.append(memory_message)
        return messages

    def _build_answer_turn(
        self,
        evidence: EvidenceLedger,
        episode: SessionEpisode,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        system = {"role": "system", "content": answer_core()}
        head = self._head(system, episode.last_exchange)
        blocks, indexer = self._pack(evidence, head=head, final=True)
        messages = [*head, {"role": "user", "content": blocks}]
        memory_message = standing_memory_message(self._memory_text)
        if memory_message is not None:
            messages.append(memory_message)
        self._check(messages)
        return messages, indexer

    def _head(
        self,
        system: dict[str, Any],
        carried: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [system, *self._history.messages, self._question, *carried]

    def _pack(
        self,
        evidence: EvidenceLedger,
        *,
        head: list[dict[str, Any]],
        final: bool,
        tool_schema_tokens: int = 0,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        instruction = FINAL_TURN_INSTRUCTION if final else CONTROL_TURN_INSTRUCTION
        instruction_block = {"type": "text", "text": instruction}
        fixed_input_tokens = estimate_messages_tokens(
            [*head, {"role": "user", "content": [instruction_block]}]
        )
        target = self._input_limit if final else self._control_target
        residual = max(0, target - tool_schema_tokens - fixed_input_tokens)
        while True:
            blocks, indexer = evidence.transform(residual_tokens=residual)
            content = [*blocks, instruction_block]
            rendered_tokens = (
                estimate_messages_tokens([*head, {"role": "user", "content": content}])
                + tool_schema_tokens
            )
            if rendered_tokens <= target:
                return content, indexer
            if residual == 0:
                raise AnswerInputOverflowError(
                    "Fixed research evidence handles exceed the resolved model input target"
                )
            residual = max(0, residual - (rendered_tokens - target))

    def _check(self, messages: list[dict[str, Any]]) -> None:
        self._check_input_tokens(estimate_messages_tokens(messages))

    def _check_input_tokens(self, input_tokens: int) -> None:
        if input_tokens > self._input_limit:
            raise AnswerInputOverflowError(
                "Research input exceeds the resolved model input limit: "
                f"{input_tokens} > {self._input_limit} estimated input tokens"
            )


def _question_message(
    query: str,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
) -> dict[str, Any]:
    manifest = _resource_manifest_context(resource_manifest)
    if not query_images and not manifest:
        return {"role": "user", "content": query}
    content: list[dict[str, Any]] = [{"type": "text", "text": query}]
    if manifest:
        content.append({"type": "text", "text": manifest})
    content.extend(query_images or [])
    return {"role": "user", "content": content}


def _is_control_evidence_message(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return bool(
        message.get("role") == "user"
        and isinstance(content, list)
        and content
        and isinstance(content[-1], dict)
        and content[-1].get("text") == CONTROL_TURN_INSTRUCTION
    )


def _empty_tool_message(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function") or {}
    return {
        "role": "tool",
        "tool_call_id": str(call.get("id") or ""),
        "name": str(function.get("name") or ""),
        "content": "",
    }


def _resource_manifest_context(manifest: tuple[ResourceManifestEntry, ...]) -> str:
    if not manifest:
        return ""
    lines = ["## Registered request-local resources"]
    for entry in manifest:
        filename = safe_source_filename(entry.filename or "resource")
        kind = "image" if (entry.declared_mime or "").lower().startswith("image/") else "resource"
        lines.append(f"- [resource: {entry.resource_id}] {filename} ({kind})")
    lines.append("Use only these opaque resource ids with read or inspect.")
    return "\n".join(lines)


__all__ = ["ContextAssembler"]
