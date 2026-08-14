# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Assemble one research request from the run's memory under one capacity."""

import asyncio
from typing import Any

from dlightrag_ai.tokens import estimate_messages_tokens

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.core.answer.capacity import FINAL_GENERATION_CAPACITY_RESERVE, AnswerCapacity
from dlightrag.core.answer.errors import AnswerInputOverflowError
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.resources.models import ResourceManifestEntry
from dlightrag.prompts import (
    CONTROL_TURN_INSTRUCTION,
    FINAL_TURN_INSTRUCTION,
    agent_control_prompt,
    answer_core,
)
from dlightrag.sourcing.source_contract import safe_source_filename


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
        capacity: AnswerCapacity,
        *,
        query: str,
        history: PriorTurns,
        query_images: list[dict[str, Any]] | None,
        resource_manifest: tuple[ResourceManifestEntry, ...],
    ) -> None:
        self._capacity = capacity
        self._input_budget = max(
            1, capacity.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
        )
        self._history = history
        self._question = _question_message(query, query_images, resource_manifest)

    async def control_turn(
        self,
        *,
        evidence: EvidenceLedger,
        episode: RunEpisode,
    ) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._build_control_turn, evidence, episode)

    async def answer_turn(
        self,
        *,
        evidence: EvidenceLedger,
        episode: RunEpisode,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        return await asyncio.to_thread(self._build_answer_turn, evidence, episode)

    def _build_control_turn(
        self,
        evidence: EvidenceLedger,
        episode: RunEpisode,
    ) -> list[dict[str, Any]]:
        system = {"role": "system", "content": agent_control_prompt()}
        head = self._head(system, episode.messages(), evidence, final=False)
        messages = list(head)
        if evidence.row_count:
            blocks, _ = self._pack(evidence, head=head, final=False)
            messages.append({"role": "user", "content": blocks})
        self._check(messages)
        return messages

    def _build_answer_turn(
        self,
        evidence: EvidenceLedger,
        episode: RunEpisode,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        system = {"role": "system", "content": answer_core()}
        head = self._head(system, episode.last_exchange, evidence, final=True)
        blocks, indexer = self._pack(evidence, head=head, final=True)
        messages = [*head, {"role": "user", "content": blocks}]
        self._check(messages)
        return messages, indexer

    def _head(
        self,
        system: dict[str, Any],
        carried: list[dict[str, Any]],
        evidence: EvidenceLedger,
        *,
        final: bool,
    ) -> list[dict[str, Any]]:
        # Evidence outranks old chat, so the block it would render with the whole window to
        # itself is reserved before history is fitted into what remains.
        wanted, _ = self._pack(evidence, head=[], final=final)
        reserved = estimate_messages_tokens([{"role": "user", "content": wanted}])
        kept = self._history.fit(
            max(1, self._input_budget - reserved),
            lambda history: estimate_messages_tokens([system, *history, self._question, *carried]),
        )
        return [system, *kept, self._question, *carried]

    def _pack(
        self,
        evidence: EvidenceLedger,
        *,
        head: list[dict[str, Any]],
        final: bool,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        blocks, indexer = evidence.transform(
            self._capacity, fixed_input_tokens=estimate_messages_tokens(head)
        )
        instruction = FINAL_TURN_INSTRUCTION if final else CONTROL_TURN_INSTRUCTION
        return [*blocks, {"type": "text", "text": instruction}], indexer

    def _check(self, messages: list[dict[str, Any]]) -> None:
        input_tokens = estimate_messages_tokens(messages)
        if input_tokens > self._input_budget:
            raise AnswerInputOverflowError(
                "Research input does not fit beside the generation reserve: "
                f"{input_tokens} > {self._input_budget} estimated input tokens"
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


def _resource_manifest_context(manifest: tuple[ResourceManifestEntry, ...]) -> str:
    if not manifest:
        return ""
    lines = ["## Registered request-local resources"]
    for entry in manifest:
        filename = safe_source_filename(entry.filename or "resource")
        kind = "image" if (entry.declared_mime or "").lower().startswith("image/") else "resource"
        lines.append(f"- [resource: {entry.resource_id}] {filename} ({kind})")
    lines.append("Use only these opaque resource ids with read_resource or inspect_resource.")
    return "\n".join(lines)


__all__ = ["ContextAssembler"]
