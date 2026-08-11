# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Assemble one research request from the run's memory under one capacity."""

from typing import Any

from dlightrag.citations.indexer import CitationIndexer
from dlightrag.core.agent.episode import RunEpisode
from dlightrag.core.agent.evidence import EvidenceLedger
from dlightrag.core.answer.capacity import FINAL_GENERATION_CAPACITY_RESERVE, AnswerCapacity
from dlightrag.core.answer.errors import AnswerInputOverflowError
from dlightrag.core.resources.models import ResourceManifestEntry
from dlightrag.prompts import (
    CONTROL_TURN_INSTRUCTION,
    FINAL_TURN_INSTRUCTION,
    agent_control_prompt,
    answer_core,
)
from dlightrag.sourcing.source_contract import safe_source_filename
from dlightrag.utils.tokens import estimate_messages_tokens


class ContextAssembler:
    """Build each turn's messages from the stores, never by appending to the last turn.

    A control turn replays the episode and packs the ledger; the answer turn
    replaces the control prompt, carries one exchange instead of the episode, and
    therefore packs more evidence into the same window.
    """

    def __init__(self, capacity: AnswerCapacity) -> None:
        self._capacity = capacity
        self._input_budget = max(
            1, capacity.context_window_tokens - FINAL_GENERATION_CAPACITY_RESERVE
        )

    def opening_messages(
        self,
        query: str,
        *,
        conversation_history: list[dict[str, Any]] | None,
        query_images: list[dict[str, Any]] | None,
        resource_manifest: tuple[ResourceManifestEntry, ...],
    ) -> list[dict[str, Any]]:
        """The head every turn of this run repeats verbatim."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": agent_control_prompt()},
            *(conversation_history or []),
        ]
        manifest = _resource_manifest_context(resource_manifest)
        if not query_images and not manifest:
            messages.append({"role": "user", "content": query})
            return messages
        content: list[dict[str, Any]] = [{"type": "text", "text": query}]
        if manifest:
            content.append({"type": "text", "text": manifest})
        content.extend(query_images or [])
        messages.append({"role": "user", "content": content})
        return messages

    def control_turn(
        self,
        *,
        opening: list[dict[str, Any]],
        evidence: EvidenceLedger,
        episode: RunEpisode,
    ) -> list[dict[str, Any]]:
        replayed = episode.messages()
        messages = [*opening, *replayed]
        if evidence.row_count:
            blocks, _ = self._pack(evidence, carried=[*opening, *replayed], final=False)
            messages.append({"role": "user", "content": blocks})
        self.check(messages)
        return messages

    def answer_turn(
        self,
        *,
        opening: list[dict[str, Any]],
        evidence: EvidenceLedger,
        episode: RunEpisode,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        carried = episode.last_exchange
        blocks, indexer = self._pack(evidence, carried=[*opening, *carried], final=True)
        messages = [
            {"role": "system", "content": answer_core()},
            *opening[1:],
            *carried,
            {"role": "user", "content": blocks},
        ]
        self.check(messages)
        return messages, indexer

    def check(self, messages: list[dict[str, Any]]) -> None:
        input_tokens = estimate_messages_tokens(messages)
        if input_tokens > self._input_budget:
            raise AnswerInputOverflowError(
                "Research input does not fit beside the generation reserve: "
                f"{input_tokens} > {self._input_budget} estimated input tokens"
            )

    def _pack(
        self,
        evidence: EvidenceLedger,
        *,
        carried: list[dict[str, Any]],
        final: bool,
    ) -> tuple[list[dict[str, Any]], CitationIndexer]:
        fixed = estimate_messages_tokens(carried)
        blocks, indexer = evidence.transform(self._capacity, fixed_input_tokens=fixed)
        instruction = FINAL_TURN_INSTRUCTION if final else CONTROL_TURN_INSTRUCTION
        return [*blocks, {"type": "text", "text": instruction}], indexer


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
