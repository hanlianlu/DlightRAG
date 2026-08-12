# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for how one research request is assembled from its memory."""

from typing import Any

import pytest

from dlightrag.core.agent.context import ContextAssembler
from dlightrag.core.answer.capacity import AnswerCapacity
from dlightrag.core.memory.conversation import PriorTurns
from dlightrag.core.memory.episode import RunEpisode
from dlightrag.core.memory.evidence import EvidenceLedger

_WINDOW = 80_000


def _assembler(history: list[dict[str, Any]]) -> ContextAssembler:
    return ContextAssembler(
        AnswerCapacity(_WINDOW),
        query="What changed?",
        history=PriorTurns(history),
        query_images=None,
        resource_manifest=(),
    )


def _long_history(turns: int, *, chars: int = 4_000) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for index in range(turns):
        messages.append({"role": "user", "content": f"ask {index} " + "x" * chars})
        messages.append({"role": "assistant", "content": f"reply {index} " + "y" * chars})
    return messages


def _ledger(passages: int, *, chars: int = 2_000) -> EvidenceLedger:
    evidence = EvidenceLedger()
    evidence.add_rows(
        [
            {
                "chunk_id": f"c{index}",
                "reference_id": "source-uuid",
                "full_doc_id": "doc-uuid",
                "file_path": "report.pdf",
                "content": f"passage {index} " + "e" * chars,
                "_workspace": "alpha",
                "metadata": {
                    "source_type": "file",
                    "source_uri": "file:///alpha/report.pdf",
                    "source_download_locator": "file:///alpha/report.pdf",
                },
            }
            for index in range(passages)
        ]
    )
    return evidence


async def test_a_long_conversation_sheds_turns_instead_of_overflowing() -> None:
    history = _long_history(40)
    messages = await _assembler(history).control_turn(evidence=_ledger(0), episode=RunEpisode())

    replayed = [message for message in messages if message["role"] in {"user", "assistant"}]
    assert len(replayed) < len(history)


async def test_evidence_keeps_its_share_when_the_conversation_is_long() -> None:
    evidence = _ledger(5)
    messages = await _assembler(_long_history(40)).control_turn(
        evidence=evidence, episode=RunEpisode()
    )

    packed = str(messages[-1])
    # Old chat turns go first: a long conversation must not squeeze evidence out of the window.
    assert "passage 4" in packed
    assert "Knowledge-base evidence" in packed


async def test_research_turn_packing_runs_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import threading

    from dlightrag.core.agent import context as context_module

    loop_thread = threading.get_ident()
    estimator_threads: list[int] = []
    real_estimate = context_module.estimate_messages_tokens

    def estimate(messages: list[dict[str, Any]]) -> int:
        estimator_threads.append(threading.get_ident())
        return real_estimate(messages)

    monkeypatch.setattr(context_module, "estimate_messages_tokens", estimate)
    assembler = _assembler(_long_history(2))

    await assembler.control_turn(evidence=_ledger(3), episode=RunEpisode())
    await assembler.answer_turn(evidence=_ledger(3), episode=RunEpisode())

    assert estimator_threads and loop_thread not in estimator_threads
