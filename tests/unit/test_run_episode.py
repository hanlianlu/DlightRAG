# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the research run's episodic memory."""

from typing import Any

from dlightrag.core.memory.episode import RunEpisode


def _exchange(call_id: str, *, reasoning: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "search_web", "arguments": '{"query":"x"}'},
                    "thought_signature": "signed",
                }
            ],
            "provider_state": {"reasoning_content": reasoning},
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "search_web",
            "content": "Open web added 1 new passages.",
        },
    ]


def test_a_short_run_replays_every_exchange_in_full() -> None:
    episode = RunEpisode()
    episode.record(_exchange("first", reasoning="short"))
    episode.record(_exchange("second", reasoning="short"))

    assistants = [message for message in episode.messages() if message["role"] == "assistant"]

    assert [message["tool_calls"][0]["id"] for message in assistants] == ["first", "second"]
    assert all("provider_state" in message for message in assistants)


def test_an_exchange_past_the_recent_budget_keeps_its_call_without_the_reasoning() -> None:
    episode = RunEpisode()
    episode.record(_exchange("first", reasoning="short"))
    episode.record(_exchange("second", reasoning="n" * 200_000))

    older, newer = (message for message in episode.messages() if message["role"] == "assistant")

    assert "provider_state" not in older
    assert "thought_signature" not in older["tool_calls"][0]
    assert older["tool_calls"][0]["id"] == "first"
    assert newer["provider_state"]
    assert newer["tool_calls"][0]["thought_signature"] == "signed"
