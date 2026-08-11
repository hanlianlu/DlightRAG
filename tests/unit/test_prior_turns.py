# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the session turns a request replays."""

from typing import Any

from dlightrag.core.memory.conversation import PriorTurns


def _measure(messages: list[dict[str, Any]]) -> int:
    return sum(len(str(message.get("content", ""))) for message in messages)


class TestRecent:
    def test_empty_history(self) -> None:
        assert PriorTurns().recent(max_messages=10, max_tokens=1000).messages == []

    def test_within_limits(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert PriorTurns(history).recent(max_messages=10, max_tokens=1000).messages == history

    def test_message_limit(self) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = PriorTurns(history).recent(max_messages=3, max_tokens=999999).messages
        assert len(result) == 3
        assert result[0]["content"] == "msg 7"

    def test_token_limit(self) -> None:
        # Each "x" * 400 ~ 100 tokens (400 ASCII / 4); a 250-token budget keeps two.
        history = [{"role": "user", "content": "x" * 400} for _ in range(5)]
        assert len(PriorTurns(history).recent(max_messages=50, max_tokens=250).messages) == 2

    def test_token_limit_cjk(self) -> None:
        # Each "你" * 150 ~ 100 tokens (150 CJK / 1.5).
        history = [{"role": "user", "content": "你" * 150} for _ in range(5)]
        assert len(PriorTurns(history).recent(max_messages=50, max_tokens=250).messages) == 2

    def test_zero_budget_keeps_nothing(self) -> None:
        history = [{"role": "user", "content": "message"}]
        assert PriorTurns(history).recent(max_messages=0, max_tokens=10_000).messages == []
        assert PriorTurns(history).recent(max_messages=10, max_tokens=0).messages == []


class TestFit:
    def test_everything_stays_when_it_already_fits(self) -> None:
        history = [
            {"role": "user", "content": "ask"},
            {"role": "assistant", "content": "reply"},
        ]
        assert PriorTurns(history).fit(1000, _measure) == history

    def test_a_user_turn_and_its_reply_leave_together(self) -> None:
        history = [
            {"role": "user", "content": "old ask"},
            {"role": "assistant", "content": "old reply"},
            {"role": "user", "content": "new"},
        ]
        assert PriorTurns(history).fit(10, _measure) == [{"role": "user", "content": "new"}]

    def test_a_request_that_never_fits_sheds_every_turn(self) -> None:
        history = [{"role": "user", "content": "x" * 100} for _ in range(4)]
        assert PriorTurns(history).fit(1, _measure) == []
