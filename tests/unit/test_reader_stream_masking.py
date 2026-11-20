# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe-prefix release: a reader's token frames never carry a protected passage whole.

The stream is the one reader surface whose text arrives in pieces, so the property these
tests pin is not "the finished answer is masked" but "no piece, and no concatenation of
pieces, ever carries a fingerprint window of a protected document". A run that straddles a
token boundary is the hard case: the renderer cannot know it has ended until text beyond it
arrives, so it withholds the passage until then.
"""

import json
import random
from datetime import UTC, datetime
from typing import Any

import pytest

from dlightrag.adapters.http.browser.answer_events import (
    STREAM_HOLD_VISIBLE_CHARS,
    ReaderFrameRenderer,
    browser_frame,
)
from dlightrag.application.runs import RunEvent
from dlightrag.engine.agent.skill_protection import (
    FINGERPRINT_CHARS,
    MASK_PLACEHOLDER,
    ProtectedTextIndex,
)
from tests.unit.web.answer_run_fixtures import stored_result

#: One protected document, long enough to be a passage a reader surface replaces.
PROTECTED = (
    "The run ledger is the first thing a calculation states: every input carries the period "
    "it was observed in, the source that stated it, and the mechanism by which it reaches "
    "the conclusion. An input nobody can date is an input nobody can defend."
)
#: The same document as a copier might wrap it, so raw characters and normalized
#: characters are not the same length -- the holdback must be measured in the latter.
PROTECTED_WRAPPED = (
    "The run ledger is the first thing a calculation states:\nevery input carries the period\n"
    "it was observed in, the source that stated it, and the mechanism by which it reaches\n"
    "the conclusion. An input nobody can date is an input nobody can defend."
)
PREFIX = "Here is the passage I read: "
SUFFIX = " That is the whole method."

#: Fixed seed, so a failure names one reproducible split rather than a flaky one.
SEED = 20261020


def _index(document: str = PROTECTED) -> ProtectedTextIndex:
    return ProtectedTextIndex.build([("skill", "references/method.md", document.encode("utf-8"))])


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        event_type=event_type,
        payload=payload,
        created_at=datetime.now(UTC),
    )


def _frame_data(frame: str) -> Any:
    return json.loads(frame.split("data: ", 1)[1].strip())


def _visible_count(text: str) -> int:
    return sum(1 for character in text if character.isprintable() and not character.isspace())


def _chunks(text: str, rng: random.Random, *, largest: int) -> list[str]:
    """Cut ``text`` at random boundaries, never at the same one twice by construction."""
    chunks: list[str] = []
    cursor = 0
    while cursor < len(text):
        size = rng.randint(1, largest)
        chunks.append(text[cursor : cursor + size])
        cursor += size
    return chunks


def _fingerprint_windows(document: str) -> list[str]:
    """Every window of the protected text a streamed frame must never contain."""
    return [
        document[offset : offset + FINGERPRINT_CHARS]
        for offset in range(len(document) - FINGERPRINT_CHARS + 1)
    ]


async def _stream(
    renderer: ReaderFrameRenderer, chunks: list[str], *, first_sequence: int = 1
) -> list[str]:
    """Feed one chunk per token event and return the text each frame released."""
    frames = [
        await renderer(_event(first_sequence + index, "token", {"text": chunk}))
        for index, chunk in enumerate(chunks)
    ]
    return [_frame_data(frame) for frame in frames]


async def _stream_reasoning(
    renderer: ReaderFrameRenderer, chunks: list[str], *, first_sequence: int = 1
) -> list[str]:
    """Feed one chunk per reasoning event and return the text each frame released."""
    frames = [
        await renderer(_event(first_sequence + index, "reasoning", {"text": chunk}))
        for index, chunk in enumerate(chunks)
    ]
    return [_frame_data(frame) for frame in frames]


async def test_no_token_frame_releases_a_fingerprint_window_of_the_passage() -> None:
    """Fifty randomly split streams, and no frame of any of them shows a protected window."""
    index = _index()
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    masked = f"{PREFIX}{MASK_PLACEHOLDER}{SUFFIX}"
    rng = random.Random(SEED)
    for trial in range(50):
        renderer = ReaderFrameRenderer(index, index.mask)
        released = await _stream(renderer, _chunks(answer, rng, largest=13))
        joined = "".join(released)
        for window in _fingerprint_windows(PROTECTED):
            assert all(window not in text for text in released), f"trial {trial}"
            assert window not in joined, f"trial {trial}"
        assert joined.count(MASK_PLACEHOLDER) == 1, f"trial {trial}"
        # The output is the masked answer with at most the carried tail still withheld.
        assert masked.startswith(joined), f"trial {trial}"
        assert len(masked) - len(joined) <= len(SUFFIX), f"trial {trial}"


async def test_a_wrapped_passage_is_held_by_normalized_not_raw_length() -> None:
    """A line-wrapped copy has fewer normalized than raw characters; both stay hidden."""
    index = _index(PROTECTED_WRAPPED)
    answer = f"{PREFIX}{PROTECTED_WRAPPED}{SUFFIX}"
    masked = f"{PREFIX}{MASK_PLACEHOLDER}{SUFFIX}"
    rng = random.Random(SEED)
    for trial in range(50):
        renderer = ReaderFrameRenderer(index, index.mask)
        released = await _stream(renderer, _chunks(answer, rng, largest=13))
        joined = "".join(released)
        for window in _fingerprint_windows(PROTECTED_WRAPPED):
            assert all(window not in text for text in released), f"trial {trial}"
            assert window not in joined, f"trial {trial}"
        assert masked.count(MASK_PLACEHOLDER) == 1
        assert joined.count(MASK_PLACEHOLDER) == 1, f"trial {trial}"
        assert masked.startswith(joined), f"trial {trial}"


async def test_every_two_way_split_of_the_answer_is_safe() -> None:
    """Every possible single boundary, because a leak that needs one offset is still a leak."""
    index = _index()
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    for cut in range(1, len(answer)):
        renderer = ReaderFrameRenderer(index, index.mask)
        joined = "".join(await _stream(renderer, [answer[:cut], answer[cut:]]))
        for window in _fingerprint_windows(PROTECTED):
            assert window not in joined, f"cut {cut}"


async def test_a_plain_answer_is_released_in_order_with_only_the_tail_delayed() -> None:
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask)
    text = "A plain answer with no protected document text in it at all, delivered in pieces."
    released = await _stream(renderer, _chunks(text, random.Random(SEED), largest=9))
    joined = "".join(released)
    assert text.startswith(joined)
    # The delay is bounded by the fingerprint window the renderer must be able to scan.
    assert _visible_count(text[len(joined) :]) == STREAM_HOLD_VISIBLE_CHARS
    assert len(released) > 1


async def test_a_token_frame_is_an_sse_token_frame_whose_data_is_the_released_text() -> None:
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask)
    frame = await renderer(_event(7, "token", {"text": "x" * 100}))
    lines = frame.split("\n")
    assert lines[0] == "id: 7"
    assert lines[1] == "event: token"
    assert _frame_data(frame) == "x" * (100 - STREAM_HOLD_VISIBLE_CHARS)


async def test_a_reset_frame_discards_the_carry_and_restarts_the_stream() -> None:
    index = _index()
    renderer = ReaderFrameRenderer(index, index.mask)
    await _stream(renderer, ["A" * 100])
    reset = await renderer(_event(2, "reset", {}))
    assert "event: reset" in reset
    released = await _stream(renderer, ["B" * 100], first_sequence=3)
    assert released == ["B" * (100 - STREAM_HOLD_VISIBLE_CHARS)]
    assert "A" not in "".join(released)


async def test_a_terminal_frame_discards_the_carry() -> None:
    """A ``done`` frame cannot carry a tail; after it, nothing of the old draft remains."""
    index = _index()
    renderer = ReaderFrameRenderer(index, index.mask)
    await _stream(renderer, ["A" * 100])
    done = await renderer(_event(2, "done", {"status": "succeeded", "result": stored_result()}))
    assert '"succeeded"' in done
    released = await _stream(renderer, ["B" * 100], first_sequence=3)
    assert released == ["B" * (100 - STREAM_HOLD_VISIBLE_CHARS)]


async def test_the_done_frame_shows_the_same_masked_answer_the_stream_withheld() -> None:
    """The stream delays a protected passage; the terminal presentation must not reveal it."""
    index = _index()
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    renderer = ReaderFrameRenderer(index, index.mask)
    released = await _stream(renderer, _chunks(answer, random.Random(SEED), largest=11))
    assert MASK_PLACEHOLDER in "".join(released)
    done = await renderer(
        _event(99, "done", {"status": "succeeded", "result": stored_result(answer=answer)})
    )
    payload = _frame_data(done)
    presentation = payload["presentation"]
    assert PROTECTED not in presentation["answer_text"]
    assert presentation["answer_text"] == f"{PREFIX}{MASK_PLACEHOLDER}{SUFFIX}"
    for part in presentation["parts"]:
        rendered = f"{part.get('text', '')}{part.get('html', '')}"
        assert PROTECTED not in rendered


async def test_an_unprotected_answer_is_rendered_by_browser_frame_unchanged() -> None:
    """The stateful renderer composes with the frame projection rather than replacing it."""
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask, run_id="run-1")
    event = _event(4, "progress", {"phase": "planning"})
    assert await renderer(event) == await browser_frame(event)


async def test_no_reasoning_frame_releases_a_fingerprint_window_of_the_passage() -> None:
    """Reasoning is a second reader-facing text stream, so it gets the same safe-prefix gate."""
    index = _index()
    reasoning = f"{PREFIX}{PROTECTED}{SUFFIX}"
    masked = f"{PREFIX}{MASK_PLACEHOLDER}{SUFFIX}"
    rng = random.Random(SEED)
    for trial in range(50):
        renderer = ReaderFrameRenderer(index, index.mask)
        released = await _stream_reasoning(renderer, _chunks(reasoning, rng, largest=13))
        joined = "".join(released)
        for window in _fingerprint_windows(PROTECTED):
            assert all(window not in text for text in released), f"trial {trial}"
            assert window not in joined, f"trial {trial}"
        assert joined.count(MASK_PLACEHOLDER) == 1, f"trial {trial}"
        assert masked.startswith(joined), f"trial {trial}"


async def test_a_reasoning_frame_is_an_sse_reasoning_frame_carrying_released_text() -> None:
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask)
    frame = await renderer(_event(8, "reasoning", {"text": "x" * 100}))
    lines = frame.split("\n")
    assert lines[0] == "id: 8"
    assert lines[1] == "event: reasoning"
    assert _frame_data(frame) == "x" * (100 - STREAM_HOLD_VISIBLE_CHARS)


async def test_browser_frame_projects_a_reasoning_event_as_its_text() -> None:
    """The stateless projection names the payload a string, symmetric with ``token``."""
    frame = await browser_frame(_event(5, "reasoning", {"text": "thinking"}))
    assert "\nevent: reasoning\n" in frame
    assert _frame_data(frame) == "thinking"


async def test_reasoning_then_token_keep_arrival_order_with_independent_carries() -> None:
    """A reasoning→token switch keeps order and neither stream inherits the other's tail."""
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask)
    reasoning_frame = await renderer(_event(1, "reasoning", {"text": "a" * 100}))
    token_frame = await renderer(_event(2, "token", {"text": "b" * 100}))

    assert "\nevent: reasoning\n" in reasoning_frame
    assert "\nevent: token\n" in token_frame
    released = 100 - STREAM_HOLD_VISIBLE_CHARS
    assert _frame_data(reasoning_frame) == "a" * released
    assert _frame_data(token_frame) == "b" * released


@pytest.mark.parametrize(
    ("event_type", "payload"),
    [
        ("reset", {}),
        ("done", {"status": "cancelled"}),
        ("error", {"kind": "answer_stream_failed", "message": "broken"}),
    ],
)
async def test_a_forget_event_discards_the_reasoning_carry(
    event_type: str, payload: dict[str, Any]
) -> None:
    """A reset or terminal frame forgets the reasoning block as well as the answer draft."""
    index = _index()
    renderer = ReaderFrameRenderer(index, index.mask)
    await _stream_reasoning(renderer, ["A" * 100])
    forget = await renderer(_event(2, event_type, payload))
    assert f"event: {event_type}" in forget

    released = await _stream_reasoning(renderer, ["B" * 100], first_sequence=3)
    assert released == ["B" * (100 - STREAM_HOLD_VISIBLE_CHARS)]
    assert "A" not in "".join(released)


@pytest.mark.parametrize("tool_event", ["tool_start", "tool_progress", "tool_end"])
async def test_a_tool_event_closes_the_reasoning_block_carry(tool_event: str) -> None:
    """A Tool call ends a thinking block, so its withheld tail cannot bleed into the next one."""
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask)
    first = await renderer(_event(1, "reasoning", {"text": "a" * 100}))
    assert _frame_data(first) == "a" * (100 - STREAM_HOLD_VISIBLE_CHARS)

    tool = await renderer(_event(2, tool_event, {"tool_name": "view", "call_id": "call-1"}))
    assert f"\nevent: {tool_event}\n" in tool

    second = await renderer(_event(3, "reasoning", {"text": "b" * 100}))
    released = _frame_data(second)
    assert released == "b" * (100 - STREAM_HOLD_VISIBLE_CHARS)
    assert "a" not in released


async def test_a_tool_event_does_not_reset_the_token_carry() -> None:
    """An Answer's text resumes after a Tool call, so its own carry must survive."""
    index = ProtectedTextIndex.empty()
    renderer = ReaderFrameRenderer(index, index.mask)
    first = await renderer(_event(1, "token", {"text": "c" * 100}))
    assert _frame_data(first) == "c" * (100 - STREAM_HOLD_VISIBLE_CHARS)

    await renderer(_event(2, "tool_start", {"tool_name": "view", "call_id": "call-1"}))

    second = await renderer(_event(3, "token", {"text": "d" * 100}))
    assert _frame_data(second) == "c" * STREAM_HOLD_VISIBLE_CHARS + "d" * (
        100 - STREAM_HOLD_VISIBLE_CHARS
    )
