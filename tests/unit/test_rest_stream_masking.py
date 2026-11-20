# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe-prefix release on the REST event stream, and what a short quotation may show.

The REST ``/runs/{run_id}/events`` stream used to hand a ``token`` frame's text to the wire
verbatim while the browser path held it, so the official SDK's token callback was the one
surface a protected passage could cross. These tests pin the two properties that close it:
one transport-neutral :class:`~dlightrag.adapters.http.streaming.token_masking.TokenFrameMasker`
behind both projections, and two thresholds inside it -- detect and hold at a fingerprint
window (40), decide between ``[...]`` and the quotation itself at the finished answer's
threshold (80), so a streamed draft never shows something the settled answer will not.
"""

import json
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from dlightrag.adapters.http.browser.answer_events import (
    ReaderFrameRenderer,
    browser_frame,
)
from dlightrag.adapters.http.rest.routes import runs as runs_route
from dlightrag.adapters.http.rest.routes.runs import run_event_renderer, run_frame
from dlightrag.adapters.http.streaming.answer_stream import follow_frames
from dlightrag.adapters.http.streaming.token_masking import (
    STREAM_HOLD_VISIBLE_CHARS,
    TokenFrameMasker,
)
from dlightrag.application.runs import RunEvent
from dlightrag.engine.agent.skill_protection import (
    FINGERPRINT_CHARS,
    MASK_PLACEHOLDER,
    MIN_MASK_CHARS,
    ProtectedTextIndex,
)

#: One protected document, long enough that a quoted sentence of it is a verified run.
PROTECTED = (
    "The run ledger is the first thing a calculation states: every input carries the period "
    "it was observed in, the source that stated it, and the mechanism by which it reaches "
    "the conclusion. An input nobody can date is an input nobody can defend."
)
PREFIX = "Here is the passage I read: "
SUFFIX = " That is the whole method."
#: The same sentence with no leading space, so a slice of the document ends exactly where
#: the suffix begins and a quotation's verified length is the length of the slice.
CLEAN_SUFFIX = "That is the whole method."
#: A quotation below the finished answer's threshold: the settled answer shows it verbatim.
SHORT_QUOTE = PROTECTED[10:83]
#: A quotation at it: the settled answer replaces it, so the stream may replace it too.
LONG_QUOTE = PROTECTED[:120]
#: A document whose raw length dwarfs its normalized length, because every word is followed
#: by a long whitespace run the normalization collapses. The stream's thresholds are measured
#: in normalized coordinates too: measured in raw ones, a copy of this would be released while
#: the settled answer replaces it.
_PADDING = " " * 200
PADDED_LONG = "".join(f"ledger{index:02d}{_PADDING}" for index in range(12))
PADDED_SHORT = _PADDING.join(
    (
        "alpha",
        "bravo",
        "charl",
        "delta",
        "echoo",
        "foxtr",
        "golff",
        "hotel",
        "india",
        "julie",
        "kilo",
        "lima",
        "mike",
    )
)
RUN_ID = "019893f4-0000-7000-8000-000000000001"
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


def _rest_renderer(index: ProtectedTextIndex) -> Any:
    return run_event_renderer(
        downloadable_workspaces=None,
        visual_workspaces=None,
        run_id=RUN_ID,
        run_kind="answer",
        mask_reader_text=index.mask,
        protected_index=index,
    )


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


def _token_texts(renderer: Any, chunks: list[str], *, first_sequence: int = 1) -> list[str]:
    """Feed one chunk per token event through a synchronous frame renderer."""
    return [
        _frame_data(renderer(_event(first_sequence + offset, "token", {"text": chunk})))["text"]
        for offset, chunk in enumerate(chunks)
    ]


def _reasoning_texts(renderer: Any, chunks: list[str], *, first_sequence: int = 1) -> list[str]:
    """Feed one chunk per reasoning event through a synchronous frame renderer."""
    return [
        _frame_data(renderer(_event(first_sequence + offset, "reasoning", {"text": chunk})))["text"]
        for offset, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# The shared release component: detect at 40, replace at 80
# ---------------------------------------------------------------------------


def test_a_short_quotation_streams_verbatim_once_its_end_is_known() -> None:
    """A run below ``MIN_MASK_CHARS`` reads whole in the settled answer, so the draft shows it."""
    index = _index()
    answer = f"{PREFIX}{SHORT_QUOTE}{CLEAN_SUFFIX}"
    assert len(SHORT_QUOTE) == 73
    assert len(SHORT_QUOTE) < MIN_MASK_CHARS
    assert [finding.length for finding in index.scan(answer, minimum=FINGERPRINT_CHARS)] == [73]

    rng = random.Random(SEED)
    for trial in range(30):
        masker = TokenFrameMasker(index)
        released = [masker.release(chunk) for chunk in _chunks(answer, rng, largest=13)]
        joined = "".join(released)
        assert MASK_PLACEHOLDER not in joined, f"trial {trial}"
        assert all(MASK_PLACEHOLDER not in text for text in released), f"trial {trial}"
        assert SHORT_QUOTE in joined, f"trial {trial}"


def test_a_quotation_past_the_replacement_threshold_is_replaced_under_every_split() -> None:
    """A 120-character quotation is replaced, and no piece or concatenation leaks a window."""
    index = _index()
    answer = f"{PREFIX}{LONG_QUOTE}{CLEAN_SUFFIX}"
    assert len(LONG_QUOTE) == 120
    assert [finding.length for finding in index.scan(answer, minimum=FINGERPRINT_CHARS)] == [120]

    rng = random.Random(SEED)
    for trial in range(50):
        masker = TokenFrameMasker(index)
        released = [masker.release(chunk) for chunk in _chunks(answer, rng, largest=13)]
        joined = "".join(released)
        for window in _fingerprint_windows(LONG_QUOTE):
            assert all(window not in text for text in released), f"trial {trial}"
            assert window not in joined, f"trial {trial}"
        assert joined.count(MASK_PLACEHOLDER) == 1, f"trial {trial}"


def test_a_padded_run_past_the_threshold_is_replaced_in_normalized_coordinates() -> None:
    """Long raw, short normalized: the settled answer replaces it, so the stream must too."""
    index = _index(PADDED_LONG)
    answer = f"{PREFIX}{PADDED_LONG}{SUFFIX}"
    assert index.scan(PADDED_LONG, minimum=MIN_MASK_CHARS)

    rng = random.Random(SEED)
    for trial in range(15):
        masker = TokenFrameMasker(index)
        released = [masker.release(chunk) for chunk in _chunks(answer, rng, largest=13)]
        joined = "".join(released)
        assert MASK_PLACEHOLDER in joined, f"trial {trial}"
        assert PADDED_LONG not in joined, f"trial {trial}"
        for text in [*released, joined]:
            assert not index.scan(text, minimum=FINGERPRINT_CHARS), f"trial {trial}"


def test_a_padded_quotation_below_the_threshold_is_shown_verbatim() -> None:
    """Short normalized, long raw: the settled answer shows it, so the draft must not hide it."""
    index = _index(PADDED_SHORT)
    answer = f"{PREFIX}{PADDED_SHORT}{CLEAN_SUFFIX}"
    assert not index.scan(answer, minimum=MIN_MASK_CHARS)

    rng = random.Random(SEED)
    for trial in range(15):
        masker = TokenFrameMasker(index)
        released = [masker.release(chunk) for chunk in _chunks(answer, rng, largest=13)]
        joined = "".join(released)
        assert MASK_PLACEHOLDER not in joined, f"trial {trial}"
        assert PADDED_SHORT in joined, f"trial {trial}"


async def test_both_transports_release_the_same_token_text() -> None:
    """One masker semantics: the browser's rendered frames and REST's canonical frames agree."""
    index = _index()
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    chunks = _chunks(answer, random.Random(SEED), largest=13)
    browser = ReaderFrameRenderer(index, index.mask)
    browser_texts = [
        _frame_data(await browser(_event(offset + 1, "token", {"text": chunk})))
        for offset, chunk in enumerate(chunks)
    ]

    assert _token_texts(_rest_renderer(index), chunks) == browser_texts


# ---------------------------------------------------------------------------
# REST wiring: the frames the route serves
# ---------------------------------------------------------------------------


def test_rest_token_frames_never_carry_a_fingerprint_window() -> None:
    """The regression: REST used to send ``payload["text"]`` verbatim while the browser held it."""
    index = _index()
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    masked = f"{PREFIX}{MASK_PLACEHOLDER}{SUFFIX}"
    rng = random.Random(SEED)
    for trial in range(30):
        released = _token_texts(_rest_renderer(index), _chunks(answer, rng, largest=13))
        joined = "".join(released)
        for window in _fingerprint_windows(PROTECTED):
            assert all(window not in text for text in released), f"trial {trial}"
            assert window not in joined, f"trial {trial}"
        assert joined.count(MASK_PLACEHOLDER) == 1, f"trial {trial}"
        assert masked.startswith(joined), f"trial {trial}"
        assert len(masked) - len(joined) <= len(SUFFIX), f"trial {trial}"


def test_a_rest_token_frame_rendered_without_a_masker_still_passes_the_gate() -> None:
    """A caller with no subscription carry gets the stateless gate rather than raw text."""
    index = _index()
    frame = run_frame(
        _event(5, "token", {"text": PROTECTED}),
        downloadable_workspaces=None,
        visual_workspaces=None,
        run_id=RUN_ID,
        run_kind="answer",
        mask_reader_text=index.mask,
    )
    assert _frame_data(frame)["text"] == MASK_PLACEHOLDER


def test_rest_reset_restarts_the_stream_from_an_empty_carry() -> None:
    """A held passage of the replaced draft cannot bleed into the draft that follows it."""
    index = _index()
    renderer = _rest_renderer(index)
    draft = _frame_data(renderer(_event(1, "token", {"text": f"Here {PROTECTED[:60]}"})))["text"]
    assert draft == "Here "

    reset = renderer(_event(2, "reset", {}))
    assert "event: reset" in reset

    released = _frame_data(renderer(_event(3, "token", {"text": "B" * 100})))["text"]
    assert released == "B" * (100 - STREAM_HOLD_VISIBLE_CHARS)
    for window in _fingerprint_windows(PROTECTED[:60]):
        assert window not in released


def test_rest_terminal_event_discards_the_carry() -> None:
    """A terminal frame never carries a tail; after it the old draft is gone."""
    index = _index()
    renderer = _rest_renderer(index)
    renderer(_event(1, "token", {"text": "A" * 100}))
    done = renderer(_event(2, "done", {"status": "cancelled"}))
    assert "event: done" in done

    released = _frame_data(renderer(_event(3, "token", {"text": "B" * 100})))["text"]
    assert released == "B" * (100 - STREAM_HOLD_VISIBLE_CHARS)


async def test_the_rest_events_stream_masks_tokens_and_restarts_after_reset() -> None:
    """The frame generator behind ``StreamingResponse``: body has no window, and reset is a cut."""
    index = _index()
    renderer = _rest_renderer(index)
    draft = f"Draft zero begins. {PROTECTED[:60]}"
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    chunks = _chunks(answer, random.Random(SEED), largest=11)
    events = [
        _event(1, "token", {"text": draft}),
        _event(2, "reset", {}),
        *[_event(3 + offset, "token", {"text": chunk}) for offset, chunk in enumerate(chunks)],
        _event(3 + len(chunks), "done", {"status": "cancelled"}),
    ]

    async def committed() -> AsyncIterator[RunEvent]:
        for event in events:
            yield event

    frames = [frame async for frame in follow_frames(committed(), renderer)]

    assert [frame.split("\n")[0] for frame in frames] == [
        f"id: {event.sequence}" for event in events
    ]
    body = "".join(frame.split("data: ", 1)[1].strip() for frame in frames)
    for window in _fingerprint_windows(PROTECTED):
        assert window not in body
    assert PROTECTED not in body
    assert MASK_PLACEHOLDER in body
    # The first draft's held fragment never crossed the wire, before the reset or after it.
    assert PROTECTED[:60] not in body

    token_payloads = [_frame_data(frame) for frame in frames if "\nevent: token\n" in frame]
    fresh = TokenFrameMasker(index)
    expected = [fresh.release(chunk) for chunk in chunks]
    assert [payload["text"] for payload in token_payloads[1:]] == expected


async def test_the_rest_events_route_serves_the_masked_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The route itself, not only its pieces: the subscription it opens masks through the app."""
    index = _index()
    answer = f"{PREFIX}{PROTECTED}{SUFFIX}"
    chunks = _chunks(answer, random.Random(SEED), largest=11)
    events = [
        _event(1, "token", {"text": "Intro. "}),
        *[_event(2 + offset, "token", {"text": chunk}) for offset, chunk in enumerate(chunks)],
        _event(2 + len(chunks), "done", {"status": "cancelled"}),
    ]

    async def subscribed(**_kwargs: Any) -> AsyncIterator[RunEvent]:
        for event in events:
            yield event

    application = SimpleNamespace(
        runs=SimpleNamespace(subscribe=lambda **_kwargs: subscribed()),
        skill_protection=SimpleNamespace(index=index, mask=index.mask),
        retrieval=SimpleNamespace(),
    )
    record = SimpleNamespace(
        access_scope_id="owner-1",
        events_trimmed_at=None,
        run_kind="answer",
        request_input=lambda: {"workspaces": []},
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(application=application)),
        headers={},
        query_params={},
    )
    monkeypatch.setattr(runs_route, "_authorized_run", AsyncMock(return_value=record))
    monkeypatch.setattr(runs_route, "authorized_workspaces", AsyncMock(return_value=set()))

    response = await runs_route.stream_run_events(
        "run-1", cast(Any, request), user=cast(Any, object())
    )

    frames: list[str] = []
    async for frame in cast(AsyncIterator[Any], response.body_iterator):
        frames.append(frame if isinstance(frame, str) else bytes(frame).decode())
    body = "".join(frame.split("data: ", 1)[1].strip() for frame in frames)
    for window in _fingerprint_windows(PROTECTED):
        assert window not in body
    assert PROTECTED not in body
    assert MASK_PLACEHOLDER in body
    assert [frame.split("\n")[0] for frame in frames] == [
        f"id: {event.sequence}" for event in events
    ]


# ---------------------------------------------------------------------------
# The memory body on the same event streams
# ---------------------------------------------------------------------------


async def test_the_settled_memory_body_passes_the_content_gate_on_both_streams() -> None:
    """The operator body is free text; it can quote a protected document like any answer."""
    index = _index()
    payload = {
        "body": PROTECTED,
        "kind": "fact",
        "outcome": "created",
        "operation": "remember",
        "memory_ids": ["memory-1"],
    }
    event = _event(9, "memory_operation_settled", payload)

    browser = _frame_data(await browser_frame(event, mask_reader_text=index.mask))
    rest = _frame_data(
        run_frame(
            event,
            downloadable_workspaces=None,
            visual_workspaces=None,
            run_id=RUN_ID,
            run_kind="answer",
            mask_reader_text=index.mask,
        )
    )

    assert browser["body"] == MASK_PLACEHOLDER
    assert rest["body"] == MASK_PLACEHOLDER
    assert PROTECTED not in json.dumps(browser)
    assert PROTECTED not in json.dumps(rest)


# ---------------------------------------------------------------------------
# The live reasoning stream on the same edges
# ---------------------------------------------------------------------------


async def test_both_transports_release_the_same_reasoning_text() -> None:
    """Reasoning reads through the same safe-prefix component as tokens on both edges."""
    index = _index()
    reasoning = f"{PREFIX}{PROTECTED}{SUFFIX}"
    chunks = _chunks(reasoning, random.Random(SEED), largest=13)
    browser = ReaderFrameRenderer(index, index.mask)
    browser_texts = [
        _frame_data(await browser(_event(offset + 1, "reasoning", {"text": chunk})))
        for offset, chunk in enumerate(chunks)
    ]

    assert _reasoning_texts(_rest_renderer(index), chunks) == browser_texts


def test_rest_reasoning_frames_never_carry_a_fingerprint_window() -> None:
    """The same regression as tokens: REST reasoning frames cannot ship protected text raw."""
    index = _index()
    reasoning = f"{PREFIX}{PROTECTED}{SUFFIX}"
    masked = f"{PREFIX}{MASK_PLACEHOLDER}{SUFFIX}"
    rng = random.Random(SEED)
    for trial in range(30):
        released = _reasoning_texts(_rest_renderer(index), _chunks(reasoning, rng, largest=13))
        joined = "".join(released)
        for window in _fingerprint_windows(PROTECTED):
            assert all(window not in text for text in released), f"trial {trial}"
            assert window not in joined, f"trial {trial}"
        assert joined.count(MASK_PLACEHOLDER) == 1, f"trial {trial}"
        assert masked.startswith(joined), f"trial {trial}"


def test_a_rest_reasoning_frame_rendered_without_a_masker_still_passes_the_gate() -> None:
    """A caller with no subscription carry gets the stateless gate rather than raw text."""
    index = _index()
    frame = run_frame(
        _event(5, "reasoning", {"text": PROTECTED}),
        downloadable_workspaces=None,
        visual_workspaces=None,
        run_id=RUN_ID,
        run_kind="answer",
        mask_reader_text=index.mask,
    )
    assert _frame_data(frame)["text"] == MASK_PLACEHOLDER


def test_rest_reset_restarts_the_reasoning_carry() -> None:
    """A held reasoning passage of the replaced draft cannot bleed into the draft after it."""
    index = _index()
    renderer = _rest_renderer(index)
    draft = _frame_data(renderer(_event(1, "reasoning", {"text": f"Here {PROTECTED[:60]}"})))[
        "text"
    ]
    assert draft == "Here "

    reset = renderer(_event(2, "reset", {}))
    assert "event: reset" in reset

    released = _frame_data(renderer(_event(3, "reasoning", {"text": "B" * 100})))["text"]
    assert released == "B" * (100 - STREAM_HOLD_VISIBLE_CHARS)
    for window in _fingerprint_windows(PROTECTED[:60]):
        assert window not in released


async def test_the_rest_events_stream_keeps_reasoning_before_the_tokens_it_precedes() -> None:
    """The frame generator preserves arrival order across the two text channels."""
    index = _index()
    renderer = _rest_renderer(index)
    reasoning = f"{PREFIX}{PROTECTED}{SUFFIX}"
    reasoning_chunks = _chunks(reasoning, random.Random(SEED), largest=11)
    token_chunks = _chunks(reasoning, random.Random(SEED), largest=11)
    events = [
        *[
            _event(1 + offset, "reasoning", {"text": chunk})
            for offset, chunk in enumerate(reasoning_chunks)
        ],
        *[
            _event(1 + len(reasoning_chunks) + offset, "token", {"text": chunk})
            for offset, chunk in enumerate(token_chunks)
        ],
        _event(2 + len(reasoning_chunks) + len(token_chunks), "done", {"status": "cancelled"}),
    ]

    async def committed() -> AsyncIterator[RunEvent]:
        for event in events:
            yield event

    frames = [frame async for frame in follow_frames(committed(), renderer)]

    assert [frame.split("\n")[0] for frame in frames] == [
        f"id: {event.sequence}" for event in events
    ]
    assert [frame.split("\n")[1] for frame in frames] == [
        *["event: reasoning"] * len(reasoning_chunks),
        *["event: token"] * len(token_chunks),
        "event: done",
    ]
    body = "".join(frame.split("data: ", 1)[1].strip() for frame in frames)
    for window in _fingerprint_windows(PROTECTED):
        assert window not in body


@pytest.mark.parametrize("tool_event", ["tool_start", "tool_progress", "tool_end"])
def test_rest_tool_event_closes_the_reasoning_block_carry(tool_event: str) -> None:
    """A Tool call ends a thinking block, so its withheld tail cannot bleed into the next one."""
    index = ProtectedTextIndex.empty()
    renderer = _rest_renderer(index)
    first = _frame_data(renderer(_event(1, "reasoning", {"text": "a" * 100})))
    assert first["text"] == "a" * (100 - STREAM_HOLD_VISIBLE_CHARS)

    tool = renderer(_event(2, tool_event, {}))
    assert f"\nevent: {tool_event}\n" in tool

    second = _frame_data(renderer(_event(3, "reasoning", {"text": "b" * 100})))
    assert second["text"] == "b" * (100 - STREAM_HOLD_VISIBLE_CHARS)
    assert "a" not in second["text"]
