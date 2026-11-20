# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The live preview of a leased page: what it captures, what it costs, and who drives it.

The CDP screencast is driven through a fake page and a fake CDP session, so a test reads
exactly what the browser was asked to encode and when: nothing is captured while nobody is
watching, one screencast serves every reader, a reader that falls behind loses frames
instead of growing a queue, and a session whose readers are all behind steps its own budget
down a rung at a time — and back up — without ever flapping.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import time
from collections.abc import Callable
from typing import Any, cast

import pytest

from dlightrag.adapters.agent_browser.live import (
    FRAME_HEADER_BYTES,
    LiveBrowserSession,
    LiveBrowserSessions,
    LiveInput,
    PlaywrightLivePage,
    _Pace,
)
from dlightrag.adapters.agent_browser.live import (
    Frame as _Frame,
)
from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
    BrowserHolder,
)
from dlightrag.engine.answer.agent_browser.live import (
    LiveControl,
    LiveControlState,
    LivePreviewBudget,
    LivePreviewSettings,
    LivePreviewStep,
)
from dlightrag.engine.answer.agent_browser.run import (
    RunAgentBrowser,
)
from tests.support.agent_browser import FakeLeases, FakeLivePage, FakeProvider, browser_settings
from tests.support.dns import public_dns

HOLDER = BrowserHolder("owner", "11111111-1111-1111-1111-111111111111", "worker-1", 1)
#: The bytes one frame is, and the base64 CDP hands the same frame over as.
JPEG = b"\xff\xd8jpeg\xff\xd9"
JPEG_BASE64 = base64.b64encode(JPEG).decode()
BUDGET = LivePreviewBudget(frames_per_second=4.0, max_width=1280, jpeg_quality=68)
SETTINGS = LivePreviewSettings(
    budget=BUDGET,
    session_seconds=30.0,
    control_idle_seconds=0.2,
)
#: The budget the step tests walk down from, at the numbers the deployment configures.
STEP_BASE = LivePreviewBudget(frames_per_second=24.0, max_width=1280, jpeg_quality=70)
SLOW = LivePreviewSettings(
    budget=LivePreviewBudget(frames_per_second=1.0, max_width=1280, jpeg_quality=60),
    session_seconds=30.0,
    control_idle_seconds=30.0,
)


def jpeg_of(frame: bytes | None) -> bytes:
    """The JPEG inside one delivered frame, with the header that precedes it removed."""
    assert frame is not None
    return frame[FRAME_HEADER_BYTES:]


def sequence_of(frame: bytes | None) -> int:
    """The sequence number one delivered frame states in its own header."""
    assert frame is not None
    return struct.unpack("!Q", frame[:8])[0]


def captured_at_of(frame: bytes | None) -> float:
    """The capture time one delivered frame states in its own header, epoch seconds."""
    assert frame is not None
    return struct.unpack("!d", frame[8:FRAME_HEADER_BYTES])[0]


class FakeCdp:
    """A CDP session that records what it was asked and replays frames a test pushes."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.handlers: dict[str, Any] = {}
        self.detached = False
        self.acks = 0
        self.frame_number = 0

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    async def send(self, method: str, params: dict[str, Any] | None = None) -> None:
        if self.detached:
            raise RuntimeError("the CDP session is gone")
        self.sent.append((method, params or {}))
        if method == "Page.screencastFrameAck":
            self.acks += 1

    async def detach(self) -> None:
        self.detached = True

    @property
    def methods(self) -> list[str]:
        return [method for method, _ in self.sent]

    @property
    def encodes(self) -> list[dict[str, Any]]:
        """Every set of encoding parameters this screencast was asked for, oldest first."""
        return [params for method, params in self.sent if method == "Page.startScreencast"]

    def frame(self, data: str = JPEG_BASE64, *, captured_at: float | None = None) -> None:
        """Hand the screencast one more frame, as the browser would.

        ``captured_at`` is what Chromium puts in the frame's own metadata: the moment of the
        capture in *seconds* since the epoch, as CDP reports it — a real frame off the
        bundled pool carried 1791948557.2566 while this process read 1791948559.690. Handing
        milliseconds to this instead is the mistake that put the header in 1970, where a
        dock reads its own lag as 1.8 billion seconds.
        """
        self.frame_number += 1
        metadata: dict[str, Any] = {}
        if captured_at is not None:
            metadata["timestamp"] = captured_at
        self.handlers["Page.screencastFrame"](
            {"data": data, "sessionId": self.frame_number, "metadata": metadata}
        )

    @property
    def acked(self) -> list[int]:
        """Every frame id this screencast was asked to acknowledge, oldest first."""
        return [
            int(params["sessionId"])
            for method, params in self.sent
            if method == "Page.screencastFrameAck"
        ]


class FakePage:
    """The Playwright page a live session watches, without a browser behind it."""

    def __init__(self) -> None:
        self.cdps: list[FakeCdp] = []
        self.context = self
        self.handlers: dict[str, Any] = {}
        self.closed = False
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()
        self.navigated: list[str] = []

    @property
    def cdp(self) -> FakeCdp:
        """The CDP session of the screencast running now, as the browser's page has one."""
        assert self.cdps, "no screencast was ever started"
        return self.cdps[-1]

    def methods(self) -> list[str]:
        """Every CDP method this page's screencasts were asked for, oldest first."""
        return [method for cdp in self.cdps for method in cdp.methods]

    def encodes(self) -> list[dict[str, Any]]:
        """Every set of encoding parameters this page was asked for, oldest first."""
        return [params for cdp in self.cdps for params in cdp.encodes]

    def acks(self) -> int:
        """How many frames every screencast of this page acknowledged."""
        return sum(cdp.acks for cdp in self.cdps)

    async def new_cdp_session(self, page: Any) -> FakeCdp:
        assert page is self
        cdp = FakeCdp()
        self.cdps.append(cdp)
        return cdp

    def on(self, event: str, handler: Any) -> None:
        self.handlers[event] = handler

    def close(self) -> None:
        """Close the page, as the browser does when its context or the browser goes."""
        self.closed = True
        for handler in self.handlers.values():
            handler(self)

    def is_closed(self) -> bool:
        return self.closed

    async def goto(self, url: str, *, wait_until: str, timeout: float) -> None:
        self.navigated.append(url)


class FakeMouse:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    async def move(self, x: float, y: float) -> None:
        self.events.append(("move", x, y))

    async def click(self, x: float, y: float, *, button: str, click_count: int) -> None:
        self.events.append(("click", x, y, button, click_count))

    async def down(self, *, button: str, click_count: int) -> None:
        self.events.append(("down", button, click_count))

    async def up(self, *, button: str, click_count: int) -> None:
        self.events.append(("up", button, click_count))

    async def wheel(self, delta_x: float, delta_y: float) -> None:
        self.events.append(("wheel", delta_x, delta_y))


class FakeKeyboard:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    async def press(self, key: str) -> None:
        self.events.append(("press", key))

    async def down(self, key: str) -> None:
        self.events.append(("down", key))

    async def up(self, key: str) -> None:
        self.events.append(("up", key))

    async def insert_text(self, text: str) -> None:
        self.events.append(("type", text))


class _FakeBrowser:
    """The one piece of ``PlaywrightLeasedBrowser`` a live session reads."""

    def __init__(self, page: FakePage) -> None:
        self._page = page

    def live_page(self) -> FakePage:
        return self._page


def session(
    page: FakePage | None = None,
    *,
    settings: LivePreviewSettings = SETTINGS,
    control: LiveControl | None = None,
    record: list[LiveControlState] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[LiveBrowserSession, FakePage]:
    """A live session over a fake page, with the Playwright page it was built on."""
    playwright_page = page if page is not None else FakePage()
    surface = PlaywrightLivePage(
        cast(Any, _FakeBrowser(playwright_page)), control if control is not None else LiveControl()
    )
    recorded = record if record is not None else []
    live = LiveBrowserSession(
        owner_id=HOLDER.owner_id,
        run_id=HOLDER.run_id,
        page=surface,
        settings=settings,
        write_control=_recorder(recorded),
        clock=clock,
    )
    return live, playwright_page


def _recorder(recorded: list[LiveControlState]) -> Any:
    async def record(state: LiveControlState) -> None:
        recorded.append(state)

    return record


async def _watching(live: LiveBrowserSession, page: FakePage) -> Any:
    """One reader, subscribed and with a screencast running; returns its iterator."""
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    assert "Page.startScreencast" in page.methods()
    return reader


class FakeClock:
    """A clock a test moves by hand, so a window of delivery evidence is a few lines."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StepHarness:
    """One reader whose reading a test controls, and a clock that moves a frame at a time.

    Frames are handed on exactly as the capture loop hands them on, and what the reader did
    with the last one is the test's to choose: a frame the reader has not read when the next
    one arrives is a displaced one, which is what the session counts. Nothing here is a
    second implementation of the decision — every verdict a window produces is read off the
    session itself, which is the thing under test.
    """

    #: The frames in one window, small enough that a test writes a window as one line.
    WINDOW = 3

    def __init__(self, *, base: LivePreviewBudget = STEP_BASE, keep_watching: bool = False) -> None:
        #: Whether the one reader stays subscribed across windows. A test of what a rung
        #: does to the screencast wants the screencast to survive between them, which is
        #: what one reader reading through the whole session does.
        self.keep_watching = keep_watching
        self.clock = FakeClock()
        self.live, self.page = session(
            settings=LivePreviewSettings(
                budget=base, session_seconds=30.0, control_idle_seconds=30.0
            ),
            clock=self.clock,
        )
        self.live._pace = _Pace(self.clock, window=self.WINDOW)  # noqa: SLF001
        self.reader = self.live.frames()
        self.frame_id = 0

    async def subscribe(self) -> None:
        """Subscribe one reader, with the first update the session always sends."""
        await anext(self.reader)

    async def window(self, share: float) -> None:
        """One window of frames, ``share`` of which displaced one the reader never read.

        ``share`` is what the reader did inside the window: one is a reader that has stopped
        reading entirely — every frame arrives while the previous one is still in its slot —
        and zero is one that reads each frame before the next arrives. A frame arrives every
        few tenths of a second, so nothing here reads as a page that went quiet.
        """
        for index in range(self.WINDOW):
            self.clock.advance(0.4)
            await self.live._deliver(self.frame())  # noqa: SLF001 - the pace is what the test drives
            if index / self.WINDOW >= share:
                await anext(self.reader)
        if not self.keep_watching:
            # A reader that leaves at the end of a window and comes back takes the session's
            # screencast with it, which is not what "the session kept stepping" means.
            await self.reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]
            self.reader = self.live.frames()
            await anext(self.reader)

    def frame(self) -> Any:
        """One frame as the capture loop hands it on: the JPEG, and where it came from."""
        self.frame_id += 1
        return _Frame(
            jpeg=JPEG,
            frame_id=self.frame_id,
            sequence=self.frame_id,
            captured_at=1_700_000_000.0 + self.frame_id,
        )

    async def quiet(self, seconds: float) -> None:
        """Let the page go quiet: the frames either side of the gap are not one window."""
        self.clock.advance(seconds)

    @property
    def budget(self) -> LivePreviewBudget:
        """What the session is delivering at now, as the state message would carry it."""
        return self.live._budget  # noqa: SLF001 - the budget is the thing under test


async def test_a_run_without_a_lease_has_no_live_session():
    leases = FakeLeases(holder=None)
    sessions = LiveBrowserSessions(provider=FakeProvider(), leases=leases, settings=SETTINGS)

    with pytest.raises(AgentBrowserError) as refused:
        await sessions.open(owner_id=HOLDER.owner_id, run_id=HOLDER.run_id)

    assert refused.value.reason == "not_configured"


async def test_a_lease_another_process_holds_is_not_watched_here():
    leases = FakeLeases(holder=HOLDER)
    sessions = LiveBrowserSessions(provider=FakeProvider(), leases=leases, settings=SETTINGS)

    with pytest.raises(AgentBrowserError) as refused:
        await sessions.open(owner_id=HOLDER.owner_id, run_id=HOLDER.run_id)

    assert refused.value.reason == "unreachable"


async def test_nobody_watching_captures_nothing():
    live, page = session()

    # No subscriber has ever asked for a frame, so no screencast was ever started.
    await asyncio.sleep(0.05)
    assert page.methods() == []


async def test_a_subscriber_starts_one_screencast_and_leaving_stops_it():
    live, page = session()
    reader = live.frames()

    first = await anext(reader)
    assert first.attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    assert page.methods()[0] == "Page.startScreencast"
    # One watcher of a page nobody drives is encoded at the session's one budget.
    assert page.cdp.encodes[0]["maxWidth"] == BUDGET.max_width
    assert page.cdp.encodes[0]["quality"] == BUDGET.jpeg_quality

    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]
    for _ in range(200):
        if "Page.stopScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    assert "Page.stopScreencast" in page.methods()

    # And with the last reader gone, a frame the page still sends is not encoded at all.
    assert page.acks() == 0


async def test_frames_arrive_while_a_subscriber_is_attached():
    live, page = session()
    reader = live.frames()
    assert (await anext(reader)).attached is True

    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    page.cdp.frame()

    update = await asyncio.wait_for(anext(reader), 2)
    assert jpeg_of(update.frame) == JPEG
    assert sequence_of(update.frame) == 1
    # And the state says the same number: the frame's header and the update beside it are
    # two readings of one count, so a reader that only sees the update is not blind to it.
    assert update.sequence == 1
    assert update.control == "agent"
    assert update.driver is None
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_frame_carries_its_capture_time_and_a_number_that_only_goes_up():
    live, page = session()
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)

    # A browser states when it captured the page in the frame's own metadata, in seconds
    # since the epoch — the same clock this process reads, in the unit CDP reports — and
    # what a reader is handed is that moment as it stands, beside the number of the frame
    # itself.
    captured = time.time()
    page.cdp.frame(captured_at=captured)
    first = await asyncio.wait_for(anext(reader), 2)
    page.cdp.frame()
    second = await asyncio.wait_for(anext(reader), 2)

    translated = captured_at_of(first.frame)
    # Read as a time of day it is when the page was captured, to within the reading's own
    # resolution. What it is *not* is that moment divided by a thousand, or read as if the
    # protocol reported milliseconds: either would be 1.7 billion seconds away from this,
    # which is exactly what a dock's lag readout said while the header carried 1970.
    assert translated == pytest.approx(captured, abs=1.0)
    assert translated == pytest.approx(time.time(), abs=5.0)
    assert sequence_of(first.frame) == 1
    assert sequence_of(second.frame) == 2
    # The update beside each frame states the same count, and only ever the newest one this
    # session has delivered: that is the number a reader compares with what it has drawn.
    assert first.sequence == 1
    assert second.sequence == 2
    # A frame whose metadata carried no reading at all is stamped with the moment it arrived
    # here, which is a little later and never earlier than the truth.
    assert captured_at_of(second.frame) == pytest.approx(time.time(), abs=5.0)
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_frame_is_acknowledged_when_it_is_superseded_and_not_when_it_arrives():
    live, page = session()
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)

    # The ack is CDP's flow control: a frame acknowledged on receipt lets the page produce
    # at its own pace while the frames pile up behind the reader. Nothing is acknowledged
    # until a newer frame makes the one in hand worthless.
    page.cdp.frame()
    await asyncio.wait_for(anext(reader), 2)
    assert page.cdp.acked == []

    page.cdp.frame()
    await asyncio.wait_for(anext(reader), 2)
    assert page.cdp.acked == [1]
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_one_screencast_serves_every_subscriber():
    live, page = session()
    first, second = live.frames(), live.frames()
    assert (await anext(first)).attached is True
    assert (await anext(second)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    page.cdp.frame()

    updates = await asyncio.gather(
        asyncio.wait_for(anext(first), 2), asyncio.wait_for(anext(second), 2)
    )
    assert [jpeg_of(update.frame) for update in updates] == [JPEG, JPEG]
    assert page.methods().count("Page.startScreencast") == 1
    await first.aclose()  # pyright: ignore[reportAttributeAccessIssue]
    await second.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_late_reader_gets_the_newest_frame_and_never_a_backlog():
    live, page = session(settings=SLOW)
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)

    # The first frame is delivered, and then two arrive while the reader is not reading:
    # the slot holds the newest one only, so the older of the two is dropped, never queued.
    # CDP hands each one over as base64; what a reader is handed is a header and the JPEG.
    first = base64.b64encode(b"first").decode()
    second = base64.b64encode(b"second").decode()
    third = base64.b64encode(b"third").decode()
    page.cdp.frame(first)
    delivered = await asyncio.wait_for(anext(reader), 2)
    assert jpeg_of(delivered.frame) == b"first"
    assert sequence_of(delivered.frame) == 1
    assert delivered.sequence == 1
    page.cdp.frame(second)
    page.cdp.frame(third)
    newest = await asyncio.wait_for(anext(reader), 3)
    assert jpeg_of(newest.frame) == b"third"
    # The number counts what this session *delivered*, not what the page encoded: the
    # frame nobody was sent consumes none, so the numbers a reader compares are its own
    # consecutive count. Numbering the captures instead would hand this frame a 3 here.
    assert sequence_of(newest.frame) == 2
    assert newest.sequence == 2
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_input_is_refused_until_a_person_takes_control():
    live, page = session()
    with pytest.raises(AgentBrowserError) as refused:
        await live.send(LiveInput(kind="mouse", action="click", x=4, y=4), token="v1")
    assert refused.value.reason == "taken_over"
    assert page.mouse.events == []


async def test_a_person_drives_the_page_they_took_over_and_a_retry_runs_once(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    recorded: list[LiveControlState] = []
    live, page = session(record=recorded)

    seat = await live.take_control()
    assert seat.control == "human"
    assert seat.token is not None
    assert seat.driver is not None
    assert live.driver == seat.driver
    assert recorded == ["human"]

    token = seat.token
    await live.send(LiveInput(kind="mouse", action="click", x=10, y=20), token=token, key="one")
    await live.send(LiveInput(kind="mouse", action="click", x=10, y=20), token=token, key="one")
    await live.send(LiveInput(kind="key", action="press", key="Enter"), token=token)
    await live.send(LiveInput(kind="wheel", delta_y=-240), token=token)
    await live.send(LiveInput(kind="navigate", url="https://example.com/"), token=token)

    assert page.mouse.events == [
        ("click", 10, 20, "left", 1),
        ("move", 0.0, 0.0),
        ("wheel", 0.0, -240),
    ]
    assert page.keyboard.events == [("press", "Enter")]
    assert page.navigated == ["https://example.com/"]

    assert await live.release_control(token=token) == "agent"
    assert await live.release_control(token=token) == "agent"
    assert live.driver is None
    assert recorded == ["human", "agent"]
    with pytest.raises(AgentBrowserError):
        await live.send(LiveInput(kind="key", action="press", key="Enter"), token=token)


async def test_a_reader_that_only_watched_cannot_drive_or_take_the_seat():
    """The broadcast is the whole of what a second reader gets, and it holds no credential.

    This is the security property the whole seat rests on: the state every reader is sent
    names the seat as *held*, and the token that drives it exists only in the answer to the
    request that took it. So a reader that replays every byte of the broadcast — which is
    exactly what a second viewer of the same Run can do — has nothing to present.
    """
    live, page = session()

    first = await live.take_control()
    assert first.token is not None

    # Everything the second reader can see: the state, which carries the label and no token.
    reader = live.frames()
    seen = await anext(reader)
    assert seen.control == "human"
    assert seen.driver == first.driver
    replayed = json.dumps(
        {
            "control": seen.control,
            "driver": seen.driver,
            "frames_per_second": seen.budget.frames_per_second if seen.budget else None,
            "jpeg_quality": seen.budget.jpeg_quality if seen.budget else None,
            "max_width": seen.budget.max_width if seen.budget else None,
        }
    )
    assert first.token not in replayed

    # A second reader asking for the seat is refused, seat held or not: nothing in a
    # request says who is asking, so "take it" is never answered with the credential of a
    # seat that is already held.
    with pytest.raises(AgentBrowserError) as refused:
        await live.take_control()
    assert refused.value.reason == "taken_over"

    # And presenting the broadcast — the label, a guess, or nothing — drives nothing.
    for replayed_token in (seen.driver or "", "", "v1", "viewer"):
        with pytest.raises(AgentBrowserError) as refused_input:
            await live.send(LiveInput(kind="mouse", action="click", x=1, y=1), token=replayed_token)
        assert refused_input.value.reason == "taken_over"
    # Nor does it give the seat up: a hand-back reaches the seat only for the credential
    # that took it, and everything else is refused as ``taken_over`` with the seat still
    # held. The session's own release — its end, its idle timer — is another path, and not
    # one a credential a reader can present ever reaches.
    for replayed_token in (seen.driver or "", "", "viewer", "v1"):
        with pytest.raises(AgentBrowserError) as refused_release:
            await live.release_control(token=replayed_token)
        assert refused_release.value.reason == "taken_over"
    # A caller that presents no credential at all: the transport refuses a hand-back with
    # no header before it reaches the session, and this guard is the last word on it —
    # ``token is None`` used to read as "the session's own release" and gave the seat up to
    # anybody who asked without one. The type is bypassed deliberately, because the guard
    # is what a caller that got here anyway meets.
    absent: Any = None
    with pytest.raises(AgentBrowserError) as refused_absent:
        await live.release_control(token=absent)
    assert refused_absent.value.reason == "taken_over"
    assert live.control == "human"
    assert page.mouse.events == []
    assert live.driver == first.driver

    # The holder still drives with the credential it was answered with.
    await live.send(LiveInput(kind="mouse", action="click", x=2, y=2), token=first.token)
    assert page.mouse.events == [("click", 2, 2, "left", 1)]
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_handing_the_seat_back_burns_the_credential_it_was_held_with():
    """A credential is spent by the hand-back: the seat taken next is a different one."""
    live, page = session()

    first = await live.take_control()
    assert first.token is not None
    assert await live.release_control(token=first.token) == "agent"

    # The token that took the seat cannot take it again, drive with it, or hand it back.
    with pytest.raises(AgentBrowserError) as refused:
        await live.send(LiveInput(kind="key", action="press", key="Enter"), token=first.token)
    assert refused.value.reason == "taken_over"

    second = await live.take_control()
    assert second.token is not None
    assert second.token != first.token
    assert second.driver != first.driver
    with pytest.raises(AgentBrowserError) as replayed:
        await live.send(LiveInput(kind="key", action="press", key="Enter"), token=first.token)
    assert replayed.value.reason == "taken_over"

    await live.send(LiveInput(kind="key", action="press", key="Enter"), token=second.token)
    assert page.keyboard.events == [("press", "Enter")]


async def test_an_input_event_the_wire_overtook_is_dropped_rather_than_run():
    """Input travels as separate requests, and separate requests can arrive out of order."""
    live, page = session()
    seat = await live.take_control()
    token = seat.token
    assert token is not None

    async def moved(x: float, y: float, sequence: int) -> None:
        await live.send(
            LiveInput(kind="mouse", action="move", x=x, y=y, sequence=sequence),
            token=token,
            key=f"move-{sequence}",
        )

    await moved(10, 10, 1)
    await moved(20, 20, 2)
    # Event one arrives after event two has run: it describes where the pointer was rather
    # than where it is, so running it would drag the page back.
    await moved(11, 11, 1)
    await moved(30, 30, 3)
    # An event that states no order — a click, which is never coalesced — still runs: the
    # ordering is only ever used to drop an event that named itself as an older one.
    await live.send(LiveInput(kind="mouse", action="click", x=40, y=40), token=token, key="c")

    assert [event for event in page.mouse.events if event[0] == "move"] == [
        ("move", 10, 10),
        ("move", 20, 20),
        ("move", 30, 30),
    ]
    assert ("click", 40, 40, "left", 1) in page.mouse.events


async def test_one_viewer_drives_and_a_second_stays_an_observer():
    live, page = session()

    first = await live.take_control()
    assert first.control == "human" and first.token is not None
    # The seat is one, and it is the first viewer's: the second can neither take it,
    # type into the page, nor hand back a seat it never held.
    with pytest.raises(AgentBrowserError) as refused:
        await live.take_control()
    assert refused.value.reason == "taken_over"
    with pytest.raises(AgentBrowserError) as refused_input:
        await live.send(LiveInput(kind="mouse", action="click", x=1, y=1), token="second")
    assert refused_input.value.reason == "taken_over"
    with pytest.raises(AgentBrowserError):
        await live.release_control(token="second")

    assert live.control == "human"
    assert live.driver == first.driver
    assert page.mouse.events == []

    # The observer can watch, and every update says the seat is held.
    reader = live.frames()
    assert (await anext(reader)).driver == first.driver
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]

    assert await live.release_control(token=first.token) == "agent"
    second = await live.take_control()
    assert second.control == "human"
    assert live.driver == second.driver


async def test_one_budget_serves_the_whole_session_whoever_drives():
    live, page = session()

    # The seat is authority over the page, not an allowance of frames: taking it and
    # handing it back change nothing about what is delivered, and no frame changes it.
    assert live._budget == BUDGET  # noqa: SLF001
    assert live._interval() == pytest.approx(1 / BUDGET.frames_per_second)  # noqa: SLF001
    seat = await live.take_control()
    assert seat.token is not None
    assert live._interval() == pytest.approx(1 / BUDGET.frames_per_second)  # noqa: SLF001
    await live.release_control(token=seat.token)
    assert live._interval() == pytest.approx(1 / BUDGET.frames_per_second)  # noqa: SLF001
    assert page.methods() == []


async def test_a_handover_sends_the_state_that_names_the_seat_and_no_new_encoding():
    live, page = session()
    reader = await _watching(live, page)
    assert page.encodes() == [
        {
            "format": "jpeg",
            "quality": BUDGET.jpeg_quality,
            "maxWidth": BUDGET.max_width,
            "everyNthFrame": 1,
        }
    ]

    seat = await live.take_control()
    assert seat.token is not None
    # A handover is recorded and told to every reader; the screencast runs on unchanged.
    handed = await asyncio.wait_for(anext(reader), 2)
    assert handed.driver == seat.driver and handed.control == "human"
    assert page.methods().count("Page.startScreencast") == 1

    await live.release_control(token=seat.token)
    handed_back = await asyncio.wait_for(anext(reader), 2)
    assert handed_back.driver is None and handed_back.control == "agent"
    assert page.methods().count("Page.startScreencast") == 1
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_every_update_carries_the_budget_the_stream_is_delivered_at():
    live, _page = session()
    reader = live.frames()

    first = await anext(reader)
    assert first.budget == BUDGET
    assert first.budget_fields() == {
        "frames_per_second": BUDGET.frames_per_second,
        "jpeg_quality": BUDGET.jpeg_quality,
        "max_width": BUDGET.max_width,
    }
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_session_whose_readers_are_behind_steps_its_budget_down():
    step = StepHarness()
    await step.subscribe()
    for _ in range(2):  # the windows in a row the hysteresis asks for
        await step.window(share=1.0)

    # Two windows where every frame displaced one nobody had read: the picture lightens.
    assert step.budget.jpeg_quality < STEP_BASE.jpeg_quality
    assert step.budget.frames_per_second == STEP_BASE.frames_per_second
    assert step.budget.max_width == STEP_BASE.max_width


async def test_a_session_that_has_been_clear_steps_its_budget_back_up():
    step = StepHarness()
    await step.subscribe()
    for _ in range(4):
        await step.window(share=1.0)
    lowered = step.budget
    assert lowered.jpeg_quality < STEP_BASE.jpeg_quality

    for _ in range(2):
        await step.window(share=0.0)

    assert step.budget.jpeg_quality > lowered.jpeg_quality


async def test_a_window_that_goes_either_way_does_not_move_the_rung():
    step = StepHarness()
    await step.subscribe()

    # Behind, clear, behind, clear: neither streak reaches two, so nothing moves. The
    # frame a slow reader costs arrives in bursts, and this is what keeps it from flapping.
    for share in (1.0, 0.0, 1.0, 0.0):
        await step.window(share=share)

    assert step.budget == STEP_BASE


async def test_the_step_down_stops_at_a_floor_that_still_shows_the_page():
    step = StepHarness()
    await step.subscribe()
    for _ in range(40):
        await step.window(share=1.0)

    # Forty windows of nothing but displaced frames: the ladder ends, and what is left is
    # a stream that still moves and still shows the page rather than nothing at all.
    assert step.budget.frames_per_second == 10.0
    assert step.budget.jpeg_quality == 45
    assert step.budget.max_width == STEP_BASE.max_width


async def test_a_stepped_down_session_climbs_back_to_the_budget_it_started_at():
    """The ladder's ceiling is the deployment's budget, not a number of the ladder's own.

    A deployment that configured more than the ladder's old 24 fps ceiling stepped down and
    never climbed back: ``up`` capped the rate at 24 whatever the base was, so a session
    starting at 60 descended to 10/45 and topped out at 24/45 for as long as it lived.
    """
    taller = LivePreviewBudget(frames_per_second=60.0, max_width=1600, jpeg_quality=80)
    step = StepHarness(base=taller)
    await step.subscribe()
    for _ in range(40):
        await step.window(share=1.0)
    assert step.budget.frames_per_second == 10.0
    assert step.budget.jpeg_quality == 45

    for _ in range(80):
        await step.window(share=0.0)

    # Back where it started: the rate first, then the quality, and never above the base.
    assert step.budget == taller


async def test_the_rate_a_session_climbs_towards_is_the_configured_one():
    taller = LivePreviewBudget(frames_per_second=30.0, max_width=1280, jpeg_quality=70)
    from_below = LivePreviewStep.up(
        LivePreviewBudget(frames_per_second=20.0, max_width=1280, jpeg_quality=45), taller
    )
    assert from_below.frames_per_second == 30.0

    # And a deployment configured below the ladder's own floor cannot be climbed above.
    cheaper = LivePreviewBudget(frames_per_second=4.0, max_width=800, jpeg_quality=50)
    assert (
        LivePreviewStep.up(
            LivePreviewBudget(frames_per_second=4.0, max_width=800, jpeg_quality=45), cheaper
        )
        == cheaper
    )


async def test_the_rung_a_session_logged_says_which_way_it_moved(caplog: pytest.LogCaptureFixture):
    """The rung log is what an operator reads afterwards, and docs/operations.md quotes it."""
    step = StepHarness()
    await step.subscribe()
    with caplog.at_level("INFO", logger="dlightrag.adapters.agent_browser.live"):
        for _ in range(2):
            await step.window(share=1.0)

    assert "stepped down to" in caplog.text
    assert f"at quality {step.budget.jpeg_quality}" in caplog.text


async def test_a_deployment_that_configures_less_than_the_ladder_keeps_its_own_ceiling():
    cheaper = LivePreviewBudget(frames_per_second=6.0, max_width=800, jpeg_quality=50)
    step = StepHarness(base=cheaper)
    await step.subscribe()
    for _ in range(4):
        await step.window(share=1.0)
    lowered = step.budget

    for _ in range(40):
        await step.window(share=0.0)

    # Stepping back up never overtakes what the deployment asked for in the first place.
    assert lowered.jpeg_quality < cheaper.jpeg_quality
    assert step.budget == cheaper


async def test_a_window_a_page_left_quiet_is_not_a_window_that_was_behind():
    step = StepHarness()
    await step.subscribe()
    await step.window(share=1.0)
    after_one = step.budget

    # The page goes quiet for longer than a window, which is not a session that is behind:
    # nothing arrived to displace anything, so the frames either side of the gap are not one
    # run of windows, and what the first one was does not count towards the second.
    await step.quiet(5.0)
    await step.window(share=1.0)
    assert step.budget == after_one

    await step.window(share=1.0)
    # Two windows after the gap, and not one before it plus one after: the ladder moved once.
    assert step.budget.jpeg_quality < after_one.jpeg_quality
    assert step.budget == LivePreviewStep.down(STEP_BASE, STEP_BASE)


async def test_stepping_down_re_encodes_the_running_screencast_at_the_new_quality():
    step = StepHarness(keep_watching=True)
    page = step.page
    await step.subscribe()
    await _watching_started(step.live, page)
    assert page.encodes()[-1]["quality"] == STEP_BASE.jpeg_quality

    for _ in range(2):
        await step.window(share=1.0)

    # The quality is the browser's own screencast parameter, so a rung is applied to the
    # screencast that is already running rather than waiting for the stream to be restarted.
    assert page.methods().count("Page.startScreencast") == 2
    assert page.encodes()[-1]["quality"] == step.budget.jpeg_quality


async def _watching_started(live: LiveBrowserSession, page: FakePage) -> None:
    """Wait until the subscription has started its screencast, as a reader's first frame does."""
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the subscription never started a screencast")


async def test_control_goes_back_once_the_person_stops_touching_the_page():
    recorded: list[LiveControlState] = []
    live, _page = session(record=recorded)
    seat = await live.take_control()
    assert seat.token is not None

    for _ in range(200):
        if live.control == "agent":
            break
        await asyncio.sleep(0.01)

    assert live.control == "agent"
    assert live.driver is None
    # The idle release spends the credential too: nobody may drive with a seat nobody holds.
    with pytest.raises(AgentBrowserError):
        await live.send(LiveInput(kind="key", action="press", key="Enter"), token=seat.token)
    assert recorded == ["human", "agent"]


async def test_an_expired_session_says_expired_rather_than_no_browser():
    """The cap on one live session is not a Run that never had a browser."""
    capped = LivePreviewSettings(
        budget=BUDGET,
        session_seconds=0.2,
        control_idle_seconds=30.0,
    )
    live, _page = session(settings=capped)
    reader = live.frames()
    assert (await anext(reader)).attached is True
    update = await asyncio.wait_for(anext(reader), 3)
    assert update.ended == "expired"
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(AgentBrowserError) as refused:
        await live.take_control()
    assert refused.value.reason == "expired"


async def test_a_quiet_page_does_not_hold_the_session_open_past_its_cap():
    capped = LivePreviewSettings(
        budget=BUDGET,
        session_seconds=0.2,
        control_idle_seconds=30.0,
    )
    live, _page = session(settings=capped)
    reader = live.frames()
    assert (await anext(reader)).attached is True

    update = await asyncio.wait_for(anext(reader), 3)
    assert update.ended == "expired"
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]
    assert live.attached is False


async def test_closing_a_session_stops_its_frames():
    live, page = session()
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)

    await live.close()
    for _ in range(200):
        if "Page.stopScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    assert "Page.stopScreencast" in page.methods()
    ended = await asyncio.wait_for(anext(reader), 2)
    assert ended.ended == "closed"
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(reader), 2)


async def test_the_last_reader_leaving_stops_the_capture_and_keeps_the_session():
    live, page = session()
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]
    for _ in range(200):
        if "Page.stopScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)

    # The session is the Run's, not the reader's: the next reader starts a screencast again.
    assert live.attached is True
    assert (await live.take_control()).control == "human"
    second = live.frames()
    assert (await anext(second)).control == "human"
    for _ in range(200):
        if page.methods().count("Page.startScreencast") == 2:
            break
        await asyncio.sleep(0.01)
    assert page.methods().count("Page.startScreencast") == 2
    await second.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_page_that_goes_away_ends_the_session_without_an_error():
    live, page = session()
    reader = live.frames()
    assert (await anext(reader)).attached is True
    for _ in range(200):
        if "Page.startScreencast" in page.methods():
            break
        await asyncio.sleep(0.01)

    page.close()
    update = await asyncio.wait_for(anext(reader), 3)
    assert update.ended != ""
    await reader.aclose()  # pyright: ignore[reportAttributeAccessIssue]


async def test_a_lease_that_ends_closes_the_session_of_its_readers():
    page = FakePage()
    surface = PlaywrightLivePage(cast(Any, _FakeBrowser(page)), LiveControl())

    class _SurfaceProvider:
        """A pool that leases nothing and hands one live surface back, as ``page_for`` does."""

        def page_for(self, holder: BrowserHolder) -> PlaywrightLivePage:
            return surface

    leases = FakeLeases(holder=HOLDER)
    sessions = LiveBrowserSessions(provider=_SurfaceProvider(), leases=leases, settings=SETTINGS)

    live = await sessions.open(owner_id=HOLDER.owner_id, run_id=HOLDER.run_id)
    reader = live.frames()
    assert (await anext(reader)).attached is True

    # The lease is gone, so the page it showed is gone: the session ends now, rather than
    # surviving until the next reader happens to open one.
    await sessions.retired(HOLDER)
    ended = await asyncio.wait_for(anext(reader), 2)
    assert ended.ended == "closed"
    assert ended.attached is False
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(reader), 2)

    # And a later open leases afresh rather than finding the closed session.
    again = await sessions.open(owner_id=HOLDER.owner_id, run_id=HOLDER.run_id)
    assert again is not live
    assert again.attached is True
    await sessions.aclose()


async def test_the_agent_is_refused_while_a_person_drives_the_run():
    control = LiveControl()
    provider = FakeProvider(live=FakeLivePage(control))
    run = RunAgentBrowser(provider, HOLDER, browser_settings())

    assert run.taken_over() is False
    await run.render("https://example.com/")
    control.take()
    assert run.taken_over() is True
    control.release()
    assert run.taken_over() is False


async def test_a_lease_record_that_cannot_be_written_does_not_stop_the_person_driving():
    async def failing(state: LiveControlState) -> None:
        raise ConnectionError("the database is down")

    live, _page = session()
    live._write_control = failing  # noqa: SLF001 - the store is the piece under test here

    assert (await live.take_control()).control == "human"
    assert live.control == "human"
