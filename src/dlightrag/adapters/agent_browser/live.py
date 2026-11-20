# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Run's live preview, and the co-drive that follows it.

The browser a Run leases is already there; this module turns the page it is driving into
frames a person can watch, and, on request, lets that person drive it. Four things decide
the shape of it:

- **Frames are demand-driven.** A CDP screencast exists only while somebody is watching:
  the first subscriber starts it, the last one to leave stops it, and an unwatched page is
  never encoded. Playwright delivers frames to a plain callback and this module acks one
  only as it hands it on, so the budget in force also bounds what the browser encodes.
- **Frames are change-driven, and budgeted, never paced.** An unchanged page produces no
  frame, so there is no idle rate to decay to; what is configured is a ceiling. One budget
  serves the whole session whoever drives, and the width and quality are the browser's own
  screencast parameters, so they are set once. A session whose viewers cannot keep up steps
  itself down a rung at a time — quality first, then the rate — and back up once they can;
  that is the only thing that ever changes what the page is encoded at.
- **One encode serves everybody.** The screencast is per page, not per subscriber; each
  subscriber holds one slot, and a frame that arrives while that slot is full replaces
  what is in it. A slow reader loses resolution, never memory: no queue grows. Those
  displacements are also how the session knows it is behind, which is why they are counted
  rather than merely dropped.
- **Every frame says which frame it is.** The browser hands a frame over with the moment it
  was captured, and the session stamps it with a number that only ever goes up, so a reader
  can drop a frame older than the one it has already drawn and say how far behind the page
  it is. Both travel in front of the JPEG rather than beside it, because the text channel
  carries state and nothing here may race it.
- **One driver.** The seat holds one viewer at a time, and the credential for it is minted
  when the seat is taken and answered only to the request that took it. A second viewer
  watches, sees that the seat is held in every update, and is refused if it asks to drive
  or presents anything but the credential the seat handed out: it can neither type into
  the page nor hand the seat back. Nothing that every reader can see is a credential.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import secrets
import struct
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

from dlightrag.adapters.agent_browser.leased_browser import PlaywrightLeasedBrowser
from dlightrag.engine.answer.agent_browser.contracts import (
    BrowserHolder,
    BrowserLeases,
    browser_failure,
    page_failure,
)
from dlightrag.engine.answer.agent_browser.live import (
    LiveControl,
    LiveControlState,
    LiveInput,
    LivePreviewBudget,
    LivePreviewSettings,
    LivePreviewStep,
    LiveSeat,
    LiveUpdate,
)
from dlightrag.engine.network_admission import (
    PublicHttpPolicyError,
    avalidate_public_http_url,
)

logger = logging.getLogger(__name__)

#: How long a co-driven navigation may take before the page is left as it stands.
_NAVIGATE_SECONDS = 30.0
#: Playwright takes a navigation timeout in milliseconds.
_NAVIGATE_MS = _NAVIGATE_SECONDS * 1000
#: The slowest a session ever delivers, whatever it is configured with.
_MIN_FRAMES_PER_SECOND = 0.1
#: How many input events' idempotency keys one session remembers, in memory.
_REMEMBERED_INPUT_KEYS = 64
#: How many bytes of secrets one seat mints: 256 bits of credential, and a label beside it.
_SEAT_TOKEN_BYTES = 32
_SEAT_LABEL_BYTES = 4
#: The frame header every delivered frame wears, in front of the JPEG bytes.
#:
#: ``!Qd`` — the sequence number a reader compares, then the capture time in epoch seconds
#: as a float64. Fixed-width and little ceremony on purpose: it is decoded by
#: ``DataView.getBigUint64``/``getFloat64`` on the client and by ``struct.unpack`` here, and
#: it cannot race the text channel the way a metadata message beside the frame could (a
#: message that arrived late would describe the wrong picture). ``FRAME_HEADER_BYTES`` is
#: stated for the reader of this module and for the tests that check the layout.
FRAME_HEADER = "!Qd"
FRAME_HEADER_BYTES = struct.calcsize(FRAME_HEADER)

#: What it takes for a session to move a rung: a window of frames, and how many of them.
#:
#: The window counts frames rather than seconds because every frame is evidence and a second
#: is not: a window that closed on the clock would decide on however many frames a page
#: happened to paint in it, and a quiet page would decide on none. Twenty-four frames is
#: about a second of a page that is repainting at the budget in force.
_STEP_WINDOW_FRAMES = 24
#: A window is behind when this share of its frames displaced one nobody had read. A quarter
#: of the frames arriving while every reader's slot is still full is visible in the picture,
#: and it is the point at which a lighter frame is worth more than the one being replaced.
_STEP_DOWN_BEHIND = 0.25
#: A window is clear at a twentieth: the small remainder is one frame arriving just as the
#: previous one was read, which no lighter budget makes disappear.
_STEP_UP_CLEAR = 0.05
#: The windows in a row a session must be behind, or clear, before a rung moves.
_STEP_WINDOWS = 2
#: How long a gap between frames makes the next one start a fresh window: a page that went
#: quiet is neither behind nor clear, and the frames either side of the gap are not one run.
_QUIET_SECONDS = 2.0


#: A subscriber's slot holds the newest event only; this tells its reader to stop.
_CLOSED = object()


class _Subscriber:
    """One reader's slot: the newest event it has not read yet, and nothing else."""

    __slots__ = ("_slot",)

    def __init__(self) -> None:
        self._slot: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)

    def offer(self, event: Any) -> bool:
        """Keep ``event`` as the newest one; True when it displaced an undelivered one.

        A full slot means this reader has not read the frame before this one yet, so what is
        dropped is a frame the viewer never saw. Nothing queues, and that count is the whole
        of what a session knows about whether its readers are keeping up.
        """
        displaced = self._slot.full()
        if displaced:
            self._slot.get_nowait()
        self._slot.put_nowait(event)
        return displaced

    async def next(self) -> Any:
        return await self._slot.get()


class _Pace:
    """Whether the readers of one session kept up, over the last whole window of frames.

    A frame that arrives while the previous one is still undelivered — in a reader's slot —
    is a frame that arrived faster than it could be handed on, and the share of those in a
    window is the whole of what a session knows about being asked for more than it is
    delivering. The window is a count of frames rather than a length of time, so a page that
    paints slowly is judged on the frames it did produce, and a page that goes quiet is
    judged on nothing at all.
    """

    __slots__ = ("_clock", "_window", "_since", "_delivered", "_displaced", "quiet")

    def __init__(self, clock: Callable[[], float], window: int = _STEP_WINDOW_FRAMES) -> None:
        self._clock = clock
        self._window = window
        self._since = clock()
        self._delivered = 0
        self._displaced = 0
        #: Whether the frame just counted arrived after a gap: a page that went quiet is
        #: neither behind nor clear, so the windows either side of the gap are not one run.
        self.quiet = False

    def frame(self, *, displaced: bool) -> str:
        """Count one delivered frame; ``behind`` or ``clear`` when this one closed a window.

        A frame more than ``_QUIET_SECONDS`` after the last one starts a fresh window, and
        says so in ``quiet``: what happened before a page went quiet is not evidence about
        what it is doing now.
        """
        now = self._clock()
        self.quiet = now - self._since > _QUIET_SECONDS
        if self.quiet:
            self._delivered = 0
            self._displaced = 0
        self._since = now
        self._delivered += 1
        self._displaced += int(displaced)
        if self._delivered < self._window:
            return ""
        share = self._displaced / self._delivered
        self._delivered = 0
        self._displaced = 0
        if share >= _STEP_DOWN_BEHIND:
            return "behind"
        return "clear" if share < _STEP_UP_CLEAR else ""


def _screencast_params(budget: LivePreviewBudget) -> dict[str, Any]:
    """The parameters one screencast is started, or reconfigured, with."""
    return {
        "format": "jpeg",
        "quality": budget.jpeg_quality,
        "maxWidth": budget.max_width,
        "everyNthFrame": 1,
    }


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame the page encoded, with what a reader needs to place it in time."""

    jpeg: bytes
    """The JPEG the browser encoded, to the byte."""
    frame_id: int
    """CDP's own id for it, which is what acknowledging it names."""
    captured_at: float
    """When the page was captured, in epoch seconds, as the browser's metadata reported it."""
    sequence: int = 0
    """This session's own count of the frames it delivered, oldest first, never repeated.

    The count belongs to the stream rather than to the capture, so it is written as the
    frame is handed to a reader: a page that encoded three frames while this session
    delivered two numbers them 1 and 2, and the one nobody was sent carries none. What a
    reader reads off the wire is therefore one consecutive count, never a gap and never a
    number twice — the property its own ``#received`` drops a frame by.
    """

    def wire(self) -> bytes:
        """The frame as it is delivered: the header, then the JPEG bytes themselves.

        The header is ``!Qd`` — the sequence number, then the capture time — in front of the
        JPEG. Every transport hands these bytes on unchanged, so the one encode the page
        paid for is still the last copy of it.
        """
        return struct.pack(FRAME_HEADER, self.sequence, self.captured_at) + self.jpeg


class _Screencast:
    """The CDP screencast of one page, acked when its frame is superseded, never on receipt.

    The ack is the flow control: until a frame is acknowledged the browser will not spend
    another encode, exactly as the protocol documents, so *when* this module acks is what
    decides whether anything can be waiting behind the frame in hand. Acknowledging as a
    frame is taken lets the browser produce while the reader is still being served, and
    whatever it produces has to be held somewhere between here and the page. Acknowledging
    it as the next frame supersedes it means this session never has more than one frame
    outstanding: the newest frame is the one in hand, the one it replaced is gone, and the
    page cannot run ahead of the stream by more than the one encode it is next asked for.

    Measured rather than argued, on the bundled pool: a page that repainted a gradient
    through every pixel at 1280×720 painted at 59.8 fps, was delivered at 22.8 fps, and the
    frame a reader received was a median of 33.3 ms and a p95 of 50.0 ms old — two to three
    painted frames — under this policy *and* under ack-on-receipt, identically. So the
    backlog this policy removes was not what made that page's preview late; what it removes
    is the possibility of one, which is not a thing a measurement of one page can promise.

    CDP hands a frame over as base64 text, and it is decoded here, once, at the boundary
    where the browser's protocol ends: what every reader and every transport then handles
    is the JPEG the browser encoded, to the byte.

    An acknowledgement may be lost without harm: a screencast that is reconfigured — which
    is what a rung is — has its last frame delivered again, so the frame this module let go
    of unacknowledged is the frame it is handed next.
    """

    def __init__(self, page: Page) -> None:
        self._page = page
        self._session: Any = None
        self._ready = asyncio.Event()
        self._pending: Frame | None = None
        self._failed = False
        self._budget: LivePreviewBudget | None = None

    async def start(self, budget: LivePreviewBudget) -> None:
        # What this stream is: CDP's screencast is a change-driven preview, not a video
        # pipeline, and what it carries depends on what the page makes it encode rather
        # than on a rate of its own. Measured through the real frames socket on the bundled
        # pool, a page repainting every 50 ms arrives at 93–97 % of the budget in force:
        # 11.6 fps under a budget of 12, 22.6 under 24 and 27.9 under 30, over 15 s windows,
        # the rest being this acknowledge-and-pace loop. Bytes are the cost that grows with
        # the page, not frames: a canvas whose every pixel changes was still delivered at
        # 20 fps under a 24 fps budget, but at 430 kB a frame, 8.4 MB/s and more than one
        # core of encoding. What it cannot see at any rate is what is not the page — an
        # OS-level dialog, a PDF viewer, a native picker never appear in it — and showing
        # those needs a headful browser and a WebRTC media path, which is a different
        # deployment and not a bigger budget.
        session = await self._page.context.new_cdp_session(self._page)
        self._session = session
        session.on("Page.screencastFrame", self._on_frame)
        # A page that closes is the end of the wait, not something to wait out.
        self._page.on("close", self._on_close)
        self._budget = budget
        try:
            await session.send("Page.startScreencast", _screencast_params(budget))
        except BaseException:
            await self.stop()
            raise

    async def reconfigure(self, budget: LivePreviewBudget) -> None:
        """Ask the browser to encode the same screencast at a new width and quality.

        Sending ``Page.startScreencast`` again reconfigures the screencast that is already
        running: the ack flow, and the frame the page is in the middle of, survive it. A
        session that steps down or back up calls this once per rung; no frame ever does.
        """
        session = self._session
        if session is None or budget == self._budget:
            return
        self._budget = budget
        with suppress(PlaywrightError):
            await session.send("Page.startScreencast", _screencast_params(budget))

    async def stop(self) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        with suppress(PlaywrightError):
            # The page, its context, or the browser went with the screencast.
            await session.send("Page.stopScreencast")
        with suppress(PlaywrightError):
            await session.detach()
        self._pending = None
        self._budget = None
        self._ready.set()

    async def newest(self, acknowledges: int | None = None) -> Frame | None:
        """The next frame, acknowledging the one it supersedes, or None once the page is gone.

        Nothing is acknowledged until a frame arrives to supersede it, so the browser's own
        flow control — not this session's hope — is what bounds how far ahead the page may
        run. ``acknowledges`` is the frame the caller has finished with, and it is let go of
        here rather than at the moment it was taken, which is the whole of the difference
        between a preview that follows the page and one that follows its own backlog.
        """
        if acknowledges is not None:
            await self._ack(acknowledges)
        while True:
            if self._failed:
                return None
            pending = self._pending
            if pending is not None:
                self._pending = None
                return pending
            if self._page.is_closed():
                return None
            # Clear, then look again: a frame that arrived between the two would otherwise
            # leave a set event with nothing behind it, and this wait would sit on it.
            self._ready.clear()
            if self._pending is not None:
                continue
            await self._ready.wait()

    async def _ack(self, frame_id: int) -> None:
        """Acknowledge one frame, which is what lets the browser spend its next encode."""
        session = self._session
        if session is None:
            return
        try:
            await session.send("Page.screencastFrameAck", {"sessionId": frame_id})
        except PlaywrightError:
            # The page, its context, or the browser went with the screencast. What is
            # waiting is not going to be read, and the session ends on the next look.
            self._failed = True

    def _on_frame(self, event: dict[str, Any]) -> None:
        data = event.get("data")
        if not isinstance(data, str):
            return
        try:
            # The browser's own base64, decoded once for every reader this frame will reach.
            jpeg = base64.b64decode(data, validate=True)
        except binascii.Error, ValueError:
            logger.info("A live browser frame arrived that was not base64 JPEG")
            return
        #: The moment the page was captured, straight off the frame's own metadata: a reader
        #: cannot see it any other way, and it is what says how old the picture already was
        #: when this session was handed it. The frame carries no sequence number yet: the
        #: count is the session's own, and it is written as the frame is delivered — a frame
        #: this capture supersedes before anybody reads it is never numbered.
        self._pending = Frame(
            jpeg=jpeg,
            frame_id=int(event.get("sessionId") or 0),
            captured_at=self._captured_at(event),
        )
        self._ready.set()

    def _captured_at(self, event: dict[str, Any]) -> float:
        """When the page was captured, as epoch seconds, from the frame's own metadata.

        CDP states the moment in *seconds* since the epoch, and the header carries it as it
        stands: read against the bundled pool, a frame's ``metadata.timestamp`` was
        1791948557.2566 while this process's ``time.time()`` was 1791948559.6900, and the
        metadata carried no monotonic reading beside it — one clock, so a reader's own
        ``Date.now()`` can be subtracted from what arrives with no offset to calibrate.

        Reading that number as milliseconds and dividing it by a thousand is a mistake with
        a symptom rather than a subtlety: the header then says 1970, and a dock's lag
        readout — the distance between the capture and its own clock — says 1.8 billion
        seconds. What a frame's unit has to satisfy is that distance, so the test's fake
        speaks the protocol's own seconds.

        A frame whose metadata carries no such reading — a browser that reports none — is
        stamped with the moment it arrived instead, which is a little later and never
        earlier, so a reader is never told a frame is newer than it is.
        """
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            stamp = metadata.get("timestamp")
            if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp > 0:
                return float(stamp)
        return time.time()

    def _on_close(self, _page: Any = None) -> None:
        self._pending = None
        self._ready.set()


class PlaywrightLivePage:
    """The live surface of one Run's browser: the page it drives, and who drives it.

    ``PlaywrightLeasedBrowser`` opens and closes its Run's Agent Pages; this reads the page
    that is open now, so a session follows the Run from page to page and ends when the Run
    has none. It is what ``PooledBrowserProvider.page_for`` answers with.
    """

    __slots__ = ("_browser", "_control")

    def __init__(self, browser: PlaywrightLeasedBrowser, control: LiveControl) -> None:
        self._browser = browser
        self._control = control

    @property
    def control(self) -> LiveControl:
        """The seat itself, which only the live session takes and releases."""
        return self._control

    def taken_over(self) -> bool:
        """Whether a person holds this browser's driving seat now."""
        return self._control.taken_over

    def take_drive_notice(self) -> bool:
        """Whether a person has driven it since this was last asked, which answers it once."""
        return self._control.take_drive_notice()

    def page(self) -> Page | None:
        """The Playwright page the Run drives now, or None while it drives none."""
        return self._browser.live_page()


class LiveBrowserSession:
    """One Run's live preview: its page's frames, and the driving seat of that page.

    A session is bound to a Run by its owner and its id, and to the browser the Run has
    leased: it never leases one, never renders one, and never releases one. Its frames come
    from the page the Run is driving, its input goes to that same page, and the lease it
    sits on ends the session with it.
    """

    def __init__(
        self,
        *,
        owner_id: str,
        run_id: str,
        page: PlaywrightLivePage,
        settings: LivePreviewSettings,
        write_control: Callable[[LiveControlState], Awaitable[None]],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._owner_id = owner_id
        self._run_id = run_id
        self._page = page
        self._settings = settings
        self._write_control = write_control
        self._clock = clock
        self._control = page.control
        self._subscribers: list[_Subscriber] = []
        self._pump: asyncio.Task[None] | None = None
        self._screencast: _Screencast | None = None
        self._idle: asyncio.Task[None] | None = None
        self._input = asyncio.Event()
        self._deadline = clock() + settings.session_seconds
        #: Who holds the driving seat: the label every reader is told, and the credential
        #: only the request that took the seat is answered with. The two are separate on
        #: purpose — the label is broadcast and the credential never is — and both are
        #: cleared the moment the seat goes back to the agent.
        self._driver: str | None = None
        self._seat: str | None = None
        self._ended = ""
        self._closed = False
        #: What this session delivers at now: the configured budget until its viewers fall
        #: behind, and one rung back up whenever they stop being behind.
        self._budget = settings.budget
        self._pace = _Pace(clock)
        self._behind_windows = 0
        self._clear_windows = 0
        #: The frames this session has delivered, counted once each and never repeated, so a
        #: reader can tell a frame older than the one it drew from the one that follows it.
        self._sequence = 0
        #: The input events already run, newest last, so a retried one runs once. In memory
        #: only: an input event is never written anywhere (session cap: 64 keys).
        self._keys: dict[str, None] = {}
        #: The highest input sequence this seat has run, so an event the wire overtook is
        #: dropped rather than run over the state a later one already left.
        self._input_sequence = 0

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def control(self) -> LiveControlState:
        return self._control.state

    @property
    def driver(self) -> str | None:
        """The seat holder's display label, or None while the agent drives.

        A label and not a credential: it travels in every update, which is to say that every
        reader of the Run sees it. What drives the page is ``_seat``, and that is written
        down in exactly one place — the answer to the request that took the seat.
        """
        return self._driver if self._control.taken_over else None

    @property
    def attached(self) -> bool:
        """Whether the Run drives a page this session can show right now."""
        return not self._closed and self._page.page() is not None

    async def frames(self) -> AsyncGenerator[LiveUpdate]:
        """The newest frame of the Run's page, for as long as this reader is subscribed.

        The first subscriber starts the screencast and the last one to leave stops it, so a
        page nobody watches is never encoded. What a slow reader misses is dropped, not
        queued.
        """
        if self._closed:
            # The session is already over: say why, and subscribe to nothing.
            yield self._update()
            return
        subscriber = _Subscriber()
        self._subscribers.append(subscriber)
        self._start_pump()
        try:
            yield self._update()
            while True:
                event = await subscriber.next()
                if event is _CLOSED:
                    # The session is over: say why once, then end this reader's stream.
                    if self._ended:
                        yield self._update()
                    return
                yield self._update(event)
                if self._ended:
                    return
        finally:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
            if not self._subscribers:
                await self._stop_pump()

    async def take_control(self) -> LiveSeat:
        """Give the driving seat to the caller, minting the credential that holds it.

        The seat is one, and the credential is the whole of who holds it: this mints a fresh
        secret as the request is answered, and writes it down nowhere else. A caller arriving
        at a seat somebody — or somebody's earlier request — already holds is refused as
        ``taken_over`` and stays an observer, whether or not it is in fact the holder: there
        is nothing in a request that would let this session tell the two apart, and a seat
        that could be re-claimed by asking again would hand its credential to any reader of
        the Run that knew when to ask. What the holder does instead is what it took the seat
        for — driving — and handing it back is what gives it up.
        """
        self._require_live()
        if self._control.taken_over:
            raise page_failure("taken_over")
        self._control.take()
        self._driver = self._seat_label()
        self._seat = secrets.token_urlsafe(_SEAT_TOKEN_BYTES)
        # A new seat is a new sender: the events the last holder numbered say nothing about
        # the ones this one will send, so its first event is its first again.
        self._input_sequence = 0
        await self._publish_control("human")
        self._input.clear()
        self._idle = asyncio.create_task(
            self._release_when_idle(), name="live-browser-control-idle"
        )
        return LiveSeat(control=self._control.state, driver=self._driver, token=self._seat)

    async def release_control(self, *, token: str) -> LiveControlState:
        """Give the driving seat back to the agent, for the credential that holds it.

        ``token`` is the credential the take answered that one request with: the holder hands
        its own seat back with it, and nothing else does. A token that is not the holder's —
        another reader's, the display label every reader is shown, an empty string — is
        refused as ``taken_over``, so a reader that only watched the stream can neither take
        the seat nor give somebody else's back. Handing back burns the credential: the seat
        taken next is a different one, and asking again with the spent one changes nothing.

        The session's own releases — its end, and the idle timer that ends a hold nobody is
        using — do not come through here: they present no credential, which is what the
        private ``_release`` beside this is for.
        """
        if self._control.taken_over and token != self._seat:
            raise page_failure("taken_over")
        return await self._release()

    async def _release(self) -> LiveControlState:
        """Give the seat back to the agent whoever holds it, and wake every reader.

        This is the session's own path rather than a request's: the end of the session and
        the idle release both arrive here with nobody left to present a credential. It is
        private for exactly that reason — a caller that could reach it would be a hand-back
        that needs no credential at all.
        """
        idle, self._idle = self._idle, None
        if idle is not None and idle is not asyncio.current_task():
            idle.cancel()
        if self._control.release():
            self._driver = None
            self._seat = None
            await self._publish_control("agent")
        return self._control.state

    async def send(self, event: LiveInput, *, token: str, key: str = "") -> None:
        """Run one input event on the page, for the credential that holds control of it.

        ``token`` is the credential the seat handed out, never a reader's own name for
        itself: a viewer that only watched the stream has no credential at all, which is
        what makes the broadcast state useless for driving. ``key`` is the caller's
        idempotency key: an event already run under it is a no-op, which is what a retried
        POST after a timeout should be. An event numbered at or below one this seat has
        already run is dropped for ordering rather than run: it describes a page that the
        event which overtook it has already changed.
        """
        self._require_live()
        if not self._control.taken_over or self._seat is None or token != self._seat:
            raise page_failure("taken_over")
        if key and key in self._keys:
            return
        if event.sequence and event.sequence <= self._input_sequence:
            return
        page = self._page.page()
        if page is None:
            raise page_failure("no_page")
        await self._dispatch(page, event)
        if event.sequence:
            self._input_sequence = event.sequence
        if key:
            self._keys[key] = None
            if len(self._keys) > _REMEMBERED_INPUT_KEYS:
                self._keys.pop(next(iter(self._keys)))
        self._input.set()

    async def close(self) -> None:
        """End the session: stop the screencast and give the driving seat back.

        The browser stays the Run's: a live session never releases a lease.
        """
        if self._closed:
            return
        self._closed = True
        self._ended = self._ended or "closed"
        await self._release()
        await self._stop_pump()
        self._end_subscribers()

    async def _finish(self, reason: str) -> None:
        """End the session from its own capture loop, which must not cancel itself."""
        self._ended = self._ended or reason
        self._closed = True
        await self._release()
        self._end_subscribers()

    # ---- internals -------------------------------------------------------

    def _require_live(self) -> None:
        """Refuse a write to a session that is over, in the words that say which way.

        An expired session is not a Run without a browser: the browser is there and the cap
        on one live session is what ran out, so it says ``expired`` and the reader is told
        to start a fresh preview rather than that this Run never had a page.
        """
        if self._clock() >= self._deadline:
            raise browser_failure("expired")
        if self._closed:
            raise browser_failure("not_configured")

    @staticmethod
    def _seat_label() -> str:
        """A fresh display label for whoever is taking the seat.

        It is what every reader is told — ``viewer-3f9a1c04`` — and it is deliberately not a
        credential: it is short, it is broadcast, and it is minted again for every holder so
        that no reader can be mistaken for the one before it. The credential the holder
        drives with is ``secrets``-minted and never leaves the answer to the request.
        """
        return f"viewer-{secrets.token_hex(_SEAT_LABEL_BYTES)}"

    async def _publish_control(self, state: LiveControlState) -> None:
        """Announce a control handover: record it for operators, and wake every reader.

        The seat is authority over the page, not an allowance of frames, so a handover does
        not change what anybody is sent: every reader is woken with the state that names who
        drives now.
        """
        try:
            await self._write_control(state)
        except Exception:
            # The record is for operators; a store that is down does not stop the person
            # who is driving, and it does not stop the agent from standing down.
            logger.warning("Failed to record live browser control state", exc_info=True)
        for subscriber in tuple(self._subscribers):
            subscriber.offer(None)

    async def _release_when_idle(self) -> None:
        """Hand the driving seat back once nobody has touched the page for long enough.

        Nobody is here to present a credential: the hold ended because its holder stopped
        using it, so this is the session's own release rather than a hand-back.
        """
        while True:
            try:
                await asyncio.wait_for(self._input.wait(), self._settings.control_idle_seconds)
            except TimeoutError:
                break
            self._input.clear()
        await self._release()

    def _start_pump(self) -> None:
        if (self._pump is None or self._pump.done()) and not self._closed:
            self._pump = asyncio.create_task(self._pump_frames(), name="live-browser-frames")

    async def _stop_pump(self) -> None:
        pump, self._pump = self._pump, None
        if pump is None:
            return
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)

    async def _pump_frames(self) -> None:
        """The one screencast of this Run's page, and the pace its frames are handed on at.

        One frame is in hand at a time and the browser is not asked for another until this
        loop asks for it: the frame the reader has finished with is acknowledged as the next
        one is waited on, so what Chromium may queue ahead of this session is bounded by how
        fast the session delivers rather than by how fast the page paints.
        """
        page = self._page.page()
        if page is None:
            await self._finish("detached")
            return
        screencast = _Screencast(page)
        try:
            await screencast.start(self._budget)
        except PlaywrightError:
            logger.warning("Failed to start a live browser screencast", exc_info=True)
            await self._finish("unreachable")
            return
        self._screencast = screencast
        #: The frame this loop has finished with, acknowledged as the next one is waited on.
        acknowledged: int | None = None
        reason = ""
        try:
            while not reason:
                remaining = self._deadline - self._clock()
                if remaining <= 0:
                    reason = "expired"
                    break
                try:
                    # The wait is bounded by the session's cap, and a cancelled wait is the
                    # cap arriving: nothing inside the screencast turns it into a silence.
                    frame = await asyncio.wait_for(screencast.newest(acknowledged), remaining)
                except TimeoutError:
                    reason = "expired"
                    break
                except PlaywrightError:
                    reason = "unreachable"
                    break
                if frame is None:
                    reason = "detached"
                    break
                await self._deliver(frame)
                acknowledged = frame.frame_id
                await asyncio.sleep(self._interval())
        except asyncio.CancelledError:
            # The last reader left: the capture stops and the session stays open, so the
            # next one starts a screencast again rather than finding a finished session.
            raise
        finally:
            self._screencast = None
            await screencast.stop()
        await self._finish(reason)

    async def _deliver(self, frame: Frame) -> None:
        """Hand one frame to every reader, and move a rung when the windows say so.

        A reader whose slot is still full never saw the frame before this one, which is the
        whole of the evidence that this session is being asked for more than it is handing
        on. What each reader does with that is backpressure's business — it keeps the newest
        frame and no queue — and what the session does with it is ``_step``.
        """
        self._sequence += 1
        delivered = replace(frame, sequence=self._sequence)
        displaced = False
        for subscriber in tuple(self._subscribers):
            displaced = subscriber.offer(delivered) or displaced
        verdict = self._pace.frame(displaced=displaced)
        if self._pace.quiet:
            # A page that went quiet since the last window says nothing about whether its
            # readers are keeping up, so it ends the run rather than joining it.
            self._behind_windows = 0
            self._clear_windows = 0
        if verdict in {"behind", "clear"}:
            await self._step(verdict)

    async def _step(self, verdict: str) -> None:
        """Move the session one rung, once the windows in a row all agree.

        One window is a hiccup, and both directions need ``_STEP_WINDOWS`` of them, so a
        session that is behind and clear by turns stays where it is instead of flapping. The
        rung is logged once as it changes, which is what an operator reads afterwards.
        """
        if verdict == "behind":
            self._clear_windows = 0
            self._behind_windows += 1
            if self._behind_windows < _STEP_WINDOWS:
                return
            stepped = LivePreviewStep.down(self._budget, self._settings.budget)
        else:
            self._behind_windows = 0
            self._clear_windows += 1
            if self._clear_windows < _STEP_WINDOWS:
                return
            stepped = LivePreviewStep.up(self._budget, self._settings.budget)
        self._behind_windows = 0
        self._clear_windows = 0
        if stepped == self._budget:
            # The floor, or the budget the deployment configured: nothing left to move, and
            # nothing to log — a session that sits at the floor is at the floor, not noisy.
            return
        logger.info(
            "Live browser session stepped %s to %g fps at quality %d",
            "down" if verdict == "behind" else "up",
            stepped.frames_per_second,
            stepped.jpeg_quality,
        )
        self._budget = stepped
        screencast = self._screencast
        if screencast is not None:
            # The quality is the browser's own screencast parameter, so the rung is applied
            # to the running screencast rather than to the frames the session lets through.
            await screencast.reconfigure(stepped)
        for subscriber in tuple(self._subscribers):
            subscriber.offer(None)

    def _interval(self) -> float:
        """How long to wait between frames: the ceiling of the budget in force.

        A budget is a ceiling and not a pace: frames arrive when the page changes, so this
        only bounds how many of them one watcher is sent, and a quiet page still sends none.
        """
        return 1.0 / max(self._budget.frames_per_second, _MIN_FRAMES_PER_SECOND)

    def _update(self, frame: Frame | None = None) -> LiveUpdate:
        """One state update, carrying the frame it describes in the bytes a reader reads.

        A frame goes out as its header and its JPEG — the sequence number and the moment it
        was captured, then what the browser encoded — so what a transport sends and what a
        reader decodes are the same bytes, whether they arrived together or not.
        """
        return LiveUpdate(
            control=self._control.state,
            attached=self.attached,
            frame=None if frame is None else frame.wire(),
            # The count the frame's own header carries, and for a frame-less update the
            # newest one this session has delivered: one number a reader can compare both
            # against, so "nothing newer" is a fact it can read rather than infer.
            sequence=self._sequence,
            driver=self.driver,
            budget=self._budget,
            at=time.time(),
            ended=self._ended,
        )

    def _end_subscribers(self) -> None:
        for subscriber in tuple(self._subscribers):
            subscriber.offer(_CLOSED)

    async def _dispatch(self, page: Page, event: LiveInput) -> None:
        """Run one event on the page the Run drives, as the driver's own browser would."""
        mouse = page.mouse
        try:
            match event.kind:
                case "mouse":
                    await self._mouse(mouse, event)
                case "wheel":
                    await mouse.move(event.x, event.y)
                    await mouse.wheel(event.delta_x, event.delta_y)
                case "key":
                    await self._key(page, event)
                case "navigate":
                    await self._navigate(page, event)
        except PlaywrightError as exc:
            logger.info("A live browser input event failed (%s)", type(exc).__name__)
            raise page_failure(
                "action_failed",
                action=event.action or event.kind,
                target="the page",
                detail=type(exc).__name__,
            ) from exc

    async def _mouse(self, mouse: Any, event: LiveInput) -> None:
        match event.action:
            case "move":
                await mouse.move(event.x, event.y)
            case "down":
                await mouse.move(event.x, event.y)
                await mouse.down(button=event.button, click_count=event.click_count)
            case "up":
                await mouse.up(button=event.button, click_count=event.click_count)
            case _:
                await mouse.click(
                    event.x, event.y, button=event.button, click_count=event.click_count
                )

    async def _key(self, page: Page, event: LiveInput) -> None:
        match event.action:
            case "down":
                await page.keyboard.down(event.key)
            case "up":
                await page.keyboard.up(event.key)
            case "type":
                await page.keyboard.insert_text(event.text)
            case _:
                await page.keyboard.press(event.key)

    async def _navigate(self, page: Page, event: LiveInput) -> None:
        """A co-driven navigation is admitted on the same terms as the agent's own."""
        try:
            await avalidate_public_http_url(event.url)
        except PublicHttpPolicyError as exc:
            raise page_failure("navigation_failed", detail=str(exc)) from exc
        await page.goto(event.url, wait_until="load", timeout=_NAVIGATE_MS)


class LiveBrowserSessions:
    """Every live session this process serves: at most one per Run, shared by its readers.

    The session list is what makes one screencast serve every reader of one page. A Run
    this process holds no browser for has no session, and opening one says why: the Run
    holds no lease, or the browser belongs to another process. A lease that ends here ends
    its session too, through ``retired``, so a watcher is told rather than left looking at
    a page that is gone.
    """

    def __init__(
        self,
        *,
        provider: Any,
        leases: BrowserLeases,
        settings: LivePreviewSettings,
    ) -> None:
        self._provider = provider
        self._leases = leases
        self._settings = settings
        self._sessions: dict[tuple[str, str], LiveBrowserSession] = {}

    async def open(self, *, owner_id: str, run_id: str) -> LiveBrowserSession:
        """The live session of one Run, opened for a Run this owner holds a lease on."""
        key = (owner_id, run_id)
        existing = self._sessions.get(key)
        if existing is not None and existing.attached:
            return existing
        if existing is not None:
            await self.close(key)
        holder = await self._holder(owner_id, run_id)
        if holder is None:
            raise browser_failure("not_configured")
        page = self._provider.page_for(holder)
        if page is None:
            raise browser_failure("unreachable")
        session = LiveBrowserSession(
            owner_id=owner_id,
            run_id=run_id,
            page=page,
            settings=self._settings,
            write_control=lambda state: self._write_control(holder, state),
        )
        self._sessions[key] = session
        return session

    async def retired(self, holder: BrowserHolder) -> None:
        """Close the session of a Run whose lease just ended in this process.

        The page the session was showing is gone with the lease, so its readers are told the
        session is over now instead of at the next ``open()``.
        """
        await self.close((holder.owner_id, holder.run_id))

    async def close(self, key: tuple[str, str]) -> None:
        session = self._sessions.pop(key, None)
        if session is not None:
            await session.close()

    async def aclose(self) -> None:
        for key in tuple(self._sessions):
            await self.close(key)

    async def _holder(self, owner_id: str, run_id: str) -> BrowserHolder | None:
        try:
            return await self._leases.holder_for(owner_id=owner_id, run_id=run_id)
        except Exception as exc:
            logger.error("Live browser lease lookup failed (%s)", type(exc).__name__)
            raise browser_failure("unreachable") from exc

    async def _write_control(self, holder: BrowserHolder, state: LiveControlState) -> None:
        await self._leases.set_control(holder, state)


__all__ = [
    "LiveBrowserSession",
    "LiveBrowserSessions",
    "LiveInput",
    "LiveUpdate",
    "PlaywrightLivePage",
]
