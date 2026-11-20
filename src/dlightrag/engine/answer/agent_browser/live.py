# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Who drives a Run's browser while a person is watching it.

A live session shows the page a Run already leased and, on request, hands the person
watching it the driving seat. The seat is one flag on the Run's browser: while a person
holds it, the ``browser`` tool refuses every call with ``taken_over`` and the Run goes on.
The flag lives here, in the Engine, because two sides read it — the session that a person
takes control through, and the tool that must then stand down — and neither may import the
other (ADR 0032).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Protocol

type LiveControlState = Literal["agent", "human"]
"""Who drives a Run's browser: its agent, or the person watching it."""


@dataclass(frozen=True, slots=True)
class LivePreviewBudget:
    """What one watcher may cost while a page is watched.

    The frame rate is a ceiling, not a target: the page's own screencast is change-driven,
    so an unchanged page produces no frame at all. Width and quality are what the browser
    is asked to encode at. Who drives does not change any of the three — a person looking
    at their own input is served the same frames as one watching the agent — so the budget
    moves only when the session itself cannot keep it up.
    """

    frames_per_second: float
    """The most frames one watcher is sent per second."""
    max_width: int
    """The widest frame the browser encodes; the page's own aspect ratio is kept."""
    jpeg_quality: int
    """The JPEG quality the browser encodes each frame with."""


#: How far a session may step down from its configured budget, and in what order.
#:
#: Quality goes first because a page is read rather than watched: text and controls stay
#: legible at a lower quality, while a lower rate makes every scroll and click feel late,
#: and it is the rate a person driving the page is looking at. So a session that cannot
#: keep up lightens the frame before it slows it down, and only a session still behind at
#: the quality floor gives up frames — 24 to 15 is the last rung before the floor, which is
#: a stream that still moves.
_MIN_STEP_QUALITY = 45
_MIN_STEP_FRAMES_PER_SECOND = 10.0


@dataclass(frozen=True, slots=True)
class LivePreviewSettings:
    """What a deployment spends on a live preview, from ``answer.agent.browser.live``.

    One budget for the whole session, whoever drives: a reader who has taken the seat sees
    the same page at the same rate as one watching the agent, because the seat is authority
    over the page rather than a bigger allowance of frames. Nothing here is an idle rate:
    frames are change-driven, so an idle page is not a slow stream but no stream. The budget
    is where the session starts; a page whose viewers cannot keep up is stepped down from it
    by the session itself, and stepped back up once they can.
    """

    budget: LivePreviewBudget
    """What one watcher costs while the session can deliver it."""
    session_seconds: float
    """The hard cap on one live session, whatever it is doing."""
    control_idle_seconds: float
    """How long a person may hold control without touching the page before it goes back."""


class LivePreviewStep:
    """One rung of the ladder a session steps down and back up.

    It answers two questions about a budget — the next rung below it, and the next above —
    and nothing else, so the session that uses it holds no ladder of its own. The step is a
    property of the mechanism rather than a setting: what a deployment configures is where
    the ladder starts, and only a session that cannot deliver that walks down it.
    """

    @staticmethod
    def down(budget: LivePreviewBudget, base: LivePreviewBudget) -> LivePreviewBudget:
        """One rung below ``budget``, never above what the deployment configured."""
        if budget.jpeg_quality > _MIN_STEP_QUALITY:
            lower = replace(
                budget, jpeg_quality=max(round(budget.jpeg_quality * 0.75), _MIN_STEP_QUALITY)
            )
        else:
            lower = replace(
                budget,
                frames_per_second=max(
                    round(budget.frames_per_second * 0.625, 1), _MIN_STEP_FRAMES_PER_SECOND
                ),
            )
        return _clamp(lower, base)

    @staticmethod
    def up(budget: LivePreviewBudget, base: LivePreviewBudget) -> LivePreviewBudget:
        """One rung above ``budget``; the frame rate comes back before the quality does.

        The rate climbs towards the one the deployment configured, not towards a number of
        the ladder's own: a session configured above the ladder's old ceiling would
        otherwise never climb back to where it started.
        """
        if budget.frames_per_second < base.frames_per_second:
            higher = replace(
                budget,
                frames_per_second=min(
                    round(budget.frames_per_second * 1.6, 1), base.frames_per_second
                ),
            )
        else:
            higher = replace(budget, jpeg_quality=round(budget.jpeg_quality / 0.75))
        return _clamp(higher, base)


def _clamp(budget: LivePreviewBudget, base: LivePreviewBudget) -> LivePreviewBudget:
    """Keep a stepped budget between where the deployment started and the floor.

    A rung is never more expensive than the configured budget — a deployment that runs
    cheaper than the ladder's own numbers keeps its own ceiling — and never cheaper than
    ``_MIN_STEP_QUALITY`` and ``_MIN_STEP_FRAMES_PER_SECOND``.
    """
    return LivePreviewBudget(
        frames_per_second=min(
            max(budget.frames_per_second, _MIN_STEP_FRAMES_PER_SECOND), base.frames_per_second
        ),
        max_width=min(budget.max_width, base.max_width),
        jpeg_quality=min(max(budget.jpeg_quality, _MIN_STEP_QUALITY), base.jpeg_quality),
    )


class LiveControl:
    """The driving seat of one Run's browser: the agent holds it until a person takes it.

    Two facts, no I/O, and no clock: who holds the seat now, which a session takes and
    releases and the Run's ``browser`` tool reads before every call, and whether a person has
    held it since the agent was last told, which the tool reads once with its answer. Both
    belong to the lease, so they die with the browser.
    """

    __slots__ = ("_taken_over", "_driven")

    def __init__(self) -> None:
        self._taken_over = False
        self._driven = False

    @property
    def taken_over(self) -> bool:
        """Whether a person holds this Run's browser now."""
        return self._taken_over

    @property
    def state(self) -> LiveControlState:
        """Who drives now, as the lease record and every reader name it."""
        return "human" if self._taken_over else "agent"

    def take(self) -> bool:
        """Give the seat to a person; True when this call changed who holds it.

        Taking it is also what the seat remembers: a hand-back leaves the page as the person
        left it, so the agent's next call is told a person drove it rather than reasoning
        from a mental model of a page that has moved.
        """
        changed = not self._taken_over
        self._taken_over = True
        self._driven = True
        return changed

    def take_drive_notice(self) -> bool:
        """Whether a person has driven this browser since this was last asked.

        Asking answers it once: the first call of the Session that learns it says so, and a
        session that never asks leaves it for the next one. It is not ``taken_over``: the
        seat is the agent's again, and only what the page holds has changed.
        """
        driven, self._driven = self._driven, False
        return driven

    def release(self) -> bool:
        """Give the seat back to the agent; True when this call changed who holds it."""
        changed = self._taken_over
        self._taken_over = False
        return changed


@dataclass(frozen=True, slots=True)
class LiveInput:
    """One input event a person sent to the page they took over.

    It is the transport's vocabulary rather than the browser's: what each kind means is the
    adapter's to run, and it is stated here so a transport can build one without importing
    a browser driver.
    """

    kind: Literal["mouse", "wheel", "key", "navigate"]
    action: str = ""
    x: float = 0.0
    y: float = 0.0
    button: Literal["left", "right", "middle"] = "left"
    click_count: int = 1
    delta_x: float = 0.0
    delta_y: float = 0.0
    key: str = ""
    text: str = ""
    url: str = ""
    sequence: int = 0
    """The sender's own count of the events it has sent, oldest first, or 0 for no ordering.

    Input travels as separate requests, and separate requests can be overtaken: an event that
    arrives after a later one has already run describes a page that no longer exists — a
    pointer where it was, a click on what was there. A sender that numbers its events lets
    the page say so, and an event numbered at or below one already run is dropped rather
    than replayed over the state the newer one left. Zero means the sender states no order,
    which is what a coalesced or retried write does, and it is never dropped for ordering.
    """


@dataclass(frozen=True, slots=True)
class LiveSeat:
    """The driving seat as the one request that took it is answered.

    The credential is here and nowhere else. Taking the seat mints it, the answer to that
    request is the only place it is ever written down, and every write a driver makes
    presents it again; the stream every reader of the Run can see carries ``driver``, which
    is a display label and not a credential (ADR 0032, and the co-drive's own rule).
    """

    control: LiveControlState
    """Who drives the Run's browser now."""
    driver: str | None
    """The display label of whoever holds the seat, None while the agent drives."""
    token: str | None = None
    """The credential the holder presents to drive, and to hand the seat back.

    It belongs to the one connection that asked for the seat: minted when that request is
    answered, replaced whenever the seat changes hands, and gone the moment the seat is
    handed back or the session ends. Nothing else can drive with it, and no reader of the
    stream is ever shown it.
    """


@dataclass(frozen=True, slots=True)
class LiveUpdate:
    """One event of a live stream: the newest frame, in the state it was sent in.

    Every event carries the state, so a reader that missed a frame-less event — a slow one
    holds the newest event only — still reads the current control state from the next.
    """

    control: LiveControlState
    attached: bool
    frame: bytes | None = None
    """The newest frame, as the browser encoded it: JPEG bytes, handed on unchanged.

    A transport carries these bytes as its own binary payload — never as text — so the
    one encode the page paid for is also the last copy of it: nothing here re-encodes,
    inflates or re-wraps a frame.
    """
    sequence: int = 0
    """The newest frame this session has delivered, as of this update, or 0 before any.

    The same count a delivered frame's own header carries, and always the newest one: a text
    update travels for every event, a frame-less one included, so a reader that compares this
    with the newest frame it has drawn can tell a page that has stopped changing — nothing
    newer was delivered — from a stream that is behind it, where a frame that was delivered
    has not been drawn yet. A quiet session repeats it in every heartbeat.
    """
    driver: str | None = None
    """The display label of whoever holds the driving seat, None while the agent does.

    A label, never a credential: it is broadcast to every reader of the Run, so anything
    that travels here is public to all of them. A reader that did not take the seat is an
    observer — it watches, and neither drives nor hands the seat back — and it learns that
    from the seat it does not hold rather than from this string.
    """
    budget: LivePreviewBudget | None = None
    """What this stream is being delivered at now, and why it looks the way it does.

    It travels with every update because it changes without anybody asking: a session whose
    viewers cannot keep up steps itself down, and a reader that is told the rung it is on
    can say why the picture got lighter instead of leaving it a mystery.
    """
    at: float = 0.0
    ended: str = ""
    """Why the session is over: ``expired``, ``closed``, ``unreachable``, ``detached`` or empty.

    Every value is a word the adapter that ends a session actually sets: ``expired`` is the
    session's hard cap arriving, ``closed`` the session being closed or its lease ending,
    ``unreachable`` a page whose screencast would not start, ``detached`` a Run that drives
    no page, and empty a session that is still live.
    """

    def budget_fields(self) -> dict[str, float | int | None]:
        """The budget as the flat fields a state message carries, or all None without one."""
        budget = self.budget
        if budget is None:
            return {"frames_per_second": None, "jpeg_quality": None, "max_width": None}
        return {
            "frames_per_second": budget.frames_per_second,
            "jpeg_quality": budget.jpeg_quality,
            "max_width": budget.max_width,
        }


class LiveBrowserPage(Protocol):
    """The live surface of the browser a Run leases: whether a person holds its seat.

    ``PooledBrowserProvider.page_for`` answers with one of these, or with None when this
    process leases that Run no browser at all. The Engine asks one question of it — the
    one the ``browser`` tool must answer before every call — and the surface answers for
    its own seat rather than handing the seat out to be interrogated.
    """

    def taken_over(self) -> bool:
        """Whether a person holds this browser's driving seat now."""
        ...

    def take_drive_notice(self) -> bool:
        """Whether a person has driven this browser since this was last asked.

        The one read the ``browser`` tool makes after a call it answered: a seat that has
        been handed back leaves the page as the person left it, and the call that learns it
        says so. Asking answers it once.
        """
        ...


__all__ = [
    "LiveBrowserPage",
    "LiveControl",
    "LiveControlState",
    "LiveInput",
    "LivePreviewBudget",
    "LivePreviewSettings",
    "LivePreviewStep",
    "LiveSeat",
    "LiveUpdate",
]
