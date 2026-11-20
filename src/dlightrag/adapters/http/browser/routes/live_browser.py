# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The live preview of a Run's browser, and the co-drive that follows it.

Four routes, all scoped to one Run the caller owns and all answered by the Application's
live browser use case: the frames it already renders, the driving seat, the input events a
person who took that seat sends back, and the last picture a Run kept for a reader with
nothing live to watch. The frames are one WebSocket — the page's own JPEG, behind a fixed
header, on the binary opcode, the state that describes it on the text one — and each frame
is the page's own JPEG, encoded once however many readers are watching. Because both of
those are driven by the page, a page that stops changing produces nothing for minutes at a
time, so the socket also carries a heartbeat: a text message with the moment it was sent and
the number of the newest frame the session has delivered, which is what holds a proxied
WebSocket open across a quiet page (see ``HEARTBEAT_SECONDS``) and what lets a reader tell
a page that has stopped changing from one whose frame it has not drawn yet.

The driving seat holds one credential and one only, and the frames socket — which every
reader of the Run receives — never carries it: taking the seat mints a token as it is
answered and the answer to that one request is where it exists. Nothing here records a
frame or an input event, and none of it is cached, so every answer carries
``Cache-Control: private, no-store``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, WebSocket
from pydantic import BaseModel, ConfigDict, Field
from starlette.websockets import WebSocketDisconnect, WebSocketDisconnected

from dlightrag.adapters.http.application import get_application
from dlightrag.adapters.http.browser.auth import (
    WebSocketHandshakeRefused,
    authenticate_web_socket,
)
from dlightrag.adapters.http.browser.deps import get_live_browser_service
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.live_browser.models import LiveSession
from dlightrag.application.live_browser.service import LiveBrowserService
from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
)
from dlightrag.engine.answer.agent_browser.live import (
    LiveControlState,
    LiveInput,
    LiveUpdate,
)
from dlightrag.engine.answer.client_contracts import model_dump_json_safe

router = APIRouter()

#: The deliberate close codes of the frames socket, all sent before any frame. A socket
#: that asks to watch again after one of these is answered the same way: the close reason
#: carries the browser failure's own word (``not_configured``, ``unreachable``, ``no_page``,
#: ``expired``), which is what the dock classifies into one of its four states.
WS_UNAUTHENTICATED = 4401
WS_CROSS_ORIGIN = 4403
WS_NO_RUN = 4404
WS_NO_SESSION = 4409

#: How long the frames socket may go without a frame or a state message before it sends a
#: heartbeat anyway: a text message ``{"at": <epoch ms>, "sequence": <int>}``, carrying no
#: state and no frame, whose sequence is the newest frame the session has delivered.
#:
#: Both halves of the protocol are driven by the page, so a quiet page sends nothing at all
#: for as long as it stays quiet, and a proxy in front of this route will not wait that long:
#: Cloudflare closes a proxied WebSocket after about 100 seconds without data, so a reader
#: watching a quiet page would be dropped and reconnected about every hundred seconds. This
#: answers that window with a margin wide enough that one heartbeat lost to a slow moment is
#: not a lost socket, and it is a data message rather than a ``ping`` frame because only what
#: this route sends is liveness the reader's own environment can see. The sequence it repeats
#: is the same one every state message states, unchanged since the last update: a reader that
#: has drawn that frame reads "nothing newer was delivered" from it, which is the difference
#: between a still page and a stream that is behind it.
HEARTBEAT_SECONDS = 30.0

IdempotencyKey = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=200)]
#: The credential a driver presents. Taking the seat mints one and answers with it; every
#: later write of that seat carries it. The frames socket never carries it and no state
#: update ever names it, so a second reader of the same Run — which can read all of those —
#: holds no credential at all and cannot drive.
SeatCredential = Annotated[str, Header(alias="X-Live-Seat", min_length=1, max_length=128)]

#: Every answer of this surface is per-authorization and per-moment: never cached.
LIVE_HEADERS = {"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}


class _WebLiveInput(BaseModel):
    """One input event a person sends to the page they took over."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    type: Literal["mouse", "wheel", "key", "navigate"]
    action: Literal["move", "down", "up", "click", "press", "type", "wheel", "go"] = "click"
    x: float = Field(default=0.0, ge=-32_000, le=32_000)
    y: float = Field(default=0.0, ge=-32_000, le=32_000)
    button: Literal["left", "right", "middle"] = "left"
    click_count: int = Field(default=1, ge=1, le=3)
    delta_x: float = Field(default=0.0, ge=-10_000, le=10_000)
    delta_y: float = Field(default=0.0, ge=-10_000, le=10_000)
    key: str = Field(default="", max_length=64)
    text: str = Field(default="", max_length=4_096)
    url: str = Field(default="", max_length=2_048)
    #: The sender's own count of its events, oldest first. Zero states no order, and the
    #: session drops an event numbered at or below one it has already run.
    sequence: int = Field(default=0, ge=0, le=2_000_000_000)


class _WebLiveControl(BaseModel):
    """Who should drive the Run's browser next."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["agent", "human"]


class _WebLiveControlState(BaseModel):
    """Who drives the Run's browser now, and the seat's own answer to the caller.

    ``token`` is the credential, and it is here only for the request that took the seat:
    it is minted as that request is answered, replaced whenever the seat changes hands, and
    gone when the seat is handed back. A hand-back names no token because there is none
    left to name, and a reader that never took the seat never sees one.
    """

    control: LiveControlState
    driver: str | None
    token: str | None = None


@router.websocket("/runs/{run_id}/browser/live/frames")
async def live_browser_frames(
    websocket: WebSocket,
    run_id: str,
    service: Annotated[LiveBrowserService, Depends(get_live_browser_service)],
) -> None:
    """Stream the frames of the page this Run's browser is driving, over one socket.

    The binary opcode carries the newest frame: a fixed 16-byte header — the frame's own
    sequence number, then when the page was captured — and then the JPEG exactly as the
    browser encoded it. The text opcode carries the state that frame arrived in, so a reader
    that missed a frame still reads who drives it from the next update. A session only
    captures while somebody is reading it: this socket is what starts it, and its end is what
    stops it. The stream is read-only and it holds no credential: every update says the seat
    is held, and nothing here says by whom in any way a reader could drive with.
    """
    try:
        caller = authenticate_web_socket(websocket, get_application(websocket).config)
    except WebSocketHandshakeRefused as exc:
        code = WS_UNAUTHENTICATED if exc.kind == "unauthenticated" else WS_CROSS_ORIGIN
        await _refuse(websocket, code, str(exc))
        return
    owner_id = owner_id_from_user(caller)
    if await get_application(websocket).runs.get(owner_id=owner_id, run_id=run_id) is None:
        await _refuse(websocket, WS_NO_RUN, "Run not found")
        return
    try:
        session = await service.open(owner_id=owner_id, run_id=run_id)
    except AgentBrowserError as exc:
        await _refuse(websocket, WS_NO_SESSION, exc.reason)
        return
    await websocket.accept()
    await _stream(websocket, session)


@router.post(
    "/runs/{run_id}/browser/live/control",
    response_model=_WebLiveControlState,
)
async def live_browser_control(
    run_id: str,
    body: _WebLiveControl,
    request: Request,
    response: Response,
    service: Annotated[LiveBrowserService, Depends(get_live_browser_service)],
    seat: Annotated[str | None, Header(alias="X-Live-Seat", min_length=1, max_length=128)] = None,
) -> _WebLiveControlState:
    """Take the Run's browser, or hand it back, answering the holder its own credential.

    Taking the seat needs nothing but the caller's session, and the answer carries the token
    that holds it: that answer, and nothing else in this surface, is where the credential
    ever appears. A second caller asking for a seat somebody else holds is refused as
    ``taken_over`` and stays an observer, with no token to drive with. Handing the seat back
    presents the token that took it, and burns it; a hand-back that presents nothing at all —
    no ``X-Live-Seat`` header — is refused in the same words as one that presents a
    stranger's, because a reader that only watched the stream holds no seat to give up.
    """
    response.headers.update(LIVE_HEADERS)
    session = await _live_session(request, run_id, service)
    try:
        if body.state == "human":
            taken = await session.take_control()
            return _WebLiveControlState(
                control=taken.control, driver=taken.driver, token=taken.token
            )
        if seat is None:
            raise _no_credential()
        control = await session.release_control(token=seat)
    except AgentBrowserError as exc:
        raise _refused(exc) from None
    return _WebLiveControlState(control=control, driver=session.driver)


@router.post("/runs/{run_id}/browser/live/input", status_code=204)
async def live_browser_input(
    run_id: str,
    body: _WebLiveInput,
    request: Request,
    response: Response,
    service: Annotated[LiveBrowserService, Depends(get_live_browser_service)],
    seat: SeatCredential,
    idempotency_key: IdempotencyKey,
) -> None:
    """Send one input event to the page, for the seat's own credential.

    The credential is the whole of the authority here: a reader of the Run that watched the
    stream, replayed everything it carried, and took the seat never did holds nothing to
    present, and is refused as ``taken_over``. The event is not stored and not echoed;
    ``Idempotency-Key`` makes a retried event a no-op within the session, and the key is
    remembered in memory only. This is the only way an event reaches the page: the frames
    socket carries the read stream alone.
    """
    response.headers.update(LIVE_HEADERS)
    session = await _live_session(request, run_id, service)
    if session.control != "human":
        raise HTTPException(
            status_code=409,
            detail={
                "kind": "taken_over",
                "message": "Take control of the browser before sending an input event.",
            },
        )
    try:
        await session.send(_live_input(body), token=seat, key=idempotency_key)
    except AgentBrowserError as exc:
        raise _refused(exc) from None


@router.get("/runs/{run_id}/browser/live/screenshot")
async def live_browser_screenshot(
    run_id: str,
    request: Request,
    service: Annotated[LiveBrowserService, Depends(get_live_browser_service)],
) -> Response:
    """Serve the last picture the Run's browser took, for a reader with nothing live.

    The bytes are durable Run state — the attachment the ``browser`` tool stored — and the
    corpus visual-asset channel cannot address them from the client, because that channel
    names a chunk image of a workspace while a Run's attachment has no address the client is
    ever told. So the address is resolved here and the bytes are served once, never cached.
    """
    owner_id = owner_id_from_user(getattr(request.state, "user_context", None))
    if await get_application(request).runs.get(owner_id=owner_id, run_id=run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    screenshot = await service.screenshot(owner_id=owner_id, run_id=run_id)
    if screenshot is None:
        raise HTTPException(status_code=404, detail="This Run kept no screenshot")
    return Response(
        content=screenshot.content, media_type=screenshot.media_type, headers=LIVE_HEADERS
    )


async def _refuse(websocket: WebSocket, code: int, reason: str) -> None:
    """Refuse an upgrade with the deliberate code, and never a frame.

    A close sent before the handshake is accepted reaches the browser as a plain 403 with
    no close code, so the socket is accepted and closed at once instead: the viewer reads
    which refusal it was, and the close still comes before any frame.
    """
    await websocket.accept()
    await websocket.close(code=code, reason=reason)


async def _stream(websocket: WebSocket, session: LiveSession) -> None:
    """One viewer's frames until its socket closes, or until the session ends.

    The two directions are separate tasks because they are independent: the session may end
    while the viewer is quiet, and the viewer may leave while the page is not repainting. The
    first one to finish ends the socket, and the frame subscription is closed exactly once.
    """
    reading = asyncio.create_task(_read_messages(websocket), name="live-frames-messages")
    writing = asyncio.create_task(_write_frames(websocket, session), name="live-frames")
    try:
        await asyncio.wait({reading, writing}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in (reading, writing):
            task.cancel()
        await asyncio.gather(reading, writing, return_exceptions=True)
        with contextlib.suppress(WebSocketDisconnected, WebSocketDisconnect):
            # A session that ended, or a viewer that left: both are a normal close, and a
            # viewer whose browser is already gone is not told twice.
            await websocket.close()


async def _read_messages(websocket: WebSocket) -> None:
    """Hold this socket's reading half open until the viewer's browser closes it.

    Nothing a viewer sends here can touch the page: control handovers and input events are
    writes of their own — ``POST …/control`` and ``POST …/input``, with the CSRF header and
    idempotency key every write carries. What arrives is read and dropped; it is read at all
    so that a viewer leaving ends the stream at once rather than at the next frame.
    """
    while True:
        try:
            await websocket.receive_text()
        except WebSocketDisconnect:
            return


async def _write_frames(websocket: WebSocket, session: LiveSession) -> None:
    """Send the session's updates: the state as text, the newest frame as binary.

    ``send_bytes`` awaits the socket's own buffer, so a viewer that reads slowly is sent
    fewer frames rather than older ones: the session keeps the newest frame only, and a
    frame that arrives while this one is still on its way replaces it.

    Frames and state both come from the page, so this is also where the socket is kept
    alive through a page that has nothing to say: one heartbeat goes out whenever
    ``HEARTBEAT_SECONDS`` pass with no message of either kind. The wait for the session's
    next update is raced against that clock rather than beside it, so there is exactly one
    writer on this socket and a heartbeat can never land between a state message and the
    frame it describes. The wait is a task of its own because waiting for the *next* update
    must not be cancelled by a heartbeat: what a viewer missed is still the newest thing
    the page produced. A heartbeat repeats the sequence of the last state message it
    followed, which is still the newest frame the session has delivered: that is what lets
    a reader tell a still page from a stream it has fallen behind.
    """
    frames = session.frames()
    update = asyncio.create_task(_next_update(frames))
    sent = time.monotonic()
    delivered = 0
    try:
        while True:
            quiet = max(HEARTBEAT_SECONDS - (time.monotonic() - sent), 0.0)
            beat = asyncio.create_task(asyncio.sleep(quiet))
            done, _ = await asyncio.wait({update, beat}, return_when=asyncio.FIRST_COMPLETED)
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)
            if update in done:
                newest = update.result()
                if newest is None:
                    return
                delivered = newest.sequence
                update = asyncio.create_task(_next_update(frames))
                await websocket.send_text(_state_message(newest))
                if newest.frame is not None:
                    await websocket.send_bytes(newest.frame)
            else:
                await websocket.send_text(_heartbeat_message(delivered))
            sent = time.monotonic()
    finally:
        # Exactly once, whether this ended by itself, by the socket closing, or by the
        # session ending: the last reader leaving is what stops the page's screencast.
        update.cancel()
        await asyncio.gather(update, return_exceptions=True)
        await frames.aclose()


async def _next_update(frames: AsyncIterator[LiveUpdate]) -> LiveUpdate | None:
    """The session's next update, or None when it has no more to send.

    The end is a value rather than an exception because this runs as a task of its own: a
    task that raised ``StopAsyncIteration`` would report it as a failure of the wait, while
    the end of a session is the ordinary way a preview finishes.
    """
    try:
        return await anext(frames)
    except StopAsyncIteration:
        return None


def _heartbeat_message(sequence: int) -> str:
    """One keepalive: the moment it was sent, and the newest frame it saw delivered.

    It is the text opcode with none of a state message's fields, so a reader that knows the
    state shape reads it as a keepalive and no reader can mistake it for a new state. The
    sequence is the one the last state message stated, repeated unchanged: it says nothing
    new was delivered, which is exactly what a page that has stopped changing looks like.
    """
    return json.dumps({"at": int(time.time() * 1000), "sequence": sequence})


def _state_message(update: LiveUpdate) -> str:
    """The state one update arrived in, as the text half of the socket's own protocol.

    The frame is not in here: it is the binary message that follows, to the byte, because
    encoding bytes into a string is what this transport stopped doing. The budget is in
    here because a session steps itself down without being asked, and the reader that is
    shown a lighter picture is the one that can say why. The sequence is the newest frame
    the session has delivered, which is what a reader compares with its own newest drawn
    frame: a reader behind that number is behind the stream, and one level with it is
    looking at a page that has not changed.
    """
    return json.dumps(
        model_dump_json_safe(
            {
                "control": update.control,
                "driver": update.driver,
                "attached": update.attached,
                "ended": update.ended,
                "sequence": update.sequence,
                **update.budget_fields(),
                "at": update.at,
            }
        ),
        ensure_ascii=False,
    )


def _live_input(body: _WebLiveInput) -> LiveInput:
    """One wire event as the event the page runs, with the action its type implies."""
    action = body.action
    if body.type == "mouse":
        action = body.action if body.action in {"move", "down", "up", "click"} else "click"
    elif body.type == "key":
        action = body.action if body.action in {"press", "type", "down", "up"} else "press"
    elif body.type == "wheel":
        action = "wheel"
    else:
        action = "go"
    return LiveInput(
        kind=body.type,
        action=action,
        x=body.x,
        y=body.y,
        button=body.button,
        click_count=body.click_count,
        delta_x=body.delta_x,
        delta_y=body.delta_y,
        key=body.key,
        text=body.text,
        url=body.url,
        sequence=body.sequence,
    )


async def _live_session(request: Request, run_id: str, service: LiveBrowserService) -> LiveSession:
    """The Run's live session, opened for the owner of that Run and nobody else."""
    owner_id = owner_id_from_user(getattr(request.state, "user_context", None))
    if await get_application(request).runs.get(owner_id=owner_id, run_id=run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    try:
        return await service.open(owner_id=owner_id, run_id=run_id)
    except AgentBrowserError as exc:
        raise _refused(exc) from None


def _refused(exc: AgentBrowserError) -> HTTPException:
    """A live-session refusal as this transport reports it: the reason, and its sentence.

    The refusal wears the browser command envelope every sibling route's typed failure
    wears, so the error handler passes it through as the object it is rather than
    flattening it to a sentence the reader cannot classify.
    """
    return HTTPException(
        status_code=409,
        detail={"kind": exc.reason, "message": exc.public_message},
    )


def _no_credential() -> HTTPException:
    """A hand-back that presents no credential, refused as the seat's own refusal.

    Handing the seat back is the holder's move, and the credential is the whole of who holds
    it: a write that presents none is refused exactly like one that presents a stranger's —
    the same ``taken_over`` every other refusal of the seat wears — so a reader of the Run
    that never took the seat cannot give somebody else's up. The session is asked nothing:
    there is no credential to compare, and the answer is the same either way.
    """
    return HTTPException(
        status_code=409,
        detail={
            "kind": "taken_over",
            "message": "Handing the seat back requires the credential that took it.",
        },
    )


__all__: list[str] = ["router"]
