# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Web routes of a Run's live browser: ownership, caching, who drives, and the socket.

The routes are exercised over the real application with its services doubled, so what is
asserted here is the transport's own behaviour: the caller's Run or a 404, `private,
no-store` on every answer, the seat named by the viewer that holds it, the last screenshot
served when there is nothing live to show, and — over a fake socket — the frames socket's
own protocol: binary frames, text updates, the heartbeats a quiet session is kept alive
with, and the deliberate close codes a handshake it will not serve is refused with.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.types import Message
from starlette.websockets import WebSocket

from dlightrag.adapters.http.browser.routes.live_browser import live_browser_frames
from dlightrag.adapters.http.server import create_app
from dlightrag.application.config.loading import set_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.live_browser.models import LiveScreenshot, LiveSeat
from dlightrag.engine.ai.settings import EmbeddingSettings, ModelRoleSettings, ModelSettings
from dlightrag.engine.answer.agent_browser.contracts import (
    browser_failure,
    page_failure,
)
from dlightrag.engine.answer.agent_browser.live import LivePreviewBudget, LiveUpdate
from tests.support.application_double import application_double

RUN = "11111111-1111-1111-1111-111111111111"
WS_PATH = f"/web/api/runs/{RUN}/browser/live/frames"
ORIGIN = "http://test"
#: What the paste-token deployments store: the bearer, url-safe base64 without padding.
TOKEN = "s3cret"
COOKIE = base64.urlsafe_b64encode(TOKEN.encode()).decode().rstrip("=")


#: The credential the fake seat mints, and the display label every reader is told. The two
#: are deliberately different strings: a test that swapped one for the other would pass a
#: route that broadcast the credential, which is the defect this whole surface is about.
SEAT_TOKEN = "seat-token-9f2c41ab"
SEAT_LABEL = "viewer-1"


class FakeSession:
    """A live session as the transport reads it: frames a test pushes, and a held seat.

    It holds one seat, exactly as the real session does, and it is stingy with the token the
    same way: the credential comes back from a take and from nowhere else, and a write that
    presents anything but it is refused.
    """

    def __init__(
        self,
        frames: list[LiveUpdate] | None = None,
        *,
        hold: bool = False,
        token: str = SEAT_TOKEN,
        label: str = SEAT_LABEL,
    ) -> None:
        self.control = "agent"
        self.driver: str | None = None
        self.unsubscribed = 0
        self._frames = frames or []
        self._hold = hold
        self._token = token
        self._label = label
        self.taken: list[str | None] = []
        #: Every credential a hand-back reached this session with, in order, and the ones
        #: that actually gave the seat up. The two differ: a hand-back the route refuses
        #: never arrives here at all, and one carrying somebody else's credential arrives
        #: and is refused, while only the holder's own appears in both.
        self.asked: list[str] = []
        self.released: list[str] = []
        self.sent: list[tuple[Any, str, str]] = []

    async def frames(self) -> AsyncGenerator[LiveUpdate]:
        """The updates a test pushed, then either the end, or a session that stays open."""
        try:
            for update in self._frames:
                yield update
            while self._hold:
                await asyncio.sleep(0.01)
        finally:
            # One reader's subscription ending, exactly as the real session counts it.
            self.unsubscribed += 1

    async def take_control(self) -> LiveSeat:
        if self.control == "human":
            raise page_failure("taken_over")
        self.driver = self._label
        self.control = "human"
        self.taken.append(None)
        return LiveSeat(control="human", driver=self._label, token=self._token)

    async def release_control(self, *, token: str) -> str:
        # The real session compares the credential a caller presents against the one it
        # minted, and a hand-back that reaches it with anything else — a stranger's token,
        # the broadcast label — is refused without giving the seat up.
        self.asked.append(token)
        if self.control == "human" and token != self._token:
            raise page_failure("taken_over")
        # Only a hand-back that actually gives the seat up is recorded, exactly as the real
        # session only publishes on a change: asking twice with the spent credential is a
        # no-op, not a second release.
        if self.control == "human":
            self.driver = None
            self.control = "agent"
            self.released.append(token)
        return self.control

    async def send(self, event: Any, *, token: str, key: str = "") -> None:
        if self.control != "human" or token != self._token:
            raise page_failure("taken_over")
        self.sent.append((event, token, key))


class FakeSocket(WebSocket):
    """One frames socket: Starlette's own WebSocket over a test's private ASGI channel.

    What is faked is the transport — the scope it reads its caller from, the messages a
    viewer sends, and the capture of what the route answered — so the handler, its message
    shapes and its close codes are the ones the deployment runs.
    """

    def __init__(
        self,
        app: Any,
        *,
        origin: str | None = ORIGIN,
        cookie: str | None = None,
    ) -> None:
        headers = [
            (b"host", b"test"),
            (b"connection", b"Upgrade"),
            (b"upgrade", b"websocket"),
        ]
        if origin is not None:
            headers.append((b"origin", origin.encode()))
        if cookie is not None:
            headers.append((b"cookie", f"dlightrag_web_auth={cookie}".encode()))
        scope: dict[str, Any] = {
            "type": "websocket",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "scheme": "ws",
            "server": ("test", 80),
            "client": ("127.0.0.1", 5555),
            "root_path": "",
            "path": WS_PATH,
            "raw_path": WS_PATH.encode(),
            "query_string": b"",
            "headers": headers,
            "app": app,
        }
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.incoming.put_nowait({"type": "websocket.connect"})
        self.sent: list[Message] = []
        super().__init__(scope, self._inbound, self._outbound)

    async def _inbound(self) -> Message:
        return await self.incoming.get()

    async def _outbound(self, message: Message) -> None:
        self.sent.append(message)

    @property
    def frames(self) -> list[bytes]:
        """Every binary message the route sent: one frame each."""
        return [
            message["bytes"]
            for message in self.sent
            if message["type"] == "websocket.send" and "bytes" in message
        ]

    @property
    def updates(self) -> list[dict[str, Any]]:
        """Every text message the route sent, as the state object it is."""
        return [
            json.loads(message["text"])
            for message in self.sent
            if message["type"] == "websocket.send" and "text" in message
        ]

    @property
    def closure(self) -> Message | None:
        """The close the route sent, or None while the socket is still open."""
        return next(
            (message for message in self.sent if message["type"] == "websocket.close"), None
        )

    async def hang_up(self) -> None:
        """The viewer's browser closing the socket."""
        await self.incoming.put({"type": "websocket.disconnect", "code": 1000})


def _config(tmp_path: Path, **access: Any) -> DlightragConfig:
    settings: dict[str, Any] = {"deployment": {"working_dir": str(tmp_path)}}
    if access:
        settings["access"] = access
    return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        **settings,
        models={
            "chat": ModelRoleSettings(default=ModelSettings(model="gpt-5.4-mini", api_key="test")),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="test",
                startup_probe=False,
            ),
        },
    )


def _socket(config: DlightragConfig, application: Any, **kwargs: Any) -> FakeSocket:
    """One frames socket over a test's application, as the route is handed it."""
    app = SimpleNamespace(state=SimpleNamespace(application=application))
    return FakeSocket(app, **kwargs)


@pytest.fixture()
async def route_client(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Any]]:
    set_config(_config(tmp_path))
    application = application_double(_config(tmp_path))
    application.runs.get.return_value = {"run_id": RUN}
    app = create_app()
    app.state.application = application
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", follow_redirects=False
    ) as client:
        yield client, application


@pytest.fixture()
def application(tmp_path: Path) -> Any:
    """One application double, with the Run this caller owns and one live session."""
    set_config(_config(tmp_path))
    double = application_double(_config(tmp_path))
    double.runs.get.return_value = {"run_id": RUN}
    return double


# ---- the frames socket ---------------------------------------------------


async def test_a_frame_arrives_as_bytes_and_its_state_as_text(
    application: Any,
) -> None:
    session = FakeSession(
        [
            LiveUpdate(control="human", attached=True, driver="viewer-1", at=1.0),
            LiveUpdate(
                control="human",
                attached=True,
                driver="viewer-1",
                frame=b"\xff\xd8jpeg\xff\xd9",
                sequence=1,
                at=2.0,
            ),
        ]
    )
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)

    await live_browser_frames(socket, RUN, application.live_browser)

    # The whole of the frame is one binary message, to the byte: no base64, no wrapper.
    assert socket.frames == [b"\xff\xd8jpeg\xff\xd9"]
    # Every text message states the newest frame the session has delivered, so a reader can
    # compare it with the newest frame it has drawn instead of guessing from silence.
    assert socket.updates == [
        {
            "control": "human",
            "driver": "viewer-1",
            "attached": True,
            "ended": "",
            "sequence": 0,
            "at": 1.0,
            "frames_per_second": None,
            "jpeg_quality": None,
            "max_width": None,
        },
        {
            "control": "human",
            "driver": "viewer-1",
            "attached": True,
            "ended": "",
            "sequence": 1,
            "at": 2.0,
            "frames_per_second": None,
            "jpeg_quality": None,
            "max_width": None,
        },
    ]
    assert all("frame" not in update for update in socket.updates)
    assert socket.closure == {"type": "websocket.close", "code": 1000, "reason": ""}
    assert session.unsubscribed == 1
    assert application.live_browser.open.await_args.kwargs == {
        "owner_id": application.runs.get.await_args.kwargs["owner_id"],
        "run_id": RUN,
    }


async def test_the_state_message_carries_the_rung_the_stream_is_delivered_at(
    application: Any,
) -> None:
    session = FakeSession(
        [
            LiveUpdate(
                control="agent",
                attached=True,
                at=1.0,
                budget=LivePreviewBudget(frames_per_second=15.0, max_width=1280, jpeg_quality=45),
            )
        ]
    )
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)

    await live_browser_frames(socket, RUN, application.live_browser)

    # A session steps itself down without being asked, so the reader is told what it is being
    # served at: the rung travels with the state rather than only in the operator's log.
    assert socket.updates[0]["frames_per_second"] == 15.0
    assert socket.updates[0]["jpeg_quality"] == 45
    assert socket.updates[0]["max_width"] == 1280


async def test_the_socket_ends_when_the_session_does(application: Any) -> None:
    session = FakeSession(
        [LiveUpdate(control="agent", attached=False, ended="expired", at=3.0)], hold=False
    )
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)

    await asyncio.wait_for(live_browser_frames(socket, RUN, application.live_browser), 5)

    assert socket.closure is not None
    assert socket.updates[-1]["ended"] == "expired"
    # The subscription ended with the session, and only once.
    assert session.unsubscribed == 1


async def test_a_viewer_that_leaves_unsubscribes_exactly_once(application: Any) -> None:
    session = FakeSession([LiveUpdate(control="agent", attached=True, at=1.0)], hold=True)
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)
    serving = asyncio.create_task(live_browser_frames(socket, RUN, application.live_browser))
    for _ in range(100):
        if socket.frames or socket.updates:
            break
        await asyncio.sleep(0.01)

    await socket.hang_up()
    await asyncio.wait_for(serving, 5)

    assert session.unsubscribed == 1
    # A socket the viewer closed is closed back once, and no frame outlives the reader.
    assert socket.closure == {"type": "websocket.close", "code": 1000, "reason": ""}


async def test_a_quiet_session_is_kept_alive_by_heartbeats_and_nothing_else(
    application: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A page that has not changed sends neither frame nor state, and the socket lives.

    The quiet window is scaled down from the ``HEARTBEAT_SECONDS`` the deployment runs with,
    so what is asserted is the route's own behaviour: nothing else goes out, and the one
    thing that does carries only the moment it was sent.
    """
    monkeypatch.setattr(
        "dlightrag.adapters.http.browser.routes.live_browser.HEARTBEAT_SECONDS", 0.05
    )
    session = FakeSession([], hold=True)
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)
    serving = asyncio.create_task(live_browser_frames(socket, RUN, application.live_browser))
    await asyncio.sleep(0.4)
    await socket.hang_up()
    await asyncio.wait_for(serving, 5)

    beats = [message for message in socket.updates if set(message) == {"at", "sequence"}]
    assert 3 <= len(beats) <= 12
    # Nothing else was sent: a quiet session has no state to send and no frame to send, so
    # every text message is a heartbeat and no binary message is anything at all.
    assert socket.updates == beats
    assert socket.frames == []
    # Each heartbeat is the moment it went out, so the cadence is what the clock says: at
    # least two full quiet windows between the first and the last. It repeats the newest
    # frame the session delivered, which a session that never delivered one states as zero.
    moments = [beat["at"] for beat in beats]
    assert moments == sorted(moments)
    assert moments[-1] - moments[0] >= 100
    assert {beat["sequence"] for beat in beats} == {0}
    # And the keepalive is the socket's whole traffic either way.
    assert session.unsubscribed == 1


async def test_a_heartbeat_repeats_the_newest_frame_the_session_delivered(
    application: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quiet moment says which frame was the last one, unchanged since it arrived.

    That number is what a dock compares with the newest frame it has drawn: level with it is
    a page that has stopped changing, and behind it is a frame the dock is late on. A
    heartbeat that only said when it was sent could tell a reader neither.
    """
    monkeypatch.setattr(
        "dlightrag.adapters.http.browser.routes.live_browser.HEARTBEAT_SECONDS", 0.05
    )
    session = FakeSession(
        [
            LiveUpdate(control="agent", attached=True, frame=b"\xff\xd8first", sequence=1, at=1.0),
            LiveUpdate(control="agent", attached=True, sequence=1, at=1.5),
        ],
        hold=True,
    )
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)
    serving = asyncio.create_task(live_browser_frames(socket, RUN, application.live_browser))
    await asyncio.sleep(0.3)
    await socket.hang_up()
    await asyncio.wait_for(serving, 5)

    beats = [message for message in socket.updates if set(message) == {"at", "sequence"}]
    assert beats
    assert {beat["sequence"] for beat in beats} == {1}
    assert socket.frames == [b"\xff\xd8first"]


async def test_a_state_message_restarts_the_heartbeat_clock(
    application: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A heartbeat answers silence, not a clock: a page that says something is not silent."""
    monkeypatch.setattr(
        "dlightrag.adapters.http.browser.routes.live_browser.HEARTBEAT_SECONDS", 0.5
    )
    session = FakeSession(
        [LiveUpdate(control="agent", attached=True, at=float(index)) for index in range(20)],
        hold=True,
    )
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)
    serving = asyncio.create_task(live_browser_frames(socket, RUN, application.live_browser))

    # The burst is the page's own state, sent well inside one quiet window: no heartbeat
    # interrupts it, because the clock restarted at every message.
    await asyncio.sleep(0.2)
    assert len(socket.updates) == 20
    assert all("control" in message for message in socket.updates)

    # The page then goes quiet, and the heartbeat is what its silence is answered with.
    await asyncio.sleep(0.6)
    await socket.hang_up()
    await asyncio.wait_for(serving, 5)

    assert [message for message in socket.updates if set(message) == {"at", "sequence"}]


async def test_a_run_this_caller_does_not_own_is_closed_before_any_frame(
    application: Any,
) -> None:
    application.runs.get.return_value = None
    socket = _socket(application.config, application)

    await live_browser_frames(socket, RUN, application.live_browser)

    assert socket.sent[0]["type"] == "websocket.accept"
    assert socket.closure is not None
    assert socket.closure["code"] == 4404
    assert socket.frames == []
    application.live_browser.open.assert_not_awaited()


async def test_a_deployment_without_a_browser_closes_with_its_reason(application: Any) -> None:
    application.live_browser.open.side_effect = browser_failure("not_configured")
    socket = _socket(application.config, application)

    await live_browser_frames(socket, RUN, application.live_browser)

    assert socket.closure is not None
    assert socket.closure["code"] == 4409
    # The reason is the browser failure's own word, which is what the dock classifies.
    assert socket.closure["reason"] == "not_configured"
    assert socket.frames == []


# ---- the handshake -------------------------------------------------------


async def test_an_unauthenticated_socket_is_closed_before_any_frame(tmp_path: Path) -> None:
    config = _config(tmp_path, auth_mode="simple", api_token=TOKEN)
    set_config(config)
    application = application_double(config)
    application.runs.get.return_value = {"run_id": RUN}
    socket = _socket(config, application)

    await live_browser_frames(socket, RUN, application.live_browser)

    assert socket.sent[0]["type"] == "websocket.accept"
    assert socket.closure is not None
    assert socket.closure["code"] == 4401
    assert socket.frames == []
    application.runs.get.assert_not_awaited()
    application.live_browser.open.assert_not_awaited()


async def test_the_session_cookie_is_what_authenticates_the_handshake(tmp_path: Path) -> None:
    config = _config(tmp_path, auth_mode="simple", api_token=TOKEN)
    set_config(config)
    application = application_double(config)
    application.runs.get.return_value = {"run_id": RUN}
    application.live_browser.open.return_value = FakeSession(
        [LiveUpdate(control="agent", attached=True, at=1.0)]
    )
    socket = _socket(config, application, cookie=COOKIE)

    await live_browser_frames(socket, RUN, application.live_browser)

    assert socket.closure is not None
    assert socket.closure["code"] == 1000
    assert socket.updates != []
    # A cookie that does not decode to the token is no credential at all.
    refused = _socket(config, application, cookie=COOKIE[:-1] + "x")
    application.live_browser.open.return_value = FakeSession([])
    await live_browser_frames(refused, RUN, application.live_browser)
    assert refused.closure is not None
    assert refused.closure["code"] == 4401


async def test_a_foreign_origin_is_closed_before_any_frame(application: Any) -> None:
    socket = _socket(application.config, application, origin="https://elsewhere.example")

    await live_browser_frames(socket, RUN, application.live_browser)

    assert socket.closure is not None
    assert socket.closure["code"] == 4403
    assert socket.frames == []
    application.live_browser.open.assert_not_awaited()


async def test_a_socket_with_no_origin_at_all_is_closed(application: Any) -> None:
    socket = _socket(application.config, application, origin=None)

    await live_browser_frames(socket, RUN, application.live_browser)

    assert socket.closure is not None
    assert socket.closure["code"] == 4403
    application.live_browser.open.assert_not_awaited()


# ---- the rest of the surface ---------------------------------------------


async def test_taking_the_seat_answers_a_credential_and_is_never_cached(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    session = FakeSession()
    application.live_browser.open.return_value = session

    response = await client.post(
        f"/web/api/runs/{RUN}/browser/live/control",
        json={"state": "human"},
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    # The answer to the request that took the seat is the only place the credential exists.
    assert response.json() == {
        "control": "human",
        "driver": SEAT_LABEL,
        "token": SEAT_TOKEN,
    }
    assert session.taken == [None]


async def test_a_second_reader_cannot_take_the_seat(route_client: tuple[AsyncClient, Any]) -> None:
    client, application = route_client
    application.live_browser.open.return_value = FakeSession()

    first = await client.post(
        f"/web/api/runs/{RUN}/browser/live/control",
        json={"state": "human"},
    )
    second = await client.post(
        f"/web/api/runs/{RUN}/browser/live/control",
        json={"state": "human"},
    )

    assert first.json()["driver"] == SEAT_LABEL
    assert second.status_code == 409
    assert second.json()["kind"] == "taken_over"


async def test_the_stream_never_carries_the_credential(application: Any) -> None:
    """A second reader of the same Run reads the broadcast, and no credential is in it.

    This is the defect the seat was rebuilt for: every update used to name the holder's own
    token, so any reader of the Run could replay it and drive. What the stream carries now
    is a display label, and the credential is in the answer to the take and nowhere else.
    """
    session = FakeSession(
        [
            LiveUpdate(control="human", attached=True, driver=SEAT_LABEL, at=1.0),
            LiveUpdate(
                control="human", attached=True, driver=SEAT_LABEL, frame=b"\xff\xd8jpeg", at=2.0
            ),
        ]
    )
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)

    await live_browser_frames(socket, RUN, application.live_browser)

    assert [update["driver"] for update in socket.updates] == [SEAT_LABEL, SEAT_LABEL]
    # The credential appears in no text message and in no binary one: the label is what
    # every reader is told, and it is not what drives.
    assert all(SEAT_TOKEN not in json.dumps(update) for update in socket.updates)
    assert all(SEAT_TOKEN.encode() not in frame for frame in socket.frames)

    # What the take answers with is the label the stream carried and a credential that
    # was never in it: the two are different strings on purpose, and only one is broadcast.
    seat = await session.take_control()
    assert seat.driver == SEAT_LABEL
    assert seat.token == SEAT_TOKEN
    assert seat.token != seat.driver


async def test_a_reader_that_replays_the_broadcast_cannot_drive(
    route_client: tuple[AsyncClient, Any],
) -> None:
    """Everything a second reader can see, replayed at the write routes, drives nothing."""
    client, application = route_client
    session = FakeSession([LiveUpdate(control="human", attached=True, driver=SEAT_LABEL, at=1.0)])
    session.control, session.driver = "human", SEAT_LABEL
    application.live_browser.open.return_value = session
    socket = _socket(application.config, application)
    await live_browser_frames(socket, RUN, application.live_browser)
    replayed = socket.updates[0]
    assert replayed["driver"] == SEAT_LABEL

    # The reader learned the seat is held, and by which label; neither is a credential.
    # Presenting nothing at all is refused before the session is even asked.
    uncredentialed = await client.post(
        f"/web/api/runs/{RUN}/browser/live/input",
        json={"type": "mouse", "action": "click", "x": 3, "y": 4},
        headers={"Idempotency-Key": "one"},
    )
    assert uncredentialed.status_code == 422
    for replayed_credential in (replayed["driver"], "guess", "viewer-2"):
        drove = await client.post(
            f"/web/api/runs/{RUN}/browser/live/input",
            json={"type": "mouse", "action": "click", "x": 3, "y": 4},
            headers={"X-Live-Seat": replayed_credential, "Idempotency-Key": "one"},
        )
        assert drove.status_code == 409, replayed_credential
        assert drove.json()["kind"] == "taken_over"
    assert session.sent == []

    # The holder drives with the credential its own take was answered with.
    session.control = "human"
    drove = await client.post(
        f"/web/api/runs/{RUN}/browser/live/input",
        json={"type": "mouse", "action": "click", "x": 3, "y": 4},
        headers={"X-Live-Seat": SEAT_TOKEN, "Idempotency-Key": "one"},
    )
    assert drove.status_code == 204
    assert [(event.x, token, key) for event, token, key in session.sent] == [
        (3.0, SEAT_TOKEN, "one")
    ]


async def test_handing_the_seat_back_requires_the_credential_and_spends_it(
    route_client: tuple[AsyncClient, Any],
) -> None:
    """Only the credential that took the seat gives it up, and giving it up spends it.

    Every way of *not* presenting the credential is refused here, and the one that is easy
    to miss is the first: a hand-back with no ``X-Live-Seat`` header at all. A reader of the
    Run sees the seat held and the label it is held under, so a route that treated a missing
    credential as "nobody in particular" would let any such reader give somebody else's seat
    up — which is what this test pins.
    """
    client, application = route_client
    session = FakeSession()
    application.live_browser.open.return_value = session
    took = await client.post(f"/web/api/runs/{RUN}/browser/live/control", json={"state": "human"})
    assert took.json()["token"] == SEAT_TOKEN

    # No credential at all is refused before the session is ever asked: there is nothing to
    # compare, and the answer is the seat's own refusal rather than a validation error.
    absent = await client.post(f"/web/api/runs/{RUN}/browser/live/control", json={"state": "agent"})
    assert absent.status_code == 409
    assert absent.json()["kind"] == "taken_over"
    assert session.asked == []

    # A credential that is not the seat's — the broadcast label it names the holder by, and
    # another viewer's token — is refused too, and the seat stays the taker's.
    for stranger in (SEAT_LABEL, "viewer-2-token"):
        refused = await client.post(
            f"/web/api/runs/{RUN}/browser/live/control",
            json={"state": "agent"},
            headers={"X-Live-Seat": stranger},
        )
        assert refused.status_code == 409, stranger
        assert refused.json()["kind"] == "taken_over"
        assert session.control == "human"
    assert session.asked == [SEAT_LABEL, "viewer-2-token"]
    assert session.released == []

    # The holder's own credential hands the seat back, and it does so exactly once: the
    # hand-back that follows presents the credential it spent, changes nothing, and is not
    # a second release.
    handed = await client.post(
        f"/web/api/runs/{RUN}/browser/live/control",
        json={"state": "agent"},
        headers={"X-Live-Seat": SEAT_TOKEN},
    )
    assert handed.status_code == 200
    assert handed.json() == {"control": "agent", "driver": None, "token": None}
    assert session.control == "agent"
    assert session.released == [SEAT_TOKEN]
    again = await client.post(
        f"/web/api/runs/{RUN}/browser/live/control",
        json={"state": "agent"},
        headers={"X-Live-Seat": SEAT_TOKEN},
    )
    assert again.status_code == 200
    assert again.json() == {"control": "agent", "driver": None, "token": None}
    assert session.released == [SEAT_TOKEN]

    # And the credential the hand-back spent drives nothing afterwards.
    after = await client.post(
        f"/web/api/runs/{RUN}/browser/live/input",
        json={"type": "mouse", "action": "click"},
        headers={"X-Live-Seat": SEAT_TOKEN, "Idempotency-Key": "one"},
    )
    assert after.status_code == 409


async def test_an_input_event_carries_its_credential_and_is_never_cached(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    session = FakeSession()
    session.control, session.driver = "human", SEAT_LABEL
    application.live_browser.open.return_value = session

    response = await client.post(
        f"/web/api/runs/{RUN}/browser/live/input",
        json={"type": "mouse", "action": "click", "x": 3, "y": 4, "sequence": 7},
        headers={"X-Live-Seat": SEAT_TOKEN, "Idempotency-Key": "one"},
    )

    assert response.status_code == 204
    assert response.headers["cache-control"] == "private, no-store"
    assert [(event.x, token, key) for event, token, key in session.sent] == [
        (3.0, SEAT_TOKEN, "one")
    ]
    assert session.sent[0][0].sequence == 7


async def test_input_without_a_credential_is_refused(route_client: tuple[AsyncClient, Any]) -> None:
    client, application = route_client
    application.live_browser.open.return_value = FakeSession()

    response = await client.post(
        f"/web/api/runs/{RUN}/browser/live/input",
        json={"type": "mouse", "action": "click"},
        headers={"Idempotency-Key": "one"},
    )

    assert response.status_code == 422
    application.live_browser.open.assert_not_awaited()


async def test_the_last_screenshot_is_served_for_a_reader_with_nothing_live(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    application.live_browser.screenshot.return_value = LiveScreenshot(
        content=b"\x89PNG\r\n\x1a\n", media_type="image/png"
    )

    response = await client.get(f"/web/api/runs/{RUN}/browser/live/screenshot")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.content == b"\x89PNG\r\n\x1a\n"
    assert application.live_browser.screenshot.await_args.kwargs == {
        "owner_id": application.runs.get.await_args.kwargs["owner_id"],
        "run_id": RUN,
    }


async def test_a_run_that_kept_no_screenshot_answers_404(
    route_client: tuple[AsyncClient, Any],
) -> None:
    client, application = route_client
    application.live_browser.screenshot.return_value = None

    response = await client.get(f"/web/api/runs/{RUN}/browser/live/screenshot")

    assert response.status_code == 404
