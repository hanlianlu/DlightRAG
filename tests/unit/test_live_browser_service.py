# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Application use case behind the live browser routes: opening, and the fallback picture."""

from __future__ import annotations

from typing import Any

import pytest

from dlightrag.application.answer_runs.service import RunResourceDescriptor
from dlightrag.application.live_browser.service import LiveBrowserService
from dlightrag.engine.answer.agent_browser.contracts import (
    AgentBrowserError,
)

OWNER = "owner"
RUN = "11111111-1111-1111-1111-111111111111"


class FakeSessions:
    """The live sessions port: one session, and what it was opened for."""

    def __init__(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.closed = 0

    async def open(self, *, owner_id: str, run_id: str) -> Any:
        self.opened.append((owner_id, run_id))
        return "session"

    async def aclose(self) -> None:
        self.closed += 1


class FakeRuns:
    """The Run attachment port: what the Run stored under each name."""

    def __init__(
        self, stored: dict[str, tuple[RunResourceDescriptor, bytes]] | None = None
    ) -> None:
        self.stored = stored or {}
        self.asked: list[tuple[str, str, str]] = []

    async def latest_run_attachment(
        self, *, owner_id: str, run_id: str, safe_name: str
    ) -> tuple[RunResourceDescriptor, bytes] | None:
        self.asked.append((owner_id, run_id, safe_name))
        return self.stored.get(safe_name)


async def test_a_deployment_with_no_browser_refuses_as_not_configured() -> None:
    service = LiveBrowserService(sessions=None, runs=FakeRuns())

    with pytest.raises(AgentBrowserError) as refused:
        await service.open(owner_id=OWNER, run_id=RUN)

    assert refused.value.reason == "not_configured"
    await service.aclose()


async def test_a_session_is_opened_for_the_owner_that_asked() -> None:
    sessions = FakeSessions()
    service = LiveBrowserService(sessions=sessions, runs=FakeRuns())

    assert await service.open(owner_id=OWNER, run_id=RUN) == "session"
    assert sessions.opened == [(OWNER, RUN)]

    await service.aclose()
    assert sessions.closed == 1


async def test_the_fallback_picture_is_the_screenshot_the_run_stored() -> None:
    runs = FakeRuns(
        {
            "screenshot": (
                RunResourceDescriptor(
                    resource_id="res-shot-1",
                    registry="resource",
                    reference_kind=None,
                    ordinal=0,
                    filename="screenshot",
                    mime_type="image/jpeg",
                    digest="a" * 64,
                ),
                b"jpeg\xff\xd9",
            )
        }
    )
    service = LiveBrowserService(sessions=None, runs=runs)

    screenshot = await service.screenshot(owner_id=OWNER, run_id=RUN)

    assert screenshot is not None
    assert (screenshot.content, screenshot.media_type) == (b"jpeg\xff\xd9", "image/jpeg")
    # The name asked for is the one the browser tool gives its attachment.
    assert runs.asked == [(OWNER, RUN, "screenshot")]


async def test_a_run_that_kept_no_screenshot_has_no_fallback_picture() -> None:
    service = LiveBrowserService(sessions=None, runs=FakeRuns())

    assert await service.screenshot(owner_id=OWNER, run_id=RUN) is None
