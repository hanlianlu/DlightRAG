# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The live preview of a Run's browser: the use case the Web transport calls.

A Run's browser is leased by the Run, never by this service: it opens the live view of a
page that already exists and ends with the lease that page sits on. The transport asks two
questions — the Run's session, and the picture to show when the session has nothing live —
and each is answered from durable state or through the port, so no HTTP route reaches an
adapter (ADR 0032).
"""

from __future__ import annotations

import logging

from dlightrag.engine.answer.agent_browser.contracts import browser_failure

from .models import LiveScreenshot, LiveSession, LiveSessions, RunAttachmentReader

logger = logging.getLogger(__name__)

#: What the ``browser`` tool names the screenshot it attaches to its result. The name is
#: durable Run state's own vocabulary, so a reader of that state can find the last picture.
SCREENSHOT_ATTACHMENT_NAME = "screenshot"


class LiveBrowserService:
    """A deployment's live browser preview: opening one Run's session, and its last picture.

    Where the deployment composes no Agent Browser, ``sessions`` is absent and opening says
    so as ``not_configured`` — the same word every other browser surface uses. The
    screenshots a Run stored stay readable either way, because they are Run state and not
    the browser's.
    """

    def __init__(self, *, sessions: LiveSessions | None, runs: RunAttachmentReader) -> None:
        self._sessions = sessions
        self._runs = runs

    async def open(self, *, owner_id: str, run_id: str) -> LiveSession:
        """Open one Run's live session, for an owner that holds a lease on that Run."""
        if self._sessions is None:
            raise browser_failure("not_configured")
        return await self._sessions.open(owner_id=owner_id, run_id=run_id)

    async def screenshot(self, *, owner_id: str, run_id: str) -> LiveScreenshot | None:
        """The most recent screenshot the Run's browser took, if it kept one.

        This is what a reader is shown when nothing is attached: the live page is gone, and
        its last picture is durable state of the Run rather than anything this reader
        cached. The corpus visual-asset channel cannot answer it from the client — that
        channel addresses a chunk image of one workspace, and a Run's attachment has no
        address a reader is ever told — so the address is resolved here and the bytes are
        served once, under the same no-store rule as the live routes.
        """
        stored = await self._runs.latest_run_attachment(
            owner_id=owner_id, run_id=run_id, safe_name=SCREENSHOT_ATTACHMENT_NAME
        )
        if stored is None:
            return None
        descriptor, content = stored
        return LiveScreenshot(content=content, media_type=descriptor.mime_type)

    async def aclose(self) -> None:
        """End every live session this process serves; the browsers stay their Runs'."""
        if self._sessions is not None:
            await self._sessions.aclose()


__all__ = ["SCREENSHOT_ATTACHMENT_NAME", "LiveBrowserService"]
