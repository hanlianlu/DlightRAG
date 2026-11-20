# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the Application asks of a Run's live browser, and who answers.

The live preview is an Engine capability reached through a port (ADR 0032): the session
that renders frames and arbitrates the driving seat belongs to an adapter, and the Web
transport talks to this vocabulary instead. Nothing here drives a browser, and a test can
double every question it states.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Protocol

from dlightrag.application.answer_runs.service import RunResourceDescriptor
from dlightrag.engine.answer.agent_browser.live import (
    LiveControlState,
    LiveInput,
    LiveSeat,
    LiveUpdate,
)


@dataclass(frozen=True, slots=True)
class LiveScreenshot:
    """The most recent picture of a Run's browser, and what those bytes are."""

    content: bytes
    media_type: str


class LiveSession(Protocol):
    """One Run's live preview, as the transport reads it.

    ``frames`` produces the newest frame only while somebody is reading it, so a session
    nobody watches costs nothing; it is a generator because the reader's own end is what
    ends the subscription. The seat holds one holder: taking it answers with the credential
    that holder drives with, and that answer is the only place the credential exists — what
    ``driver`` names, and what ``frames`` carries to every reader, is a label.
    """

    @property
    def control(self) -> LiveControlState:
        """Who drives the Run's browser now."""
        ...

    @property
    def driver(self) -> str | None:
        """The seat holder's display label, or None while the agent drives."""
        ...

    def frames(self) -> AsyncGenerator[LiveUpdate]:
        """The newest frame, for as long as this reader is subscribed."""
        ...

    async def take_control(self) -> LiveSeat:
        """Give the seat to the caller, answering with the credential that holds it.

        A second caller is refused as ``taken_over``: the seat is one, and the credential
        travels in this answer rather than in anything every reader can see.
        """
        ...

    async def release_control(self, *, token: str) -> LiveControlState:
        """Give the seat back, for the credential that took it.

        ``token`` is the credential the take answered that one request with — never the
        display label every reader is shown, and never left out: a hand-back that presents
        nothing is refused as ``taken_over`` exactly like one that presents a stranger's, so
        a reader that only watched the stream cannot give somebody else's seat up. The
        session's own releases (its end, its idle timer) are its own and do not travel here.
        """
        ...

    async def send(self, event: LiveInput, *, token: str, key: str = "") -> None:
        """Run one input event, for the credential the seat handed out."""
        ...


class LiveSessions(Protocol):
    """The live sessions a deployment serves: at most one per Run, shared by its readers."""

    async def open(self, *, owner_id: str, run_id: str) -> LiveSession: ...

    async def aclose(self) -> None: ...


class RunAttachmentReader(Protocol):
    """A Run's durable attachments, read whole by the name the tool that stored one gave it."""

    async def latest_run_attachment(
        self, *, owner_id: str, run_id: str, safe_name: str
    ) -> tuple[RunResourceDescriptor, bytes] | None: ...


__all__ = [
    "LiveScreenshot",
    "LiveSeat",
    "LiveSession",
    "LiveSessions",
    "RunAttachmentReader",
]
