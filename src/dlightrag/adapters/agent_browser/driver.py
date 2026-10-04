# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a render and an Agent Page share of the Playwright driver."""

from __future__ import annotations

import re

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page, Response

from dlightrag.engine.answer.agent_browser import AgentBrowserError, browser_failure

#: The only part of a navigation failure the model reads: Chromium's network error token.
_NETWORK_ERROR = re.compile(r"net::ERR_[A-Z0-9_]+")
#: Closing a context, or disconnecting a browser, that does not answer must not hold a
#: render, a call, or a Run's settlement open.
CLOSE_SECONDS = 10.0


def milliseconds(seconds: float) -> float:
    """Playwright's timeouts are milliseconds, and 0 would mean no limit at all."""
    return max(1.0, seconds * 1000)


def navigation_status(page: Page, response: Response) -> int | None:
    """The status of a response that answers a navigation of the page's main frame, else None."""
    if response.request.is_navigation_request() and response.frame == page.main_frame:
        return response.status
    return None


def starts_download(exc: PlaywrightError) -> bool:
    """Whether a navigation failed because its URL answers with a file, which Chromium downloads."""
    return "Download is starting" in str(exc)


def network_failure(exc: PlaywrightError) -> AgentBrowserError | None:
    """The failure Chromium's network error token names, or None when the error holds none."""
    token = _NETWORK_ERROR.search(str(exc))
    return browser_failure("navigation_failed", detail=token.group()) if token else None


async def page_content(page: Page, timeout: float) -> str:
    """The serialized DOM; a navigation in progress is waited out once."""
    try:
        return await page.content()
    except PlaywrightError:
        await page.wait_for_load_state("load", timeout=milliseconds(timeout))
        return await page.content()
