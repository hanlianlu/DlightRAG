# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a render and an Agent Page share of the Playwright driver."""

from __future__ import annotations

import re

from playwright.async_api import Error as PlaywrightError
from playwright.async_api import Page

#: The only part of a navigation failure the model reads: Chromium's network error token.
NETWORK_ERROR = re.compile(r"net::ERR_[A-Z0-9_]+")
#: Closing a context, or disconnecting a browser, that does not answer must not hold a
#: render, a call, or a Run's settlement open.
CLOSE_SECONDS = 10.0


def milliseconds(seconds: float) -> float:
    """Playwright's timeouts are milliseconds, and 0 would mean no limit at all."""
    return max(1.0, seconds * 1000)


async def page_content(page: Page, timeout: float) -> str:
    """The serialized DOM; a navigation in progress is waited out once."""
    try:
        return await page.content()
    except PlaywrightError:
        await page.wait_for_load_state("load", timeout=milliseconds(timeout))
        return await page.content()
