# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for E2E tests."""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, Playwright

from dlightrag.api.server import create_app


@pytest.fixture
def app():
    """Create FastAPI app with web routes enabled."""
    return create_app(include_web=True)


@pytest.fixture
def page(app, playwright: Playwright) -> Page:
    """Create a Playwright page wired to the FastAPI app."""
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page_obj = context.new_page()
    yield page_obj
    context.close()
    browser.close()
