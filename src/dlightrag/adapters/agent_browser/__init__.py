# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one Agent Browser implementation: Playwright over a pool of run-servers (ADR 0032)."""

from dlightrag.adapters.agent_browser.pool import PooledBrowserProvider

__all__ = ["PooledBrowserProvider"]
