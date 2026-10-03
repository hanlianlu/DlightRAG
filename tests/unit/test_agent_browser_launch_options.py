# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The launch options the Agent Browser sends a pool member over the wire."""

import json

from dlightrag.adapters.agent_browser.pool import launch_options_header


def test_a_launch_asks_for_a_proxied_headless_chromium_with_its_sandbox() -> None:
    header = launch_options_header("http://egress:3128", sandbox=True)

    assert (
        header == '{"headless":true,"chromiumSandbox":true,"proxy":{"server":"http://egress:3128"}}'
    )


def test_a_launch_without_the_sandbox_asks_for_nothing_else_different() -> None:
    sandboxed = json.loads(launch_options_header("http://egress:3128", sandbox=True))
    unsandboxed = json.loads(launch_options_header("http://egress:3128", sandbox=False))

    assert unsandboxed == {k: v for k, v in sandboxed.items() if k != "chromiumSandbox"}
