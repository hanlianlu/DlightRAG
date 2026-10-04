# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Compose stack's Agent Browser pool is written from the one number that sizes it."""

import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.agent_browser_pool import render

_COMPOSE = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
_SERVICES = ("dlightrag-api", "dlightrag-mcp", "dlightrag-reader")
_PEERS = (*_SERVICES, "agent-browser-egress")


def _declaring(members: int) -> str:
    """The shipped compose text with its members=N changed to ``members``."""
    return re.sub(r"(?<=agent-browser-pool:anchors members=)\d+", str(members), _COMPOSE)


def test_the_shipped_compose_file_is_written_for_the_size_it_declares() -> None:
    assert render(_COMPOSE) == _COMPOSE, "run scripts/agent_browser_pool.py"


@pytest.mark.parametrize("members", [1, 3])
def test_each_member_is_alone_on_an_internal_network_its_peers_join(members: int) -> None:
    stack = yaml.safe_load(render(_declaring(members)))
    names = [f"agent-browser-{index}" for index in range(1, members + 1)]

    pool = sorted(
        name
        for name in stack["services"]
        if name.startswith("agent-browser-") and name != "agent-browser-egress"
    )
    assert pool == names
    for name in names:
        assert stack["services"][name]["networks"] == [name]
        assert stack["networks"][name] == {"internal": True}
    for peer in _PEERS:
        assert set(stack["services"][peer]["networks"]) == {"default", *names}
    for service in _SERVICES:
        endpoints = stack["services"][service]["environment"][
            "DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS"
        ]
        assert json.loads(endpoints) == [f"ws://{name}:3000/" for name in names]


def test_a_pool_has_at_least_one_member() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        render(_declaring(0))
