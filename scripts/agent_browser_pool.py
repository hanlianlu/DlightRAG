#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Write docker-compose.yml's Agent Browser pool for the size it declares.

Each member of the pool is a Playwright run-server on an internal network of its own, which
the egress proxy and the application services join, and whose endpoint is bound into those
services (ADR 0032). Compose has no loop, so the three blocks marked ``agent-browser-pool``
are written from the ``members=N`` on the first one. After changing that number, run:

    uv run python scripts/agent_browser_pool.py
"""

from __future__ import annotations

import re
from pathlib import Path

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"

_BLOCK = re.compile(
    r"^(?P<head>(?P<indent> *)# >>> agent-browser-pool:(?P<name>\w+)[^\n]*\n)"
    r".*?"
    r"^(?P<tail>(?P=indent)# <<< agent-browser-pool:(?P=name)\n)",
    re.M | re.S,
)
_MEMBERS = re.compile(r"^# >>> agent-browser-pool:anchors members=(\d+)$", re.M)


def declared_members(text: str) -> int:
    """The pool size the compose text declares."""
    match = _MEMBERS.search(text)
    if match is None:
        raise ValueError("docker-compose.yml declares no agent-browser-pool members=N")
    return int(match.group(1))


def _names(members: int) -> list[str]:
    return [f"agent-browser-{index}" for index in range(1, members + 1)]


def _anchors(members: int) -> str:
    """The networks every member's peer joins, and the endpoints bound into the services."""
    names = _names(members)
    endpoints = ",".join(f'"ws://{name}:3000/"' for name in names)
    return (
        "x-agent-browser-networks: &agent-browser-networks\n"
        + "".join(f"  {name}:\n" for name in names)
        + "\nx-agent-browser-bindings: &agent-browser-bindings\n"
        + f"  DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS: '[{endpoints}]'\n"
        + "  DLIGHTRAG_ANSWER__AGENT__BROWSER__EGRESS_PROXY: http://agent-browser-egress:3128\n"
    )


def _services(members: int) -> str:
    """One service per member, each on the network that is its own."""
    return "\n".join(
        f"  {name}:\n    <<: *agent-browser\n    networks: [{name}]\n" for name in _names(members)
    )


def _networks(members: int) -> str:
    """One internal network per member, so no member reaches another or the Web."""
    return "".join(f"  {name}:\n    internal: true\n" for name in _names(members))


_BODIES = {"anchors": _anchors, "services": _services, "networks": _networks}


def render(text: str) -> str:
    """The compose text with its pool blocks written for the size it declares."""
    members = declared_members(text)
    if members < 1:
        raise ValueError("the pool needs at least one member")
    found: set[str] = set()

    def block(match: re.Match[str]) -> str:
        found.add(match["name"])
        return match["head"] + _BODIES[match["name"]](members) + match["tail"]

    rendered = _BLOCK.sub(block, text)
    if missing := sorted(set(_BODIES) - found):
        raise ValueError(f"docker-compose.yml lacks the agent-browser-pool blocks {missing}")
    return rendered


def main() -> None:
    text = COMPOSE.read_text(encoding="utf-8")
    COMPOSE.write_text(render(text), encoding="utf-8")
    print(f"agent-browser pool: {declared_members(text)} member(s)")


if __name__ == "__main__":
    main()
