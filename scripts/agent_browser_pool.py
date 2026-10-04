#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Write docker-compose.yml's Agent Browser pool from the one number that sizes it.

Each member of the pool is a Playwright run-server on an internal network of its own, which
the egress proxy and the application services join, and whose endpoint is bound into those
services (ADR 0032). Compose has no loop, so the three blocks marked ``agent-browser-pool``
are written from the ``members=N`` on the first one:

    uv run python scripts/agent_browser_pool.py --members 3   # resize the pool
    uv run python scripts/agent_browser_pool.py               # rewrite for the declared size
    uv run python scripts/agent_browser_pool.py --check       # fail if the blocks disagree
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.yml"

_BLOCK = re.compile(
    r"^(?P<indent> *)# >>> agent-browser-pool:(?P<name>\w+).*\n"
    r".*?"
    r"^(?P=indent)# <<< agent-browser-pool:(?P=name)\n",
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


def render(text: str, members: int | None = None) -> str:
    """The compose text with its pool blocks written for ``members``, by default the number
    it declares."""
    count = declared_members(text) if members is None else members
    if count < 1:
        raise ValueError("the pool needs at least one member")
    found: set[str] = set()

    def block(match: re.Match[str]) -> str:
        name, indent = match["name"], match["indent"]
        found.add(name)
        size = f" members={count}" if name == "anchors" else ""
        return (
            f"{indent}# >>> agent-browser-pool:{name}{size}\n"
            + _BODIES[name](count)
            + f"{indent}# <<< agent-browser-pool:{name}\n"
        )

    rendered = _BLOCK.sub(block, text)
    if missing := sorted(set(_BODIES) - found):
        raise ValueError(f"docker-compose.yml lacks the agent-browser-pool blocks {missing}")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--members", type=int, help="the pool size to write")
    mode.add_argument("--check", action="store_true", help="fail if the blocks disagree")
    args = parser.parse_args()

    text = COMPOSE.read_text(encoding="utf-8")
    rendered = render(text, args.members)
    if args.check:
        if rendered != text:
            sys.exit(
                "docker-compose.yml's agent-browser-pool blocks disagree with its members=N; "
                "run scripts/agent_browser_pool.py"
            )
        return
    COMPOSE.write_text(rendered, encoding="utf-8")
    print(f"agent-browser pool: {declared_members(rendered)} member(s)")


if __name__ == "__main__":
    main()
