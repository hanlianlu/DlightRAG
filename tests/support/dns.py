# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One deterministic DNS answer for tests that must not reach the network.

Callers install this on the process-wide ``socket.getaddrinfo`` through
``dlightrag.engine.network_admission``, so it answers only the hosts a caller means to fake and
delegates every other lookup to the real resolver. A fake that answered everything once sent a
PostgreSQL connect on ``localhost`` to the pinned public address, where it hung for the client's
full connect timeout and failed twenty-six integration tests.
"""

from __future__ import annotations

import socket
from typing import Any

PUBLIC_ADDRESS = "93.184.216.34"
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "::"})

_real_getaddrinfo = socket.getaddrinfo


def public_dns(host: str, port: int, *args: Any, **kwargs: Any) -> list[Any]:
    """Resolve any host to one pinned public address, except loopback, which stays real."""
    if host in LOOPBACK_HOSTS:
        return _real_getaddrinfo(host, port, *args, **kwargs)
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_ADDRESS, port))]
