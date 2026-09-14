# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared PostgreSQL connection defaults for integration tests.

Honors the same environment variables the CI job exports
(``PGHOST``/``PGPORT``/``PGUSER``/``PGPASSWORD``/``PGDATABASE``) so the
suite can run against an isolated container without editing test code.
"""

from __future__ import annotations

import os
from typing import Any

# `localhost` is ambiguous on a machine that also runs a host PostgreSQL on [::1]: the resolver
# prefers that IPv6 instance over the container's IPv4 mapping, so the suite silently tested against
# a server whose `dlightrag` role is not the superuser CI provides - which surfaced as unrelated
# "permission denied to terminate process" and "must be superuser to create extension" failures.
# Pin the container mapping and keep PGHOST for CI and other deployments.
PG_CONN_KWARGS: dict[str, Any] = dict(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "dlightrag"),
    password=os.environ.get("PGPASSWORD", "dlightrag"),
    database=os.environ.get("PGDATABASE", "dlightrag"),
)
