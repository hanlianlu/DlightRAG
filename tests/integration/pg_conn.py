# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared PostgreSQL connection defaults for integration tests.

Honors the same environment variables the CI job exports
(``PGHOST``/``PGPORT``/``PGUSER``/``PGPASSWORD``/``PGDATABASE``) so the
suite can run against an isolated container without editing test code.
"""

from __future__ import annotations

import os
from typing import Any

PG_CONN_KWARGS: dict[str, Any] = dict(
    host=os.environ.get("PGHOST", "localhost"),
    port=int(os.environ.get("PGPORT", "5432")),
    user=os.environ.get("PGUSER", "dlightrag"),
    password=os.environ.get("PGPASSWORD", "dlightrag"),
    database=os.environ.get("PGDATABASE", "dlightrag"),
)
