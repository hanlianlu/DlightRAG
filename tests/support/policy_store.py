# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Runtime Policy store that keeps its documents in memory, as the real one keeps them in a row.

It holds what the real store holds - the document as it was written, who wrote it, and when -
and nothing else, so a test reads back exactly what the product stored.
"""

import copy
import datetime
from collections.abc import Mapping
from typing import Any

from dlightrag.application.policies.service import StoredPolicy

_FIRST_WRITE = datetime.datetime(2026, 10, 9, 8, 0, tzinfo=datetime.UTC)


class InMemoryPolicyStore:
    """Documents by key; every write is a minute after the one before it."""

    def __init__(self) -> None:
        self.rows: dict[str, StoredPolicy] = {}
        self.writes = 0

    async def read_runtime_policy(self, key: str) -> StoredPolicy | None:
        return self.rows.get(key)

    async def write_runtime_policy(
        self, key: str, value: Mapping[str, Any], actor: str
    ) -> StoredPolicy:
        stored = StoredPolicy(
            value=copy.deepcopy(dict(value)),
            updated_at=_FIRST_WRITE + datetime.timedelta(minutes=self.writes),
            updated_by=actor,
        )
        self.writes += 1
        self.rows[key] = stored
        return stored
