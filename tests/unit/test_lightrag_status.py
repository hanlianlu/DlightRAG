# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Every upstream outcome is compared by one normalized LightRAG status."""

from enum import Enum
from types import SimpleNamespace

import pytest

from dlightrag.engine.rag.lightrag.status import lightrag_status


class _DocStatus(Enum):
    PROCESSED = "Processed"


@pytest.mark.parametrize(
    "record",
    [
        {"status": "processed"},
        {"status": " Processed "},
        {"status": _DocStatus.PROCESSED},
        SimpleNamespace(status=_DocStatus.PROCESSED),
    ],
    ids=["mapping", "padded", "enum", "object"],
)
def test_a_status_reads_the_same_however_lightrag_reports_it(record: object) -> None:
    assert lightrag_status(record) == "processed"


@pytest.mark.parametrize("record", [None, {}, {"status": None}, SimpleNamespace()])
def test_a_missing_status_reads_as_empty(record: object) -> None:
    assert lightrag_status(record) == ""
