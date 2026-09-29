# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Run's durable dependency retries: bounded backoff and a per-Run deferral cap."""

from typing import Any

import pytest

from dlightrag.engine.dependencies import (
    MAX_DEPENDENCY_DEFERRALS,
    DependencyComponent,
    DependencyRetriesExhausted,
    next_dependency_retry,
)


def test_a_run_fails_after_its_outage_deferrals_across_every_component() -> None:
    checkpoint: dict[str, Any] = {}
    components: tuple[DependencyComponent, ...] = ("providers", "parser")
    for component in components * (MAX_DEPENDENCY_DEFERRALS // 2):
        retry, _delay = next_dependency_retry(checkpoint, component)
        checkpoint = {**checkpoint, **retry}
    assert checkpoint["dependency_deferrals"] == MAX_DEPENDENCY_DEFERRALS

    with pytest.raises(DependencyRetriesExhausted) as raised:
        next_dependency_retry(checkpoint, "corpus_storage")

    assert raised.value.kind == "dependency_unavailable"
    assert raised.value.component == "corpus_storage"
    assert raised.value.public_message == (
        "Corpus storage stayed unavailable through 10 retries, so this Run stopped. "
        "Try it again later."
    )


def test_a_wait_keeps_its_backoff_but_spends_no_deferral() -> None:
    checkpoint = {
        "corpus_unavailable_attempt": 3,
        "dependency_deferrals": MAX_DEPENDENCY_DEFERRALS,
    }

    retry, delay = next_dependency_retry(checkpoint, "corpus_storage", base_seconds=2, outage=False)

    assert retry == {
        "corpus_unavailable_attempt": 4,
        "dependency_deferrals": MAX_DEPENDENCY_DEFERRALS,
    }
    assert delay == 16


def test_a_first_wait_records_no_deferral_count() -> None:
    retry, delay = next_dependency_retry(None, "corpus_storage", outage=False)

    assert retry == {"corpus_unavailable_attempt": 1}
    assert delay == 5


def test_an_unreadable_deferral_count_counts_from_zero() -> None:
    retry, _delay = next_dependency_retry({"dependency_deferrals": "many"}, "parser")

    assert retry == {"parser_unavailable_attempt": 1, "dependency_deferrals": 1}
