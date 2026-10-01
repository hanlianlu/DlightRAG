# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cheap CI guard for the opt-in RunRuntime load shape."""

from tests.load.runtime_workload import latency_summary, mutation_action, query_submission


def test_10k_workload_shape_is_exact_and_deterministic() -> None:
    submissions = [query_submission(index) for index in range(10_000)]

    assert sum(item.kind == "retrieval" for item in submissions) == 1_000
    assert sum(item.mode == "fast" for item in submissions) == 4_000
    assert sum(item.mode == "research" for item in submissions) == 5_000
    assert {item.workspace_count for item in submissions if item.kind == "retrieval"} == {
        1,
        10,
        50,
        100,
    }
    assert query_submission(0) == query_submission(0)


def test_mutation_workload_covers_every_production_action_exactly() -> None:
    actions = [mutation_action(index) for index in range(1_000)]

    assert {
        action: actions.count(action)
        for action in ("ingest", "replace", "delete", "retry", "reset")
    } == {
        "ingest": 200,
        "replace": 200,
        "delete": 200,
        "retry": 200,
        "reset": 200,
    }


def test_latency_summary_uses_stable_nearest_rank_percentiles() -> None:
    summary = latency_summary([value / 1000 for value in range(1, 101)])

    assert summary == {
        "count": 100,
        "p50_ms": 50.0,
        "p95_ms": 95.0,
        "p99_ms": 99.0,
        "max_ms": 100.0,
    }
