# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the automatic CI gates guarantee, rather than how their steps are spelled."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = _ROOT / ".github/workflows/ci.yml"
_PG_TESTS = "Run deterministic durable PostgreSQL integration tests"

#: The one place a PostgreSQL suite is kept out of CI, with the reason it cannot run there.
_PG_SUITES_OUTSIDE_CI = {
    "tests/integration/test_format_routes_pg.py": (
        "its non-Latin route reads a local embedded-font PDF that is not bundled, so it skips"
    ),
}


def _job(name: str) -> dict[str, Any]:
    return yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"][name]


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_every_postgresql_suite_runs_in_ci_or_says_why_not() -> None:
    command = _named_step(_job("integration"), _PG_TESTS)["run"]
    selected = set(re.findall(r"tests/integration/test_\w+\.py", command))
    suites = {
        path.relative_to(_ROOT).as_posix()
        for path in (_ROOT / "tests/integration").glob("test_*.py")
    }

    assert sorted(selected - suites) == [], "CI names a suite that does not exist"
    assert sorted(suites - selected - _PG_SUITES_OUTSIDE_CI.keys()) == [], (
        "a PostgreSQL suite neither runs in CI nor says why not"
    )
    assert sorted(_PG_SUITES_OUTSIDE_CI.keys() - (suites - selected)) == [], (
        "an exclusion names a suite that is gone or that CI runs"
    )


@pytest.mark.parametrize(
    ("job", "tests", "guard"),
    [
        ("integration", _PG_TESTS, "Reject skipped PostgreSQL integration tests"),
        (
            "browser-e2e",
            "Run mocked browser E2E tests",
            "Reject empty or skipped browser E2E results",
        ),
    ],
)
def test_ci_fails_a_job_whose_tests_skipped_or_never_ran(job: str, tests: str, guard: str) -> None:
    spec = _job(job)
    report = re.search(r"--junitxml=(\S+)", _named_step(spec, tests)["run"])
    guard_command = _named_step(spec, guard)["run"]

    assert report is not None
    assert report.group(1) in guard_command
    assert "tests > 0" in guard_command
    assert "skipped == 0" in guard_command


def test_ci_postgresql_image_is_pinned_by_digest() -> None:
    assert re.fullmatch(
        r"ghcr\.io/\$\{\{ github\.repository_owner \}\}/dlightrag-postgres@sha256:[0-9a-f]{64}",
        _job("integration")["env"]["POSTGRES_IMAGE"],
    )


def test_ci_uses_no_secret_and_calls_no_paid_service() -> None:
    workflow = json.dumps(yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))).lower()

    for forbidden in (
        "secrets.",
        "github_token",
        "credentials",
        "api_key",
        "openai",
        "anthropic",
        "ragas",
    ):
        assert forbidden not in workflow
