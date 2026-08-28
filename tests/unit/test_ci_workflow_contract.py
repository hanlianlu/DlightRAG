# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Contracts for the automatic durable PostgreSQL pull-request gate."""

import json
import re
from pathlib import Path
from typing import Any

import yaml

_CI_WORKFLOW = Path(".github/workflows/ci.yml")
_DURABLE_PG_SUITES = {
    "tests/integration/test_agent_session_pg.py",
    "tests/integration/test_answer_runs_pg.py",
    "tests/integration/test_answer_run_api_pg.py",
    "tests/integration/test_answer_run_coordinator_pg.py",
    "tests/integration/test_memory_pg.py",
    "tests/integration/test_run_cancellation_pg.py",
}
_REQUIRED_ENV = {
    "PGHOST": "localhost",
    "PGPORT": "5432",
    "PGUSER": "dlightrag",
    "PGPASSWORD": "dlightrag",
    "PGDATABASE": "dlightrag",
}


def _integration_job() -> dict[str, Any]:
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["integration"]


def _workflow_triggers() -> set[str]:
    workflow = yaml.load(_CI_WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return set(workflow["on"])


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_pg_integration_job_is_automatic_and_uses_an_ephemeral_pinned_database() -> None:
    job = _integration_job()
    start = _named_step(job, "Start clean PostgreSQL")
    cleanup = _named_step(job, "Remove PostgreSQL container")
    start_command = start["run"]

    assert {"pull_request", "push", "workflow_dispatch"} <= _workflow_triggers()
    assert "if" not in job
    assert re.fullmatch(
        r"ghcr\.io/\$\{\{ github\.repository_owner \}\}/"
        r"dlightrag-postgres@sha256:[0-9a-f]{64}",
        job["env"]["POSTGRES_IMAGE"],
    )
    assert {key: str(job["env"][key]) for key in _REQUIRED_ENV} == _REQUIRED_ENV
    assert "docker run --detach --name dlightrag-ci-postgres" in start_command
    assert "--rm" not in start_command
    assert "--volume" not in start_command
    assert "postgres -c shared_preload_libraries=pg_textsearch,pg_jieba" in start_command
    assert "docker inspect" in start_command
    assert ".State.Status" in start_command
    assert ".State.Health.Status" in start_command
    assert "docker logs dlightrag-ci-postgres || true" in start_command
    assert cleanup["if"] == "always()"
    assert "docker rm --force dlightrag-ci-postgres" in cleanup["run"]


def test_pg_integration_job_rejects_skips_without_external_evaluation() -> None:
    job = _integration_job()
    test_step = _named_step(job, "Run deterministic durable PostgreSQL integration tests")
    skip_guard = _named_step(job, "Reject skipped PostgreSQL integration tests")
    test_command = test_step["run"]
    guard_command = skip_guard["run"]
    selected_suites = set(re.findall(r"tests/integration/test_[a-z0-9_]+\.py", test_command))
    serialized_job = json.dumps(job).lower()

    assert selected_suites == _DURABLE_PG_SUITES
    assert "--junitxml=.test-results/postgres.xml" in test_command
    assert ".test-results/postgres.xml" in guard_command
    assert "tests > 0" in guard_command
    assert "skipped == 0" in guard_command
    for forbidden in (
        "ragas",
        "scripts/ragas_eval.py",
        "credentials",
        "secrets.",
        "github_token",
        "api_key",
        "openai",
        "anthropic",
    ):
        assert forbidden not in serialized_job
