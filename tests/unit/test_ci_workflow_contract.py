# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the automatic CI gates guarantee, rather than how their steps are spelled."""

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = _ROOT / ".github/workflows/ci.yml"
_PRE_COMMIT = _ROOT / ".pre-commit-config.yaml"
_PG_TESTS = "Run deterministic durable PostgreSQL integration tests"

#: The one place a PostgreSQL suite is kept out of CI, with the reason it cannot run there.
_PG_SUITES_OUTSIDE_CI: dict[str, str] = {}


def _job(name: str) -> dict[str, Any]:
    return yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))["jobs"][name]


def _named_step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


def test_every_gate_runs_on_each_push_and_pull_request_and_cannot_pass_by_skipping() -> None:
    """A skipped job counts as passed, so a gate that may not run is no gate."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))  # YAML 1.1 reads a bare `on` as true

    assert {"push", "pull_request"} <= set(triggers)
    jobs = ["fast", "integration", "browser-e2e"]
    for name in jobs:
        job = workflow["jobs"][name]
        assert "if" not in job and "needs" not in job, name
        assert not any(step.get("continue-on-error") for step in job["steps"]), name
    fast = "\n".join(step.get("run", "") for step in workflow["jobs"]["fast"]["steps"])
    for gate in (
        "make lint",
        "make typecheck",
        "make architecture-check",
        "make platform-check",
        "make frontend-lint",
        "make frontend-test",
        "make test-unit",
    ):
        assert gate in fast, gate


def test_commit_hooks_check_the_change_while_ci_checks_the_repository() -> None:
    """A hook judges what the commit changes; the whole-repository gates stay in CI.

    A repository-wide gate in a commit hook turns baseline debt left by an older release into a
    blocked commit for every agent on the branch, which is what an agent-first harness cannot have.
    The fast CI job still runs every whole-repository gate over the committed repository.
    """
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    fast = "\n".join(step.get("run", "") for step in workflow["jobs"]["fast"]["steps"])
    for gate in (
        "make lint",
        "make lint-security",
        "make format-check",
        "make typecheck",
        "make architecture-check",
    ):
        assert gate in fast, gate

    hooks = {
        hook["id"]: hook
        for repo in yaml.safe_load(_PRE_COMMIT.read_text(encoding="utf-8"))["repos"]
        for hook in repo.get("hooks", [])
    }
    for hook_id in ("pyright", "ruff-security"):
        hook = hooks[hook_id]
        assert hook.get("types") == ["python"], hook_id
        # `pass_filenames: false` runs the tool over the repository; the default passes the
        # commit's files, which is what the hook must judge.
        assert hook.get("pass_filenames") is not False, hook_id


def test_the_chart_tests_run_where_resvg_and_the_font_are_installed() -> None:
    """The PNG tests fail, not skip, in CI; the jobs that run them must provide what they need."""
    steps = _job("fast")["steps"]
    runs = [step.get("run", "") for step in steps]
    install = next(i for i, run in enumerate(runs) if "scripts/install-chart-tools.sh" in run)

    assert "GITHUB_PATH" in runs[install]
    assert "FONTS_DIR" in runs[install]
    assert install < runs.index("make chart-render-test") < runs.index("make test-unit")

    # The browser job builds real reports with the CLI, which draws with the installed ECharts.
    browser = [step.get("run", "") for step in _job("browser-e2e")["steps"]]
    tests = next(i for i, run in enumerate(browser) if "pytest tests/e2e" in run)
    assert "make chart-render-install" in browser[:tests]


def test_every_postgresql_suite_runs_in_ci_or_says_why_not() -> None:
    command = _named_step(_job("integration"), _PG_TESTS)["run"]
    selected = set(re.findall(r"tests/integration/test_\w+\.py", command))
    # The contract is about the committed repository, not a developer's staged or untracked work:
    # a local WIP suite is not in the checkout CI scans and cannot be a suite CI forgot.
    # ``git ls-tree HEAD`` is that committed set, so a clean checkout and a dirty one answer the
    # same question.
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(_ROOT),
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
            "--",
            "tests/integration",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    suites = {
        path for path in tracked if path.endswith(".py") and Path(path).name.startswith("test_")
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


def test_ci_builds_the_postgresql_image_it_runs() -> None:
    """The image is built here, not pulled: CI runs exactly what a deployment builds
    from postgres/, with no registry in between."""
    job = _job("integration")
    build = _named_step(job, "Build the PostgreSQL image")

    assert job["env"]["POSTGRES_IMAGE"] == "dlightrag-postgres:pg18"
    assert build["with"]["context"] == "postgres/"
    assert build["with"]["tags"] == "dlightrag-postgres:pg18"
    assert build["with"]["load"] is True


def test_ci_uses_no_secret_and_calls_no_paid_service() -> None:
    workflow = json.dumps(yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))).lower()

    for forbidden in (
        "secrets.",
        "github_token",
        "credentials",
        "api_key",
        "openai",
        "anthropic",
    ):
        assert forbidden not in workflow


def _publishing_steps() -> list[tuple[str, dict[str, Any]]]:
    """Every workflow step that can publish an image, whatever it builds with."""
    published: list[tuple[str, dict[str, Any]]] = []
    for path in sorted((_ROOT / ".github/workflows").glob("*.yml")):
        workflow = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for job in (workflow.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                with_ = step.get("with") or {}
                pushes = str(with_.get("push", "")).lower() == "true" or bool(
                    re.search(r"\bdocker\s+push\b", str(step.get("run") or ""))
                )
                if pushes:
                    published.append((path.name, step))
    return published


def test_no_workflow_pulls_or_publishes_an_image() -> None:
    """The repository ships as a repository: a deployment clones it and builds the
    PostgreSQL image from postgres/, so no workflow may reach a registry and no
    credential, package visibility or digest pin exists to go stale."""
    assert _publishing_steps() == []
    for workflow, step in _publishing_steps():
        assert (step.get("with") or {}).get("context") == "postgres/", workflow
    for path in sorted((_ROOT / ".github/workflows").glob("*.yml")):
        assert "ghcr.io" not in path.read_text(encoding="utf-8"), path.name


def test_the_postgres_image_copies_no_application_source() -> None:
    """The image builds from postgres/ alone, and one file in it is ours: init.sql."""
    dockerfile = (_ROOT / "postgres/Dockerfile").read_text(encoding="utf-8")

    copies = [line for line in dockerfile.splitlines() if line.startswith(("COPY", "ADD"))]

    assert copies == ["COPY init.sql /docker-entrypoint-initdb.d/"]
