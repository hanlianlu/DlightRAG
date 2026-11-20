# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The MinerU manager: env rules, version range, installer, service, launcher.

The shell component this replaced is gone; the artifacts it rendered were captured
into `tests/unit/fixtures/mineru/` first, and the render-parity tests below compare
against those captures. The one intended difference is the removed Gradio WebUI, and
the lifecycle commands still clear what an older install left behind (ADR 0054).
"""

import ast
import dataclasses
import json
import os
import plistlib
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from dlightrag_mineru import (
    cli,
    env,
    install,
    launchd,
    service,
    systemd,
    title_aided,
    version_range,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures" / "mineru"
CLIENT = Path("lightrag/parser/external/mineru/client.py")


def _write_client(environment: Path, body: str) -> Path:
    path = environment / "lib" / "python3.13" / "site-packages" / CLIENT
    path.parent.mkdir(parents=True)
    path.write_text(body, encoding="utf-8")
    return path


def _fake_binary(directory: Path, name: str, log: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{log}"\nexit 0\n', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


# ── env ────────────────────────────────────────────────────────────────────


def test_environment_wins_over_the_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.mineru"
    env_file.write_text("MINERU_API_PORT=9000\n", encoding="utf-8")
    values = env.effective(("MINERU_API_PORT",), environ={"MINERU_API_PORT": "8210"}, path=env_file)
    assert values == {"MINERU_API_PORT": "8210"}


def test_a_set_but_empty_environment_value_is_not_absent(tmp_path: Path) -> None:
    """``MINERU_MAX_VERSION=`` lifts the ceiling; an unset one keeps the default.

    The shell read this through ``${VAR-default}``, which defaults on unset only.
    """
    env_file = tmp_path / ".env.mineru"
    env_file.write_text("", encoding="utf-8")
    values = env.effective(
        ("MINERU_MAX_VERSION",), environ={"MINERU_MAX_VERSION": ""}, path=env_file
    )
    assert values == {"MINERU_MAX_VERSION": ""}


def test_the_file_fills_a_gap_and_the_last_occurrence_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.mineru"
    env_file.write_text(
        '# comment\nMINERU_API_HOST=10.0.0.1\nMINERU_API_HOST="127.0.0.1"\n', encoding="utf-8"
    )
    assert env.effective(("MINERU_API_HOST",), environ={}, path=env_file) == {
        "MINERU_API_HOST": "127.0.0.1"
    }


def test_a_key_absent_from_both_is_absent_from_the_result(tmp_path: Path) -> None:
    assert env.effective(("MINERU_API_PORT",), environ={}, path=tmp_path / "missing") == {}


def test_env_file_follows_mineru_env_file_and_the_project_root(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert env.env_file_path(environ={}, start=nested) == tmp_path / ".env.mineru"
    relative = env.env_file_path(environ={"MINERU_ENV_FILE": "custom.env"}, start=nested)
    assert relative == tmp_path / "custom.env"


def test_env_file_honors_an_absolute_path_and_an_empty_fallback(tmp_path: Path) -> None:
    absolute = tmp_path / "custom.env"
    assert env.env_file_path(environ={"MINERU_ENV_FILE": str(absolute)}, start=tmp_path) == absolute
    assert (
        env.env_file_path(environ={"MINERU_ENV_FILE": ""}, start=tmp_path)
        == tmp_path / ".env.mineru"
    )


def test_a_set_but_empty_file_value_is_deliberately_empty(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.mineru"
    env_file.write_text("MINERU_MAX_VERSION=\n", encoding="utf-8")
    values = env.effective(("MINERU_MAX_VERSION", "MINERU_MIN_VERSION"), environ={}, path=env_file)
    assert values == {"MINERU_MAX_VERSION": ""}


# ── version range ──────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("url = '/tasks'\n", version_range.V3),
        ("url = '/v1/parse/jobs'\n", version_range.V4),
        ("url = '/v1/uploads'\n", version_range.V4),
        ("nothing here\n", version_range.UNKNOWN),
    ],
)
def test_classify_reads_the_client_the_client_calls(
    tmp_path: Path, body: str, expected: str
) -> None:
    assert version_range.classify(body) == expected


def test_resolve_reports_the_family_and_where_it_was_read(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    client = _write_client(environment, "url = '/tasks'\n")
    family, source = version_range.resolve(explicit=environment, environ={})
    assert (family, source) == (version_range.V3, client)


def test_resolve_without_a_client_is_unknown(tmp_path: Path) -> None:
    assert version_range.resolve(explicit=tmp_path / "venv", environ={}) == (
        version_range.UNKNOWN,
        None,
    )


def test_derived_range_follows_the_family_and_the_platform() -> None:
    assert version_range.derived_range(version_range.V3, is_apple_silicon=True) == (
        "3.4.5",
        "4.0",
        "core,mlx",
    )
    assert version_range.derived_range(version_range.V3, is_apple_silicon=False) == (
        "3.4.5",
        "4.0",
        "core",
    )
    assert version_range.derived_range(version_range.V4, is_apple_silicon=True) == ("4.0", "", "")
    assert version_range.derived_range(version_range.UNKNOWN, is_apple_silicon=False) == (
        "3.4.5",
        "4.0",
        "core",
    )


def test_an_exact_pin_at_or_above_the_ceiling_is_refused() -> None:
    refusal = version_range.unsupported_pin("4.0", "4.0", version_range.V3)
    assert refusal is not None
    assert "Upgrade LightRAG first" in refusal
    assert version_range.unsupported_pin("3.4.5", "4.0", version_range.V3) is None


def test_project_environment_prefers_uv_then_virtualenv_then_dotvenv(tmp_path: Path) -> None:
    uv = tmp_path / "uv-env"
    virtualenv = tmp_path / "virtualenv"

    assert version_range.project_environment(explicit=uv, environ={}, start=tmp_path) == uv
    assert (
        version_range.project_environment(
            environ={"UV_PROJECT_ENVIRONMENT": str(uv), "VIRTUAL_ENV": str(virtualenv)},
            start=tmp_path,
        )
        == uv
    )
    assert (
        version_range.project_environment(environ={"VIRTUAL_ENV": str(virtualenv)}, start=tmp_path)
        == virtualenv
    )
    assert version_range.project_environment(environ={}, start=tmp_path) == tmp_path / ".venv"


def test_a_lifted_ceiling_allows_any_pin() -> None:
    assert version_range.unsupported_pin("4.0.10", "", version_range.V4) is None


# ── install ────────────────────────────────────────────────────────────────


def _project(tmp_path: Path) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return tmp_path


def test_plan_builds_the_spec_from_the_client_and_the_platform(tmp_path: Path) -> None:
    project = _project(tmp_path)
    environment = tmp_path / "client-env"
    _write_client(environment, "url = '/tasks'\n")
    resolved = install.plan(environ={}, start=project, project_environment=environment)
    assert isinstance(resolved, install.InstallPlan)
    assert resolved.family == version_range.V3
    assert resolved.package.startswith("mineru[") and resolved.package.endswith(">=3.4.5,<4.0")


def test_plan_lets_an_empty_ceiling_through(tmp_path: Path) -> None:
    project = _project(tmp_path)
    environment = tmp_path / "client-env"
    _write_client(environment, "url = '/tasks'\n")
    resolved = install.plan(
        environ={"MINERU_MAX_VERSION": ""}, start=project, project_environment=environment
    )
    assert isinstance(resolved, install.InstallPlan)
    assert "<4.0" not in resolved.package


def test_plan_resolves_a_relative_venv_against_the_project_root(tmp_path: Path) -> None:
    project = _project(tmp_path)
    resolved = install.plan(
        environ={"MINERU_SERVICE_VENV": "envs/mineru"},
        start=project,
        project_environment=tmp_path / "client-env",
    )
    assert isinstance(resolved, install.InstallPlan)
    assert resolved.environment == (project / "envs" / "mineru").resolve()


def test_plan_refuses_a_pin_the_client_cannot_call(tmp_path: Path) -> None:
    project = _project(tmp_path)
    environment = tmp_path / "client-env"
    _write_client(environment, "url = '/tasks'\n")
    refusal = install.plan(
        environ={"MINERU_VERSION": "4.1"}, start=project, project_environment=environment
    )
    assert isinstance(refusal, install.Refusal)
    assert refusal.code == 64
    assert "not supported" in refusal.message


def test_plan_builds_the_v3_v4_and_unknown_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install, "is_apple_silicon", lambda: False)
    project = _project(tmp_path)
    v3 = tmp_path / "v3-env"
    _write_client(v3, "url = '/tasks'\n")
    v4 = tmp_path / "v4-env"
    _write_client(v4, "url = '/v1/parse/jobs'\n")
    missing_env = {"MINERU_ENV_FILE": str(tmp_path / "missing.env")}

    v3_plan = install.plan(environ=missing_env, start=project, project_environment=v3)
    v4_plan = install.plan(environ=missing_env, start=project, project_environment=v4)
    unknown_plan = install.plan(
        environ=missing_env, start=project, project_environment=tmp_path / "empty"
    )

    assert isinstance(v3_plan, install.InstallPlan)
    assert v3_plan.package == "mineru[core]>=3.4.5,<4.0"
    assert isinstance(v4_plan, install.InstallPlan)
    assert v4_plan.package == "mineru>=4.0"
    assert isinstance(unknown_plan, install.InstallPlan)
    assert unknown_plan.package == "mineru[core]>=3.4.5,<4.0"


def test_plan_honors_version_min_and_extras_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install, "is_apple_silicon", lambda: False)
    project = _project(tmp_path)
    client = tmp_path / "v3-env"
    _write_client(client, "url = '/tasks'\n")

    pinned = install.plan(
        environ={
            "MINERU_ENV_FILE": str(tmp_path / "missing.env"),
            "MINERU_VERSION": "3.4.5",
            "MINERU_INSTALL_EXTRAS": "core,lmdeploy",
        },
        start=project,
        project_environment=client,
    )
    assert isinstance(pinned, install.InstallPlan)
    assert pinned.package == "mineru[core,lmdeploy]==3.4.5,>=3.4.5,<4.0"

    floored = install.plan(
        environ={"MINERU_ENV_FILE": str(tmp_path / "missing.env"), "MINERU_MIN_VERSION": "3.5.0"},
        start=project,
        project_environment=client,
    )
    assert isinstance(floored, install.InstallPlan)
    assert floored.package == "mineru[core]>=3.5.0,<4.0"


def test_materialize_places_the_guards_in_site_packages_and_the_launcher_in_the_venv(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    (environment / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    launcher = install.materialize(environment)

    directory = environment / "dlightrag-mineru"
    site_packages = environment / "lib" / "python3.13" / "site-packages"
    package = Path(install.__file__).parent

    assert launcher == directory / "launcher.py"
    assert launcher.stat().st_mode & stat.S_IXUSR
    assert (directory / "_env.py").read_bytes() == (package / "env.py").read_bytes()
    # The guards ride a .pth, so they are active for any start of this interpreter,
    # not only when the launcher starts it.
    assert (site_packages / install.GUARD_MODULE).read_bytes() == (
        package / "_shim" / "_dlightrag_mineru_shim.py"
    ).read_bytes()
    assert (site_packages / install.PTH_NAME).read_text(encoding="utf-8") == install.PTH_CONTENT
    assert not list(directory.glob("*.tmp"))
    assert install.launcher_command(environment) == (
        str(environment / "bin" / "python"),
        str(directory / "launcher.py"),
    )


def test_install_drops_the_guard_pth_before_the_venv_changes_underneath_it(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "venv"
    site_packages = environment / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    install.materialize(environment)

    install.remove_guards(environment)
    assert not (site_packages / install.PTH_NAME).exists()
    # The module may stay: without the .pth nothing imports it.
    assert (site_packages / install.GUARD_MODULE).exists()


# ── render parity with the shell's captured artifacts ──────────────────────


def _spec(tmp_path: Path, home: Path) -> service.ServiceSpec:
    environment = tmp_path / "venv"
    environment.mkdir(exist_ok=True)
    return service.spec(
        environment=environment,
        working_directory=ROOT,
        env_file=home / ".env.mineru",
        launchd_home=home,
    )


def _normalize(text: str, spec: service.ServiceSpec, home: Path) -> str:
    program = str(spec.program[-1])
    return (
        text.replace(str(program), "__PROGRAM__")
        .replace(str(spec.working_directory), "__PROJECT_ROOT__")
        .replace(str(home), "__LAUNCHD_HOME__")
    )


def test_launchd_plist_keeps_every_shell_key_but_the_program(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    spec = _spec(tmp_path, home)

    expected = plistlib.loads((FIXTURES / "launchd-api.plist").read_bytes())
    rendered = plistlib.loads(
        _normalize(launchd.render(spec).decode("utf-8"), spec, home).encode("utf-8")
    )
    rendered["ProgramArguments"] = ["__PROGRAM__"]
    assert rendered == expected
    assert plistlib.loads(launchd.render(spec))["ProgramArguments"] == list(spec.program)


def _unit_pairs(text: str) -> dict[str, str]:
    """Key/value pairs, with the unit-file quoting removed for comparison."""
    pairs: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("[") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        pairs[key] = value.replace('"', "")
    return pairs


def test_systemd_unit_keeps_every_shell_line_but_exec_start(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    spec = _spec(tmp_path, home)

    fixture = (FIXTURES / "systemd-api.service").read_text(encoding="utf-8")
    rendered = systemd.render(spec)
    expected = _unit_pairs(fixture)
    actual = _unit_pairs(rendered)
    expected.pop("ExecStart")
    actual.pop("ExecStart")

    assert actual == {
        key: value.replace("__PROJECT_ROOT__", str(ROOT)).replace("__LAUNCHD_HOME__", str(home))
        for key, value in expected.items()
    }
    exec_start = next(line for line in rendered.splitlines() if line.startswith("ExecStart="))
    assert shlex.split(exec_start.removeprefix("ExecStart=")) == list(spec.program)


def test_systemd_quotes_a_program_path_that_contains_a_space(tmp_path: Path) -> None:
    environment = tmp_path / "a venv"
    environment.mkdir()
    spec = service.spec(
        environment=environment,
        working_directory=tmp_path,
        env_file=tmp_path / ".env.mineru",
        launchd_home=tmp_path,
    )
    exec_start = next(
        line for line in systemd.render(spec).splitlines() if line.startswith("ExecStart=")
    )
    assert shlex.split(exec_start.removeprefix("ExecStart=")) == list(spec.program)


def test_systemd_refuses_a_path_carrying_a_control_character(tmp_path: Path) -> None:
    spec = service.ServiceSpec(
        name="api",
        description="d",
        environment=tmp_path / "venv",
        program=("python", "/tmp/x\nExecStartPre=/bin/sh -c true"),
        working_directory=tmp_path,
        env_file=tmp_path / ".env.mineru",
        stdout_log=tmp_path / "out.log",
        stderr_log=tmp_path / "err.log",
    )
    with pytest.raises(ValueError, match="control character"):
        systemd.render(spec)


# ── command parity ─────────────────────────────────────────────────────────


def _recorder(tmp_path: Path, name: str) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    log = tmp_path / f"{name}.log"
    _fake_binary(bin_dir, name, log)
    return bin_dir, log


def test_launchd_install_bootstraps_the_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    bin_dir, log = _recorder(tmp_path, "launchctl")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("MINERU_LAUNCHD_HOME", str(home))

    spec = _spec(tmp_path, home)
    assert launchd.run("install", spec) == 0

    commands = log.read_text(encoding="utf-8").splitlines()
    assert commands == [
        f"bootout gui/{os.getuid()}/{spec.label}",
        f"bootstrap gui/{os.getuid()} {launchd.plist_path(spec)}",
    ]
    assert plistlib.loads(launchd.plist_path(spec).read_bytes())["Label"] == spec.label


def test_launchd_start_stop_and_uninstall_drive_the_expected_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    bin_dir, log = _recorder(tmp_path, "launchctl")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("MINERU_LAUNCHD_HOME", str(home))
    spec = _spec(tmp_path, home)

    assert launchd.run("start", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"bootstrap gui/{os.getuid()} {launchd.plist_path(spec)}"
    ]

    log.write_text("", encoding="utf-8")
    assert launchd.run("stop", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{os.getuid()}/{spec.label}",
    ]

    log.write_text("", encoding="utf-8")
    assert launchd.run("uninstall", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{os.getuid()}/{spec.label}",
    ]


def test_launchd_retries_a_bootstrap_that_keeps_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "count"
    binary = bin_dir / "launchctl"
    binary.write_text(
        "#!/usr/bin/env bash\n"
        f'if [[ "$1" == "bootstrap" ]]; then echo x >> "{counter}"; '
        f'[[ "$(wc -l < "{counter}")" -ge 5 ]] || exit 1; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    spec = _spec(tmp_path, home)
    assert launchd.run("install", spec) == 0
    assert len(counter.read_text(encoding="utf-8").splitlines()) == 5


def test_systemd_install_disables_then_enables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir, log = _recorder(tmp_path, "systemctl")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    spec = _spec(tmp_path, tmp_path / "home")

    assert systemd.run("install", spec) == 0
    commands = log.read_text(encoding="utf-8").splitlines()
    assert commands == [
        f"--user disable --now {spec.unit}",
        "--user daemon-reload",
        f"--user enable --now {spec.unit}",
    ]
    assert systemd.unit_path(spec).read_text(encoding="utf-8") == systemd.render(spec)


def test_systemd_start_stop_status_and_uninstall_drive_the_expected_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir, log = _recorder(tmp_path, "systemctl")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    spec = _spec(tmp_path, tmp_path / "home")

    assert systemd.run("start", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [f"--user start {spec.unit}"]

    log.write_text("", encoding="utf-8")
    assert systemd.run("stop", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"--user stop {spec.unit}",
    ]

    log.write_text("", encoding="utf-8")
    assert systemd.run("status", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [f"--user status {spec.unit}"]

    log.write_text("", encoding="utf-8")
    assert systemd.run("uninstall", spec) == 0
    assert log.read_text(encoding="utf-8").splitlines() == [
        f"--user disable --now {spec.unit}",
        "--user daemon-reload",
    ]


def test_a_missing_systemctl_refuses_with_the_foreground_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    spec = _spec(tmp_path, tmp_path / "home")
    assert systemd.run("install", spec) == 127
    assert "foreground" in capsys.readouterr().err


# ── the launcher inside the service venv ───────────────────────────────────


def _installed_venv(tmp_path: Path) -> tuple[Path, Path]:
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    install.materialize(environment)
    dump = tmp_path / "dump.json"
    binary = environment / "bin" / "mineru-api"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['FAKE_DUMP'], 'w') as fh:\n"
        "    json.dump({'argv': sys.argv, 'env': dict(os.environ)}, fh)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return environment, dump


def test_the_launcher_loads_the_env_file_and_execs_mineru(tmp_path: Path) -> None:
    environment, dump = _installed_venv(tmp_path)
    env_file = tmp_path / ".env.mineru"
    env_file.write_text(
        "MINERU_API_HOST=127.0.0.1\nMINERU_API_PORT=8321\nMINERU_TITLE_AIDED_API_KEY=sk-secret\n",
        encoding="utf-8",
    )
    launcher = environment / "dlightrag-mineru" / "launcher.py"

    completed = subprocess.run(
        [sys.executable, str(launcher), "--extra"],
        env={
            "PATH": os.environ["PATH"],
            "MINERU_ENV_FILE": str(env_file),
            "FAKE_DUMP": str(dump),
            "MINERU_TITLE_AIDED_API_KEY": "sk-inherited",
        },
        check=False,
    )
    assert completed.returncode == 0

    payload = json.loads(dump.read_text(encoding="utf-8"))
    assert payload["argv"][1:] == ["--host", "127.0.0.1", "--port", "8321", "--extra"]
    assert "MINERU_TITLE_AIDED_API_KEY" not in payload["env"], (
        "a credential must not reach the parser"
    )


def test_the_launcher_reports_a_missing_mineru_and_a_bad_binding(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    (environment / "bin").mkdir(parents=True)
    (environment / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    install.materialize(environment)
    launcher = environment / "dlightrag-mineru" / "launcher.py"
    env_file = tmp_path / ".env.mineru"
    env_file.write_text("MINERU_API_PORT=not-a-port\n", encoding="utf-8")

    missing = subprocess.run(
        [sys.executable, str(launcher)],
        env={"PATH": os.environ["PATH"], "MINERU_ENV_FILE": str(env_file)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode == 127
    assert "mineru-api was not found" in missing.stderr

    (environment / "bin" / "mineru-api").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    bad_port = subprocess.run(
        [sys.executable, str(launcher)],
        env={"PATH": os.environ["PATH"], "MINERU_ENV_FILE": str(env_file)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_port.returncode == 64
    assert "not a host and a port" in bad_port.stderr


def test_the_shim_and_launcher_compile_under_the_service_interpreter() -> None:
    interpreter = ROOT / ".venv-mineru" / "bin" / "python"
    if not interpreter.is_file():
        pytest.skip("no service venv on this host")
    shim = Path(install.__file__).parent / "_shim"
    for name in ("launcher.py", "_dlightrag_mineru_shim.py"):
        completed = subprocess.run(
            [str(interpreter), "-m", "py_compile", str(shim / name)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


# ── the CLI surface ────────────────────────────────────────────────────────


def test_an_unknown_service_action_returns_64(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["service", "explode"]) == 64
    assert "Usage: dlightrag-mineru service" in capsys.readouterr().err


def test_service_refuses_before_the_launcher_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    assert cli.main(["service", "install"]) == 127
    assert "dlightrag-mineru install" in capsys.readouterr().err


def test_help_is_available_for_the_wheel_gate(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "MinerU" in capsys.readouterr().out


# ── title-aided ────────────────────────────────────────────────────────────


def test_disabling_keeps_every_other_key(tmp_path: Path) -> None:
    target = tmp_path / "mineru.json"
    target.write_text(
        json.dumps(
            {"models-dir": "/models", "llm-aided-config": {"title_aided": {"enable": True}}}
        ),
        encoding="utf-8",
    )
    assert title_aided.disable_config(target) == [f"==> Disabled title-aided config in {target}"]
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["models-dir"] == "/models"
    assert data["llm-aided-config"]["title_aided"] == {"enable": False}


def test_enabling_writes_0600_atomically_and_never_echoes_the_key(tmp_path: Path) -> None:
    target = tmp_path / "mineru.json"
    request = title_aided.Request(
        target=target, api_key="sk-secret", base_url="https://api.example/v1", model="m-1"
    )
    lines = title_aided.write_config(request)

    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".*.tmp"))
    assert "sk-secret" not in "\n".join(lines)
    assert "set (9 chars)" in "\n".join(lines)


def test_from_environment_reads_the_file_and_injects_the_resolver(tmp_path: Path) -> None:
    env_file = tmp_path / ".env.mineru"
    env_file.write_text(
        "MINERU_TITLE_AIDED_ENABLE=true\n"
        "MINERU_TITLE_AIDED_API_KEY=sk-abc\n"
        "MINERU_TITLE_AIDED_BASE_URL=https://api.example/v1\n"
        "MINERU_TITLE_AIDED_MODEL=m-1\n"
        "MINERU_TITLE_AIDED_ENABLE_THINKING=TRUE\n",
        encoding="utf-8",
    )
    monkeypatch_free = {"MINERU_ENV_FILE": str(env_file)}
    target = tmp_path / "mineru.json"
    asked: list[tuple[str, str]] = []

    code = title_aided.from_environment(
        environ=monkeypatch_free,
        target=target,
        resolver=lambda base_url, model: asked.append((base_url, model)) or {"x": 1},
    )

    assert code == 0
    # TRUE means "leave reasoning to the provider", in any case, as the shell read it.
    assert asked == []
    assert (
        json.loads(target.read_text(encoding="utf-8"))["llm-aided-config"]["title_aided"][
            "extra_body"
        ]
        == {}
    )


def test_from_environment_reports_every_missing_value(tmp_path: Path) -> None:
    reported: list[str] = []
    code = title_aided.from_environment(
        environ={"MINERU_ENV_FILE": str(tmp_path / "missing")},
        target=tmp_path / "mineru.json",
        report=lambda lines: reported.extend(lines),
    )
    assert code == 1
    assert sum(line.startswith("ERROR:") for line in reported) == 3


# ── purge: a destructive command with guards ───────────────────────────────


def _service_venv(tmp_path: Path) -> Path:
    environment = tmp_path / "venv"
    (environment / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    (environment / "bin").mkdir()
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    install.materialize(environment)
    return environment


def test_purge_refuses_a_directory_that_is_not_a_service_environment(tmp_path: Path) -> None:
    assert install.purge(tmp_path / "venv", confirmed=True) == 1


def test_purge_says_what_it_would_remove_before_removing_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environment = _service_venv(tmp_path)
    assert install.purge(environment, dry_run=True) == 0
    assert environment.is_dir()
    assert "would remove" in capsys.readouterr().out


def test_purge_requires_confirmation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    environment = _service_venv(tmp_path)
    assert install.purge(environment) == 64
    assert environment.is_dir()
    assert "--yes" in capsys.readouterr().out


def test_purge_removes_the_venv_it_installed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environment = _service_venv(tmp_path)
    assert install.purge(environment, confirmed=True) == 0
    assert not environment.exists()
    assert "service uninstall" in capsys.readouterr().out


# ── the unit as systemd itself reads it ────────────────────────────────────


def test_the_rendered_unit_passes_systemd_analyze(tmp_path: Path) -> None:
    """systemd validates the unit, not just us.

    CI is Linux, so a missing validator fails there rather than skipping quietly —
    the same rule the chart-render tests follow.
    """
    analyzer = shutil.which("systemd-analyze")
    if analyzer is None:
        if os.environ.get("CI"):
            pytest.fail("systemd-analyze must exist in CI so this gate cannot silently not run")
        pytest.skip("systemd-analyze ships with Linux")

    # A venv path with a space: that is the input the quoting exists for, and an
    # unquoted ExecStart would hand systemd a truncated argv[0].
    environment = tmp_path / "a venv"
    (environment / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    (environment / "bin").mkdir()
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    launcher = install.materialize(environment)
    spec = service.spec(
        environment=environment,
        working_directory=tmp_path,
        env_file=tmp_path / ".env.mineru",
        launchd_home=tmp_path,
    )
    # Real executables: verify should judge the unit, not complain about a fake path.
    spec = dataclasses.replace(spec, program=(sys.executable, str(launcher)))
    unit = tmp_path / spec.unit
    unit.write_text(systemd.render(spec), encoding="utf-8")

    completed = subprocess.run(
        [analyzer, "verify", str(unit)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


# ── the modules the service venv has to parse ──────────────────────────────


@pytest.mark.parametrize(
    "relative", ["env.py", "_shim/launcher.py", "_shim/_dlightrag_mineru_shim.py"]
)
def test_the_copied_modules_parse_on_the_service_pythons_grammar(relative: str) -> None:
    """The service venv is 3.13 while this project targets 3.14.

    ``env.py`` is copied verbatim into that venv, so a 3.14-only spelling the formatter
    is happy with (a bare ``except A, B:``) would stop the sidecar dead. Parsing with
    the older grammar catches it everywhere, venv or no venv.
    """
    source = (Path(install.__file__).parent / relative).read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 13))


# ── an unreadable env file is loud ─────────────────────────────────────────


def test_an_unreadable_env_file_is_an_error_not_an_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / ".env.mineru"
    env_file.write_text("MINERU_API_PORT=8210\n", encoding="utf-8")
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-0 file")
    env_file.chmod(0)
    with pytest.raises(env.EnvFileError):
        env.effective(("MINERU_API_PORT",), environ={}, path=env_file)


def test_a_missing_env_file_stays_silent(tmp_path: Path) -> None:
    assert env.effective(("MINERU_API_PORT",), environ={}, path=tmp_path / "gone") == {}


def test_a_path_that_is_not_a_directory_is_an_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("", encoding="utf-8")
    with pytest.raises(env.EnvFileError):
        env.effective(("MINERU_API_PORT",), environ={}, path=blocker / ".env.mineru")


def test_the_cli_reports_an_unreadable_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env_file = tmp_path / ".env.mineru"
    env_file.write_text("", encoding="utf-8")
    if os.geteuid() == 0:
        pytest.skip("root reads a mode-0 file")
    env_file.chmod(0)
    monkeypatch.setenv("MINERU_ENV_FILE", str(env_file))
    assert cli.main(["title-aided", "--target", str(tmp_path / "out.json")]) == 1
    assert "ERROR" in capsys.readouterr().err


# ── status reports what the manager says ───────────────────────────────────


@pytest.mark.parametrize("manager", ["launchctl", "systemctl"])
def test_status_propagates_a_failing_manager(
    manager: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    binary = bin_dir / manager
    binary.write_text("#!/usr/bin/env bash\nexit 3\n", encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("MINERU_LAUNCHD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    spec = _spec(tmp_path, tmp_path / "home")
    module = launchd if manager == "launchctl" else systemd
    assert module.run("status", spec) == 3


# ── the guards ride the venv, and go away while it is rewritten ────────────


def test_install_pulls_the_guard_pth_before_uv_replaces_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project(tmp_path)
    environment = project / ".venv-mineru"
    site_packages = environment / "lib" / "python3.13" / "site-packages"
    site_packages.mkdir(parents=True)
    (environment / "bin").mkdir()
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    install.materialize(environment)

    log = tmp_path / "uv.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "CALL $*" >> "{log}"\n'
        f'ls "{site_packages}" >> "{log}" 2>&1\n'
        "exit 0\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.chdir(project)

    assert install.install() == 0
    calls = log.read_text(encoding="utf-8").splitlines()
    pip_at = next(index for index, line in enumerate(calls) if line.startswith("CALL pip install"))
    assert install.PTH_NAME not in calls[pip_at:], "the guards must be off during the upgrade"
    assert (site_packages / install.PTH_NAME).is_file(), "and back afterwards"


def test_the_guard_pth_loads_at_interpreter_start(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    created = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"cannot create a venv here: {created.stderr.strip()}")
    install.materialize(environment)

    probe = subprocess.run(
        [
            str(environment / "bin" / "python"),
            "-c",
            "import sys; print('_dlightrag_mineru_shim' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # The guards import MinerU and Pillow, absent from this bare venv, so the import may
    # fail — but the name can only appear if site ran our .pth line.
    assert "_dlightrag_mineru_shim" in probe.stdout + probe.stderr


# ── purge's guards, and the platform refusal ───────────────────────────────


def test_purge_refuses_a_venv_without_our_manifest(tmp_path: Path) -> None:
    environment = tmp_path / "venv"
    (environment / "dlightrag-mineru").mkdir(parents=True)
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (environment / "dlightrag-mineru" / install.LAUNCHER_NAME).write_text("", encoding="utf-8")
    assert install.purge(environment, confirmed=True) == 1
    assert environment.is_dir()


def test_the_cli_wires_purge_to_the_service_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project = _project(tmp_path)
    environment = project / ".venv-mineru"
    (environment / "lib" / "python3.13" / "site-packages").mkdir(parents=True)
    (environment / "bin").mkdir()
    (environment / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    install.materialize(environment)
    monkeypatch.chdir(project)

    assert cli.main(["purge", "--dry-run"]) == 0
    assert str(environment) in capsys.readouterr().out
    assert environment.is_dir()


@pytest.mark.parametrize(
    "command", [["install"], ["api"], ["service", "status"], ["purge", "--yes"]]
)
def test_a_non_posix_host_is_refused(
    command: list[str], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(os, "name", "nt")
    assert cli.main(command) == 64
    assert "macOS, Linux, and WSL2" in capsys.readouterr().err
