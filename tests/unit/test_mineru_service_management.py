# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for local MinerU service management helpers."""

from __future__ import annotations

import os
import plistlib
import stat
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MINERU_SCRIPTS = ROOT / "scripts" / "mineru"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_pyproject_keeps_mineru_out_of_dlightrag_runtime() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    requirements = list(pyproject["project"]["dependencies"])
    optional = pyproject["project"].get("optional-dependencies", {})

    assert all(not requirement.startswith("mineru") for requirement in requirements)
    assert all(not extra.startswith("mineru") for extra in optional)


def test_makefile_delegates_mineru_defaults_to_scripts() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "MINERU_INSTALL_EXTRAS ?=" not in makefile
    assert "MINERU_API_HOST ?=" not in makefile
    assert "MINERU_API_PORT ?=" not in makefile
    assert "MINERU_SERVICE_VENV ?=" not in makefile
    assert "\nmineru-install:\n\tscripts/mineru/install.sh\n" in makefile
    assert "\nmineru-api:\n\tscripts/mineru/api.sh\n" in makefile


def test_makefile_targets_are_thin_script_wrappers() -> None:
    install = subprocess.run(
        ["make", "-n", "mineru-install"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    api = subprocess.run(
        ["make", "-n", "mineru-api"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert install.stdout.strip() == "scripts/mineru/install.sh"
    assert api.stdout.strip() == "scripts/mineru/api.sh"


def test_mineru_helper_defaults_to_separate_env_file() -> None:
    env_script = (MINERU_SCRIPTS / "env.sh").read_text(encoding="utf-8")

    assert 'mineru_env_file="${MINERU_ENV_FILE:-$mineru_repo_root/.env.mineru}"' in env_script


def test_mineru_env_example_documents_install_extras() -> None:
    example = (ROOT / ".env.mineru.example").read_text(encoding="utf-8")

    assert "MINERU_VERSION=3.2.3" in example
    assert "MINERU_INSTALL_EXTRAS=core,mlx" in example
    assert "# MINERU_INSTALL_EXTRAS=core" in example
    assert "MINERU_SERVICE_VENV=.venv-mineru" in example
    assert "MINERU_API_PORT=8210" in example


def test_makefile_exposes_mineru_launch_agent_targets() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "\nmineru-service-install:\n\tscripts/mineru/launch_agent.sh install\n" in makefile
    assert "\nmineru-service-start:\n\tscripts/mineru/launch_agent.sh start\n" in makefile
    assert "\nmineru-service-stop:\n\tscripts/mineru/launch_agent.sh stop\n" in makefile
    assert "\nmineru-service-status:\n\tscripts/mineru/launch_agent.sh status\n" in makefile
    assert "\nmineru-service-logs:\n\tscripts/mineru/launch_agent.sh logs\n" in makefile
    assert "\nmineru-service-uninstall:\n\tscripts/mineru/launch_agent.sh uninstall\n" in makefile


def test_legacy_top_level_mineru_helpers_are_removed() -> None:
    assert not (ROOT / "scripts" / "install_mineru_service.sh").exists()
    assert not (ROOT / "scripts" / "start_mineru_api.sh").exists()
    assert not (ROOT / "scripts" / "mineru_launch_agent.sh").exists()
    assert not (ROOT / "scripts" / "mineru_env.sh").exists()


def test_makefile_passes_command_line_overrides_to_scripts(tmp_path: Path) -> None:
    capture = tmp_path / "uv.txt"
    bin_dir = tmp_path / "bin"
    service_env = tmp_path / "mineru-env"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    _write_executable(
        fake_uv,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)

    subprocess.run(
        [
            "make",
            f"MINERU_SERVICE_VENV={service_env}",
            "MINERU_VERSION=3.2.3",
            "MINERU_INSTALL_EXTRAS=core,vllm",
            "mineru-install",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,vllm]==3.2.3",
    ]


def test_mineru_installer_creates_dedicated_service_env(tmp_path: Path) -> None:
    capture = tmp_path / "uv.txt"
    bin_dir = tmp_path / "bin"
    service_env = tmp_path / "mineru-env"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    _write_executable(
        fake_uv,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    env.pop("MINERU_INSTALL_EXTRAS", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    default_extras = (
        "core,mlx"
        if os.uname().sysname == "Darwin" and os.uname().machine in {"arm64", "aarch64"}
        else "core"
    )
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[{default_extras}]",
    ]


def test_mineru_installer_defaults_to_mlx_on_apple_silicon(tmp_path: Path) -> None:
    capture = tmp_path / "uv.txt"
    bin_dir = tmp_path / "bin"
    service_env = tmp_path / "mineru-env"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uname = bin_dir / "uname"
    _write_executable(
        fake_uv,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )
    _write_executable(
        fake_uname,
        '#!/usr/bin/env bash\nif [[ "$1" == "-s" ]]; then echo Darwin; else echo arm64; fi\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    env.pop("MINERU_INSTALL_EXTRAS", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,mlx]",
    ]


def test_mineru_installer_defaults_to_core_off_apple_silicon(tmp_path: Path) -> None:
    capture = tmp_path / "uv.txt"
    bin_dir = tmp_path / "bin"
    service_env = tmp_path / "mineru-env"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uname = bin_dir / "uname"
    _write_executable(
        fake_uv,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )
    _write_executable(
        fake_uname,
        '#!/usr/bin/env bash\nif [[ "$1" == "-s" ]]; then echo Linux; else echo x86_64; fi\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    env.pop("MINERU_INSTALL_EXTRAS", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core]",
    ]


def test_mineru_installer_reads_mineru_values_from_env_file(tmp_path: Path) -> None:
    capture = tmp_path / "uv.txt"
    env_file = tmp_path / ".env"
    bin_dir = tmp_path / "bin"
    service_env = tmp_path / "mineru-env"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    _write_executable(
        fake_uv,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )
    env_file.write_text(
        f"MINERU_SERVICE_VENV={service_env}\n"
        "MINERU_VERSION=3.2.3\n"
        "MINERU_INSTALL_EXTRAS=core,lmdeploy\n"
        "MINERU_LOCAL_ENDPOINT=http://ignored-by-installer:8210\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_ENV_FILE"] = str(env_file)
    env.pop("MINERU_SERVICE_VENV", None)
    env.pop("MINERU_INSTALL_EXTRAS", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,lmdeploy]==3.2.3",
    ]


def test_mineru_installer_can_pin_version(tmp_path: Path) -> None:
    capture = tmp_path / "uv.txt"
    bin_dir = tmp_path / "bin"
    service_env = tmp_path / "mineru-env"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    _write_executable(
        fake_uv,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_VERSION"] = "3.2.3"
    env["MINERU_INSTALL_EXTRAS"] = "core,mlx"
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,mlx]==3.2.3",
    ]


def test_mineru_launcher_uses_default_port_and_passes_args(tmp_path: Path) -> None:
    capture = tmp_path / "args.txt"
    service_env = tmp_path / "mineru-env"
    service_bin = service_env / "bin"
    service_bin.mkdir(parents=True)
    fake_mineru_api = service_bin / "mineru-api"
    _write_executable(
        fake_mineru_api,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env.pop("MINERU_API_HOST", None)
    env.pop("MINERU_API_PORT", None)

    script = MINERU_SCRIPTS / "api.sh"
    subprocess.run(
        [str(script), "--enable-vlm-preload", "true"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "--host",
        "127.0.0.1",
        "--port",
        "8210",
        "--enable-vlm-preload",
        "true",
    ]


def test_mineru_launcher_allows_host_and_port_override(tmp_path: Path) -> None:
    capture = tmp_path / "args.txt"
    service_env = tmp_path / "mineru-env"
    service_bin = service_env / "bin"
    service_bin.mkdir(parents=True)
    fake_mineru_api = service_bin / "mineru-api"
    _write_executable(
        fake_mineru_api,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_API_HOST"] = "0.0.0.0"
    env["MINERU_API_PORT"] = "9001"

    script = MINERU_SCRIPTS / "api.sh"
    subprocess.run([str(script)], cwd=ROOT, env=env, check=True)

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "--host",
        "0.0.0.0",
        "--port",
        "9001",
    ]


def test_mineru_launcher_reads_mineru_values_from_env_file(tmp_path: Path) -> None:
    capture = tmp_path / "args.txt"
    env_file = tmp_path / ".env"
    service_env = tmp_path / "mineru-env"
    service_bin = service_env / "bin"
    service_bin.mkdir(parents=True)
    fake_mineru_api = service_bin / "mineru-api"
    _write_executable(
        fake_mineru_api,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MINERU_CAPTURE"\n',
    )
    env_file.write_text(
        f"MINERU_SERVICE_VENV={service_env}\n"
        "MINERU_API_HOST=127.9.9.9\n"
        "MINERU_API_PORT=9999\n"
        "MINERU_LOCAL_ENDPOINT=http://ignored-by-launcher:8210\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_ENV_FILE"] = str(env_file)
    env.pop("MINERU_SERVICE_VENV", None)
    env.pop("MINERU_API_HOST", None)
    env.pop("MINERU_API_PORT", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "api.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "--host",
        "127.9.9.9",
        "--port",
        "9999",
    ]


def test_mineru_launch_agent_install_writes_plist_and_bootstraps(tmp_path: Path) -> None:
    capture = tmp_path / "launchctl.txt"
    env_file = tmp_path / ".env"
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env_file.write_text("MINERU_API_PORT=8210\n", encoding="utf-8")
    fake_launchctl = bin_dir / "launchctl"
    _write_executable(
        fake_launchctl,
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n'
        'if [[ "$1" == "print" ]]; then exit 113; fi\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_ENV_FILE"] = str(env_file)
    env["MINERU_LAUNCHD_HOME"] = str(home)

    subprocess.run(
        [str(MINERU_SCRIPTS / "launch_agent.sh"), "install"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    label = "com.hanlianlyu.dlightrag.mineru-api"
    plist_path = home / "Library" / "LaunchAgents" / f"{label}.plist"
    plist = plistlib.loads(plist_path.read_bytes())

    assert plist["Label"] == label
    assert plist["ProgramArguments"] == [str(MINERU_SCRIPTS / "api.sh")]
    assert plist["WorkingDirectory"] == str(ROOT)
    assert plist["RunAtLoad"] is True
    assert plist["KeepAlive"] is True
    assert plist["EnvironmentVariables"] == {"MINERU_ENV_FILE": str(env_file)}
    assert plist["StandardOutPath"] == str(
        home / "Library" / "Logs" / "dlightrag" / "mineru-api.out.log"
    )
    assert plist["StandardErrorPath"] == str(
        home / "Library" / "Logs" / "dlightrag" / "mineru-api.err.log"
    )
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{os.getuid()}/{label}",
        f"bootstrap gui/{os.getuid()} {plist_path}",
    ]


def test_mineru_launch_agent_stop_unloads_by_label(tmp_path: Path) -> None:
    capture = tmp_path / "launchctl.txt"
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_launchctl = bin_dir / "launchctl"
    _write_executable(
        fake_launchctl,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_LAUNCHD_HOME"] = str(home)

    subprocess.run(
        [str(MINERU_SCRIPTS / "launch_agent.sh"), "stop"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{os.getuid()}/com.hanlianlyu.dlightrag.mineru-api",
    ]
