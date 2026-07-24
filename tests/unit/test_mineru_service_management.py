# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for local MinerU service management helpers."""

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


def _makefile_target_command(target: str) -> str:
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines):
        if line == f"{target}:":
            command = lines[idx + 1]
            assert command.startswith("\t")
            return command.strip()
    raise AssertionError(f"Makefile target not found: {target}")


def test_pyproject_keeps_mineru_out_of_dlightrag_runtime() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    requirements = list(pyproject["project"]["dependencies"])
    optional = pyproject["project"].get("optional-dependencies", {})

    assert all(not requirement.startswith("mineru") for requirement in requirements)
    assert all(not extra.startswith("mineru") for extra in optional)


def test_makefile_targets_are_thin_script_wrappers() -> None:
    assert _makefile_target_command("mineru-install") == "scripts/mineru/install.sh"
    assert _makefile_target_command("mineru-api") == "scripts/mineru/api.sh"
    assert _makefile_target_command("mineru-gradio") == "scripts/mineru/gradio.sh"


def test_mineru_helper_defaults_to_separate_env_file() -> None:
    env_script = (MINERU_SCRIPTS / "env.sh").read_text(encoding="utf-8")

    assert 'mineru_env_file="${MINERU_ENV_FILE:-$mineru_repo_root/.env.mineru}"' in env_script


def test_makefile_dispatches_mineru_service_targets() -> None:
    for action in ("install", "start", "stop", "status", "logs", "uninstall"):
        assert (
            _makefile_target_command(f"mineru-service-{action}")
            == f"scripts/mineru/service.sh {action}"
        )


def test_mineru_service_dispatcher_routes_per_os() -> None:
    dispatcher = MINERU_SCRIPTS / "service.sh"
    systemd = MINERU_SCRIPTS / "systemd_service.sh"
    assert dispatcher.exists() and os.access(dispatcher, os.X_OK)
    assert systemd.exists() and os.access(systemd, os.X_OK)
    body = dispatcher.read_text(encoding="utf-8")
    assert "launch_agent.sh" in body  # macOS route
    assert "systemd_service.sh" in body  # Linux / WSL2 route


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
        f"venv --python 3.13 {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[{default_extras}]>=3.4.4",
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
        f"venv --python 3.13 {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,mlx]>=3.4.4",
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
        f"venv --python 3.13 {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core]>=3.4.4",
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
        f"venv --python 3.13 {service_env}",
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
        f"venv --python 3.13 {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,mlx]==3.2.3",
    ]


def test_mineru_installer_honors_min_version_floor(tmp_path: Path) -> None:
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
    env["MINERU_MIN_VERSION"] = "3.5.0"
    env["MINERU_INSTALL_EXTRAS"] = "core,mlx"
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    env.pop("MINERU_VERSION", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv --python 3.13 {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,mlx]>=3.5.0",
    ]


def test_mineru_installer_pins_python_interpreter(tmp_path: Path) -> None:
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
    env["MINERU_PYTHON"] = "3.12"
    env["MINERU_INSTALL_EXTRAS"] = "core,mlx"
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    env.pop("MINERU_VERSION", None)
    env.pop("MINERU_MIN_VERSION", None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "install.sh")],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"venv --python 3.12 {service_env}",
        f"pip install --python {service_env}/bin/python -U mineru[core,mlx]>=3.4.4",
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


def test_mineru_launch_agent_install_writes_both_plists_and_bootstraps(tmp_path: Path) -> None:
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

    la = home / "Library" / "LaunchAgents"
    logs = home / "Library" / "Logs" / "dlightrag"
    api_label = "com.hanlianlyu.dlightrag.mineru-api"
    gradio_label = "com.hanlianlyu.dlightrag.mineru-gradio"
    api_plist_path = la / f"{api_label}.plist"
    gradio_plist_path = la / f"{gradio_label}.plist"

    api_plist = plistlib.loads(api_plist_path.read_bytes())
    assert api_plist["Label"] == api_label
    assert api_plist["ProgramArguments"] == [str(MINERU_SCRIPTS / "api.sh")]
    assert api_plist["WorkingDirectory"] == str(ROOT)
    assert api_plist["RunAtLoad"] is True
    assert api_plist["KeepAlive"] is True
    assert api_plist["EnvironmentVariables"] == {"MINERU_ENV_FILE": str(env_file)}
    assert api_plist["StandardOutPath"] == str(logs / "mineru-api.out.log")
    assert api_plist["StandardErrorPath"] == str(logs / "mineru-api.err.log")

    gradio_plist = plistlib.loads(gradio_plist_path.read_bytes())
    assert gradio_plist["Label"] == gradio_label
    assert gradio_plist["ProgramArguments"] == [str(MINERU_SCRIPTS / "gradio.sh")]
    assert gradio_plist["RunAtLoad"] is True
    assert gradio_plist["KeepAlive"] is True
    assert gradio_plist["EnvironmentVariables"] == {"MINERU_ENV_FILE": str(env_file)}
    assert gradio_plist["StandardOutPath"] == str(logs / "mineru-gradio.out.log")
    assert gradio_plist["StandardErrorPath"] == str(logs / "mineru-gradio.err.log")

    uid = os.getuid()
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{uid}/{api_label}",
        f"bootout gui/{uid}/{gradio_label}",
        f"bootstrap gui/{uid} {api_plist_path}",
        f"bootstrap gui/{uid} {gradio_plist_path}",
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

    uid = os.getuid()
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{uid}/com.hanlianlyu.dlightrag.mineru-api",
        f"bootout gui/{uid}/com.hanlianlyu.dlightrag.mineru-gradio",
    ]


def test_mineru_gradio_launcher_reuses_api_and_uses_defaults(tmp_path: Path) -> None:
    capture = tmp_path / "args.txt"
    service_env = tmp_path / "mineru-env"
    service_bin = service_env / "bin"
    service_bin.mkdir(parents=True)
    fake_gradio = service_bin / "mineru-gradio"
    _write_executable(
        fake_gradio,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    for key in (
        "MINERU_API_HOST",
        "MINERU_API_PORT",
        "MINERU_GRADIO_HOST",
        "MINERU_GRADIO_PORT",
        "MINERU_GRADIO_API_URL",
    ):
        env.pop(key, None)

    subprocess.run(
        [str(MINERU_SCRIPTS / "gradio.sh"), "--enable-example", "true"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "--api-url",
        "http://127.0.0.1:8210",
        "--server-name",
        "127.0.0.1",
        "--server-port",
        "7860",
        "--enable-example",
        "true",
    ]


def test_mineru_gradio_launcher_honors_overrides_and_rewrites_wildcard(tmp_path: Path) -> None:
    capture = tmp_path / "args.txt"
    service_env = tmp_path / "mineru-env"
    service_bin = service_env / "bin"
    service_bin.mkdir(parents=True)
    fake_gradio = service_bin / "mineru-gradio"
    _write_executable(
        fake_gradio,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$MINERU_CAPTURE"\n',
    )

    env = os.environ.copy()
    env["MINERU_CAPTURE"] = str(capture)
    env["MINERU_SERVICE_VENV"] = str(service_env)
    env["MINERU_ENV_FILE"] = str(tmp_path / "missing.env")
    env["MINERU_API_HOST"] = "0.0.0.0"
    env["MINERU_API_PORT"] = "8300"
    env["MINERU_GRADIO_HOST"] = "0.0.0.0"
    env["MINERU_GRADIO_PORT"] = "7999"
    env.pop("MINERU_GRADIO_API_URL", None)

    subprocess.run([str(MINERU_SCRIPTS / "gradio.sh")], cwd=ROOT, env=env, check=True)

    assert capture.read_text(encoding="utf-8").splitlines() == [
        "--api-url",
        "http://127.0.0.1:8300",
        "--server-name",
        "0.0.0.0",
        "--server-port",
        "7999",
    ]


def test_mineru_launch_agent_can_disable_gradio_webui(tmp_path: Path) -> None:
    capture = tmp_path / "launchctl.txt"
    env_file = tmp_path / ".env"
    home = tmp_path / "home"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    env_file.write_text("MINERU_GRADIO_ENABLE=false\n", encoding="utf-8")
    fake_launchctl = bin_dir / "launchctl"
    _write_executable(
        fake_launchctl,
        '#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$MINERU_CAPTURE"\n',
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

    la = home / "Library" / "LaunchAgents"
    api_plist_path = la / "com.hanlianlyu.dlightrag.mineru-api.plist"
    gradio_plist_path = la / "com.hanlianlyu.dlightrag.mineru-gradio.plist"
    assert api_plist_path.exists()
    assert not gradio_plist_path.exists()

    uid = os.getuid()
    assert capture.read_text(encoding="utf-8").splitlines() == [
        f"bootout gui/{uid}/com.hanlianlyu.dlightrag.mineru-api",
        f"bootout gui/{uid}/com.hanlianlyu.dlightrag.mineru-gradio",
        f"bootstrap gui/{uid} {api_plist_path}",
    ]
