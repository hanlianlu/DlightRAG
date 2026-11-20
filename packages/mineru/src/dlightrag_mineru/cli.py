# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""``dlightrag-mineru``: install, run, and background-service the MinerU sidecar.

Every command resolves the same two things first -- the service venv and the env
file -- so the foreground path, the background path, and the installer can never
disagree about which sidecar they are talking about (ADR 0054).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from dlightrag_mineru import env, install, service, title_aided

POSIX_ONLY = (
    "dlightrag-mineru {command} runs a host service through launchd or systemd "
    "--user, so it supports macOS, Linux, and WSL2. This host is {platform}."
)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command in {"install", "api", "service", "purge"}:
            if code := _nonposix_exit(arguments.command):
                return code
        if arguments.command == "install":
            return install.install(project_environment=arguments.project_env)
        if arguments.command == "api":
            return _api(arguments.extra)
        if arguments.command == "service":
            return _service(arguments.action)
        if arguments.command == "purge":
            return _purge(arguments)
        return _title_aided(arguments)
    except env.EnvFileError as exc:
        # An unreadable env file must not look like an absent one: the defaults would
        # start a sidecar on the wrong host, port, or effort.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def _nonposix_exit(command: str) -> int:
    """0 when this host can run the command, otherwise the exit code to use."""
    if os.name == "posix":
        return 0
    print(
        POSIX_ONLY.format(command=command, platform=platform.system() or os.name), file=sys.stderr
    )
    return 64


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dlightrag-mineru",
        description="Manage the host-side MinerU sidecar DlightRAG parses through.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    install_parser = commands.add_parser(
        "install", help="Create the service env and install MinerU into it"
    )
    install_parser.add_argument(
        "--project-env",
        type=Path,
        default=None,
        help="The environment whose LightRAG client decides the MinerU range",
    )

    api_parser = commands.add_parser("api", help="Run the parsing backend in the foreground")
    api_parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to mineru-api",
    )

    service_parser = commands.add_parser("service", help="Manage the background service")
    service_parser.add_argument(
        "action",
        metavar="{install,start,stop,status,logs,uninstall}",
        help="Which lifecycle action to run",
    )

    purge_parser = commands.add_parser("purge", help="Remove the service venv this tool installed")
    purge_parser.add_argument("--yes", action="store_true", help="Confirm the removal")
    purge_parser.add_argument("--dry-run", action="store_true", help="Say what would be removed")

    title_parser = commands.add_parser(
        "title-aided", help="Write MinerU's title-aided LLM config to ~/mineru.json"
    )
    title_parser.add_argument("--disable", action="store_true", help="Turn the feature off")
    title_parser.add_argument("--target", type=Path, default=None, help="The config to merge into")
    title_parser.add_argument("--api-key", default=None)
    title_parser.add_argument("--base-url", default=None)
    title_parser.add_argument("--model", default=None)
    title_parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave reasoning to the provider's default instead of disabling it",
    )
    title_parser.add_argument(
        "--extra-body",
        default=None,
        help="JSON request fields that disable reasoning, resolved by the caller",
    )
    return parser


def _api(extra: Sequence[str]) -> int:
    environment, env_file = _paths()
    launcher = environment / install.SHIM_DIRECTORY / install.LAUNCHER_NAME
    interpreter = environment / "bin" / "python"
    if not launcher.is_file() or not interpreter.is_file():
        print(service.LAUNCHER_MISSING.format(environment=environment), file=sys.stderr)
        return 127
    # The launcher resolves the same file from its own directory rules; naming it here
    # keeps the foreground and background paths on one file.
    os.environ.setdefault("MINERU_ENV_FILE", str(env_file))
    os.execv(str(interpreter), [str(interpreter), str(launcher), *extra])  # noqa: S606 - argv, no shell
    return 127  # unreachable: execv replaces this process


def _purge(arguments: argparse.Namespace) -> int:
    environment, _ = _paths()
    return install.purge(environment, confirmed=arguments.yes, dry_run=arguments.dry_run)


def _service(action: str) -> int:
    environment, env_file = _paths()
    specification = service.spec(
        environment=environment,
        working_directory=env.project_root(),
        env_file=env_file,
    )
    try:
        return service.run(action, spec=specification)
    except ValueError as exc:
        # A path the unit file cannot carry safely (systemd quotation).
        print(f"ERROR: {exc}", file=sys.stderr)
        return 64


def _title_aided(arguments: argparse.Namespace) -> int:
    """One command for both callers: this CLI, and the application's resolver.

    ``--extra-body`` carries fields the caller resolved; without it, and without
    ``--enable-thinking``, the request leaves reasoning to the provider's default.
    """
    extra_body = _extra_body(arguments.extra_body)
    if extra_body is None:
        return 2
    return title_aided.from_environment(
        target=arguments.target,
        disable=arguments.disable,
        api_key=arguments.api_key,
        base_url=arguments.base_url,
        model=arguments.model,
        enable_thinking=arguments.enable_thinking,
        extra_body=extra_body,
    )


def _extra_body(rendered: str | None) -> dict | None:
    if not rendered:
        return {}
    try:
        parsed = json.loads(rendered)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --extra-body is not JSON: {exc}", file=sys.stderr)
        return None
    if not isinstance(parsed, dict):
        print("ERROR: --extra-body must be a JSON object.", file=sys.stderr)
        return None
    return parsed


def _paths(*, environ: Mapping[str, str] | None = None) -> tuple[Path, Path]:
    """The service venv and the env file every command resolves first."""
    return env.service_environment(environ=environ), env.env_file_path(environ=environ)


if __name__ == "__main__":  # pragma: no cover - exercised through the console script
    raise SystemExit(main())
