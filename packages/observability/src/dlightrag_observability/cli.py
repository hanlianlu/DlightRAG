# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The stack's one command: bootstrap its env, drive Compose, and prove ingestion.

The ClickStack stack is four containers and one env file. This module owns the
env file (secrets nobody should hand-write), the Compose invocations, the
retention the collector's single TTL variable cannot express (`MODIFY TTL` per
telemetry table), and a probe that sends a span through the collector and reads
it back out of ClickHouse — the only evidence that the pipeline a deployment
just started actually ingests.

It also owns the alerting half: the notification channel, the alerts, and the
dashboard tiles ship as JSON under ``stack/``, and ``alerts push`` writes them
through the REST API, so changing what a deployment watches is a file change and
one command rather than a session of clicking.

Everything is standard library: a stack that needs a pip install to start is a
stack that fails on the machine that matters.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, NamedTuple

#: The OTLP JSON encoding of one finished span, with the service name a probe is
#: recognised by. Fixed values: a probe that changes shape is not a probe.
PROBE_SPAN_NAME = "observability-probe"
PROBE_SERVICE = "dlightrag-observability"

_DEFAULT_APP_PORT = 8300
_DEFAULT_OTLP_PORT = 4318
_DEFAULT_CH_PORT = 8123

#: The alerting artifacts, and the file whose name marks the notification channel
#: rather than an alert. One directory per API collection, so a file's location is
#: what decides the endpoint it is pushed to.
ALERTS_DIRECTORY = "alerts"
DASHBOARDS_DIRECTORY = "dashboards"
CHANNEL_FILENAME = "channel.json"

#: The REST API a push talks to, and the environment variables it reads. The key is a
#: user's ClickStack access key (Team Settings mints it); it is never written into an
#: artifact, and a webhook's URL — which carries a bot token — is referenced the same
#: way rather than committed.
API_VERSION = "v2"
API_KEY_VARIABLE = "DLIGHTRAG_OBSERVABILITY_API_KEY"
API_URL_VARIABLE = "DLIGHTRAG_OBSERVABILITY_API_URL"
DEFAULT_API_URL = "http://127.0.0.1:8000"

#: The one TTL variable the stack's `.env` carries, and the retention the CLI
#: states when nobody has set it. `apply_retention` owns the tables the collector
#: cannot reach from that variable — a table a signal creates with its first
#: write — so it reads the same statement rather than carrying a second copy of
#: it that can drift.
RETENTION_VARIABLE = "HYPERDX_OTEL_EXPORTER_TABLES_TTL"
DEFAULT_RETENTION_DAYS = 7

#: The placeholders an artifact may carry, and how long a push waits for an answer.
_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PUSH_TIMEOUT = 30.0


def stack_root() -> Path:
    """Return the directory holding ``compose.yaml`` and ``stack/``.

    A checkout keeps them beside the package. The wheel carries only the package,
    so a deployment that installs the wheel points
    ``DLIGHTRAG_OBSERVABILITY_STACK_DIR`` at wherever it vendored those files;
    the source distribution (the ``.tar.gz``) does ship them, so an operator who
    builds from it already has them. Either way, the override is how an install
    that is not a checkout finds the stack.
    """
    override = os.environ.get("DLIGHTRAG_OBSERVABILITY_STACK_DIR")
    if override:
        return Path(override).expanduser().resolve()
    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "compose.yaml").is_file():
        return candidate
    raise SystemExit(
        "compose.yaml not found; set DLIGHTRAG_OBSERVABILITY_STACK_DIR to the "
        "directory holding the stack (packages/observability in a checkout)"
    )


def render_env(example: str, *, api_key: str, session_secret: str) -> str:
    """Fill the two secrets in an env template, leaving every other line alone.

    The key is read with ``partition`` and ``strip`` rather than matched by
    prefix, so an example whose assignment carries stray whitespace still gets a
    secret instead of silently keeping an empty value.
    """
    rendered: list[str] = []
    for line in example.splitlines():
        key, separator, _value = line.partition("=")
        if separator and key.strip() == "HYPERDX_API_KEY":
            rendered.append(f"HYPERDX_API_KEY={api_key}")
        elif separator and key.strip() == "EXPRESS_SESSION_SECRET":
            rendered.append(f"EXPRESS_SESSION_SECRET={session_secret}")
        else:
            rendered.append(line)
    return "\n".join(rendered) + "\n"


def ensure_env(root: Path) -> Path:
    """Create ``stack/.env`` once, with generated secrets, mode 0600.

    The file is created with ``O_EXCL`` and mode ``0600`` in one step, so a secret
    is never written into a world-readable file and then tightened: the mode is the
    file's from its first byte. A second caller who races the first to the create
    keeps the winner's secrets, which is the "never overwritten" the command
    promises.
    """
    env_path = root / "stack" / ".env"
    if env_path.is_file():
        return env_path
    example = (root / "stack" / ".env.example").read_text(encoding="utf-8")
    rendered = render_env(
        example,
        api_key=secrets.token_urlsafe(32),
        session_secret=secrets.token_hex(32),
    )
    try:
        descriptor = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return env_path
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(rendered)
    return env_path


def compose_command(root: Path, env_path: Path, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        "dlightrag-observability",
        "--env-file",
        str(env_path),
        "-f",
        str(root / "compose.yaml"),
        *args,
    ]


def _read_env(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def probe_payload(
    *, service: str = PROBE_SERVICE, span_name: str = PROBE_SPAN_NAME
) -> dict[str, Any]:
    """Return the OTLP JSON body of one probe trace."""
    now_ns = time.time_ns()
    trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service}},
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {"name": "dlightrag-observability"},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": span_name,
                                "kind": 1,
                                "startTimeUnixNano": str(now_ns),
                                "endTimeUnixNano": str(now_ns + 1_000_000),
                                "attributes": [
                                    {
                                        "key": "dlightrag.probe",
                                        "value": {"boolValue": True},
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
    }


def send_probe(*, otlp_port: int, api_key: str, timeout: float = 10.0) -> None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{otlp_port}/v1/traces",
        data=json.dumps(probe_payload()).encode("utf-8"),
        headers={"Content-Type": "application/json", "authorization": api_key},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - literal http loopback URL
        if response.status >= 300:
            raise SystemExit(f"probe rejected: HTTP {response.status}")


def clickhouse_query(statement: str, *, port: int = _DEFAULT_CH_PORT, timeout: float = 10.0) -> str:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/?default_format=TabSeparated",
        data=statement.encode("utf-8"),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - literal http loopback URL
        return response.read().decode("utf-8").strip()


#: The tables whose retention this command owns, and the clocks a TTL can key on,
#: the newest schema's name first. The collector's single
#: `HYPERDX_OTEL_EXPORTER_TABLES_TTL` covers the tables that exist when it starts;
#: a table a signal creates with its first write is this command's.
_TELEMETRY_TABLES = (
    "SELECT name FROM system.tables"
    " WHERE database = 'default' AND engine LIKE '%MergeTree%'"
    " AND startsWith(name, 'otel_')"
    " ORDER BY name"
)

_TELEMETRY_TIME_COLUMNS = (
    "SELECT table, name FROM system.columns"
    " WHERE database = 'default' AND startsWith(type, 'DateTime')"
    " AND startsWith(table, 'otel_')"
    " ORDER BY table, position"
)

#: The column names a TTL is known to key on, best first. `Timestamp` is what the
#: collector's tables call it; `TimestampTime` is what ClickStack's newer schema
#: calls the same thing.
_TIME_COLUMNS = ("Timestamp", "TimestampTime", "TimeUnix", "Start")

#: The shape of a name this command pastes into DDL. ClickHouse's own catalog
#: names always match; one that does not is reported rather than interpolated.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class _TTLUnit(NamedTuple):
    """One unit the collector's TTL variable states, and what it is worth."""

    interval: str
    seconds: int


#: The units the collector states retention in, the ClickHouse interval function each
#: one is, and how long each one is. ClickHouse's TTL takes any of them, so the unit a
#: deployment stated is the unit its tables get.
_TTL_UNITS: Final[Mapping[str, _TTLUnit]] = {
    "s": _TTLUnit("Second", 1),
    "m": _TTLUnit("Minute", 60),
    "h": _TTLUnit("Hour", 60 * 60),
    "d": _TTLUnit("Day", 24 * 60 * 60),
    "w": _TTLUnit("Week", 7 * 24 * 60 * 60),
}


@dataclass(frozen=True, slots=True)
class Retention:
    """One retention, in the unit the deployment stated it in.

    A duration is not a day count: ``36h`` and ``2d`` are both a day and a half, and
    only the first is what a ``36h`` deployment said. Keeping the count and the unit
    makes the CLI state the collector's own TTL rather than a rounded restatement of
    it, which is the drift the single variable exists to prevent.
    """

    count: int
    #: The collector's own unit letter: ``s``, ``m``, ``h``, ``d``, or ``w``.
    unit: str

    @property
    def clause(self) -> str:
        """The TTL expression ClickHouse applies: ``36h`` is ``toIntervalHour(36)``."""
        return f"toInterval{_TTL_UNITS[self.unit].interval}({self.count})"

    @property
    def seconds(self) -> int:
        """The same retention in seconds, for comparing two statements of one duration."""
        return self.count * _TTL_UNITS[self.unit].seconds

    def __str__(self) -> str:
        return f"{self.count} {_TTL_UNITS[self.unit].interval.lower()}(s)"


#: The retention a deployment that states none gets, in the unit the documents quote.
DEFAULT_RETENTION: Final[Retention] = Retention(DEFAULT_RETENTION_DAYS, "d")


def retention_ttl(value: str | None, *, fallback: Retention = DEFAULT_RETENTION) -> Retention:
    """Read the collector's TTL variable as the retention it states.

    The collector states retention as ``<number><unit>`` (``168h``, ``30d``, ``2w``; a
    bare number is seconds), and the unit is kept: ``36h`` is ``toIntervalHour(36)``,
    never the two days a day count would have rounded it to. Anything this cannot read
    — a word, a fractional count, a duration of zero or less — falls back to the
    documented default rather than to zero, because a retention of zero is a
    deployment that deletes its telemetry as it writes it.
    """
    if not value:
        return fallback
    text = value.strip().lower()
    number, unit = text[:-1], text[-1:]
    if unit not in _TTL_UNITS:
        number, unit = text, "s"
    if not number.isdigit():
        return fallback
    count = int(number)
    if count < 1:
        return fallback
    return Retention(count, unit)


def configured_retention(root: Path) -> Retention:
    """The retention the deployment states, from its environment or its ``stack/.env``.

    The same precedence `alerts push` uses: what the process environment says wins,
    and the file `bootstrap` wrote is the fallback.
    """
    from_environment = os.environ.get(RETENTION_VARIABLE)
    if from_environment:
        return retention_ttl(from_environment)
    env_path = root / "stack" / ".env"
    if env_path.is_file():
        return retention_ttl(_read_env(env_path).get(RETENTION_VARIABLE))
    return DEFAULT_RETENTION


def retention_statement(table: str, column: str, retention: Retention) -> str:
    """The one DDL a per-table retention is, and the reason it is not in the env."""
    return f"ALTER TABLE default.{table} MODIFY TTL {column} + {retention.clause}"


def retention_time_column(candidates: Sequence[str]) -> str | None:
    """Whichever of a table's clock columns its TTL should key on.

    A table that names one of the known columns is unambiguous. A table with a
    single `DateTime` column under a name nobody listed can only mean that one,
    and a table with several unlisted clocks is one this command does not guess
    at: the caller reports it instead of altering it.
    """
    for preferred in _TIME_COLUMNS:
        if preferred in candidates:
            return preferred
    return candidates[0] if len(candidates) == 1 else None


def telemetry_tables(*, port: int = _DEFAULT_CH_PORT) -> list[str]:
    """Every telemetry table in the deployment's ClickHouse, in name order."""
    return clickhouse_query(_TELEMETRY_TABLES, port=port).split()


def telemetry_time_columns(*, port: int = _DEFAULT_CH_PORT) -> dict[str, list[str]]:
    """Each telemetry table's `DateTime` columns, in the order ClickHouse holds them."""
    columns: dict[str, list[str]] = {}
    for row in clickhouse_query(_TELEMETRY_TIME_COLUMNS, port=port).splitlines():
        table, _, name = row.partition("\t")
        if table and name:
            columns.setdefault(table, []).append(name)
    return columns


def apply_retention(retention: Retention, *, port: int = _DEFAULT_CH_PORT) -> str:
    """Give every telemetry table the deployment's retention, and report the pass.

    Idempotent: `MODIFY TTL` states the retention a table should have rather than
    adding one, so applying it twice is applying it once. A deployment that has
    not exported a signal yet has no table to alter, and one whose signal created
    a table with no clock this command recognises is reported rather than
    guessed at.
    """
    tables = telemetry_tables(port=port)
    if not tables:
        return "retention: no telemetry tables yet (nothing has been exported)"
    columns = telemetry_time_columns(port=port)
    altered: list[str] = []
    skipped: list[str] = []
    refused: list[str] = []
    for table in tables:
        column = retention_time_column(columns.get(table, []))
        if column is None:
            skipped.append(table)
            continue
        if not (_IDENTIFIER.match(table) and _IDENTIFIER.match(column)):
            refused.append(table)
            continue
        clickhouse_query(retention_statement(table, column, retention), port=port)
        altered.append(f"{table}.{column}")
    report = (
        f"retention: {retention} on {', '.join(altered)}"
        if altered
        else f"retention: {retention}, nothing to alter"
    )
    if skipped:
        report += f"; no time column to key on {', '.join(skipped)}"
    if refused:
        report += f"; not altering {', '.join(refused)} (its name is not an identifier)"
    return report


def wait_for_probe(*, ch_port: int, attempts: int = 15, delay: float = 2.0) -> str:
    # Only the module constant PROBE_SPAN_NAME is interpolated.
    statement = (
        "SELECT count() FROM default.otel_traces"  # noqa: S608
        f" WHERE SpanName = '{PROBE_SPAN_NAME}' AND Timestamp > now() - INTERVAL 5 MINUTE"
    )
    for _ in range(attempts):
        try:
            count = clickhouse_query(statement, port=ch_port)
        except urllib.error.URLError, OSError:
            count = "0"
        if count.isdigit() and int(count) > 0:
            return count
        time.sleep(delay)
    return "0"


def stack_is_up(app_port: int, *, timeout: float = 3.0) -> bool:
    """Whether the UI answers; a stack that started is up before it ingests."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{app_port}/", timeout=timeout) as response:
            return response.status < 500
    except urllib.error.URLError, OSError:
        return False


def wait_for_stack(app_port: int, *, seconds: float, delay: float = 3.0) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if stack_is_up(app_port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(delay)


def _bootstrap(args: argparse.Namespace) -> int:
    root = stack_root()
    env_path = ensure_env(root)
    values = _read_env(env_path)
    api_key = values.get("HYPERDX_API_KEY", "")
    otlp_port = int(values.get("HYPERDX_OTEL_HTTP_PORT") or _DEFAULT_OTLP_PORT)
    ch_port = int(values.get("HYPERDX_CH_HTTP_PORT") or _DEFAULT_CH_PORT)
    app_port = int(values.get("HYPERDX_APP_PORT") or _DEFAULT_APP_PORT)
    print(f"env: {env_path}")
    print(f"ui:  http://127.0.0.1:{app_port}")
    if not wait_for_stack(app_port, seconds=args.wait_seconds):
        print(
            f"the UI did not answer in {args.wait_seconds:.0f}s; is the stack up?", file=sys.stderr
        )
        return 1
    if args.skip_probe:
        return 0
    try:
        send_probe(otlp_port=otlp_port, api_key=api_key)
        print("probe sent; waiting for it to land")
    except (urllib.error.URLError, OSError) as exc:
        # The stack is up; ingestion is not provisioned yet. ClickStack defines the collector's
        # pipelines over OpAMP only once the app has a team and a source, and a fresh deployment
        # gets those when a person creates the first administrator. That is a first-run state, not
        # a failure of this command — `make hyperdx-up` must not report one.
        print(
            f"the stack is up but its collector does not accept OTLP yet ({exc}).\n"
            f"Open http://127.0.0.1:{app_port} once, create the administrator, then run this "
            f"command again to see the probe land.",
            file=sys.stderr,
        )
        return 1 if args.require_probe else 0
    try:
        # A whole-day `--days` is the operator's override; with no flag the deployment's
        # own statement is applied, in the unit it stated.
        retention = configured_retention(root) if args.days is None else Retention(args.days, "d")
        print(apply_retention(retention, port=ch_port))
    except (urllib.error.URLError, OSError) as exc:
        print(f"retention not applied ({exc})", file=sys.stderr)
    landed = wait_for_probe(ch_port=ch_port)
    if landed == "0":
        print("probe did not land in 30s", file=sys.stderr)
        return 1
    print(f"probe landed ({landed} row(s))")
    return 0


def _compose(args: argparse.Namespace, *command: str) -> int:
    root = stack_root()
    env_path = ensure_env(root)
    # The arguments are the stack's own paths and the fixed compose subcommands of `main`.
    completed = subprocess.run(compose_command(root, env_path, *command), check=False)  # noqa: S603
    return completed.returncode


def expand_environment(value: Any, *, environ: Mapping[str, str]) -> Any:
    """Replace every ``${VAR}`` a loaded artifact carries with its environment value.

    A webhook URL holds a bot token and the API key belongs to an operator, so
    neither is committed: the file names the variable, and the push resolves it. A
    file that names a variable nobody set fails here, naming it, rather than
    creating a webhook that points nowhere.
    """

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in environ:
            raise SystemExit(
                f"{name} is not set, and an artifact under stack/ names it; export it first"
            )
        return environ[name]

    if isinstance(value, str):
        return _ENV_REFERENCE.sub(replace, value)
    if isinstance(value, dict):
        return {key: expand_environment(item, environ=environ) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_environment(item, environ=environ) for item in value]
    return value


@dataclass(frozen=True)
class ArtifactPush:
    """One artifact, and the request that writes it."""

    label: str
    method: str
    path: str
    payload: dict[str, Any]

    def request(self, *, api_url: str, api_key: str) -> urllib.request.Request:
        return urllib.request.Request(  # noqa: S310 - the operator's own API URL
            f"{api_url.rstrip('/')}{self.path}",
            data=json.dumps(self.payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method=self.method,
        )


def _write_request(collection: str, document: Mapping[str, Any]) -> tuple[str, str]:
    """The method and path that create a document, or update the one it names."""
    path = f"/api/{API_VERSION}/{collection}"
    identifier = document.get("id")
    return ("PUT", f"{path}/{identifier}") if identifier else ("POST", path)


def _load_artifact(path: Path, *, environ: Mapping[str, str]) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    expanded = expand_environment(document, environ=environ)
    if not isinstance(expanded, dict):  # a document is an object; anything else is a broken file
        raise SystemExit(f"{path} does not hold a JSON object")
    return expanded


def alerting_artifacts(root: Path, *, environ: Mapping[str, str]) -> list[ArtifactPush]:
    """Every artifact the stack ships, in the order the API has to receive them.

    The channel comes first: an alert names its webhook by id, so a deployment that
    pushed alerts first would hold alerts with no destination. Files are sorted by
    name, so one run and the next send the same requests in the same order.
    """
    pushes: list[ArtifactPush] = []
    alerting = root / "stack" / ALERTS_DIRECTORY
    channel = alerting / CHANNEL_FILENAME
    if channel.is_file():
        document = _load_artifact(channel, environ=environ)
        method, path = _write_request("webhooks", document)
        pushes.append(
            ArtifactPush(
                label=f"channel {document.get('name', CHANNEL_FILENAME)}",
                method=method,
                path=path,
                payload={key: value for key, value in document.items() if key != "id"},
            )
        )
    for kind, directory in (
        ("alerts", alerting),
        ("dashboards", root / "stack" / DASHBOARDS_DIRECTORY),
    ):
        for path in sorted(directory.glob("*.json")):
            if kind == "alerts" and path.name == CHANNEL_FILENAME:
                continue
            document = _load_artifact(path, environ=environ)
            method, route = _write_request(kind, document)
            pushes.append(
                ArtifactPush(
                    label=f"{kind[:-1]} {document.get('name', path.stem)}",
                    method=method,
                    path=route,
                    payload={key: value for key, value in document.items() if key != "id"},
                )
            )
    return pushes


def push_artifacts(
    pushes: Sequence[ArtifactPush],
    *,
    api_url: str,
    api_key: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
    echo: Callable[[str], None] = print,
) -> int:
    """Send every request, and report the answer each one got.

    A rejection is fatal and carries the API's own message: the point of the
    command is to learn which file the server refused, not to walk past it.
    """
    sent = 0
    for push in pushes:
        request = push.request(api_url=api_url, api_key=api_key)
        try:
            with opener(request, timeout=_PUSH_TIMEOUT) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise SystemExit(f"{push.label}: HTTP {exc.code} {detail}") from exc
        echo(f"{status} {push.method} {push.path} — {push.label}")
        sent += 1
    return sent


def _push_alerts(args: argparse.Namespace) -> int:
    root = stack_root()
    # The alert artifacts name their webhook and chat as ${VAR} placeholders. Those
    # values live in stack/.env (written by bootstrap and edited by the operator), so
    # read that file too: the process environment still wins, which keeps an explicit
    # export working, and a `docker compose --env-file` operator is not required to
    # have exported anything.
    env_path = root / "stack" / ".env"
    stacked = _read_env(env_path) if env_path.is_file() else {}
    pushes = alerting_artifacts(root, environ={**stacked, **os.environ})
    if not pushes:
        print(
            f"nothing to push: {root / 'stack' / ALERTS_DIRECTORY} and "
            f"{root / 'stack' / DASHBOARDS_DIRECTORY} hold no JSON",
            file=sys.stderr,
        )
        return 1
    if args.dry_run:
        for push in pushes:
            print(f"{push.method} {push.path} — {push.label}")
        return 0
    api_key = os.environ.get(API_KEY_VARIABLE, "") or stacked.get(API_KEY_VARIABLE, "")
    if not api_key:
        raise SystemExit(
            f"{API_KEY_VARIABLE} is not set; export the access key ClickStack mints "
            f"under Team Settings"
        )
    api_url = os.environ.get(API_URL_VARIABLE) or stacked.get(API_URL_VARIABLE) or DEFAULT_API_URL
    sent = push_artifacts(pushes, api_url=api_url, api_key=api_key)
    print(f"pushed {sent} artifact(s) to {api_url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dlightrag-observability", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser(
        "bootstrap",
        help="write stack/.env, apply retention, send a probe, and check it landed",
    )
    bootstrap.add_argument("--skip-probe", action="store_true")
    bootstrap.add_argument(
        "--require-probe",
        action="store_true",
        help="fail when the collector does not accept OTLP yet (the first-run state)",
    )
    bootstrap.add_argument("--wait-seconds", type=float, default=60.0)
    bootstrap.add_argument(
        "--days",
        type=int,
        default=None,
        help=(
            "whole-day retention applied to every telemetry table for one run; "
            f"without it the {RETENTION_VARIABLE} the environment or stack/.env states "
            f"is applied as stated ({DEFAULT_RETENTION_DAYS} days when neither states one)"
        ),
    )
    for name, help_text in (
        ("up", "start the stack"),
        ("down", "stop the stack"),
        ("status", "show the stack's containers"),
        ("logs", "follow the stack's logs"),
        ("reset", "stop the stack and delete its volumes"),
    ):
        sub.add_parser(name, help=help_text)

    alerts = sub.add_parser("alerts", help="push the stack's channel, alerts, and dashboards")
    alerts_commands = alerts.add_subparsers(dest="alerts_command", required=True)
    push = alerts_commands.add_parser(
        "push",
        help="create or update every artifact under stack/alerts and stack/dashboards",
        description=(
            "Read the channel, the alerts, and the dashboard tiles that ship under stack/, "
            "resolve the ${VARIABLES} they name, and write them through ClickStack's REST "
            "API. An artifact with an id is updated; one without is created."
        ),
    )
    push.add_argument(
        "--dry-run",
        action="store_true",
        help="print the requests instead of sending them (still needs the artifact variables)",
    )

    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        return _bootstrap(args)
    if args.command == "alerts":
        return _push_alerts(args)
    if args.command == "up":
        return _compose(args, "up", "-d")
    if args.command == "down":
        return _compose(args, "down")
    if args.command == "status":
        return _compose(args, "ps")
    if args.command == "logs":
        return _compose(args, "logs", "--follow", "--tail", "100")
    return _compose(args, "down", "--volumes")


if __name__ == "__main__":
    raise SystemExit(main())
