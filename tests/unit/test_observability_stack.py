# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The stack CLI: an env file nobody hand-writes, and a probe that proves ingestion.

The stack itself needs Docker; these tests hold the parts that decide what a
deployment's env file and probe look like, which is where a mistake costs a
silent outage rather than a failed command. The same goes for the alerting
artifacts: what a push sends is the contract with the API, so it is held here
rather than discovered against a live stack.
"""

from __future__ import annotations

import importlib
import inspect
import io
import json
import os
import pkgutil
import re
import urllib.error
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from dlightrag_observability import cli
from dlightrag_observability import otlp as otlp_module


def test_stack_root_honours_the_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setenv("DLIGHTRAG_OBSERVABILITY_STACK_DIR", str(tmp_path))
    assert cli.stack_root() == tmp_path.resolve()


def test_stack_root_finds_the_workspace_package() -> None:
    root = cli.stack_root()
    assert (root / "compose.yaml").is_file()
    assert (root / "stack" / ".env.example").is_file()


def test_stack_root_reports_an_install_that_carries_no_stack(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A wheel carries the package and not the stack, and the command says so.

    uv's build backend includes only the module root, so `compose.yaml` reaches an
    artifact through `source-include` — the source distribution — and a wheel-only
    install has no stack beside the package. What a caller gets then is this message
    rather than a directory guessed from a path that happens to exist.
    """
    monkeypatch.delenv("DLIGHTRAG_OBSERVABILITY_STACK_DIR", raising=False)
    installed = tmp_path / "site-packages" / "dlightrag_observability" / "cli.py"
    installed.parent.mkdir(parents=True)
    installed.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "__file__", str(installed))

    with pytest.raises(SystemExit) as exit_info:
        cli.stack_root()

    assert "DLIGHTRAG_OBSERVABILITY_STACK_DIR" in str(exit_info.value)
    assert "compose.yaml not found" in str(exit_info.value)


def test_render_env_fills_exactly_the_two_secrets() -> None:
    example = "HYPERDX_API_KEY=\nEXPRESS_SESSION_SECRET=\nHYPERDX_APP_PORT=8300\n"
    rendered = cli.render_env(example, api_key="key-1", session_secret="secret-1")
    assert "HYPERDX_API_KEY=key-1" in rendered
    assert "EXPRESS_SESSION_SECRET=secret-1" in rendered
    assert "HYPERDX_APP_PORT=8300" in rendered


def test_render_env_fills_a_secret_whose_assignment_carries_whitespace() -> None:
    """A stray space must not turn a secret into an empty value the compose file keeps."""
    example = "HYPERDX_API_KEY=   \nEXPRESS_SESSION_SECRET = \nHYPERDX_APP_PORT=8300\n"
    rendered = cli.render_env(example, api_key="key-1", session_secret="secret-1")
    assert "HYPERDX_API_KEY=key-1" in rendered
    assert "EXPRESS_SESSION_SECRET=secret-1" in rendered
    assert "HYPERDX_APP_PORT=8300" in rendered


def test_ensure_env_is_written_once_with_owner_only_permissions(tmp_path: Path) -> None:
    stack = tmp_path / "stack"
    stack.mkdir()
    (stack / ".env.example").write_text(
        "HYPERDX_API_KEY=\nEXPRESS_SESSION_SECRET=\n", encoding="utf-8"
    )
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")

    first = cli.ensure_env(tmp_path)
    written = first.read_text(encoding="utf-8")
    assert written.count("HYPERDX_API_KEY=") == 1
    assert not written.startswith("HYPERDX_API_KEY=\n")
    assert first.stat().st_mode & 0o777 == 0o600

    first.write_text("HYPERDX_API_KEY=kept\n", encoding="utf-8")
    assert cli.ensure_env(tmp_path).read_text(encoding="utf-8") == "HYPERDX_API_KEY=kept\n"


def test_ensure_env_keeps_a_concurrent_creators_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The create is O_EXCL: a process that loses the race keeps the winner's secrets."""
    stack = tmp_path / "stack"
    stack.mkdir()
    (stack / ".env.example").write_text(
        "HYPERDX_API_KEY=\nEXPRESS_SESSION_SECRET=\n", encoding="utf-8"
    )
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")

    def already_created(_path: object, _flags: int, _mode: int = 0o777) -> int:
        raise FileExistsError

    monkeypatch.setattr(os, "open", already_created)
    assert cli.ensure_env(tmp_path) == stack / ".env"
    assert not (stack / ".env").exists()


def test_compose_command_names_the_project_and_the_env_file(tmp_path: Path) -> None:
    command = cli.compose_command(tmp_path, tmp_path / ".env", "up", "-d")
    assert command[:2] == ["docker", "compose"]
    assert "--project-name" in command
    assert str(tmp_path / ".env") in command
    assert command[-2:] == ["up", "-d"]


def test_the_probe_is_one_trace_with_a_stable_name() -> None:
    payload = cli.probe_payload()
    resource_spans = payload["resourceSpans"]
    assert len(resource_spans) == 1
    spans = resource_spans[0]["scopeSpans"][0]["spans"]
    assert len(spans) == 1
    assert spans[0]["name"] == cli.PROBE_SPAN_NAME
    assert spans[0]["kind"] == 1
    assert len(spans[0]["traceId"]) == 32
    assert len(spans[0]["spanId"]) == 16
    attributes = {item["key"]: item["value"] for item in spans[0]["attributes"]}
    assert attributes["dlightrag.probe"] == {"boolValue": True}
    resource_attributes = {
        item["key"]: item["value"] for item in resource_spans[0]["resource"]["attributes"]
    }
    assert resource_attributes["service.name"] == {"stringValue": cli.PROBE_SERVICE}
    # The OTLP JSON encoding wants nanoseconds as strings, not floats.
    assert isinstance(spans[0]["startTimeUnixNano"], str)
    json.dumps(payload)


def test_each_probe_is_a_new_trace() -> None:
    first = cli.probe_payload()["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    second = cli.probe_payload()["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert first["traceId"] != second["traceId"]
    assert first["spanId"] != second["spanId"]


def test_retention_is_one_statement_per_table_the_environment_cannot_express() -> None:
    statement = cli.retention_statement("otel_traces", "Timestamp", cli.Retention(7, "d"))
    assert statement == "ALTER TABLE default.otel_traces MODIFY TTL Timestamp + toIntervalDay(7)"
    # The clause is the deployment's own: a 36-hour statement is 36 hours, not two days.
    assert cli.retention_statement(
        "otel_metrics_gauge", "TimeUnix", cli.Retention(36, "h")
    ).endswith("MODIFY TTL TimeUnix + toIntervalHour(36)")


def test_a_tables_clock_column_is_the_one_its_schema_names() -> None:
    """A known name wins; one unlisted `DateTime` column can only mean that one."""
    assert cli.retention_time_column(["TimeUnix", "Timestamp"]) == "Timestamp"
    assert cli.retention_time_column(["TimestampTime"]) == "TimestampTime"
    # The trace-id lookup table's clock is `Start`; its range ends later and is not one.
    assert cli.retention_time_column(["End", "Start"]) == "Start"
    assert cli.retention_time_column(["observed_at"]) == "observed_at"
    # Two clocks nobody listed is a table this command reports rather than guesses at.
    assert cli.retention_time_column(["Seen", "Written"]) is None
    assert cli.retention_time_column([]) is None


#: The retention a deployment states in the collector's unit, a week long, which is the
#: one `stack/.env.example` ships.
_STATED_7_DAYS = cli.Retention(7, "d")


class _FakeClickHouse:
    """The two catalog reads retention makes, and the DDL it applies."""

    def __init__(self, tables: dict[str, list[str]]) -> None:
        self.tables = tables
        self.altered: list[str] = []

    def __call__(self, statement: str, **_kwargs: Any) -> str:
        if statement == cli._TELEMETRY_TABLES:  # noqa: SLF001 - the query under test
            return "\n".join(self.tables)
        if statement == cli._TELEMETRY_TIME_COLUMNS:  # noqa: SLF001 - the query under test
            return "\n".join(
                f"{table}\t{column}" for table, columns in self.tables.items() for column in columns
            )
        self.altered.append(statement)
        return ""


def test_retention_alters_every_telemetry_table_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seven-day TTLs a shell once applied by hand are one idempotent pass."""
    clickhouse = _FakeClickHouse(
        {
            "otel_logs": ["Timestamp"],
            "otel_metrics_gauge": ["TimeUnix"],
            "otel_traces": ["Timestamp"],
        }
    )
    monkeypatch.setattr(cli, "clickhouse_query", clickhouse)

    report = cli.apply_retention(_STATED_7_DAYS)

    assert clickhouse.altered == [
        "ALTER TABLE default.otel_logs MODIFY TTL Timestamp + toIntervalDay(7)",
        "ALTER TABLE default.otel_metrics_gauge MODIFY TTL TimeUnix + toIntervalDay(7)",
        "ALTER TABLE default.otel_traces MODIFY TTL Timestamp + toIntervalDay(7)",
    ]
    assert report == (
        "retention: 7 day(s) on otel_logs.Timestamp, "
        "otel_metrics_gauge.TimeUnix, otel_traces.Timestamp"
    )


def test_retention_reports_a_table_it_cannot_key_a_ttl_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A table with no recognisable clock is reported, not altered with a guess."""
    clickhouse = _FakeClickHouse({"otel_traces": ["Timestamp"], "otel_mystery": ["Seen", "Left"]})
    monkeypatch.setattr(cli, "clickhouse_query", clickhouse)

    report = cli.apply_retention(_STATED_7_DAYS)

    assert clickhouse.altered == [
        "ALTER TABLE default.otel_traces MODIFY TTL Timestamp + toIntervalDay(7)"
    ]
    assert "no time column to key on otel_mystery" in report


def test_retention_before_anything_was_exported_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deployment that has exported nothing has no table to alter."""
    monkeypatch.setattr(cli, "clickhouse_query", _FakeClickHouse({}))
    assert cli.apply_retention(_STATED_7_DAYS) == (
        "retention: no telemetry tables yet (nothing has been exported)"
    )


def test_retention_refuses_a_catalog_name_it_will_not_paste_into_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A name that is not an identifier is reported, never interpolated."""
    clickhouse = _FakeClickHouse({"otel_logs;DROP": ["Timestamp"]})
    monkeypatch.setattr(cli, "clickhouse_query", clickhouse)

    report = cli.apply_retention(_STATED_7_DAYS)

    assert clickhouse.altered == []
    assert "not altering otel_logs;DROP (its name is not an identifier)" in report


def test_retention_takes_the_days_it_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    clickhouse = _FakeClickHouse({"otel_logs": ["Timestamp"]})
    monkeypatch.setattr(cli, "clickhouse_query", clickhouse)

    assert cli.apply_retention(cli.Retention(30, "d")).startswith(
        "retention: 30 day(s) on otel_logs.Timestamp"
    )
    assert clickhouse.altered[0].endswith("toIntervalDay(30)")


@pytest.mark.parametrize(
    ("value", "retention"),
    [
        ("168h", cli.Retention(168, "h")),
        ("720h", cli.Retention(720, "h")),
        ("30d", cli.Retention(30, "d")),
        ("7d", cli.Retention(7, "d")),
        ("2w", cli.Retention(2, "w")),
        ("604800", cli.Retention(604_800, "s")),
        (" 168H ", cli.Retention(168, "h")),
        # A sub-day TTL is the case that used to drift: `36h` reached ClickHouse as two
        # days and `12h` as one, so the tables the collector's variable cannot reach
        # were kept on a different retention than the one the deployment stated.
        ("36h", cli.Retention(36, "h")),
        ("12h", cli.Retention(12, "h")),
        ("90m", cli.Retention(90, "m")),
        ("90s", cli.Retention(90, "s")),
    ],
)
def test_the_collectors_ttl_variable_is_the_one_statement_of_retention(
    value: str, retention: cli.Retention
) -> None:
    """`bootstrap` states the same retention the collector does, in the collector's unit.

    The two are read from one variable rather than written down twice, because two
    copies of a number can drift and this one decides when telemetry is deleted. The
    count and the unit both survive the read: an hour count restated in days is a
    different retention, and rounding is how one statement becomes two.
    """
    assert cli.retention_ttl(value) == retention


def test_a_sub_day_retention_reaches_clickhouse_in_the_unit_it_was_stated() -> None:
    """The interval function is the stated unit's, so no rounding is applied on the way."""
    assert cli.retention_ttl("36h").clause == "toIntervalHour(36)"
    assert cli.retention_ttl("12h").clause == "toIntervalHour(12)"
    assert cli.retention_ttl("90m").clause == "toIntervalMinute(90)"
    assert cli.retention_ttl("604800").clause == "toIntervalSecond(604800)"
    # Two statements of one duration: the shipped week equals the default, either way.
    assert cli.retention_ttl("168h").seconds == cli.retention_ttl("7d").seconds


def test_an_unreadable_retention_falls_back_to_the_documented_default() -> None:
    """A retention of zero would delete telemetry as it is written, so never guess low."""
    assert cli.retention_ttl(None) == cli.DEFAULT_RETENTION
    assert cli.retention_ttl("") == cli.DEFAULT_RETENTION
    assert cli.retention_ttl("soon") == cli.DEFAULT_RETENTION
    # Nothing this command cannot apply is rounded into a retention nobody stated: a
    # zero, a fraction, and a bare number that is not whole seconds all fall back.
    assert cli.retention_ttl("0h") == cli.DEFAULT_RETENTION
    assert cli.retention_ttl("0") == cli.DEFAULT_RETENTION
    assert cli.retention_ttl("1.5h") == cli.DEFAULT_RETENTION
    # `str.isdigit` is true of a superscript, which `int` then refuses: a duration is ASCII
    # digits, and anything else is unreadable rather than an exception.
    assert cli.retention_ttl("²h") == cli.DEFAULT_RETENTION
    assert cli.retention_ttl("٣d") == cli.DEFAULT_RETENTION


def test_configured_retention_prefers_the_environment_then_the_stack_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stack = tmp_path / "stack"
    stack.mkdir()
    (stack / ".env").write_text(f"{cli.RETENTION_VARIABLE}=720h\n", encoding="utf-8")
    monkeypatch.delenv(cli.RETENTION_VARIABLE, raising=False)

    assert cli.configured_retention(tmp_path) == cli.Retention(720, "h")

    # The environment wins, and it wins as stated: `48h` is 48 hours, not two days.
    monkeypatch.setenv(cli.RETENTION_VARIABLE, "48h")
    assert cli.configured_retention(tmp_path) == cli.Retention(48, "h")

    monkeypatch.delenv(cli.RETENTION_VARIABLE, raising=False)
    (stack / ".env").unlink()
    assert cli.configured_retention(tmp_path) == cli.DEFAULT_RETENTION


def test_the_shipped_env_example_states_the_default_retention() -> None:
    """The file an operator copies must not promise a different retention than the CLI."""
    example = (cli.stack_root() / "stack" / ".env.example").read_text(encoding="utf-8")
    stated = next(
        line.partition("=")[2].strip()
        for line in example.splitlines()
        if line.startswith(f"{cli.RETENTION_VARIABLE}=")
    )
    # The example states hours and the default is stated in days: one duration, two
    # spellings, and the CLI applies whichever one the deployment wrote down.
    retention = cli.retention_ttl(stated)
    assert retention.seconds == cli.DEFAULT_RETENTION.seconds
    assert retention.seconds == cli.DEFAULT_RETENTION_DAYS * 24 * 60 * 60


# ── The alerting artifacts: what a push sends ────────────────────────────────


_ENVIRON = {"TELEGRAM_URL": "https://example.invalid/bot", "HYPERDX_CONNECTION_ID": "connection-1"}

#: Every variable the shipped artifacts name, as a deployment's `stack/.env` would hold them.
_SHIPPED_ENVIRON = {
    "DLIGHTRAG_TELEGRAM_CHAT_ID": "chat-1",
    "DLIGHTRAG_TELEGRAM_WEBHOOK_URL": "https://example.invalid/bot",
    "HYPERDX_CONNECTION_ID": "connection-1",
    "HYPERDX_SOURCE_TRACES": "source-traces",
    "HYPERDX_SOURCE_METRICS": "source-metrics",
}


def _stack(tmp_path: Path) -> Path:
    """A miniature stack: one channel, one alert, one dashboard file, none carrying an id."""
    alerts = tmp_path / "stack" / "alerts"
    dashboards = tmp_path / "stack" / "dashboards"
    alerts.mkdir(parents=True)
    dashboards.mkdir(parents=True)
    (alerts / "channel.json").write_text(
        json.dumps(
            {
                "name": "Product · Telegram",
                "service": "generic",
                "url": "${TELEGRAM_URL}",
                "body": '{"chat_id": "1", "text": "{{title}}"}',
            }
        ),
        encoding="utf-8",
    )
    alert: dict[str, Any] = {
        "name": "Product · the api is answering 5xx",
        "displayName": "Product · api 5xx",
        "source": "inline",
        "interval": "5m",
        "threshold": 0.02,
        "thresholdType": "above",
        "channel": {"type": "webhook", "webhookId": "${HYPERDX_WEBHOOK_ID}"},
        "chartConfig": {
            "configType": "sql",
            "displayType": "number",
            "connectionId": "${HYPERDX_CONNECTION_ID}",
        },
    }
    (alerts / "api-5xx.json").write_text(json.dumps(alert), encoding="utf-8")
    (dashboards / "product.json").write_text(
        json.dumps({"name": "Product", "tiles": []}), encoding="utf-8"
    )
    return tmp_path


class _Response:
    status = 200

    def __init__(self, document: object) -> None:
        self._body = json.dumps(document).encode("utf-8")

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class _FakeApi:
    """The slice of ClickStack's external API a push talks to: list, create, and update.

    ``documents`` is what the stack already holds, as ``{collection: {name: id}}``. A
    create answers with a new id, an update answers with the id its path names, and every
    request is kept so a test can read what was sent and in what order.
    """

    def __init__(
        self,
        documents: dict[str, dict[str, str]] | None = None,
        stack: dict[str, list[dict[str, str]]] | None = None,
    ) -> None:
        self.documents = {
            collection: dict(names) for collection, names in (documents or {}).items()
        }
        # The connections and sources the stack's first seeding made, as the API lists them.
        self.stack = stack or {}
        self.sent: list[tuple[str, str, dict[str, Any] | None, urllib.request.Request]] = []

    def __call__(self, request: urllib.request.Request, **_kwargs: Any) -> _Response:
        method, path = request.get_method(), urllib.parse.urlsplit(request.full_url).path
        body = json.loads(request.data) if isinstance(request.data, bytes) else None
        self.sent.append((method, path, body, request))
        collection = path.split("/")[3]
        if collection in self.stack:
            return _Response({"data": self.stack[collection]})
        names = self.documents.setdefault(collection, {})
        key = cli._COLLECTION_KEYS[collection]  # noqa: SLF001 - the contract under test
        if method == "GET":
            data = [{"id": identifier, key: name} for name, identifier in names.items()]
            return _Response({"data": data, "meta": {"total": len(data)}})
        if method == "POST":
            assert body is not None
            names[body[key]] = identifier = f"new-{len(names) + 1}"
            return _Response({"data": {"id": identifier}})
        return _Response({"data": {"id": path.rpartition("/")[2]}})

    @property
    def writes(self) -> list[tuple[str, str]]:
        return [(method, path) for method, path, _body, _request in self.sent if method != "GET"]


def _push(root: Path, api: _FakeApi, lines: list[str] | None = None) -> int:
    return cli.push_artifacts(
        cli.alerting_artifacts(root, environ=_ENVIRON),
        api_url="http://127.0.0.1:8000/",
        api_key="access-key",
        opener=api,
        echo=(lines if lines is not None else []).append,
    )


def test_a_first_push_creates_every_artifact_and_hands_the_new_webhook_to_the_alerts(
    tmp_path: Path,
) -> None:
    """On a stack that holds nothing the push is three creates, the channel first.

    The alert names its webhook as a variable the push fills in once the channel exists:
    the id is the one the stack just minted, not one a file carried from another stack.
    """
    api = _FakeApi()
    lines: list[str] = []

    assert _push(_stack(tmp_path), api, lines) == 3

    assert api.writes == [
        ("POST", "/api/v2/webhooks"),
        ("POST", "/api/v2/alerts"),
        ("POST", "/api/v2/dashboards"),
    ]
    alert = next(
        body for method, path, body, _ in api.sent if method == "POST" and "alerts" in path
    )
    assert alert is not None
    assert alert["channel"] == {"type": "webhook", "webhookId": "new-1"}
    assert alert["chartConfig"]["connectionId"] == "connection-1"
    assert lines[0] == "200 POST /api/v2/webhooks — channel Product · Telegram"


def test_a_later_push_updates_what_carries_its_name_and_creates_the_rest(tmp_path: Path) -> None:
    """The name is the identity: a file that was pushed before is the same document now."""
    api = _FakeApi(
        {
            "webhooks": {"Product · Telegram": "hook-7"},
            "alerts": {"Product · api 5xx": "alert-1", "Someone else's alert": "alert-2"},
        }
    )

    _push(_stack(tmp_path), api)

    assert api.writes == [
        ("PUT", "/api/v2/webhooks/hook-7"),
        ("PUT", "/api/v2/alerts/alert-1"),
        ("POST", "/api/v2/dashboards"),
    ]
    alert = next(body for method, path, body, _ in api.sent if path.endswith("alerts/alert-1"))
    assert alert is not None
    assert alert["channel"]["webhookId"] == "hook-7"
    # A stack that holds other documents loses none of them, and a push twice is a push once.
    assert api.documents["alerts"] == {
        "Product · api 5xx": "alert-1",
        "Someone else's alert": "alert-2",
    }
    before = len(api.sent)
    _push(tmp_path, api)
    assert [method for method, *_ in api.sent[before:]].count("POST") == 0


def test_each_collection_is_listed_once_and_before_it_is_written(tmp_path: Path) -> None:
    api = _FakeApi()
    _push(_stack(tmp_path), api)
    assert [(method, path) for method, path, _body, _ in api.sent if method == "GET"] == [
        ("GET", "/api/v2/webhooks"),
        ("GET", "/api/v2/alerts"),
        ("GET", "/api/v2/dashboards"),
    ]
    assert [method for method, *_ in api.sent] == ["GET", "POST", "GET", "POST", "GET", "POST"]


def test_a_push_never_sends_an_id_in_a_body_or_a_variable_it_could_not_fill(tmp_path: Path) -> None:
    api = _FakeApi({"alerts": {"Product · api 5xx": "alert-1"}})
    _push(_stack(tmp_path), api)
    for method, _path, body, _request in api.sent:
        if method != "GET":
            assert body is not None
            assert "id" not in body
            assert "${" not in json.dumps(body)


def test_a_listing_shorter_than_its_total_stops_the_push(tmp_path: Path) -> None:
    """Creating what a truncated listing hid would be a duplicate, so the push stops."""

    def truncated(request: urllib.request.Request, **_kwargs: Any) -> _Response:
        assert request.get_method() == "GET"
        return _Response({"data": [], "meta": {"total": 5}})

    with pytest.raises(SystemExit, match="more exist than one listing returns"):
        cli.push_artifacts(
            cli.alerting_artifacts(_stack(tmp_path), environ=_ENVIRON),
            api_url="http://127.0.0.1:8000",
            api_key="access-key",
            opener=truncated,
            echo=lambda _line: None,
        )


def test_an_artifact_without_its_name_is_refused_naming_the_file(tmp_path: Path) -> None:
    root = _stack(tmp_path)
    document = json.loads((root / "stack" / "alerts" / "api-5xx.json").read_text(encoding="utf-8"))
    del document["displayName"]
    (root / "stack" / "alerts" / "api-5xx.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"api-5xx\.json names no `displayName`"):
        cli.alerting_artifacts(root, environ=_ENVIRON)


def test_a_variable_set_to_nothing_is_a_variable_nobody_set() -> None:
    """An empty id would be pushed as an empty string and refused by the API far from here."""
    with pytest.raises(SystemExit, match="HYPERDX_CONNECTION_ID"):
        cli.expand_environment("${HYPERDX_CONNECTION_ID}", environ={"HYPERDX_CONNECTION_ID": ""})
    assert cli.expand_environment("${LATER}", environ={}, keep=("LATER",)) == "${LATER}"


def test_a_webhook_url_comes_from_the_environment_not_the_file(tmp_path: Path) -> None:
    """A bot token is a credential; the artifact names the variable that holds it."""
    pushes = cli.alerting_artifacts(_stack(tmp_path), environ=_ENVIRON)
    assert pushes[0].payload["url"] == "https://example.invalid/bot"
    assert "${TELEGRAM_URL}" not in json.dumps([push.payload for push in pushes])


def test_a_missing_variable_stops_the_push_naming_it(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.alerting_artifacts(_stack(tmp_path), environ={})
    assert "TELEGRAM_URL" in str(raised.value)


def test_a_push_sends_the_key_as_a_bearer_token_and_the_artifact_as_its_body(
    tmp_path: Path,
) -> None:
    """One write per artifact, authenticated, and the status the API answered is reported."""
    api = _FakeApi()
    lines: list[str] = []

    _push(_stack(tmp_path), api, lines)

    requests = [request for *_, request in api.sent]
    assert all(request.get_header("Authorization") == "Bearer access-key" for request in requests)
    assert all(request.get_header("Content-type") == "application/json" for request in requests)
    assert {request.full_url.partition("/api")[0] for request in requests} == {
        "http://127.0.0.1:8000"
    }
    assert len(lines) == 3 and all(line.startswith("200 ") for line in lines)


def test_a_rejected_write_names_the_artifact_the_api_refused(tmp_path: Path) -> None:
    def refuse(request: urllib.request.Request, **_kwargs: Any) -> _Response:
        if request.get_method() == "GET":
            return _Response({"data": []})
        raise urllib.error.HTTPError(
            request.full_url, 400, "Bad Request", Message(), io.BytesIO(b'{"message": "no"}')
        )

    with pytest.raises(SystemExit, match=r"channel Product · Telegram: HTTP 400 .*no"):
        cli.push_artifacts(
            cli.alerting_artifacts(_stack(tmp_path), environ=_ENVIRON),
            api_url="http://127.0.0.1:8000",
            api_key="access-key",
            opener=refuse,
            echo=lambda _line: None,
        )


def test_the_shipped_artifacts_are_pushable_and_each_alert_watches_a_window() -> None:
    """The files under stack/ are the deployment's alerts: every one must be honest.

    Two failures this holds out: an alert whose query is not scoped to the
    window it is evaluated over (it would count every span ever recorded), and an
    alert pointing at a webhook the stack does not ship.
    """
    pushes = cli.alerting_artifacts(cli.stack_root(), environ=_SHIPPED_ENVIRON)
    channel = next(push for push in pushes if push.collection == "webhooks")
    assert channel.payload["service"] == "generic"

    alerts = [push for push in pushes if push.collection == "alerts"]
    # The browser channel's `front-end-errors` alert is not shipped, because the product
    # has no browser channel; and no 5xx share is computable from what the api exports.
    assert len(alerts) == 6
    for push in alerts:
        # The webhook is the one the push has just written: the file names the variable.
        assert push.payload["channel"]["webhookId"] == f"${{{cli.CHANNEL_ID_VARIABLE}}}"
        assert push.payload["thresholdType"] == "above"
        assert push.payload["interval"] in {"5m", "15m", "1h"}
        query = push.payload["chartConfig"]["sqlTemplate"]
        assert "$__timeFilter" in query, f"{push.label} does not filter to its evaluation window"

    dashboards = [push for push in pushes if push.collection == "dashboards"]
    assert [push.payload["name"] for push in dashboards] == ["DlightRAG"]
    tiles = dashboards[0].payload["tiles"]
    assert len(tiles) == 10
    cache_tiles = {
        tile["name"]: tile["config"]["sqlTemplate"]
        for tile in tiles
        if "cache" in tile["name"] or "fresh" in tile["name"]
    }
    # Embeddings have no prompt cache, so a rate over them reads as a regression
    # the provider never offered; the two cache tiles must both leave them out.
    assert set(cache_tiles) == {"cache hit rate by span", "fresh vs cached prompt tokens"}
    for name, query in cache_tiles.items():
        assert "SpanName != 'embed-text'" in query, f"{name} counts embeddings"
        assert "gen_ai.usage.input_cached_tokens" not in query, (
            f"{name} reads an attribute no convention defines"
        )
        assert "gen_ai.usage.cache_read.input_tokens" in query, (
            f"{name} does not read the cache-read key the registry names"
        )
    # `input_tokens` includes the cache read, so a rate divides by `input` alone and
    # the fresh count is derived by subtraction. Dividing by `input + cache_read`
    # would be the double count the exclusive domain used to require.
    hit_rate = cache_tiles["cache hit rate by span"]
    denominator = hit_rate.partition("nullIf(")[2].partition(", 0)")[0]
    assert "gen_ai.usage.input_tokens" in denominator
    assert "cache_read" not in denominator, "the cache read is already inside input_tokens"
    assert "fresh_tokens,\n" in cache_tiles["fresh vs cached prompt tokens"]


def test_the_postgres_alert_counts_backends_rather_than_label_combinations() -> None:
    """`pg_stat_activity_count` has one row per (datname, state) and one per scrape.

    Summing every row over a bucket therefore measures labels times scrapes, not
    backends: the shipped query did, and 89% of five-minute windows read over its
    own threshold while the server's real peak was a third of it. One scrape is the
    unit, so the value is the busiest single scrape; a bucket around it adds nothing
    to a `number` display, whose alert already reads the worst of its window.
    """
    document = json.loads(
        (cli.stack_root() / "stack" / "alerts" / "postgres-health.json").read_text(encoding="utf-8")
    )
    query = document["chartConfig"]["sqlTemplate"]

    # Deduplicated per scrape, then the busiest single scrape is the window's value.
    assert "SELECT DISTINCT TimeUnix AS scrape, MetricName, Value, Attributes" in query
    assert "GROUP BY scrape" in query
    assert "GROUP BY ts" not in query, "a bucket between the scrape and the alert's max"
    assert "max(backends)" in query
    assert "sumIf(Value, MetricName = 'pg_stat_activity_count')" in query
    # `max_connections` is server-global: one series. A sum over it is a category error,
    # and a sum of backends over the window is the label-times-scrape error above.
    assert "max(maxconn)" in query
    assert "sumIf(Value, MetricName = 'pg_settings_max_connections')" not in query
    # An outage reads through the same number rather than a second alert.
    assert "greatest(1 - max(up)," in query


def test_the_postgres_alert_fires_when_the_server_answers_nothing_but_pg_up() -> None:
    """A server that is down reports `pg_up = 0` and none of the other series.

    `maxconn` is then 0, and the shipped query once returned 0 for any scrape whose
    `maxconn` was 0: the very scrape that says the server is down. The connection ratio is
    the branch that needs a ceiling to divide by, so only it is guarded; `1 - max(up)` is
    read whether or not the scrape carried anything else. Worked through the query: a scrape
    of `pg_up = 0` alone is `greatest(1 - 0, 0) = 1`, over the 0.8 threshold, and a healthy
    scrape of 28 backends under an 80-connection ceiling is `greatest(0, 0.35) = 0.35`.
    """
    document = json.loads(
        (cli.stack_root() / "stack" / "alerts" / "postgres-health.json").read_text(encoding="utf-8")
    )
    query = document["chartConfig"]["sqlTemplate"]

    assert "if(max(maxconn) = 0, 0, max(backends) / max(maxconn))" in query
    assert "if(max(maxconn) = 0, 0,\n" not in query, "the ceiling guard must not wrap the outage"
    # No scrape at all is a deployment that is not watching Postgres, and stays quiet.
    assert "if(count() = 0, 0, max(pressure))" in query


def test_the_disk_alert_keeps_a_mountpoint_that_has_no_free_space_left() -> None:
    """A full disk is `free = 0`, which is the row this alert exists to report.

    `HAVING free > 0` dropped it before the utilization was computed, so a mountpoint at
    100% vanished from the result and the alert stayed `OK`. The filter now removes only a
    mountpoint that reported nothing at all (`used + free = 0`, which would divide by zero).
    """
    document = json.loads(
        (cli.stack_root() / "stack" / "alerts" / "host-disk-pressure.json").read_text(
            encoding="utf-8"
        )
    )
    query = document["chartConfig"]["sqlTemplate"]

    assert "HAVING used + free > 0" in query
    assert "HAVING free > 0" not in query
    assert "used / (used + free) AS utilization" in query


def test_the_postgres_alert_states_its_threshold_as_a_ceiling() -> None:
    """0.8 of `max_connections` is a level the deployment approaches, not one it sits at.

    The retained data's peak is 0.35 of the limit over eight days of scrapes, so a name
    or a message that said real backends *reach* 80% would describe a state this
    deployment is not in. The number is a ceiling warning — it fires with a fifth of the
    server's connections still free — and the message says so.
    """
    document = json.loads(
        (cli.stack_root() / "stack" / "alerts" / "postgres-health.json").read_text(encoding="utf-8")
    )

    assert document["threshold"] == 0.8
    assert document["thresholdType"] == "above"
    assert "ceiling" in document["message"]
    assert "80%" in document["message"]
    assert "window" in document["message"], "the value is the window's worst reading"


def _shipped_queries() -> list[tuple[str, str, str]]:
    """Every SQL the stack ships, with the label and display type that carry it.

    The artifacts are read through the same `alerting_artifacts` a push uses, so what is
    checked here is what would reach the API rather than what is on disk.
    """
    pushes = cli.alerting_artifacts(cli.stack_root(), environ=_SHIPPED_ENVIRON)
    queries: list[tuple[str, str, str]] = []
    for push in pushes:
        if push.collection == "alerts":
            config = push.payload["chartConfig"]
            queries.append((push.label, config["sqlTemplate"], config.get("displayType", "")))
        elif push.collection == "dashboards":
            for tile in push.payload["tiles"]:
                config = tile.get("config", {})
                if "sqlTemplate" in config:
                    queries.append(
                        (tile["name"], config["sqlTemplate"], config.get("displayType", ""))
                    )
    return queries


def _installed_conventions() -> tuple[frozenset[str], frozenset[str]]:
    """The attribute keys the installed conventions define, and the spellings they replaced.

    The registry is the only artifact that cannot go stale in the same direction as this
    repository: a key these authors invent is absent from it, and a key the conventions
    have moved on from is named there as replaced. The GenAI attributes live in the
    incubating modules — `opentelemetry.semconv.attributes` carries no `gen_ai_attributes`
    — which is the path the exporter's own test reads too, and
    `packages/observability/pyproject.toml` declares the distribution it arrives in.

    A replaced spelling is read from the registry's own note (`Deprecated: Replaced by
    <key>`), which is what makes `code.filepath` and `gen_ai.system` unwritable without a
    second copy of the list living here.
    """
    # Imported by name rather than by statement: this walks whatever the installed
    # distribution holds, and `opentelemetry.semconv` does not export its submodules
    # to a type checker.
    incubating = importlib.import_module("opentelemetry.semconv._incubating.attributes")
    stable = importlib.import_module("opentelemetry.semconv.attributes")

    replaced = re.compile(
        r'^([A-Z][A-Z0-9_]*): Final = "([^"\n]+)"\n"""\nDeprecated: Replaced by', re.MULTILINE
    )
    keys: set[str] = set()
    superseded: set[str] = set()
    for package in (incubating, stable):
        for module_info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package.__name__}.{module_info.name}")
            for name, value in vars(module).items():
                if name.isupper() and isinstance(value, str) and "." in value:
                    keys.add(value)
            superseded |= {match.group(2) for match in replaced.finditer(inspect.getsource(module))}
    return frozenset(keys), frozenset(superseded)


#: An attribute read out of a signal's own attribute map. The Prometheus receiver passes
#: its labels through under their bare names (`mountpoint`, `state`), so only a dotted key
#: is a name a convention owns.
_NAMED_ATTRIBUTE = re.compile(r"\b(?:Span|Resource|Log)?Attributes\['([^']+)'\]")


def test_no_shipped_artifact_reads_a_key_the_conventions_do_not_define() -> None:
    """A query is a contract with the wire, so its keys are checked against the registry.

    Two shipped artifacts once read `gen_ai.usage.input_cached_tokens`, a key no
    OpenTelemetry revision has ever defined; the exporter emitted it too, so the query and
    the exporter agreed with each other and neither agreed with the registry. A
    hand-written list of forbidden names cannot hold that — it is a second copy of the
    vocabulary, and it goes stale in the same direction as the thing it guards. These keys
    come from the installed conventions instead, and a key they mark as replaced
    (`code.filepath`, `code.lineno`, `gen_ai.system`) is as unwritable as one they never
    defined. The `dlightrag.*` namespace is ours, so it is checked against the constants
    the exporter writes rather than against a convention.
    """
    keys, superseded = _installed_conventions()
    # The exporter's published surface is its public constants: a query may read what
    # the module offers, not a key it keeps to itself (`dlightrag.observation.level`
    # is one of those, and a query reads the span's status instead).
    exported = {
        value
        for name, value in vars(otlp_module).items()
        if name.isupper() and not name.startswith("_") and isinstance(value, str)
    }
    queries = _shipped_queries()
    assert len(queries) >= 6 + 4, "every alert and every SQL tile is covered"
    for label, query, _display in queries:
        for key in _NAMED_ATTRIBUTE.findall(query):
            if key.startswith("dlightrag."):
                assert key in exported, f"{label} reads {key}, which the exporter does not publish"
            elif "." in key:
                assert key in keys, f"{label} reads {key}, which no convention defines"
                assert key not in superseded, f"{label} reads {key}, which conventions replaced"


def test_a_shipped_alert_asks_only_for_the_macros_its_display_type_is_given() -> None:
    """A raw-SQL alert is rendered with its display type's macros, and no others.

    ClickStack's alert task sets `granularity` — and with it the `intervalSeconds` the
    interval macros render into — only for a time-series display. The postgres alert
    shipped `$__timeInterval(scrape)` under `displayType: number`, so every evaluation
    failed with `Substitution intervalSeconds is not set` while the alert's state stayed
    `OK`: the silent success this holds out.
    """
    interval_macros = (
        "$__timeInterval_ms",
        "$__timeInterval",
        "$__interval_s",
        "$__interval",
        "$__rate_interval",
    )
    queries = _shipped_queries()
    assert len(queries) >= 6 + 4, "every alert and every SQL tile is covered"
    for label, query, display in queries:
        if display in {"line", "stacked_bar"}:
            continue
        for macro in interval_macros:
            assert macro not in query, (
                f"{label} is a {display or 'builder'} display, which is given no interval"
            )


def test_no_shipped_artifact_carries_a_bot_token() -> None:
    """The webhook URL, the chat id, and the bot behind them are a deployment's own.

    Every file that ships names variables, never the values. ``stack/.env`` is the one
    file that holds them, and it is excluded from the distribution; the scan covers the
    whole package, not just ``stack/``, so a token that leaks into the CLI, the
    compose file, or a document is caught too.
    """
    package = Path(cli.__file__).resolve().parents[2]
    bot_token = re.compile(r"api\.telegram\.org/bot|\b\d{8,10}:[A-Za-z0-9_-]{35}\b")
    offenders = [
        str(path.relative_to(package))
        for path in package.rglob("*")
        if path.is_file()
        and path.name not in {".env"}
        and path.suffix not in {".pyc"}
        and bot_token.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == []


_OBJECT_ID = re.compile(r"\b[0-9a-f]{24}\b")


def _shipped_artifact_files() -> list[Path]:
    stack = cli.stack_root() / "stack"
    files = sorted((stack / "alerts").glob("*.json")) + sorted(
        (stack / "dashboards").glob("*.json")
    )
    assert len(files) >= 8, "every alert, the channel, and the dashboard are covered"
    return files


def test_no_shipped_artifact_names_a_stack_instance() -> None:
    """An id belongs to the stack whose Mongo minted it, so no file may carry one.

    The alerts and the dashboard once held a connection id, a source id, their own ids, and
    the webhook's, all minted by one stack's first seeding. On any other stack a push wrote
    to documents that did not exist or read a source that held nothing. What a file names
    instead is a variable, and every variable is one `bootstrap` writes or an operator sets.
    """
    known = {
        *cli.STACK_IDS,
        cli.CHANNEL_ID_VARIABLE,
        "DLIGHTRAG_TELEGRAM_CHAT_ID",
        "DLIGHTRAG_TELEGRAM_WEBHOOK_URL",
    }
    for path in _shipped_artifact_files():
        text = path.read_text(encoding="utf-8")
        document = json.loads(text)
        assert not _OBJECT_ID.search(text), f"{path.name} carries an ObjectId"
        assert "id" not in document, f"{path.name} names its own id"
        assert all("id" not in tile for tile in document.get("tiles", [])), path.name
        assert set(cli._ENV_REFERENCE.findall(text)) <= known, path.name  # noqa: SLF001

    # An id is never shipped as a value: the push asks the stack, and an operator may override.
    example = (cli.stack_root() / "stack" / ".env.example").read_text(encoding="utf-8")
    active = {
        line.partition("=")[0]
        for line in example.splitlines()
        if "=" in line and not line.startswith("#")
    }
    assert not set(cli.STACK_IDS) & active
    assert "DLIGHTRAG_TELEGRAM_WEBHOOK_ID" not in active, "the push learns the webhook's id"


def test_every_shipped_name_carries_the_product_so_two_products_can_share_a_stack() -> None:
    """Names are the identity of a push, so a name another product could also use would be
    overwritten by it: a webhook is unique per service and name, and an alert is found by its
    display name."""
    pushes = cli.alerting_artifacts(cli.stack_root(), environ=_SHIPPED_ENVIRON)
    for push in pushes:
        assert push.name.startswith("DlightRAG"), push.label
        if push.collection == "alerts":
            assert push.name.startswith("DlightRAG · "), push.label
            assert push.payload["name"].startswith("DlightRAG · "), push.label


def test_every_query_over_product_signals_is_scoped_to_the_products_services() -> None:
    """Another product exporting into the same ClickStack must not be counted here.

    The per-process service names are `dlightrag-api`, `dlightrag-mcp` and
    `dlightrag-reader`, so the scope is a prefix.
    A query over host or collector metrics has no product to scope to and is left alone.
    """
    for label, query, _display in _shipped_queries():
        if "SpanName" in query or "SpanAttributes" in query or "otel_logs" in query:
            assert "startsWith(ServiceName, 'dlightrag')" in query, (
                f"{label} reads product signals without naming the product's services"
            )
    dashboard = next(
        push
        for push in cli.alerting_artifacts(cli.stack_root(), environ=_SHIPPED_ENVIRON)
        if push.collection == "dashboards"
    )
    for tile in dashboard.payload["tiles"]:
        config = tile["config"]
        if config.get("sourceId") == "${HYPERDX_SOURCE_TRACES}" and "select" in config:
            for item in config["select"]:
                assert "startsWith(ServiceName, 'dlightrag')" in item["where"], tile["name"]


def test_the_failing_runs_tile_reads_the_attribute_the_alert_reads() -> None:
    """The tile and the alert of one name once classified a failure by different fields.

    The alert reads `dlightrag.error.kind` and falls back to the status description for
    a span recorded before that attribute existed; the tile read only the description, so
    a failure the executor classified could show under a different kind than the one that
    paged.
    """
    pushes = cli.alerting_artifacts(cli.stack_root(), environ=_SHIPPED_ENVIRON)
    tile = next(
        tile
        for push in pushes
        if push.collection == "dashboards"
        for tile in push.payload["tiles"]
        if tile["name"] == "failing runs by kind"
    )["config"]["sqlTemplate"]
    alert = next(push for push in pushes if push.label == "alert DlightRAG · failing runs by kind")
    alerted = alert.payload["chartConfig"]["sqlTemplate"]
    attribute = "notEmpty(SpanAttributes['dlightrag.error.kind'])"
    assert attribute in tile and attribute in alerted
    # The same classification: everything between `multiIf(` and its alias is one text.
    assert (
        tile.partition("multiIf(")[2].partition(" AS error_kind")[0]
        == (alerted.partition("multiIf(")[2].partition(" AS error_kind")[0])
    )


def test_a_push_resolves_the_stack_ids_and_chat_from_the_environment() -> None:
    """The ids reach the push through the environment it is given."""
    pushes = cli.alerting_artifacts(
        cli.stack_root(), environ={**_SHIPPED_ENVIRON, "HYPERDX_CONNECTION_ID": "connection-123"}
    )
    channel = next(push for push in pushes if push.collection == "webhooks")
    assert '"chat_id": "chat-1"' in channel.payload["body"]
    alerts = [push for push in pushes if push.collection == "alerts"]
    assert len(alerts) == 6
    for push in alerts:
        assert push.payload["chartConfig"]["connectionId"] == "connection-123"
    tracked = [push for push in alerts if "sourceId" in push.payload["chartConfig"]]
    assert {push.payload["chartConfig"]["sourceId"] for push in tracked} <= {
        "source-traces",
        "source-metrics",
    }


def test_a_push_reads_the_stack_env_file_the_operator_edits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`stack/.env` is where the operator puts the values; nobody has to export them."""
    import argparse

    alerts = tmp_path / "stack" / "alerts"
    alerts.mkdir(parents=True)
    (alerts / "channel.json").write_text(
        json.dumps(
            {
                "name": "Product · Telegram",
                "service": "generic",
                "url": "${DLIGHTRAG_TELEGRAM_WEBHOOK_URL}",
                "body": '{"chat_id": "${DLIGHTRAG_TELEGRAM_CHAT_ID}", "text": "{{title}}"}',
            }
        ),
        encoding="utf-8",
    )
    (alerts / "one-alert.json").write_text(
        json.dumps(
            {
                "name": "Product · the api is answering 5xx",
                "displayName": "Product · api 5xx",
                "chartConfig": {"connectionId": "${HYPERDX_CONNECTION_ID}"},
            }
        ),
        encoding="utf-8",
    )
    variables = {
        "DLIGHTRAG_TELEGRAM_CHAT_ID": "chat-9",
        "DLIGHTRAG_TELEGRAM_WEBHOOK_URL": "https://example.invalid/bot",
        "HYPERDX_CONNECTION_ID": "connection-9",
    }
    (tmp_path / "stack" / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in variables.items()), encoding="utf-8"
    )
    monkeypatch.setenv("DLIGHTRAG_OBSERVABILITY_STACK_DIR", str(tmp_path))
    for key in variables:
        monkeypatch.delenv(key, raising=False)

    assert cli._push_alerts(argparse.Namespace(dry_run=True)) == 0  # noqa: SLF001
    printed = capsys.readouterr().out
    assert "UPSERT /api/v2/webhooks — channel Product · Telegram" in printed
    assert "UPSERT /api/v2/alerts — alert Product · api 5xx" in printed

    # A dry run sends nothing, so an id the stack would be asked for does not stop it.
    (tmp_path / "stack" / ".env").write_text(
        "DLIGHTRAG_TELEGRAM_CHAT_ID=chat-9\nDLIGHTRAG_TELEGRAM_WEBHOOK_URL=https://example.invalid/bot\n",
        encoding="utf-8",
    )
    assert cli._push_alerts(argparse.Namespace(dry_run=True)) == 0  # noqa: SLF001
    assert "UPSERT /api/v2/alerts — alert Product · api 5xx" in capsys.readouterr().out


# ── The stack's own ids, asked of its API ────────────────────────────────────

_SEEDED_SOURCES = [
    {"id": "6ac9470bc0696e82a1080997", "kind": "metric", "name": "Metrics"},
    {"id": "6ac9470bc0696e82a1080995", "kind": "trace", "name": "Traces"},
    {"id": "6ac9470bc0696e82a1080994", "kind": "log", "name": "Logs"},
    # A source an operator added later is newer, so it never displaces the one that was seeded.
    {"id": "6ad0000000000000000000aa", "kind": "trace", "name": "Traces (staging)"},
]
_SEEDED_CONNECTIONS = [
    {"id": "6ad0000000000000000000bb", "name": "Another"},
    {"id": "6ac9470bc0696e82a108098f", "name": "Local ClickHouse"},
]


class _FakeStackApi:
    """The two listings a push asks for its ids, answered as ClickStack answers them."""

    def __init__(self, connections: list[dict[str, str]], sources: list[dict[str, str]]) -> None:
        self.listings = {"connections": connections, "sources": sources}
        self.sent: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request, **_kwargs: Any) -> _Response:
        self.sent.append(request)
        collection = urllib.parse.urlsplit(request.full_url).path.rpartition("/")[2]
        return _Response({"data": self.listings[collection]})


def _resolve(environ: dict[str, str], api: _FakeStackApi) -> dict[str, str]:
    return cli.resolve_stack_ids(
        environ, api_url="http://127.0.0.1:8000/", api_key="access-key", opener=api
    )


def test_the_stack_ids_are_the_oldest_connection_and_the_oldest_source_of_each_kind() -> None:
    """What the first seeding wrote is what a push reads, found through the stack's own API."""
    api = _FakeStackApi(_SEEDED_CONNECTIONS, _SEEDED_SOURCES)

    resolved = _resolve({"UNRELATED": "kept"}, api)

    assert resolved == {
        "UNRELATED": "kept",
        "HYPERDX_CONNECTION_ID": "6ac9470bc0696e82a108098f",
        "HYPERDX_SOURCE_TRACES": "6ac9470bc0696e82a1080995",
        "HYPERDX_SOURCE_METRICS": "6ac9470bc0696e82a1080997",
    }
    # Each listing is asked for once, however many ids come out of it.
    assert [request.full_url for request in api.sent] == [
        "http://127.0.0.1:8000/api/v2/connections",
        "http://127.0.0.1:8000/api/v2/sources",
    ]
    assert all(request.get_method() == "GET" for request in api.sent)
    assert all(request.get_header("Authorization") == "Bearer access-key" for request in api.sent)


def test_a_value_the_operator_set_wins_and_is_not_asked_for() -> None:
    explicit = {
        "HYPERDX_CONNECTION_ID": "mine-connection",
        "HYPERDX_SOURCE_TRACES": "mine-traces",
        "HYPERDX_SOURCE_METRICS": "mine-metrics",
    }
    # An empty stack would stop the push if it were asked, so no request is the proof.
    api = _FakeStackApi([], [])
    assert _resolve(explicit, api) == explicit
    assert api.sent == []

    # Partly set: only what is missing is asked, and a variable set to nothing is missing.
    api = _FakeStackApi(_SEEDED_CONNECTIONS, _SEEDED_SOURCES)
    resolved = _resolve(
        {"HYPERDX_CONNECTION_ID": "mine-connection", "HYPERDX_SOURCE_TRACES": ""}, api
    )
    assert resolved["HYPERDX_CONNECTION_ID"] == "mine-connection"
    assert resolved["HYPERDX_SOURCE_TRACES"] == "6ac9470bc0696e82a1080995"
    assert [urllib.parse.urlsplit(request.full_url).path for request in api.sent] == [
        "/api/v2/sources"
    ]


@pytest.mark.parametrize(
    ("connections", "sources", "missing"),
    [
        ([], _SEEDED_SOURCES, "HYPERDX_CONNECTION_ID"),
        (_SEEDED_CONNECTIONS, [_SEEDED_SOURCES[2]], "HYPERDX_SOURCE_TRACES"),
        (_SEEDED_CONNECTIONS, _SEEDED_SOURCES[1:], "HYPERDX_SOURCE_METRICS"),
    ],
    ids=["no connection", "no trace source", "no metric source"],
)
def test_a_stack_without_what_a_query_reads_stops_the_push_and_says_how_to_get_it(
    connections: list[dict[str, str]], sources: list[dict[str, str]], missing: str
) -> None:
    """The team, and with it the sources, is created when the first administrator is."""
    with pytest.raises(SystemExit) as raised:
        _resolve({}, _FakeStackApi(connections, sources))

    message = str(raised.value)
    assert message.startswith(f"{missing}: the stack has no ")
    assert "Create the administrator and the sources in the UI first" in message
    assert "`alerts push` again" in message


def test_a_push_asks_the_stack_for_its_ids_and_writes_them_into_the_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From the environment to the wire: the alert goes out naming the stack's own connection."""
    import argparse

    root = _stack(tmp_path)
    (root / "stack" / ".env").write_text(
        "TELEGRAM_URL=https://example.invalid/bot\n", encoding="utf-8"
    )
    monkeypatch.setenv("DLIGHTRAG_OBSERVABILITY_STACK_DIR", str(root))
    monkeypatch.setenv(cli.API_KEY_VARIABLE, "access-key")
    monkeypatch.delenv(cli.API_URL_VARIABLE, raising=False)
    for name in cli.STACK_IDS:
        monkeypatch.delenv(name, raising=False)
    api = _FakeApi(stack={"connections": _SEEDED_CONNECTIONS, "sources": _SEEDED_SOURCES})

    assert cli._push_alerts(argparse.Namespace(dry_run=False), opener=api) == 0  # noqa: SLF001

    alert = next(
        body for method, path, body, _ in api.sent if method == "POST" and "alerts" in path
    )
    assert alert is not None
    assert alert["chartConfig"]["connectionId"] == "6ac9470bc0696e82a108098f"


# ── What the stack ships beside the artifacts ────────────────────────────────


def test_every_shipped_ttl_default_is_the_week_the_cli_and_the_example_state() -> None:
    """Four statements of one retention once said two things: 720h and 168h."""
    root = cli.stack_root()
    default = re.compile(rf"{cli.RETENTION_VARIABLE}:-(\w+)\}}")
    for relative in ("compose.yaml", "stack/collector-standalone.yaml"):
        found = default.findall((root / relative).read_text(encoding="utf-8"))
        assert found, f"{relative} states no default"
        for value in found:
            assert cli.retention_ttl(value).seconds == cli.DEFAULT_RETENTION.seconds, relative


def test_the_generated_secrets_never_reach_a_docker_build_context() -> None:
    """`COPY packages/ packages/` would otherwise carry `stack/.env` into a builder layer."""
    assert "**/stack/.env" in Path(".dockerignore").read_text(encoding="utf-8").splitlines()


def test_each_application_process_has_its_own_service_name() -> None:
    """One name for three processes would put them on one line of the service map."""
    import yaml

    services = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))["services"]
    for service in ("dlightrag-api", "dlightrag-mcp", "dlightrag-reader"):
        environment = services[service]["environment"]
        assert environment["DLIGHTRAG_OBSERVABILITY__SERVICE_NAME"] == service
