# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the OTLP log export renders, read off a recording log exporter.

The export is the second signal on the seam [ADR 0039][adr] governs: the records
the process already prints reach ClickHouse with the trace id of the span that
logged them, masked by the same classifier, and a dead destination costs a
record rather than a logging call. Only the network leg is the test's.

[adr]: ../../docs/adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from dlightrag_observability import otlp as otlp_module
from dlightrag_observability.otlp import shutdown_log_export, start_log_export
from dlightrag_observability.privacy import private_path_placeholder
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogRecordExporter, LogRecordExportResult

from tests.unit.conftest import OtlpExport, OtlpLogs


def _log(name: str, message: str, **extra: Any) -> None:
    """One product record: the logger namespace the application uses, and its extras."""
    logging.getLogger(f"dlightrag.{name}").warning(message, extra=extra or None)


async def test_a_record_carries_the_trace_and_span_that_logged_it(
    otlp_export: OtlpExport, start_otlp_logs: Callable[..., OtlpLogs]
) -> None:
    """A reader joins a log line to its trace: the ids are the current span's."""
    logs = start_otlp_logs()

    with otlp_export.telemetry.trace(session_id="sess-1", user_id="owner-1"):
        async with otlp_export.telemetry.observe("run-answer") as observation:
            _log("engine.answer", "the run started")
            observation.update(output={"answer": "ok"})

    (span,) = otlp_export.spans()
    assert span.context is not None
    (record,) = logs.records()
    assert logs.trace_id(record) == format(span.context.trace_id, "032x")
    assert logs.span_id(record) == format(span.context.span_id, "016x")
    assert logs.body(record) == "the run started"
    assert logs.attributes(record)["code.line.number"] > 0


def test_a_record_logged_with_no_span_open_carries_no_trace(
    start_otlp_logs: Callable[..., OtlpLogs],
) -> None:
    """Outside a Run there is nothing to correlate: the ids are the wire's zero."""
    logs = start_otlp_logs()

    _log("adapters.http.server", "no run here")

    (record,) = logs.records()
    assert logs.trace_id(record) == "0" * 32
    assert logs.span_id(record) == "0" * 16
    assert logs.body(record) == "no run here"


def test_a_secret_under_a_secret_name_is_masked_before_it_leaves(
    start_otlp_logs: Callable[..., OtlpLogs],
) -> None:
    """`extra={"api_key": ...}` is a credential by the same rule a payload is."""
    logs = start_otlp_logs()

    _log(
        "adapters.mcp.server",
        "connecting to the provider",
        api_key="sk-live-secret",
        authorization="Bearer secret",
        run_id="run-1",
    )

    (record,) = logs.records()
    attributes = logs.attributes(record)
    assert attributes["api_key"] == "[redacted]"
    assert attributes["authorization"] == "[redacted]"
    assert attributes["run_id"] == "run-1"


def test_a_record_names_its_code_location_with_the_stable_attributes(
    start_otlp_logs: Callable[..., OtlpLogs],
) -> None:
    """`code.file.path` / `code.line.number` are the registry's names.

    `code.filepath` / `code.lineno` are deprecated there, and `code.function.name`
    is meant to be the runtime's fully-qualified name rather than the bare one.
    """
    logs = start_otlp_logs()

    _log("engine.answer", "the run started")

    (record,) = logs.records()
    attributes = logs.attributes(record)
    assert "code.filepath" not in attributes
    assert "code.lineno" not in attributes
    # This suite runs from a checkout under a home directory, so the path is
    # masked (D3) while the basename a reader follows survives.
    assert attributes["code.file.path"] == private_path_placeholder(__file__)
    assert __file__ not in str(attributes["code.file.path"])
    assert attributes["code.line.number"] > 0
    assert attributes["code.function.name"] == "test_observability_logs._log"


def test_a_record_outside_any_home_directory_keeps_its_whole_path() -> None:
    """The path rule is about *home* directories, not about paths.

    A deployment's container path carries no account name, so masking it would
    cost a reader the line without protecting anyone.
    """
    record = logging.LogRecord(
        name="dlightrag.engine.answer",
        level=logging.WARNING,
        pathname="/app/.venv/lib/python3.14/site-packages/dlightrag/answer.py",
        lineno=42,
        msg="the run started",
        args=(),
        exc_info=None,
    )
    record.funcName = "execute"

    attributes = otlp_module._log_attributes(record)  # noqa: SLF001 - the renderer under test
    assert attributes["code.file.path"] == record.pathname
    assert attributes["code.function.name"] == "answer.execute"


def test_a_home_directory_in_a_path_attribute_is_masked_before_it_leaves(
    start_otlp_logs: Callable[..., OtlpLogs],
) -> None:
    """A path naming the operator is private under ADR 0039, whatever key carries it.

    The search toolchain reports where its binaries live under a path-shaped extra;
    the classifier decides, not the exporter, so a path the app logs is masked by
    the same rule that masks anything else private.
    """
    logs = start_otlp_logs()

    _log(
        "engine.agent.environment.toolchain",
        "Agent search toolchain ready",
        **{"search_toolchain.fd.path": "/Users/someone/.dlightrag/tools/fd/10.5.0/fd"},
    )

    (record,) = logs.records()
    masked = logs.attributes(record)["search_toolchain.fd.path"]
    assert masked.startswith("[private path: fd: ")
    assert "/Users/" not in str(masked)
    assert "someone" not in str(masked)


def test_a_record_carries_the_service_that_made_it_and_its_severity(
    start_otlp_logs: Callable[..., OtlpLogs],
) -> None:
    logs = start_otlp_logs()

    logging.getLogger("dlightrag.engine.ai.completion").error("the provider said no")

    (record,) = logs.records()
    assert record.resource.attributes["service.name"] == "dlightrag-api"
    assert record.resource.attributes["service.version"] == "9.9.9"
    assert record.log_record.severity_text == "ERROR"


class _FailingExporter(LogRecordExporter):
    """A destination that is not there, so what an outage costs can be read."""

    def __init__(self) -> None:
        self.failures = 0
        self.failing = True

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        del batch
        if self.failing:
            self.failures += 1
            raise ConnectionError("the collector is not answering")
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True


def test_a_dead_destination_neither_raises_nor_floods_a_logging_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The outage is the telemetry's: one report, and the logging call never hears of it."""
    exporter = _FailingExporter()
    start_log_export(log_exporter=exporter)
    try:
        with caplog.at_level(logging.DEBUG, logger="dlightrag_observability.otlp"):
            for index in range(5):
                _log("engine.answer", f"attempt {index}")
            warnings = [
                record
                for record in caplog.records
                if record.name == "dlightrag_observability.otlp"
                and record.levelno >= logging.WARNING
            ]
            assert [record.getMessage() for record in warnings] == [
                "OTLP log export is failing; telemetry is dropped until the collector answers"
            ]
            exporter.failing = False
            _log("engine.answer", "after the outage")
        recovered = [
            record
            for record in caplog.records
            if record.name == "dlightrag_observability.otlp"
            and record.levelno == logging.INFO
            and "recovered" in record.getMessage()
        ]
        assert exporter.failures == 5
        assert len(recovered) == 1
    finally:
        shutdown_log_export()


def test_no_endpoint_attaches_no_handler() -> None:
    """A development process without a destination keeps stdout as its only channel."""
    before = list(logging.getLogger().handlers)

    start_log_export(endpoint=None)

    assert logging.getLogger().handlers == before
    assert otlp_module._log_export is None  # noqa: SLF001 - the process state under test


class _RecordingExporter(LogRecordExporter):
    """An exporter whose only job is to keep the provider constructible."""

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        del batch
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True


def test_the_export_is_an_additional_handler_not_a_replacement() -> None:
    """`logging.basicConfig` keeps owning the console; the export sits beside it."""
    root = logging.getLogger()
    existing = logging.StreamHandler()
    root.addHandler(existing)
    before = set(root.handlers)

    start_log_export(log_exporter=_RecordingExporter())
    added = set(root.handlers) - before
    try:
        assert existing in root.handlers
        assert len(added) == 1
    finally:
        shutdown_log_export()
        root.removeHandler(existing)

    assert not (set(root.handlers) & added)
    assert existing not in root.handlers
