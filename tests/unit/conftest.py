# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit-test fixtures; the root conftest isolates every suite from operator inputs."""

import json
import shutil
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import pytest
from dlightrag_observability.otlp import (
    DLIGHTRAG_TOTAL,
    GEN_AI_USAGE_CACHE_READ,
    GEN_AI_USAGE_INPUT,
    GEN_AI_USAGE_OUTPUT,
    OtlpTelemetry,
    shutdown_log_export,
    shutdown_telemetry,
    start_log_export,
    start_telemetry,
)
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dlightrag.engine.agent.skills import builtin_skills_root
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.media import MODEL_IMAGE_MAX_PIXELS
from dlightrag.engine.ai.providers.base import provider_usage_buckets
from dlightrag.engine.ai.structured_transport import JSON_SCHEMA_TRANSPORT_CACHE
from dlightrag.engine.ai.telemetry import SPAN_TYPES
from dlightrag.engine.answer.capabilities import AnswerCapabilities
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from dlightrag.engine.answer.images import AnswerImagePolicy
from tests.support.skills import sealed_skills_root


@pytest.fixture(scope="session")
def _builtins_with_sealed_report(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The packaged built-in Skills, and the neutral ``sealed-report`` Skill beside them.

    One sealed artifact name has one gate, so a built-in that declares ``case.json`` too would be
    refused beside the neutral Skill; the copy leaves out any that does.
    """
    root = tmp_path_factory.mktemp("builtin-skills")
    for source in Path(str(builtin_skills_root())).iterdir():
        hub = source / "SKILL.md"
        if hub.is_file() and "sealed_artifact:" not in hub.read_text(encoding="utf-8"):
            shutil.copytree(
                source, root / source.name, ignore=shutil.ignore_patterns("__pycache__")
            )
    shutil.copytree(sealed_skills_root(), root, dirs_exist_ok=True)
    return root


@pytest.fixture
def sealed_report_packaged(
    monkeypatch: pytest.MonkeyPatch, _builtins_with_sealed_report: Path
) -> None:
    """Serve the neutral ``sealed-report`` Skill beside the built-ins a Run's catalog starts from.

    The application composes its catalog from the package, so a test that derives what a Skill
    declares about a Run (its attestation, its required reads) needs a package that holds one.
    """
    monkeypatch.setattr(
        "dlightrag.application.skills.builtin_skills_root", lambda: _builtins_with_sealed_report
    )


def answer_image_policy(**overrides: int) -> AnswerImagePolicy:
    """Shipped answer transport policy for tests; images off unless opted in."""
    fields: dict[str, int] = {
        "max_images": 0,
        "max_total_bytes": 24_000_000,
        "max_bytes_per_image": 3_000_000,
        "max_pixels": MODEL_IMAGE_MAX_PIXELS,
        "max_px": 1536,
        "min_px": 1024,
        "quality": 89,
        "min_quality": 79,
    }
    return AnswerImagePolicy(**(fields | overrides))


def answer_model_profile(**overrides: int | bool | None) -> ModelProfile:
    """Resolved answer-model facts for tests that do not exercise the catalog."""
    fields: dict[str, int | bool | None] = {
        "context_window_tokens": 1_000_000,
        "max_input_tokens": None,
        "max_output_tokens": 128_000,
        "supports_images": True,
    }
    return ModelProfile(**(fields | overrides))  # type: ignore[arg-type]


def answer_capabilities(answer: AnswerImageCapability | None = None) -> AnswerCapabilities:
    """The capability snapshot a transport test's answers double reports."""
    return AnswerCapabilities(answer=answer, vlm_status="unknown")


@pytest.fixture(autouse=True)
def _fresh_json_schema_transport_cache():
    """Keep the process-wide json_schema transport cache out of test verdicts."""
    JSON_SCHEMA_TRANSPORT_CACHE.clear()
    yield
    JSON_SCHEMA_TRANSPORT_CACHE.clear()


class OtlpExport:
    """What the process's OTLP exporter, started by ``start_telemetry``, has exported.

    Only the span exporter belongs to the test: the tracer provider, its
    sampling, the span rendering, and the masking in front of it are the
    product's, so an assertion here is an assertion about the wire.
    """

    def __init__(self, exporter: InMemorySpanExporter, telemetry: OtlpTelemetry) -> None:
        self._exporter = exporter
        # Built before the exporter starts, like the composition root builds its
        # handles: a handle resolves the installed export when a span opens.
        self.telemetry = telemetry

    def spans(self) -> tuple[ReadableSpan, ...]:
        return self._exporter.get_finished_spans()

    @staticmethod
    def attributes(span: ReadableSpan) -> dict[str, Any]:
        return dict(span.attributes or {})

    @staticmethod
    def payload(span: ReadableSpan, attribute: str) -> Any:
        """One JSON payload attribute, decoded as a reviewer's UI would decode it."""
        return json.loads(str((span.attributes or {})[attribute]))

    @staticmethod
    def usage(span: ReadableSpan) -> dict[str, int]:
        """The usage counters the span carries, none when it carries none.

        ``input`` is the inclusive prompt and ``cache_read_input_tokens`` its
        cached subset, which is the registry's nesting rather than a partition;
        ``total`` is read off the provider's own counters, never re-derived here.
        """
        attributes = span.attributes or {}
        carried: dict[str, int] = {}
        for bucket, attribute in (
            ("input", GEN_AI_USAGE_INPUT),
            ("cache_read_input_tokens", GEN_AI_USAGE_CACHE_READ),
            ("output", GEN_AI_USAGE_OUTPUT),
        ):
            value = attributes.get(attribute)
            if isinstance(value, int) and not isinstance(value, bool):
                carried[bucket] = value
        return carried

    @staticmethod
    def total(span: ReadableSpan) -> int | None:
        """The provider's own total counter, which is not the sum of the buckets.

        ``input`` already contains ``cache_read_input_tokens``, so a total is
        ``input + output`` when the provider states none, never all three.
        """
        value = (span.attributes or {}).get(DLIGHTRAG_TOTAL)
        return value if isinstance(value, int) and not isinstance(value, bool) else None


@pytest.fixture
def start_otlp_export() -> Generator[Callable[..., OtlpExport]]:
    """Start the real exporter through ``start_telemetry``, exporting into memory.

    The exporter replaces only the network leg; everything in front of it — the
    provider, the sampler, the attribute rendering, and the masker — is the
    product's, which is what makes these spans evidence about the wire.
    """

    def start(**settings: Any) -> OtlpExport:
        exporter = InMemorySpanExporter()
        telemetry = OtlpTelemetry(span_types=SPAN_TYPES, usage_buckets=provider_usage_buckets)
        start_telemetry(
            span_types=SPAN_TYPES,
            usage_buckets=provider_usage_buckets,
            endpoint="http://127.0.0.1:4318",
            service_name="dlightrag-api",
            environment="test",
            release="9.9.9",
            span_exporter=exporter,
            **settings,
        )
        return OtlpExport(exporter, telemetry)

    yield start
    shutdown_telemetry()


@pytest.fixture
def otlp_export(start_otlp_export: Callable[..., OtlpExport]) -> OtlpExport:
    return start_otlp_export()


class OtlpLogs:
    """What the process's OTLP log export, started by ``start_log_export``, has exported.

    As with ``OtlpExport``, only the exporter belongs to the test: the handler,
    the masking, and the trace context a record carries are the product's.
    """

    def __init__(self, exporter: InMemoryLogRecordExporter) -> None:
        self._exporter = exporter

    def records(self) -> tuple[ReadableLogRecord, ...]:
        return self._exporter.get_finished_logs()

    @staticmethod
    def body(record: ReadableLogRecord) -> str:
        return str(record.log_record.body)

    @staticmethod
    def attributes(record: ReadableLogRecord) -> dict[str, Any]:
        return dict(record.log_record.attributes or {})

    @staticmethod
    def trace_id(record: ReadableLogRecord) -> str:
        """The trace a record belongs to, as the wire spells it: 32 hex zeros when none."""
        return format(record.log_record.trace_id or 0, "032x")

    @staticmethod
    def span_id(record: ReadableLogRecord) -> str:
        return format(record.log_record.span_id or 0, "016x")


@pytest.fixture
def start_otlp_logs() -> Generator[Callable[..., OtlpLogs]]:
    """Attach the real log export, writing into memory instead of to the network."""

    def start(**settings: Any) -> OtlpLogs:
        exporter = InMemoryLogRecordExporter()
        start_log_export(
            log_exporter=exporter,
            service_name="dlightrag-api",
            environment="test",
            release="9.9.9",
            **settings,
        )
        # The line the export logs about itself is not the record under test.
        exporter.clear()
        return OtlpLogs(exporter)

    yield start
    shutdown_log_export()


@pytest.fixture
def serve_oembed(monkeypatch: pytest.MonkeyPatch) -> Callable[[Any], Any]:
    """Answer the playback resolver's oEmbed reads with a scripted fetch."""

    def install(fetch: Any) -> Any:
        monkeypatch.setattr(
            "dlightrag.adapters.http.browser.video_playback.fetch_public_http", fetch
        )
        return fetch

    return install
