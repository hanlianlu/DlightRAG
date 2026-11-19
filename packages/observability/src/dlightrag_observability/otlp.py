# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The OTLP export: DlightRAG's observations and log records, rendered as OTLP.

The application owns what an observation *is* — the span-name→kind registry and
the provider dialects usage arrives in are constructor data, and this module
states no span name of its own. It owns the two things the application must not
know: how an observation is rendered on the wire, and how it is transported.
The same seam carries the process's log records, so a trace and the lines it
logged meet in ClickHouse instead of only in a terminal.

Three properties shape the module, and [ADR 0039][adr] is why they are not
optional:

- **It never fails the product.** A construction that fails installs nothing and
  returns a handle whose every call is a no-op; an export that fails drops spans
  and records from a bounded queue and returns; a shutdown flushes under a
  timeout instead of waiting on a dead collector.
- **It is never noisy.** One circuit breaker per channel around the exporters
  reports the first failure at ``WARNING`` and the first success after it at
  ``INFO``; the failures between are counted, not logged, and the SDK's own
  export loggers are raised to ``CRITICAL`` so the SDK cannot log a line per
  batch either.
- **Nothing leaves unmasked.** Every payload, log message, and attribute passes
  the classifier in ``dlightrag_observability.privacy``, in this process,
  before it reaches the network.

[adr]: ../../../../docs/adr/0039-observability-is-always-on-and-the-seam-masks-private-data.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Final

from opentelemetry import trace
from opentelemetry._logs import SeverityNumber
from opentelemetry.context import get_current
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk._logs import LoggerProvider, LogRecordProcessor, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    LogRecordExporter,
    LogRecordExportResult,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from dlightrag_observability.privacy import (
    classify,
    mask_payload,
    private_path_placeholder,
)

logger = logging.getLogger(__name__)

#: The app's kind rides on every span because it is the aggregation key the
#: trace table, its dashboards, and its evaluators target, and OTel has no
#: equivalent.
OBSERVATION_KIND: Final[str] = "dlightrag.observation.kind"

#: The OpenTelemetry GenAI convention for a model, a tool, and an embedding call.
#: The registry these names come from is the primary source; the key names are
#: checked against it rather than against this module's own documentation, which is
#: how a set of self-consistent but stale names survived here once already.
GEN_AI_OPERATION: Final[str] = "gen_ai.operation.name"
GEN_AI_PROVIDER: Final[str] = "gen_ai.provider.name"
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL: Final[str] = "gen_ai.response.model"
GEN_AI_USAGE_INPUT: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT: Final[str] = "gen_ai.usage.output_tokens"
#: The cache-read counter. The name is the registry's; ``input_cached_tokens`` was
#: never one, in any revision.
GEN_AI_USAGE_CACHE_READ: Final[str] = "gen_ai.usage.cache_read.input_tokens"
GEN_AI_USAGE_COST: Final[str] = "gen_ai.usage.cost"
#: Required on an ``execute_tool`` span, and the column HyperDX's LLM view reads
#: for the tool. It is emitted at span creation, where the call site knows it.
GEN_AI_TOOL_NAME: Final[str] = "gen_ai.tool.name"
#: Conditionally required: "If the operation ended in an error".
ERROR_TYPE: Final[str] = "error.type"

#: Where a log record was emitted. The stable names; ``code.filepath`` and
#: ``code.lineno`` are deprecated in the registry, which is where these come from.
_CODE_FILE_PATH: Final[str] = "code.file.path"
_CODE_LINE_NUMBER: Final[str] = "code.line.number"
_CODE_FUNCTION_NAME: Final[str] = "code.function.name"

#: Attribution: the Agent Session and its owner, which the ``trace`` scope sets
#: and every observation inside it carries.
SESSION_ID: Final[str] = "session.id"
USER_ID: Final[str] = "user.id"

#: DlightRAG's own attributes: the payloads a reviewer reads, and the total
#: that is not one of the provider's counter dialects.
DLIGHTRAG_INPUT: Final[str] = "dlightrag.input"
DLIGHTRAG_OUTPUT: Final[str] = "dlightrag.output"
DLIGHTRAG_METADATA: Final[str] = "dlightrag.metadata"
DLIGHTRAG_MODEL_PARAMETERS: Final[str] = "dlightrag.model_parameters"
DLIGHTRAG_TOTAL: Final[str] = "dlightrag.usage.total"
DLIGHTRAG_COMPLETION_START: Final[str] = "dlightrag.completion_start_time"

#: The typed failure kind a call site already knows, such as ``run_abandoned`` or
#: ``dependency_unavailable``. It rides beside the status message rather than in
#: it: an alert groups on this attribute and reads the message only for spans
#: recorded before the kind existed, which is why the kind reaches the wire twice
#: — once here for the alert, and once as ``error.type`` for a standard consumer.
DLIGHTRAG_ERROR_KIND: Final[str] = "dlightrag.error.kind"

#: A level a caller named that is neither ``ERROR`` (the span's status) nor the
#: "no opinion" levels. The app's level authority lives behind :func:`span_status`,
#: so this key is the module's own; it is not part of the published surface and a
#: query should read the span's status instead.
_LEVEL_ATTRIBUTE: Final[str] = "dlightrag.observation.level"

#: The status description a caller named, and the fallback an alert reads when a
#: span carries no :data:`DLIGHTRAG_ERROR_KIND`. A caller that names ``ERROR``
#: reaches this through the standard span status; a caller that names another
#: level keeps the same message beside the level attribute.
DLIGHTRAG_STATUS_MESSAGE: Final[str] = "dlightrag.status_message"

#: The GenAI operation a DlightRAG kind is. Kinds the convention does not name
#: — an agent, a chain, a retriever, a plain span — keep their kind attribute and
#: carry no operation.
_OPERATIONS: Final[Mapping[str, str]] = {
    "generation": "chat",
    "tool": "execute_tool",
    "embedding": "embeddings",
}

#: The span kind each observation kind is rendered as. The convention asks a call
#: to a model or an embedding endpoint to be ``CLIENT`` — the work happens in
#: another process — and tool execution, which this process performs, to be
#: ``INTERNAL``. Everything else is DlightRAG's own orchestration and stays
#: ``INTERNAL``. Span *names* stay as they are: the convention states them as
#: SHOULD, and these names are a deliberate API.
_SPAN_KINDS: Final[Mapping[str, SpanKind]] = {
    "generation": SpanKind.CLIENT,
    "embedding": SpanKind.CLIENT,
}
_DEFAULT_SPAN_KIND: Final[SpanKind] = SpanKind.INTERNAL

#: The kind an unregistered name renders as. The registry is closed and its test
#: refuses an unknown name before it ships, so this is a last-resort shape rather
#: than a default anyone should reach.
_GENERIC_KIND: Final[str] = "span"

#: How each provider the app can produce is rendered as ``gen_ai.provider.name``.
#: The app owns its provider vocabulary — a configuration says ``gemini``, and
#: every span, cache key, and log line in the product says it too — and rendering
#: that vocabulary onto the wire is this module's job, the same way
#: :data:`_SPAN_KINDS` renders an observation kind.
#:
#: The registry's enum is the target, so a value the app spells differently is
#: translated here: ``gemini`` is the AI Studio endpoint the app's provider speaks
#: by default, which the registry names ``gcp.gemini`` (``gcp.gen_ai`` is for an
#: endpoint whose backend is unknown, which a provider literal cannot express),
#: and ``azure_cohere`` is Cohere's protocol served from an Azure deployment — the
#: platform the client is configured against, which is ``azure.ai.inference``.
#: ``jina``, ``openai_compatible``, and ``voyage`` have no registry spelling at
#: all; the convention's enum ends "otherwise a custom value MAY be used", so they
#: map to themselves rather than to a value the registry does not have or to one
#: that would claim a provider this client is not talking to.
#:
#: The table is total over the app's provider literals by test: adding one to
#: ``ChatProvider`` or ``EmbeddingSettings.provider`` without deciding its
#: rendering fails ``tests/unit/test_observability_wire.py``.
_GEN_AI_PROVIDERS: Final[Mapping[str, str]] = {
    # The chat providers (``engine/ai/contracts.ChatProvider``).
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gcp.gemini",
    # The embedding providers (``engine/ai/settings.EmbeddingSettings.provider``).
    "azure_cohere": "azure.ai.inference",
    "cohere": "cohere",
    "jina": "jina",
    "openai_compatible": "openai_compatible",
    "voyage": "voyage",
}


def _rendered_provider(provider: str) -> str:
    """The registry's name for the app's provider, or the app's own where it has none.

    A provider neither the app nor this table knows is passed through unchanged:
    the convention states this attribute as the instrumentation's best knowledge,
    and a literal the exporter cannot place is still better knowledge than a
    dropped attribute or an invented one.
    """
    return _GEN_AI_PROVIDERS.get(provider, provider)


#: Usage buckets, and the attribute each one becomes. The app's mapper answers
#: which provider counter belongs in which bucket; the wire names are ours.
#:
#: ``input`` is the *inclusive* prompt — the registry says
#: :data:`GEN_AI_USAGE_INPUT` "SHOULD include all types of input tokens, including
#: cached tokens" — while ``cache_read.input_tokens`` is the subset of it a
#: provider-managed cache served. A consumer that wants the fresh count derives
#: ``input - cache_read``; that derivation is the only place the exclusive number
#: exists now, which is what keeps the sum of ``input_tokens`` across spans an
#: honest prompt total.
_USAGE_ATTRIBUTES: Final[Mapping[str, str]] = {
    "input": GEN_AI_USAGE_INPUT,
    "output": GEN_AI_USAGE_OUTPUT,
    "cache_read_input_tokens": GEN_AI_USAGE_CACHE_READ,
    # Not a convention key: the registry states cost and usage as metrics rather
    # than putting a total on the span. It is DlightRAG's own counter and stays
    # because a provider's *stated* total is a fact no consumer can derive when
    # the provider bills differently from the buckets above.
    "total": DLIGHTRAG_TOTAL,
}

#: One span attribute is bounded, in bytes. A payload arrives already bounded by
#: the classifier; this is the second bound, so one pathological value cannot
#: write a row ClickHouse would rather not hold.
_MAX_ATTRIBUTE_BYTES: Final[int] = 64 * 1024

#: The OTLP/HTTP signal path each channel appends to the configured base.
_TRACES_SIGNAL: Final[str] = "v1/traces"
_LOGS_SIGNAL: Final[str] = "v1/logs"

#: What this package's own loggers are named. Their records are the export
#: reporting on itself, and the log handler leaves them to stdout: see
#: :meth:`_LogHandler.emit`.
_EXPORT_LOGGER_PREFIX: Final[str] = "dlightrag_observability."

#: What a :class:`logging.LogRecord` carries besides its own payload. Everything
#: else a caller attached — ``extra={"run_id": ...}``, the request id the
#: application's record factory adds — is an attribute, and so is a field whose
#: name makes its value a secret.
_RECORD_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

#: Python's levels, and the OpenTelemetry severity each one is. Everything below
#: ``INFO`` is ``DEBUG``: the exporter has one debug bucket, and a finer one would
#: be a distinction no query makes.
_SEVERITIES: Final[tuple[tuple[int, SeverityNumber, str], ...]] = (
    (logging.CRITICAL, SeverityNumber.FATAL, "FATAL"),
    (logging.ERROR, SeverityNumber.ERROR, "ERROR"),
    (logging.WARNING, SeverityNumber.WARN, "WARN"),
    (logging.INFO, SeverityNumber.INFO, "INFO"),
)

type UsageBuckets = Callable[[Mapping[str, Any]], Mapping[str, int]]


def _declared_buckets(usage: Mapping[str, Any]) -> Mapping[str, int]:
    """Read usage keys that already name their bucket.

    The default when the application supplies no dialect mapper: nothing is
    guessed, and a counter under a name no bucket owns contributes nothing
    rather than an arbitrary attribute.
    """
    return {
        key: value
        for key, value in usage.items()
        if key in _USAGE_ATTRIBUTES and isinstance(value, int) and not isinstance(value, bool)
    }


def _json_attribute(value: Any) -> str:
    """Mask one payload, then serialize it as the JSON string an attribute holds.

    Masking happens here, where the value is shaped, so a private value never
    crosses the network to be redacted ([ADR 0039][adr]). A value that cannot be
    serialized becomes a placeholder: an export that fails must not fail the
    product, and must not carry the original either.
    """
    try:
        text = json.dumps(mask_payload(value), default=str, ensure_ascii=False)
    except Exception:
        logger.debug("Unserializable observation payload (non-fatal)", exc_info=True)
        return f"[unserializable payload of type {type(value).__name__}]"
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_ATTRIBUTE_BYTES:
        return text
    marker = "... [truncated]"
    return encoded[: _MAX_ATTRIBUTE_BYTES - len(marker)].decode("utf-8", "ignore") + marker


def _bounded_text(value: object) -> str:
    """Bound and mask one free-text attribute (an error message, a level)."""
    masked = mask_payload(str(value))
    return masked if isinstance(masked, str) else str(masked)


def _reported_cost(cost_details: Mapping[str, Any]) -> float | None:
    """The one charge a provider reported, or None when it reported none.

    DlightRAG never computes cost; it only forwards what a provider stated
    (OpenRouter's ``usage.include``). A ``total`` is that charge; without one, the
    parts are summed. A price table over the traces is the operator's, and a
    reported charge outranks it.
    """
    total = cost_details.get("total")
    if isinstance(total, int | float) and not isinstance(total, bool):
        return float(total)
    parts = [
        float(value)
        for value in cost_details.values()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ]
    return sum(parts) if parts else None


class _Observation:
    """One open span, and the neutral updates a call site makes to it.

    ``update`` takes the payloads a reviewer reads (``input``, ``output``,
    ``metadata``), the provider's counters (``usage_details``, ``cost_details``),
    the outcome (``level``, ``status_message``), the typed failure a call site
    already knows (``error_kind``), the exception class that caused it
    (``error_type``), the model a server reported (``response_model``), and
    ``completion_start_time``. A key nobody named is ignored rather than guessed
    at.
    """

    def __init__(self, span: Span | None, *, usage_buckets: UsageBuckets) -> None:
        self._span = span
        self._usage_buckets = usage_buckets
        self.level_set = False
        #: Whether a call site already named the failure's type. A span does not
        #: expose its own attributes through the API, and a classification is
        #: better evidence than the exception class that reached this seam.
        self._error_type_set = False

    def update(self, **kwargs: Any) -> None:
        try:
            self._apply(dict(kwargs))
        except Exception:
            # An update is telemetry, never a product operation: a span that
            # cannot take one loses an attribute, and the caller never hears.
            logger.debug("OTLP observation update failed (non-fatal)", exc_info=True)

    def _apply(self, kwargs: dict[str, Any]) -> None:
        span = self._span
        if span is None:
            return
        level = kwargs.pop("level", None)
        status_message = kwargs.pop("status_message", None)
        error_kind = kwargs.pop("error_kind", None)
        error_type = kwargs.pop("error_type", None)
        if error_kind:
            # The kind the application already classified. The message stays too,
            # so an alert can prefer the attribute and still group old spans. It is
            # also the best ``error.type`` available: a low-cardinality identifier
            # of what went wrong, which is what the convention asks that key to be.
            text = _bounded_text(error_kind)
            span.set_attribute(DLIGHTRAG_ERROR_KIND, text)
            span.set_attribute(ERROR_TYPE, text)
            self._error_type_set = True
        elif error_type:
            # The exception a call site caught and turned into a status message.
            # Its class name is the convention's "canonical name of the exception
            # that occurred" — the seam never invents one from a message.
            span.set_attribute(ERROR_TYPE, _bounded_text(error_type))
            self._error_type_set = True
        if level is not None:
            # A caller that names the level knows the outcome better than the
            # generic "an exception left the body" mapping.
            self.level_set = True
            self._set_level(str(level), status_message)
        elif status_message is not None:
            span.set_attribute(DLIGHTRAG_STATUS_MESSAGE, _bounded_text(status_message))
        response_model = kwargs.pop("response_model", None)
        if response_model:
            # "Which model answered", which is a different fact from the alias that
            # was asked for and is what the convention's name is for.
            span.set_attribute(GEN_AI_RESPONSE_MODEL, _bounded_text(response_model))
        usage = kwargs.pop("usage_details", None)
        if usage:
            self._set_usage(usage)
        cost = kwargs.pop("cost_details", None)
        if cost:
            reported = _reported_cost(cost)
            if reported is not None:
                span.set_attribute(GEN_AI_USAGE_COST, reported)
        for key, attribute in (
            ("input", DLIGHTRAG_INPUT),
            ("output", DLIGHTRAG_OUTPUT),
            ("metadata", DLIGHTRAG_METADATA),
        ):
            value = kwargs.pop(key, None)
            if value is not None:
                span.set_attribute(attribute, _json_attribute(value))
        completion_start = kwargs.pop("completion_start_time", None)
        if completion_start is not None:
            span.set_attribute(DLIGHTRAG_COMPLETION_START, str(completion_start))
        if kwargs:
            logger.debug("Ignoring unknown observation update keys: %s", sorted(kwargs))

    def _set_level(self, level: str, status_message: Any) -> None:
        span = self._span
        if span is None:
            return
        normalized = level.strip().upper()
        if normalized == "ERROR":
            description = None if status_message is None else _bounded_text(status_message)
            span.set_status(Status(StatusCode.ERROR, description=description))
            return
        # OpenTelemetry names only ERROR; any other level the caller reports stays
        # readable as an attribute, and the span is not re-labelled a failure.
        if normalized not in {"DEFAULT", "UNSET"}:
            span.set_attribute(_LEVEL_ATTRIBUTE, normalized)
        if status_message is not None:
            span.set_attribute(DLIGHTRAG_STATUS_MESSAGE, _bounded_text(status_message))

    def _set_usage(self, usage: Mapping[str, Any]) -> None:
        span = self._span
        if span is None:
            return
        buckets = self._usage_buckets(usage)
        for bucket, attribute in _USAGE_ATTRIBUTES.items():
            value = buckets.get(bucket)
            if isinstance(value, int | float) and not isinstance(value, bool):
                span.set_attribute(attribute, value)

    def fail(self, exc: BaseException) -> None:
        """Report an exception that escaped the observation's body.

        The span's status carries the message, and ``error.type`` carries the
        exception's own class name: the convention asks for "the canonical name of
        [the] exception that occurred", and nothing else on this path knows a
        better identifier. A call site that named a better one — a classified
        ``error_kind``, or the ``error_type`` of the exception it caught — has
        already written it, and this leaves it alone.
        """
        span = self._span
        if span is None:
            return
        try:
            span.set_status(Status(StatusCode.ERROR, description=_bounded_text(exc)))
            if not self._error_type_set:
                span.set_attribute(ERROR_TYPE, type(exc).__name__)
        except Exception:
            logger.debug("OTLP failed to report an observation error (non-fatal)", exc_info=True)


class _Outage:
    """One report per outage, and nothing between its edges.

    A failing export is counted, not logged: the first failure after a healthy
    period is one ``WARNING``, the first success after a failing period is one
    ``INFO``, and a sustained outage produces no third line. The channel is named
    so a stack that answers one signal and refuses another says which one.
    """

    def __init__(self, channel: str) -> None:
        self._channel = channel
        self._failing = False
        self._failures = 0

    def record(self, *, succeeded: bool) -> None:
        if not succeeded:
            self._failures += 1
            if not self._failing:
                self._failing = True
                logger.warning(
                    "OTLP %s export is failing; telemetry is dropped until the collector answers",
                    self._channel,
                )
            return
        if self._failing:
            self._failing = False
            logger.info(
                "OTLP %s export recovered after %d failed batch(es)",
                self._channel,
                self._failures,
            )
            self._failures = 0


class _GuardedExporter(SpanExporter):
    """The span exporter that talks to the network, behind the breaker."""

    def __init__(self, exporter: SpanExporter) -> None:
        self._exporter = exporter
        self._outage = _Outage("trace")

    def export(self, spans: Any) -> SpanExportResult:
        try:
            result = self._exporter.export(spans)
        except Exception:
            # The span processor must never see the exception: it would log it,
            # and an outage would then cost one log line per batch.
            logger.debug("OTLP export raised (non-fatal)", exc_info=True)
            result = SpanExportResult.FAILURE
        self._outage.record(succeeded=result is not SpanExportResult.FAILURE)
        return result

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._exporter.force_flush(timeout_millis)


class _GuardedLogExporter(LogRecordExporter):
    """The log exporter that talks to the network, behind the same breaker."""

    def __init__(self, exporter: LogRecordExporter) -> None:
        self._exporter = exporter
        self._outage = _Outage("log")

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        try:
            result = self._exporter.export(batch)
        except Exception:
            # A batch processor that saw this would report it once per batch, and
            # a simple one would let it reach the logging call.
            logger.debug("OTLP log export raised (non-fatal)", exc_info=True)
            result = LogRecordExportResult.FAILURE
        self._outage.record(succeeded=result is not LogRecordExportResult.FAILURE)
        return result

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._exporter.force_flush(timeout_millis)


@dataclass(frozen=True)
class _Installed:
    """The provider a process started, and what its handles resolve."""

    tracer: trace.Tracer
    provider: TracerProvider
    usage_buckets: UsageBuckets


@dataclass(frozen=True)
class _LogExport:
    """The log pipeline a process attached: what its handler writes through."""

    provider: LoggerProvider
    handler: logging.Handler


def _severity(level: int) -> tuple[SeverityNumber, str]:
    for threshold, number, text in _SEVERITIES:
        if level >= threshold:
            return number, text
    return SeverityNumber.DEBUG, "DEBUG"


def _log_attributes(record: logging.LogRecord) -> dict[str, Any]:
    """Every field the record carries besides its message, each one masked.

    The classifier reads the field's own name, so a caller that logs
    ``extra={"api_key": ...}`` has it replaced here rather than in the collector,
    and a caller that logs a home directory under a path-shaped name — the search
    toolchain reports where its binaries live — has the path masked for the same
    reason and by the same rule. The application's ``request_id`` record factory
    becomes an attribute without the exporter knowing it exists. An exception's
    type, message, and stack trace ride along the same way, as text the classifier
    bounds.
    """
    extras = {key: value for key, value in vars(record).items() if key not in _RECORD_FIELDS}
    attributes = dict(mask_payload(extras))
    if record.exc_info and record.exc_info[0] is not None:
        exception_type, exception_value, exception_tb = record.exc_info
        attributes["exception.type"] = exception_type.__name__
        attributes["exception.message"] = mask_payload(str(exception_value))
        attributes["exception.stacktrace"] = mask_payload(
            "".join(traceback.format_exception(exception_type, exception_value, exception_tb))
        )
    # Where the record was emitted, which is what a reader follows to the code.
    # The stable names are ``code.file.path`` / ``code.line.number``; the older
    # ``code.filepath`` / ``code.lineno`` are deprecated in the registry, and the
    # function is qualified the way the runtime names it.
    file_path = _masked_code_path(record.pathname)
    if file_path is not None:
        attributes[_CODE_FILE_PATH] = file_path
    attributes[_CODE_FUNCTION_NAME] = _qualified_function_name(record)
    attributes[_CODE_LINE_NUMBER] = record.lineno
    return attributes


def _masked_code_path(pathname: object) -> str | None:
    """Render one record's source file, masked when it names a home directory.

    A path is what a reader follows to the code, so it is kept wherever it is
    safe to keep; one that carries the operator's account name is classified
    private and rendered by the classifier like any other private value, rather
    than exported verbatim because the exporter happened to build the key itself.
    The classifier is asked for that answer rather than recomposing its rule from
    the two predicates that make it: the exporter builds this key itself, and a
    second copy of "a path-shaped key whose value names a home directory" is a
    second rule to keep in step.
    """
    if not isinstance(pathname, str) or not pathname:
        return None
    if classify(_CODE_FILE_PATH, pathname) == "private":
        return private_path_placeholder(pathname)
    return pathname


def _qualified_function_name(record: logging.LogRecord) -> str:
    """The record's function, qualified the way the convention asks for.

    The registry wants "the method or function fully-qualified name without
    arguments ... the natural representation of the language runtime". Python's
    natural representation is ``module.qualname``; ``funcName`` alone is the bare
    name and collides across modules.
    """
    qualified = record.funcName
    module = record.module or ""
    if module and not qualified.startswith(f"{module}."):
        return f"{module}.{qualified}"
    return qualified


class _LogHandler(logging.Handler):
    """One log record per logging call, shaped and masked, never sent from here.

    The handler does no I/O and takes no lock the product holds: it renders the
    record the current span belongs to — so ClickHouse can join a log line to the
    trace that logged it — masks it through the classifier, and hands it to a
    provider whose processor owns a bounded queue. A record this cannot shape is
    dropped, because a logging call is not the place to learn that telemetry
    broke and a record nobody could mask must not be exported unmasked.
    """

    def __init__(self, provider: LoggerProvider) -> None:
        super().__init__(level=logging.NOTSET)
        self._provider = provider

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith(_EXPORT_LOGGER_PREFIX):
            # The export does not export its own voice. A breaker that logs a line
            # per failed batch — and it does, at DEBUG, with the traceback — would
            # otherwise queue the very records that just failed to leave, so an
            # outage would feed itself and a full queue would stay full.
            return
        try:
            severity_number, severity_text = _severity(record.levelno)
            self._provider.get_logger(record.name).emit(
                timestamp=int(record.created * 1e9),
                # The active span, whatever opened it: the exporter's provider is
                # the process's own, so an observation in scope is the parent a
                # reader expects and an open trace id is what ClickStack joins on.
                context=get_current(),
                severity_number=severity_number,
                severity_text=severity_text,
                # A message is text: `getMessage` renders its arguments, and the
                # classifier bounds what it renders.
                body=mask_payload(record.getMessage()),
                attributes=_log_attributes(record),
            )
        except Exception:
            return


def _resource(*, service_name: str, environment: str | None, release: str | None) -> Resource:
    """The resource identity every signal this process sends carries."""
    return Resource.create(
        {
            "service.name": service_name,
            **({"service.version": release} if release else {}),
            **({"deployment.environment.name": environment} if environment else {}),
        }
    )


def _quiet_sdk_loggers() -> None:
    """Raise the SDK's own export loggers out of the way.

    They log once per failed batch. The breaker is the one voice an outage gets,
    so the SDK's is silenced rather than filtered record by record.
    """
    logging.getLogger("opentelemetry").setLevel(logging.CRITICAL)


def _log_record_processor(
    guarded: LogRecordExporter, *, batch: bool, queue_size: int
) -> LogRecordProcessor:
    """The processor that owns the queue: bounded and off the logging call.

    A batch processor is the deployment's, and its queue drops the oldest record
    when it is full rather than making a logging call wait. A test's own exporter
    gets a simple processor instead, so a record is observable the moment it is
    logged rather than five seconds later.
    """
    if not batch:
        return SimpleLogRecordProcessor(guarded)
    return BatchLogRecordProcessor(
        guarded, max_queue_size=queue_size, max_export_batch_size=512, schedule_delay_millis=5000
    )


_installed: _Installed | None = None
_log_export: _LogExport | None = None
_session: ContextVar[str | None] = ContextVar("dlightrag_session_id", default=None)
_user: ContextVar[str | None] = ContextVar("dlightrag_user_id", default=None)


class OtlpTelemetry:
    """Render DlightRAG's observations as OpenTelemetry spans over OTLP.

    A handle rather than process state: it resolves whatever
    :func:`start_telemetry` installed, so the composition root can build the
    components that hold it before startup runs, and a process that never
    installed an export — no endpoint, or an initialization that failed — is a
    no-op telemetry that records nothing and raises nothing.

    ``span_types`` is the application's registry, keyed by its own span-name
    literals: this module states no name, so it reads the mapping as data.
    """

    def __init__(
        self,
        *,
        span_types: Mapping[Any, str],
        usage_buckets: UsageBuckets | None = None,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self._span_types = span_types
        self._usage_buckets = usage_buckets
        self._tracer = tracer

    def _export(self) -> tuple[trace.Tracer, UsageBuckets] | None:
        if self._tracer is not None:
            return self._tracer, self._usage_buckets or _declared_buckets
        installed = _installed
        if installed is None:
            return None
        return installed.tracer, self._usage_buckets or installed.usage_buckets

    @contextmanager
    def trace(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> Iterator[None]:
        """Attribute the enclosing trace; every observation in scope inherits it."""
        session_token = _session.set(session_id) if session_id else None
        user_token = _user.set(user_id) if user_id else None
        try:
            yield
        finally:
            if session_token is not None:
                _session.reset(session_token)
            if user_token is not None:
                _user.reset(user_token)

    @asynccontextmanager
    async def observe(
        self,
        name: str,
        *,
        input: Any | None = None,
        metadata: Any | None = None,
        model: str | None = None,
        provider: str | None = None,
        tool: str | None = None,
        model_parameters: dict[str, Any] | None = None,
    ) -> AsyncIterator[_Observation]:
        """Mark a DlightRAG operation as one span, nested in the open one.

        ``provider`` and ``tool`` are the two Required facts a call site knows at
        span creation and the registry cannot infer: who the model call is against
        (``gen_ai.provider.name``) and which tool is running (``gen_ai.tool.name``).
        A call site that does not know one leaves it out — the seam states what it
        was told, never a guess. ``provider`` arrives in the app's own vocabulary
        and is rendered onto the registry's enum here, in :data:`_GEN_AI_PROVIDERS`.
        """
        export = self._export()
        if export is None:
            yield _Observation(None, usage_buckets=_declared_buckets)
            return
        tracer, usage_buckets = export
        kind = self._span_types.get(name, _GENERIC_KIND)
        observation = _Observation(None, usage_buckets=usage_buckets)
        attributes = self._attributes(
            name,
            kind=kind,
            input=input,
            metadata=metadata,
            model=model,
            provider=provider,
            tool=tool,
            model_parameters=model_parameters,
        )
        with tracer.start_as_current_span(
            name,
            kind=_SPAN_KINDS.get(kind, _DEFAULT_SPAN_KIND),
            attributes=attributes,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            observation = _Observation(span, usage_buckets=usage_buckets)
            try:
                yield observation
            except asyncio.CancelledError, GeneratorExit:
                # A cancelled operation, or a generator closed rather than
                # exited, is an outcome a caller asked for, not a failure this
                # seam invents.
                raise
            except BaseException as exc:
                if not observation.level_set:
                    observation.fail(exc)
                raise

    def _attributes(
        self,
        name: str,
        *,
        kind: str,
        input: Any | None,
        metadata: Any | None,
        model: str | None,
        provider: str | None,
        tool: str | None,
        model_parameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        attributes: dict[str, Any] = {OBSERVATION_KIND: kind}
        operation = _OPERATIONS.get(kind)
        if operation is not None:
            attributes[GEN_AI_OPERATION] = operation
        if provider:
            attributes[GEN_AI_PROVIDER] = _bounded_text(_rendered_provider(provider))
        if model:
            attributes[GEN_AI_REQUEST_MODEL] = str(model)
        if tool:
            attributes[GEN_AI_TOOL_NAME] = _bounded_text(tool)
        if model_parameters:
            attributes[DLIGHTRAG_MODEL_PARAMETERS] = _json_attribute(model_parameters)
        if input is not None:
            attributes[DLIGHTRAG_INPUT] = _json_attribute(input)
        if metadata is not None:
            attributes[DLIGHTRAG_METADATA] = _json_attribute(metadata)
        session = _session.get()
        if session:
            attributes[SESSION_ID] = str(session)
        user = _user.get()
        if user:
            attributes[USER_ID] = str(user)
        return attributes


def start_telemetry(
    *,
    span_types: Mapping[Any, str],
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    service_name: str = "dlightrag",
    environment: str | None = None,
    release: str | None = None,
    sample_rate: float = 1.0,
    timeout: int | None = None,
    usage_buckets: UsageBuckets | None = None,
    span_exporter: SpanExporter | None = None,
    queue_size: int = 2048,
) -> OtlpTelemetry:
    """Start the process's OTLP export, or install nothing and say so once.

    Without an endpoint there is nothing to send to, and that is an explicit
    no-op rather than an error: monitoring is always on where a deployment
    configures a destination, and a development process that has none must not
    retry, warn, or flood. A construction that raises falls back the same way,
    so a broken destination costs telemetry and never a boot.

    ``span_exporter`` replaces the network exporter and switches to a simple
    processor: it is the seam a test reads spans through, not a deployment knob.
    """
    if endpoint is None and span_exporter is None:
        logger.info("OTLP tracing off (no endpoint configured)")
        _install(None)
        return OtlpTelemetry(span_types=span_types, usage_buckets=usage_buckets)

    provider: TracerProvider | None = None
    buckets = usage_buckets or _declared_buckets
    signal_endpoint = None if endpoint is None else _signal_endpoint(endpoint, _TRACES_SIGNAL)
    try:
        resource = _resource(service_name=service_name, environment=environment, release=release)
        sampler = TraceIdRatioBased(sample_rate) if sample_rate < 1 else None
        provider = TracerProvider(resource=resource, sampler=sampler)
        exporter: SpanExporter = (
            span_exporter
            if span_exporter is not None
            else OTLPSpanExporter(
                endpoint=signal_endpoint,
                headers=dict(headers or {}),
                timeout=timeout,
            )
        )
        guarded = _GuardedExporter(exporter)
        if span_exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(guarded))
        else:
            provider.add_span_processor(
                BatchSpanProcessor(guarded, max_queue_size=queue_size, max_export_batch_size=512)
            )
        tracer = provider.get_tracer("dlightrag")
    except Exception:
        logger.warning(
            "OTLP tracing is configured but initialization failed. Falling back to no tracing.",
            exc_info=True,
        )
        if provider is not None:
            _stop_provider(provider)
        _install(None)
        return OtlpTelemetry(span_types=span_types, usage_buckets=usage_buckets)

    # The SDK logs its own export failures, once per batch. The breaker above is
    # the one voice an outage gets, so the SDK's is raised out of the way.
    _quiet_sdk_loggers()
    _install(_Installed(tracer=tracer, provider=provider, usage_buckets=buckets))
    logger.info("OTLP tracing enabled → %s", signal_endpoint)
    return OtlpTelemetry(span_types=span_types, usage_buckets=buckets, tracer=tracer)


def start_log_export(
    *,
    endpoint: str | None = None,
    headers: Mapping[str, str] | None = None,
    service_name: str = "dlightrag",
    environment: str | None = None,
    release: str | None = None,
    timeout: int | None = None,
    log_exporter: LogRecordExporter | None = None,
    queue_size: int = 2048,
) -> None:
    """Attach the process's log export beside stdout, or attach nothing.

    One additional handler on the root logger: the records the application
    already prints are the records this sends, masked, with the trace id of the
    span that logged them, and a logging call still costs only a queue append.
    There is no second switch to turn it on — a deployment that configured an
    endpoint exports both signals, and this is the caller's to skip only where it
    deliberately wants stdout alone.

    Its own file and line are attributes, and the SDK's record factory already
    rides along, so the shape a query reads is the shape a reader expects.

    ``log_exporter`` replaces the network leg and switches to a simple processor:
    the seam a test reads records through, not a deployment knob.
    """
    global _log_export
    if _log_export is not None:
        return
    if endpoint is None and log_exporter is None:
        # The same fact `start_telemetry` states one line above; a second line for
        # it would be noise, and a process with no destination has nothing to say.
        return
    signal_endpoint = None if endpoint is None else _signal_endpoint(endpoint, _LOGS_SIGNAL)
    try:
        provider = LoggerProvider(
            resource=_resource(service_name=service_name, environment=environment, release=release)
        )
        exporter: LogRecordExporter = (
            log_exporter
            if log_exporter is not None
            else OTLPLogExporter(
                endpoint=signal_endpoint,
                headers=dict(headers or {}),
                timeout=timeout,
            )
        )
        provider.add_log_record_processor(
            _log_record_processor(
                _GuardedLogExporter(exporter), batch=log_exporter is None, queue_size=queue_size
            )
        )
        handler = _LogHandler(provider)
    except Exception:
        logger.warning(
            "OTLP logging is configured but initialization failed. Falling back to stdout only.",
            exc_info=True,
        )
        return
    _attach_log_handler(handler)
    _quiet_sdk_loggers()
    _log_export = _LogExport(provider=provider, handler=handler)
    if signal_endpoint is None:
        logger.info("OTLP log export enabled")
    else:
        logger.info("OTLP log export enabled → %s", signal_endpoint)


def shutdown_log_export(*, timeout_millis: int = 5000) -> None:
    """Detach the handler, flush what is pending under a timeout, then stop.

    Detaching first is what makes the shutdown quiet: the flush below can fail,
    and a handler still attached would turn its own report into another record to
    export. The timeout is the point, as it is for spans — a shutdown must not
    wait on a collector that is not answering.
    """
    global _log_export
    attached = _log_export
    _log_export = None
    if attached is None:
        return
    try:
        logging.getLogger().removeHandler(attached.handler)
        attached.handler.close()
    except Exception:
        logger.debug("OTLP log handler could not be detached (non-fatal)", exc_info=True)
    try:
        attached.provider.force_flush(timeout_millis)
    except Exception:
        logger.debug("OTLP log flush failed (non-fatal)", exc_info=True)
    try:
        attached.provider.shutdown()
    except Exception:
        logger.debug("OTLP log provider shutdown failed (non-fatal)", exc_info=True)


def _attach_log_handler(handler: logging.Handler) -> None:
    """Put the export beside the console, never in place of it.

    A handler on the root logger is what makes every record the process logs
    exportable, and it is also the one way this can change stdout:
    ``logging.basicConfig`` installs its console handler only while the root
    logger has none. Every entry point configures logging before the application
    starts, so the root is not bare here and this adds a second handler; where it
    is bare — a test, a one-off script — the console handler is installed first,
    which is what the app would have done.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig()
    root.addHandler(handler)


def shutdown_telemetry(*, timeout_millis: int = 5000) -> None:
    """Flush what is pending under a timeout, then stop the provider.

    The timeout is the point: a shutdown must not wait on a collector that is
    not answering, so a flush that times out costs the spans in flight and
    nothing else.
    """
    installed = _installed
    _install(None)
    if installed is None:
        return
    try:
        installed.provider.force_flush(timeout_millis)
    except Exception:
        logger.debug("OTLP flush failed (non-fatal)", exc_info=True)
    _stop_provider(installed.provider)


def _install(installed: _Installed | None) -> None:
    global _installed
    _installed = installed


def _stop_provider(provider: TracerProvider) -> None:
    try:
        provider.shutdown()
    except Exception:
        logger.debug("OTLP tracer provider shutdown failed (non-fatal)", exc_info=True)


def _signal_endpoint(endpoint: str, signal: str) -> str:
    """Return the signal URL an OTLP/HTTP exporter expects.

    A base URL gains the signal's path; one that already names it — the form the
    SDK's own environment variables take — is left alone.
    """
    path = f"/{signal}"
    base = endpoint.rstrip("/")
    return base if base.endswith(path) else f"{base}{path}"


__all__ = [
    "DLIGHTRAG_ERROR_KIND",
    "DLIGHTRAG_STATUS_MESSAGE",
    "DLIGHTRAG_TOTAL",
    "ERROR_TYPE",
    "GEN_AI_OPERATION",
    "GEN_AI_PROVIDER",
    "GEN_AI_REQUEST_MODEL",
    "GEN_AI_RESPONSE_MODEL",
    "GEN_AI_TOOL_NAME",
    "GEN_AI_USAGE_CACHE_READ",
    "GEN_AI_USAGE_COST",
    "GEN_AI_USAGE_INPUT",
    "GEN_AI_USAGE_OUTPUT",
    "OBSERVATION_KIND",
    "SESSION_ID",
    "USER_ID",
    "OtlpTelemetry",
    "UsageBuckets",
    "shutdown_log_export",
    "shutdown_telemetry",
    "start_log_export",
    "start_telemetry",
]
