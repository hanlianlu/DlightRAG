# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What the OTLP exporter renders, read off a recording span exporter.

``test_observability.py`` pins the vocabulary and the call sites; these tests read
the spans that leave the process, so a trace's tree, its attribution, its usage
buckets, and the masking in front of them are observed where ClickHouse would see
them. Only the network leg is the test's: the exporter is the product's.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any, get_args

import pytest
from dlightrag_observability import otlp as otlp_module
from dlightrag_observability.otlp import OtlpTelemetry, shutdown_telemetry, start_telemetry
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openai.types import CompletionUsage
from openai.types.completion_usage import PromptTokensDetails
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from dlightrag.engine.ai.contracts import ChatProvider
from dlightrag.engine.ai.providers.base import provider_usage_buckets, usage_to_dict
from dlightrag.engine.ai.settings import EmbeddingSettings
from dlightrag.engine.ai.telemetry import SPAN_TYPES
from tests.unit.conftest import OtlpExport


def test_an_observation_opened_in_a_request_is_a_root_and_no_framework_span_is_exported(
    otlp_export: OtlpExport,
) -> None:
    """FastAPI opens a span around every request; it must neither parent nor join a trace."""
    app = FastAPI()

    @app.get("/ask")
    async def ask() -> dict[str, str]:
        async with otlp_export.telemetry.observe("run-answer"):
            return {"answer": "ok"}

    with TestClient(app) as client:
        assert client.get("/ask").status_code == 200

    assert [(span.name, span.parent) for span in otlp_export.spans()] == [("run-answer", None)]
    # The exporter runs on a provider of its own, so a library's span in this
    # process is neither exported nor an ancestor of an observation.
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


async def test_the_exported_tree_carries_its_attribution_model_and_identity(
    otlp_export: OtlpExport,
) -> None:
    telemetry = otlp_export.telemetry

    with telemetry.trace(session_id="sess-1", user_id="owner-1"):
        async with telemetry.observe("run-answer"):
            async with telemetry.observe(
                "generate-completion", model="deepseek-flash", model_parameters={"temperature": 0.2}
            ):
                pass

    spans = {span.name: span for span in otlp_export.spans()}
    root, completion = spans["run-answer"], spans["generate-completion"]
    assert root.parent is None
    assert root.context is not None
    assert completion.parent is not None
    assert completion.parent.span_id == root.context.span_id
    for span in (root, completion):
        attributes = otlp_export.attributes(span)
        assert attributes["session.id"] == "sess-1"
        assert attributes["user.id"] == "owner-1"
        assert span.resource.attributes["service.name"] == "dlightrag-api"
        assert span.resource.attributes["service.version"] == "9.9.9"
        assert span.resource.attributes["deployment.environment.name"] == "test"
    # Orchestration this process performs is INTERNAL; a call to a model that runs
    # in another process is CLIENT, which is what the convention asks for.
    assert root.kind is trace.SpanKind.INTERNAL
    assert completion.kind is trace.SpanKind.CLIENT
    completion_attributes = otlp_export.attributes(completion)
    assert completion_attributes["gen_ai.request.model"] == "deepseek-flash"
    assert otlp_export.payload(completion, "dlightrag.model_parameters") == {"temperature": 0.2}


@pytest.mark.parametrize(
    ("name", "kind", "operation"),
    [
        ("run-answer", "agent", None),
        ("run-retrieval", "chain", None),
        ("generate-answer", "chain", None),
        ("generate-agent-turn", "generation", "chat"),
        ("compact-session", "generation", "chat"),
        ("generate-completion", "generation", "chat"),
        ("execute-agent-tool", "tool", "execute_tool"),
        ("embed-text", "embedding", "embeddings"),
        ("rerank-passages", "span", None),
        ("call-rerank-model", "span", None),
        ("retrieve-context", "retriever", None),
        ("plan-retrieval", "chain", None),
        ("highlight-sources", "chain", None),
        ("ingest-documents", "chain", None),
    ],
)
async def test_every_span_carries_its_kind_and_the_genai_operation_it_is(
    otlp_export: OtlpExport, name: str, kind: str, operation: str | None
) -> None:
    """The kind is the app's aggregation key; the operation is the convention's."""
    async with otlp_export.telemetry.observe(name):  # type: ignore[arg-type]
        pass

    (span,) = otlp_export.spans()
    attributes = otlp_export.attributes(span)
    assert attributes["dlightrag.observation.kind"] == kind
    assert attributes.get("gen_ai.operation.name") == operation


async def test_a_payload_the_caller_hands_the_seam_is_masked_before_it_leaves(
    otlp_export: OtlpExport,
) -> None:
    """A secret or inline media never crosses the network to be redacted."""
    async with otlp_export.telemetry.observe(
        "generate-completion",
        input={
            "api_key": "sk-live",
            "messages": [{"type": "image_url", "image_url": "data:image/png;base64,abc"}],
        },
        metadata={"run_id": "r-1", "blob": b"\x00\x01"},
    ) as observation:
        observation.update(output={"answer": "ok"})

    (span,) = otlp_export.spans()
    assert otlp_export.payload(span, "dlightrag.input") == {
        "api_key": "[redacted]",
        "messages": [{"type": "image_url", "image_url": "[image omitted]"}],
    }
    assert otlp_export.payload(span, "dlightrag.metadata") == {
        "run_id": "r-1",
        "blob": "[bytes omitted: 2]",
    }
    assert otlp_export.payload(span, "dlightrag.output") == {"answer": "ok"}


async def test_the_cost_a_provider_reported_rides_on_the_span(otlp_export: OtlpExport) -> None:
    """The app forwards a charge; it never computes one."""
    async with otlp_export.telemetry.observe("generate-completion") as observation:
        observation.update(cost_details={"total": 0.001})

    (span,) = otlp_export.spans()
    assert otlp_export.attributes(span)["gen_ai.usage.cost"] == 0.001


async def test_an_exception_that_escapes_names_the_failure(
    otlp_export: OtlpExport,
) -> None:
    with pytest.raises(RuntimeError):
        async with otlp_export.telemetry.observe("run-answer"):
            raise RuntimeError("the provider said no")

    (span,) = otlp_export.spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == "the provider said no"


async def test_a_reported_outcome_is_not_relabelled_by_an_exception(
    otlp_export: OtlpExport,
) -> None:
    """A Run that named its own outcome keeps it, whatever leaves the body."""
    with pytest.raises(RuntimeError):
        async with otlp_export.telemetry.observe("run-answer") as observation:
            observation.update(level="DEFAULT", output={"outcome": "cancelled"})
            raise RuntimeError("cancelled by the owner")

    (span,) = otlp_export.spans()
    assert span.status.status_code is StatusCode.UNSET


async def test_a_classified_failure_carries_its_kind_beside_the_message(
    otlp_export: OtlpExport,
) -> None:
    """An alert groups on the attribute; the message stays for spans that predate it."""
    async with otlp_export.telemetry.observe("run-answer") as observation:
        observation.update(
            error_kind="dependency_unavailable",
            status_message="The model provider stayed unavailable through 10 retries",
        )

    (span,) = otlp_export.spans()
    attributes = otlp_export.attributes(span)
    assert attributes["dlightrag.error.kind"] == "dependency_unavailable"
    assert attributes["dlightrag.status_message"] == (
        "The model provider stayed unavailable through 10 retries"
    )


async def test_the_required_genai_attributes_the_call_site_knows_are_emitted(
    otlp_export: OtlpExport,
) -> None:
    """`gen_ai.provider.name` and `gen_ai.tool.name` are Required where they apply.

    Both are facts only a call site has, so the seam takes them as arguments and
    states them; a call site that does not know one leaves it out rather than
    letting the exporter guess.
    """
    async with otlp_export.telemetry.observe(
        "execute-agent-tool", model="deepseek-flash", provider="deepseek", tool="web_search"
    ):
        pass

    (span,) = otlp_export.spans()
    attributes = otlp_export.attributes(span)
    assert attributes["gen_ai.provider.name"] == "deepseek"
    assert attributes["gen_ai.tool.name"] == "web_search"
    assert span.kind is trace.SpanKind.INTERNAL


async def test_a_call_site_that_knows_no_provider_emits_none(otlp_export: OtlpExport) -> None:
    """The seam never invents a value: an unknown provider is an absent attribute."""
    async with otlp_export.telemetry.observe("run-answer"):
        pass

    (span,) = otlp_export.spans()
    assert "gen_ai.provider.name" not in otlp_export.attributes(span)


#: The app's providers the GenAI registry's enum does not name. The convention's
#: enum ends "otherwise a custom value MAY be used", and the app's own literal is
#: better knowledge than a registry value naming some other provider.
_UNNAMED_BY_REGISTRY = frozenset({"jina", "openai_compatible", "voyage"})


def _app_provider_literals() -> frozenset[str]:
    """Every provider literal the app's own enums can produce."""
    chat = get_args(ChatProvider.__value__)
    embedding = get_args(EmbeddingSettings.model_fields["provider"].annotation)
    return frozenset(str(value) for value in (*chat, *embedding))


def test_every_provider_the_app_can_produce_is_rendered_for_the_registry() -> None:
    """The exporter's provider table is total over the app's own vocabulary.

    The app's vocabulary is not the registry's — a configuration says `gemini`
    while the registry spells that provider `gcp.gemini` — so the export carries an
    off-enum value until the rendering table translates it, which is what happened
    before this table existed. Totality is checked against the app's enums rather
    than against the table: a provider added to `ChatProvider` or
    `EmbeddingSettings.provider` fails here until someone decides its rendering.
    """
    from opentelemetry.semconv._incubating.attributes import gen_ai_attributes

    registry = {member.value for member in gen_ai_attributes.GenAiProviderNameValues}
    table = otlp_module._GEN_AI_PROVIDERS  # noqa: SLF001
    providers = _app_provider_literals()

    assert set(table) == providers
    # Every app provider either renders into the registry's enum or is one of the
    # few the registry does not name, and those render as themselves: no value is
    # invented for them and none is dropped.
    assert {provider for provider in providers if table[provider] not in registry} == (
        _UNNAMED_BY_REGISTRY
    )
    assert all(table[provider] == provider for provider in _UNNAMED_BY_REGISTRY)
    # The translations that are not the identity, read off the registry's own enum.
    assert table["gemini"] == gen_ai_attributes.GenAiProviderNameValues.GCP_GEMINI.value
    assert (
        table["azure_cohere"] == gen_ai_attributes.GenAiProviderNameValues.AZURE_AI_INFERENCE.value
    )


async def test_the_apps_provider_vocabulary_reaches_the_wire_as_the_registrys(
    otlp_export: OtlpExport,
) -> None:
    """The literal a call site states is rendered, so no off-enum value ships."""
    async with otlp_export.telemetry.observe("generate-completion", provider="gemini"):
        pass
    async with otlp_export.telemetry.observe("embed-text", provider="azure_cohere"):
        pass

    gemini, embedding = otlp_export.spans()
    assert otlp_export.attributes(gemini)["gen_ai.provider.name"] == "gcp.gemini"
    assert otlp_export.attributes(embedding)["gen_ai.provider.name"] == "azure.ai.inference"


async def test_a_provider_the_registry_does_not_name_passes_through_unchanged(
    otlp_export: OtlpExport,
) -> None:
    """The attribute is the instrumentation's best knowledge, so it is never dropped.

    A provider outside both the app's vocabulary and the registry's enum is still
    what the call site said it was talking to, and the convention's enum keeps a
    custom value for exactly that case.
    """
    async with otlp_export.telemetry.observe("generate-completion", provider="acme_llm"):
        pass

    (span,) = otlp_export.spans()
    assert otlp_export.attributes(span)["gen_ai.provider.name"] == "acme_llm"


async def test_the_model_that_answered_is_a_different_attribute_from_the_one_asked_for(
    otlp_export: OtlpExport,
) -> None:
    async with otlp_export.telemetry.observe(
        "generate-completion", model="deepseek-flash"
    ) as observation:
        observation.update(response_model="deepseek-v3.2-exp")

    (span,) = otlp_export.spans()
    attributes = otlp_export.attributes(span)
    assert attributes["gen_ai.request.model"] == "deepseek-flash"
    assert attributes["gen_ai.response.model"] == "deepseek-v3.2-exp"


async def test_an_error_span_names_the_exception_that_caused_it(
    otlp_export: OtlpExport,
) -> None:
    """`error.type` is Conditionally Required "if the operation ended in an error"."""
    with pytest.raises(RuntimeError):
        async with otlp_export.telemetry.observe("generate-completion"):
            raise RuntimeError("the provider said no")

    (span,) = otlp_export.spans()
    assert span.status.status_code is StatusCode.ERROR
    assert otlp_export.attributes(span)["error.type"] == "RuntimeError"


async def test_a_classified_failure_prefers_its_kind_over_the_exception_name(
    otlp_export: OtlpExport,
) -> None:
    """A typed failure the app already classified is the better low-cardinality id."""
    with pytest.raises(RuntimeError):
        async with otlp_export.telemetry.observe("run-answer") as observation:
            observation.update(error_kind="dependency_unavailable")
            raise RuntimeError("the provider said no")

    (span,) = otlp_export.spans()
    attributes = otlp_export.attributes(span)
    assert attributes["dlightrag.error.kind"] == "dependency_unavailable"
    assert attributes["error.type"] == "dependency_unavailable"


async def test_a_call_site_can_name_the_exception_it_caught(otlp_export: OtlpExport) -> None:
    """A call site that turns an exception into a status message still knows its class."""
    async with otlp_export.telemetry.observe("generate-completion") as observation:
        observation.update(
            level="ERROR",
            status_message="the provider said no",
            error_type="BadRequestError",
        )

    (span,) = otlp_export.spans()
    assert otlp_export.attributes(span)["error.type"] == "BadRequestError"


async def test_a_failure_that_names_no_kind_carries_no_kind(otlp_export: OtlpExport) -> None:
    """The attribute is what a call site knows, never a guess at what it meant."""
    with pytest.raises(RuntimeError):
        async with otlp_export.telemetry.observe("run-answer"):
            raise RuntimeError("the provider said no")

    (span,) = otlp_export.spans()
    assert "dlightrag.error.kind" not in otlp_export.attributes(span)


async def test_a_cancellation_is_not_an_error(otlp_export: OtlpExport) -> None:
    with pytest.raises(asyncio.CancelledError):
        async with otlp_export.telemetry.observe("generate-completion"):
            raise asyncio.CancelledError

    (span,) = otlp_export.spans()
    assert span.status.status_code is StatusCode.UNSET


def test_every_convention_key_the_exporter_writes_is_the_registry_s_own() -> None:
    """The names are checked against the installed registry, not against this suite.

    The defect this holds out was self-consistency: the exporter, its tests, and the
    documents all agreed on `gen_ai.usage.input_cached_tokens`, a key that exists in
    no OpenTelemetry revision, and on `code.filepath`/`code.lineno`, which the
    registry had already replaced. Comparing the constants to the registry's own
    strings is the only check that cannot go stale in the same direction.

    The registry publishes its stable names under ``opentelemetry.semconv.attributes``
    and the still-incubating ones — GenAI, and the names it has deprecated — under
    ``_incubating.attributes``. That private path is the only place upstream publishes
    the GenAI names, which is why ``opentelemetry-semantic-conventions`` is a declared
    dependency of ``dlightrag-observability`` rather than something the SDK happens
    to pull in.
    """
    from opentelemetry.semconv._incubating.attributes import code_attributes, gen_ai_attributes
    from opentelemetry.semconv.attributes import error_attributes

    assert otlp_module.GEN_AI_OPERATION == gen_ai_attributes.GEN_AI_OPERATION_NAME
    assert otlp_module.GEN_AI_PROVIDER == gen_ai_attributes.GEN_AI_PROVIDER_NAME
    assert otlp_module.GEN_AI_REQUEST_MODEL == gen_ai_attributes.GEN_AI_REQUEST_MODEL
    assert otlp_module.GEN_AI_RESPONSE_MODEL == gen_ai_attributes.GEN_AI_RESPONSE_MODEL
    assert otlp_module.GEN_AI_TOOL_NAME == gen_ai_attributes.GEN_AI_TOOL_NAME
    assert otlp_module.GEN_AI_USAGE_INPUT == gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS
    assert otlp_module.GEN_AI_USAGE_OUTPUT == gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS
    assert (
        otlp_module.GEN_AI_USAGE_CACHE_READ
        == gen_ai_attributes.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
    )
    assert otlp_module.ERROR_TYPE == error_attributes.ERROR_TYPE

    # The stable code names, and the deprecated ones are absent from what we emit.
    assert otlp_module._CODE_FILE_PATH == code_attributes.CODE_FILE_PATH  # noqa: SLF001
    assert otlp_module._CODE_LINE_NUMBER == code_attributes.CODE_LINE_NUMBER  # noqa: SLF001
    # The deprecated names the registry still carries, against the exporter's whole
    # stated vocabulary rather than against one constant: the assertion this replaces
    # compared the cache-read key to two code-location constants and a literal, which
    # no change to the exporter could ever have failed.
    deprecated = {
        code_attributes.CODE_FILEPATH,
        code_attributes.CODE_LINENO,
        gen_ai_attributes.GEN_AI_SYSTEM,
    }
    stated = {
        value
        for name in otlp_module.__all__
        if isinstance(value := getattr(otlp_module, name), str)
    }
    assert stated & deprecated == set()


async def test_a_sample_rate_of_zero_exports_no_trace(
    start_otlp_export: Callable[..., OtlpExport],
) -> None:
    export = start_otlp_export(sample_rate=0)

    async with export.telemetry.observe("run-answer"):
        pass

    assert export.spans() == ()


# One prompt of 673 tokens whose first 512 a prefix cache served, and a 39-token answer.
# Every wire reports it its own way, and the convention's domain is what the span must
# carry: ``input_tokens`` is the *whole* prompt (the registry says it "SHOULD include all
# types of input tokens, including cached tokens") and ``cache_read.input_tokens`` is the
# 512-token subset of it that the cache served. The fresh count, 161, is a derivation
# (``input - cache_read``) rather than a key on the wire.
_BUCKETS = {"input": 673, "cache_read_input_tokens": 512, "output": 39}
#: What ``input - cache_read`` must come to on every wire: the prompt the cache did
#: not serve, which is the number a price table prices at the uncached rate.
_FRESH = 161

_USAGE: list[Any] = [
    pytest.param(
        usage_to_dict(
            CompletionUsage(
                prompt_tokens=673,
                completion_tokens=39,
                total_tokens=712,
                prompt_tokens_details=PromptTokensDetails(cached_tokens=512),
            )
        ),
        _BUCKETS,
        id="openai-chat",
    ),
    pytest.param(
        usage_to_dict(
            CompletionUsage.model_validate(
                {
                    "prompt_tokens": 673,
                    "completion_tokens": 39,
                    "total_tokens": 712,
                    "prompt_cache_hit_tokens": 512,
                    "prompt_cache_miss_tokens": 161,
                }
            )
        ),
        _BUCKETS,
        id="deepseek",
    ),
    pytest.param(
        usage_to_dict(
            ResponseUsage(
                input_tokens=673,
                output_tokens=39,
                total_tokens=712,
                input_tokens_details=InputTokensDetails(cached_tokens=512, cache_write_tokens=0),
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            )
        ),
        _BUCKETS,
        id="responses",
    ),
    # Anthropic states the cache beside input_tokens and no total; its writes stay in input.
    pytest.param(
        {
            "input_tokens": 100,
            "cache_creation_input_tokens": 61,
            "cache_read_input_tokens": 512,
            "output_tokens": 39,
        },
        _BUCKETS,
        id="anthropic",
    ),
    pytest.param(
        {
            "prompt_tokens": 3911,
            "completion_tokens": 254,
            "total_tokens": 4165,
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 3911,
        },
        {"input": 3911, "output": 254},
        id="a-measured-miss-has-no-cache-key",
    ),
    pytest.param({"tokens_used": 5, "nested": {"a": 1}}, {}, id="unknown-dialect"),
]


@pytest.mark.parametrize(("counters", "buckets"), _USAGE)
async def test_usage_puts_the_cache_inside_the_prompt_the_convention_expects(
    otlp_export: OtlpExport, counters: dict[str, Any], buckets: dict[str, int]
) -> None:
    """``input_tokens`` is the prompt total; the cache read is a documented subset of it.

    This is what makes summing ``gen_ai.usage.input_tokens`` across spans an honest
    prompt total, and it is the opposite of the exclusive count the exporter used to
    send — a domain no convention states, and one that made 95% of measured rows
    report more cached tokens than input tokens.
    """
    async with otlp_export.telemetry.observe("generate-completion", model="m") as observation:
        observation.update(usage_details=counters)

    (span,) = otlp_export.spans()
    usage = otlp_export.usage(span)
    assert usage == buckets
    if not usage:
        return
    cached = usage.get("cache_read_input_tokens", 0)
    assert cached <= usage["input"], "a cache read is a subset of the prompt, never more"
    if cached:
        # The exclusive number is the product's own reading and stays derivable;
        # it is no longer a key the wire redefines.
        assert usage["input"] - cached == _FRESH


class _FlakyExporter(SpanExporter):
    """A destination that can be down, so an outage's one report can be read."""

    def __init__(self) -> None:
        self.failing = True
        self.batches = 0

    def export(self, spans: Any) -> SpanExportResult:
        self.batches += 1
        return SpanExportResult.FAILURE if self.failing else SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        return True


async def test_an_outage_is_reported_once_and_its_recovery_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A destination that stops answering must not cost a log line per batch."""
    exporter = _FlakyExporter()
    start_telemetry(
        span_types=SPAN_TYPES,
        usage_buckets=provider_usage_buckets,
        endpoint="http://127.0.0.1:4318",
        span_exporter=exporter,
    )
    try:
        telemetry = OtlpTelemetry(span_types=SPAN_TYPES, usage_buckets=provider_usage_buckets)
        with caplog.at_level(logging.INFO, logger="dlightrag_observability.otlp"):
            for _ in range(3):
                async with telemetry.observe("run-answer"):
                    pass
            warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
            assert len(warnings) == 1
            exporter.failing = False
            async with telemetry.observe("run-answer"):
                pass
        recovered = [
            record
            for record in caplog.records
            if record.levelno == logging.INFO and "recovered" in record.getMessage()
        ]
        assert len(recovered) == 1
    finally:
        shutdown_telemetry()


async def test_a_failed_initialization_installs_no_op_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A destination that cannot even be constructed costs telemetry, never a boot."""

    class _Unbuildable:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("no such destination")

    monkeypatch.setattr("dlightrag_observability.otlp.OTLPSpanExporter", _Unbuildable)

    telemetry = start_telemetry(
        span_types=SPAN_TYPES,
        usage_buckets=provider_usage_buckets,
        endpoint="http://127.0.0.1:9",
    )
    try:
        # Nothing was installed, so this is a no-op telemetry: the call is legal
        # and the update goes nowhere.
        assert otlp_module._installed is None  # noqa: SLF001 - the process state under test
        async with telemetry.observe("run-answer") as observation:
            observation.update(output={"answer": "ok"})
    finally:
        shutdown_telemetry()


def test_no_endpoint_is_an_explicit_no_op(caplog: pytest.LogCaptureFixture) -> None:
    """A development process without a destination says so once and stays quiet."""
    with caplog.at_level(logging.INFO, logger="dlightrag_observability.otlp"):
        start_telemetry(span_types=SPAN_TYPES, usage_buckets=provider_usage_buckets, endpoint=None)
    try:
        assert otlp_module._installed is None  # noqa: SLF001 - the process state under test
        assert [record.getMessage() for record in caplog.records] == [
            "OTLP tracing off (no endpoint configured)"
        ]
    finally:
        shutdown_telemetry()


async def test_a_second_start_keeps_the_first_provider_instead_of_leaking_it() -> None:
    """One provider per process: a repeat call must not strand the first one's export thread.

    Replacing the provider without stopping it leaves its batch processor running with
    nothing that will ever flush it, so the guard returns a handle on what is installed.
    """
    first, second = InMemorySpanExporter(), InMemorySpanExporter()
    start_telemetry(
        span_types=SPAN_TYPES,
        usage_buckets=provider_usage_buckets,
        endpoint="http://127.0.0.1:4318",
        span_exporter=first,
    )
    try:
        installed = otlp_module._installed  # noqa: SLF001 - the process state under test
        assert installed is not None
        telemetry = start_telemetry(
            span_types=SPAN_TYPES,
            usage_buckets=provider_usage_buckets,
            endpoint="http://127.0.0.1:4318",
            span_exporter=second,
        )
        # A call that asks for no export does not uninstall the one that is running either.
        start_telemetry(span_types=SPAN_TYPES, usage_buckets=provider_usage_buckets, endpoint=None)
        assert otlp_module._installed is installed  # noqa: SLF001
        async with telemetry.observe("run-answer"):
            pass
        assert len(first.get_finished_spans()) == 1
        assert second.get_finished_spans() == ()
    finally:
        shutdown_telemetry()
