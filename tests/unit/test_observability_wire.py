# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What a real Langfuse client, started by ``init_tracing``, exports.

``test_observability.py`` pins the vocabulary and what the adapter hands the client;
these tests read the spans that leave the process, so a trace's roots, its
attribution and its usage keys are observed where Langfuse would see them.
"""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from langfuse import LangfuseOtelSpanAttributes as Attributes
from openai.types import CompletionUsage
from openai.types.completion_usage import PromptTokensDetails
from openai.types.responses import ResponseUsage
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from opentelemetry import trace

from dlightrag.adapters.observability import LangfuseTelemetry
from dlightrag.engine.ai.providers.base import usage_to_dict
from tests.unit.conftest import LangfuseExport


def test_an_observation_opened_in_a_request_is_a_root_and_no_framework_span_is_exported(
    langfuse_export: LangfuseExport,
) -> None:
    """FastAPI opens a span around every request; it must neither parent nor join a trace."""
    app = FastAPI()

    @app.get("/ask")
    async def ask() -> dict[str, str]:
        async with LangfuseTelemetry().observe("run-answer"):
            return {"answer": "ok"}

    with TestClient(app) as client:
        assert client.get("/ask").status_code == 200

    assert [(span.name, span.parent) for span in langfuse_export.spans()] == [("run-answer", None)]
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)


async def test_the_exported_tree_carries_its_attribution_model_and_release(
    langfuse_export: LangfuseExport,
) -> None:
    telemetry = LangfuseTelemetry()

    with telemetry.trace(session_id="sess-1", user_id="owner-1"):
        async with telemetry.observe("run-answer"):
            async with telemetry.observe("generate-completion", model="deepseek-flash"):
                pass

    spans = {span.name: span for span in langfuse_export.spans()}
    root, completion = spans["run-answer"], spans["generate-completion"]
    assert root.parent is None
    assert root.context is not None
    assert completion.parent is not None
    assert completion.parent.span_id == root.context.span_id
    for span in (root, completion):
        assert span.attributes is not None
        assert span.attributes[Attributes.TRACE_SESSION_ID] == "sess-1"
        assert span.attributes[Attributes.TRACE_USER_ID] == "owner-1"
        assert span.resource.attributes[Attributes.RELEASE] == "9.9.9"
        assert span.resource.attributes[Attributes.ENVIRONMENT] == "test"
    assert completion.attributes is not None
    assert completion.attributes[Attributes.OBSERVATION_MODEL] == "deepseek-flash"


async def test_a_sample_rate_of_zero_exports_no_trace(
    start_langfuse_export: Callable[..., LangfuseExport],
) -> None:
    export = start_langfuse_export(langfuse_sample_rate=0)

    async with LangfuseTelemetry().observe("run-answer"):
        pass

    assert export.spans() == ()


# One prompt of 673 tokens whose first 512 a prefix cache served, and a 39-token answer.
# Every wire reports it its own way; Langfuse must see 161 + 512 + 39 = 712, each token once.
_BUCKETS = {"input": 161, "input_cached_tokens": 512, "output": 39, "total": 712}

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
        {"input": 3911, "output": 254, "total": 4165},
        id="a-measured-miss-has-no-cache-key",
    ),
    pytest.param({"tokens_used": 5, "nested": {"a": 1}}, {}, id="unknown-dialect"),
]


@pytest.mark.parametrize(("counters", "buckets"), _USAGE)
async def test_usage_counts_every_token_in_exactly_one_bucket(
    langfuse_export: LangfuseExport, counters: dict[str, Any], buckets: dict[str, int]
) -> None:
    async with LangfuseTelemetry().observe("generate-completion", model="m") as observation:
        observation.update(usage_details=counters)

    (span,) = langfuse_export.spans()
    usage = langfuse_export.usage(span)
    assert usage == buckets
    if usage:
        assert usage["total"] == sum(value for key, value in usage.items() if key != "total")
