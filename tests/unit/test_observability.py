# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Langfuse observability wrappers."""

import ast
import asyncio
import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import dlightrag.adapters.observability as observability
from dlightrag.adapters.observability import langfuse as langfuse_state
from dlightrag.adapters.observability.masking import mask_langfuse_payload
from dlightrag.engine.ai.telemetry import SPAN_TYPES, NoopTelemetry
from tests.unit.conftest import RecordingLangfuse, RecordingObservation

pytestmark = pytest.mark.usefixtures("reset_langfuse_client")


def test_no_log_call_names_a_log_record_attribute_in_extra() -> None:
    """Such a key raises KeyError at the call whenever its level is enabled.

    ``filename`` in one ``extra`` crashed ``read`` adopting an earlier Run's document.
    """
    reserved = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
    source = Path(__file__).resolve().parents[2] / "src" / "dlightrag"
    offenders = [
        f"{path.relative_to(source)}:{key.lineno} {key.value!r}"
        for path in sorted(source.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "extra" and isinstance(keyword.value, ast.Dict)
        for key in keyword.value.keys
        if isinstance(key, ast.Constant) and key.value in reserved
    ]

    assert offenders == []


def test_importing_langfuse_adapter_does_not_initialize_tracing() -> None:
    from dlightrag.adapters.observability import langfuse as langfuse_adapter

    assert langfuse_adapter.current_client() is None


def test_langfuse_masking_redacts_nested_secrets_and_binary_media() -> None:
    assert mask_langfuse_payload(
        {
            "Authorization": "Bearer secret",
            "nested": ({"account_key": "secret"}, b"abc"),
            "image": {"type": "image_url", "image_url": {"url": "data:image/png,abc"}},
        }
    ) == {
        "Authorization": "[redacted]",
        "nested": [{"account_key": "[redacted]"}, "[bytes omitted: 3]"],
        "image": {"type": "image_url", "image_url": "[image omitted]"},
    }


def test_langfuse_masking_shares_the_settings_secret_names() -> None:
    """A hyphenated header is a secret; a token count is not."""
    assert mask_langfuse_payload(
        {
            "headers": {"x-api-key": "sk-live", "X-Goog-Api-Key": "g"},
            "verification_key": "k",
            "usage": {"max_tokens": 128, "prompt_tokens": 7, "include_token": False},
        }
    ) == {
        "headers": {"x-api-key": "[redacted]", "X-Goog-Api-Key": "[redacted]"},
        "verification_key": "[redacted]",
        "usage": {"max_tokens": 128, "prompt_tokens": 7, "include_token": False},
    }


def test_langfuse_masking_hides_every_non_empty_value_under_a_secret_name() -> None:
    """A secret name hides whatever it holds, whatever its type, except a count or a flag."""
    from dataclasses import dataclass

    from pydantic import BaseModel

    class Credential(BaseModel):
        value: str

    @dataclass
    class Login:
        password: str

    masked = mask_langfuse_payload(
        {
            "auth_token": Credential(value="sk-live"),
            "password": Login(password="hunter2"),
            "tokens": frozenset({"sk-a", "sk-b"}),
            "authorization": bytearray(b"Bearer sk-live"),
            "api_key": "",
            "max_tokens": 128,
            "chunk_token_size": 1200,
            "include_token": True,
            "password_pin": 12345678,
            "otp_token": 654321,
            "client_secret": 1.5,
        }
    )

    assert masked == {
        "auth_token": "[redacted]",
        "password": "[redacted]",
        "tokens": "[redacted]",
        "authorization": "[redacted]",
        "api_key": "",
        "max_tokens": 128,
        "chunk_token_size": 1200,
        "include_token": True,
        "password_pin": "[redacted]",
        "otp_token": "[redacted]",
        "client_secret": "[redacted]",
    }


def test_a_value_whose_truth_cannot_be_read_still_hides() -> None:
    class Opaque:
        def __bool__(self) -> bool:
            raise RuntimeError("ambiguous")

    assert mask_langfuse_payload({"api_key": Opaque()}) == {"api_key": "[redacted]"}


def test_every_numeric_secret_named_setting_is_a_token_count() -> None:
    """The settings dump shows these numbers, so each must be a count, not a credential."""
    from pydantic import BaseModel

    from dlightrag.application.config import DlightragConfig
    from dlightrag.engine.ai.telemetry import hides_secret_value, is_secret_key

    def numeric_secret_fields(model: type[BaseModel], seen: set[type]) -> list[str]:
        if model in seen:
            return []
        seen.add(model)
        found: list[str] = []
        for name, field in model.model_fields.items():
            annotation = field.annotation
            for candidate in getattr(annotation, "__args__", ()) or (annotation,):
                if isinstance(candidate, type) and issubclass(candidate, BaseModel):
                    found.extend(numeric_secret_fields(candidate, seen))
                elif candidate in {int, float} and is_secret_key(name):
                    found.append(name)
        return found

    names = numeric_secret_fields(DlightragConfig, set())
    assert names, "the walk found the token-count fields it guards"
    assert not [name for name in names if hides_secret_value(name, 1)]


def test_langfuse_masking_bounds_large_text() -> None:
    masked = mask_langfuse_payload("x" * 5000)

    assert isinstance(masked, str)
    assert masked[:4000] == "x" * 4000
    assert masked.endswith("... [truncated 1000 chars]")


async def test_trace_observation_captures_input_when_enabled() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    async with observability.trace_observation("run-answer", as_type="agent", input={"query": "q"}):
        pass

    assert client.observations[-1].kwargs.get("input") == {"query": "q"}


async def test_noop_telemetry_accepts_updates_without_product_dependencies() -> None:
    async with NoopTelemetry().observe("run-answer", metadata={"source": "test"}) as observation:
        observation.update(output={"ok": True})


async def test_langfuse_telemetry_derives_the_vocabulary_type() -> None:
    """The registry owns the observation type, so no call site can disagree."""
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    async with observability.LangfuseTelemetry().observe(
        "execute-agent-tool",
        metadata={"tool": "search"},
    ) as observation:
        observation.update(output={"outcome": "ok"})

    assert client.observations[0].kwargs == {
        "as_type": "tool",
        "name": "execute-agent-tool",
        "metadata": {"tool": "search"},
    }
    assert client.observations[0].updates == [{"output": {"outcome": "ok"}}]


async def test_langfuse_telemetry_normalizes_provider_usage_and_cost() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    async with observability.LangfuseTelemetry().observe(
        "generate-completion",
    ) as observation:
        observation.update(
            usage_details={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            cost_details={"total": 0.001},
        )

    assert client.observations[0].updates == [
        {
            "usage_details": {"input": 3, "output": 2, "total": 5},
            "cost_details": {"total": 0.001},
        }
    ]


async def test_observation_update_follows_the_privacy_switch() -> None:
    """Redaction belongs to the seam: a call site cannot leak a payload by updating it."""
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=False)

    async with observability.LangfuseTelemetry().observe("run-answer") as observation:
        observation.update(
            input={"query": "secret prompt"},
            output={"answer": "secret answer"},
            metadata={"run_id": "r-1"},
        )

    assert client.observations[0].updates == [{"metadata": {"run_id": "r-1"}}]


async def test_trace_observation_redacts_input_in_privacy_mode() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=False)

    async with observability.trace_observation(
        "run-answer", as_type="agent", input={"query": "secret prompt"}
    ):
        pass

    assert "input" not in client.observations[-1].kwargs


def test_answer_trace_output_reports_the_answer_only_when_asked() -> None:
    """The seam decides capture; this shaper states what it was asked to include."""
    from dlightrag.engine.answer.execution import answer_trace_output

    captured = answer_trace_output("the answer", [], {}, capture_sensitive_data=True)
    assert captured["answer"] == "the answer"

    redacted = answer_trace_output("the answer", [], {})
    assert "answer" not in redacted
    assert redacted["answer_len"] == len("the answer")


async def test_trace_observation_records_error_text_when_enabled() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    with pytest.raises(RuntimeError):
        async with observability.trace_observation("run-answer", as_type="agent"):
            raise RuntimeError("secret provider detail")

    assert client.observations[-1].updates[-1]["status_message"] == "secret provider detail"


async def test_langfuse_telemetry_cancellation_is_not_an_error() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    with pytest.raises(asyncio.CancelledError):
        async with observability.LangfuseTelemetry().observe(
            "generate-completion",
        ):
            raise asyncio.CancelledError

    assert client.observations[-1].updates == []


async def test_an_explicit_level_outranks_the_exception_mapping() -> None:
    """A run that names its own outcome must not be re-labelled ERROR on exit."""
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    with pytest.raises(RuntimeError):
        async with observability.LangfuseTelemetry().observe("run-answer") as observation:
            observation.update(level="DEFAULT", output={"outcome": "cancelled"})
            raise RuntimeError("cancelled by the owner")

    assert [update.get("level") for update in client.observations[0].updates] == ["DEFAULT"]


async def test_trace_observation_redacts_error_text_in_privacy_mode() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=False)

    with pytest.raises(RuntimeError):
        async with observability.trace_observation("run-answer", as_type="agent"):
            raise RuntimeError("secret provider detail")

    update = client.observations[-1].updates[-1]
    assert update["status_message"] == "error"
    assert "secret provider detail" not in str(update)


def _record_propagation(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    @contextmanager
    def fake_propagate(**kwargs: Any) -> Generator[None]:
        order.append(f"session={kwargs['session_id']} user={kwargs['user_id']}")
        yield

    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagate)


async def test_trace_attribution_precedes_every_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Langfuse only propagates to observations opened inside the attribution scope."""
    order: list[str] = []
    _record_propagation(monkeypatch, order)

    class _OrderedClient(RecordingLangfuse):
        def start_as_current_observation(self, **kwargs: Any) -> RecordingObservation:
            order.append("span")
            return super().start_as_current_observation(**kwargs)

    langfuse_state.install_client(_OrderedClient(), trace_sensitive=True)
    telemetry = observability.LangfuseTelemetry()

    with telemetry.trace(session_id="sess-1", user_id="owner-1"):
        async with telemetry.observe("run-answer"):
            async with telemetry.observe("retrieve-context"):
                pass

    assert order == ["session=sess-1 user=owner-1", "span", "span"]


async def test_trace_without_attribution_claims_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    _record_propagation(monkeypatch, order)
    langfuse_state.install_client(RecordingLangfuse(), trace_sensitive=True)
    telemetry = observability.LangfuseTelemetry()

    with telemetry.trace():
        async with telemetry.observe("ingest-documents"):
            pass

    assert order == []


async def test_trace_observation_without_a_session_claims_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    _record_propagation(monkeypatch, order)
    langfuse_state.install_client(RecordingLangfuse(), trace_sensitive=True)

    with observability.LangfuseTelemetry().trace():
        pass

    assert order == []


def test_init_tracing_reads_trace_sensitive_flag() -> None:
    observability.init_tracing(
        SimpleNamespace(
            langfuse_public_key=None,
            langfuse_secret_key=None,
            langfuse_trace_sensitive_data=False,
        )
    )
    assert langfuse_state.trace_sensitive_enabled() is False


def test_trace_sensitive_enabled_reflects_flag() -> None:
    langfuse_state.install_client(langfuse_state.current_client(), trace_sensitive=False)
    assert observability.trace_sensitive_enabled() is False
    langfuse_state.install_client(langfuse_state.current_client(), trace_sensitive=True)
    assert observability.trace_sensitive_enabled() is True


async def test_trace_observation_nests_child_observations() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    async with observability.trace_observation(
        "run-answer",
        as_type="agent",
        input={"query": "q"},
        metadata={"workspaces": ["default"]},
    ) as trace:
        trace.update(output={"answer_len": 12})
        async with observability.trace_observation(
            "retrieve-context",
            as_type="retriever",
            input={"query": "q"},
            metadata={"workspaces": ["default"]},
        ):
            pass

    assert [obs.kwargs["name"] for obs in client.observations] == [
        "run-answer",
        "retrieve-context",
    ]
    assert [obs.kwargs["as_type"] for obs in client.observations] == ["agent", "retriever"]
    assert client.observations[0].kwargs["input"] == {"query": "q"}
    assert client.observations[0].kwargs["metadata"] == {"workspaces": ["default"]}
    assert client.observations[0].updates == [{"output": {"answer_len": 12}}]
    assert client.observations[1].parent is client.observations[0]


async def test_trace_observation_update_is_noop_without_client() -> None:
    langfuse_state.install_client(None, trace_sensitive=True)

    async with observability.trace_observation("run-answer", as_type="agent") as trace:
        trace.update(output={"answer_len": 12})


def test_init_tracing_does_not_call_blocking_auth_check(monkeypatch: pytest.MonkeyPatch) -> None:
    auth_called = False

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def auth_check(self) -> bool:
            nonlocal auth_called
            auth_called = True
            raise AssertionError("auth_check should not run during production startup")

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)

    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
    )
    observability.init_tracing(config)

    assert auth_called is False
    assert isinstance(langfuse_state.current_client(), FakeLangfuse)


def test_init_tracing_forwards_v4_client_options(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)

    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
        langfuse_environment="production",
        langfuse_release="2026.06.06",
        langfuse_sample_rate=0.25,
        langfuse_timeout=7,
        langfuse_flush_at=16,
        langfuse_flush_interval=2.5,
    )
    observability.init_tracing(config)

    assert captured["environment"] == "production"
    assert captured["release"] == "2026.06.06"
    assert captured["sample_rate"] == 0.25
    assert captured["timeout"] == 7
    assert captured["flush_at"] == 16
    assert captured["flush_interval"] == 2.5
    assert callable(captured["mask"])


def test_langfuse_mask_redacts_secrets_and_omits_images(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)
    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
    )

    observability.init_tracing(config)
    mask = captured["mask"]
    masked = mask(
        {
            "api_key": "sk-secret",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ],
        }
    )

    assert masked == {
        "api_key": "[redacted]",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": "[image omitted]"},
        ],
    }


def test_init_tracing_clears_previous_client_when_keys_missing() -> None:
    langfuse_state.install_client(RecordingLangfuse(), trace_sensitive=True)

    config = SimpleNamespace(
        langfuse_public_key=None,
        langfuse_secret_key=None,
    )
    observability.init_tracing(config)

    assert langfuse_state.current_client() is None


def test_shutdown_tracing_uses_sdk_shutdown_and_clears_client() -> None:
    client = RecordingLangfuse()
    langfuse_state.install_client(client, trace_sensitive=True)

    observability.shutdown_tracing()

    assert client.shutdown_called is True
    assert client.flushed is False
    assert langfuse_state.current_client() is None


def test_a_client_that_fails_to_start_leaves_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class UnreachableLangfuse:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("unreachable")

    monkeypatch.setattr("langfuse.Langfuse", UnreachableLangfuse)

    observability.init_tracing(
        SimpleNamespace(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            langfuse_host="https://cloud.langfuse.com",
        )
    )

    assert langfuse_state.current_client() is None


_SPAN_VOCABULARY_SOURCE = Path(__file__).resolve().parents[2] / "src" / "dlightrag"
_CONTRACT_DOC = Path(__file__).resolve().parents[2] / "docs" / "observability.md"

_V4_OBSERVATION_TYPES = frozenset(
    {"agent", "chain", "embedding", "generation", "retriever", "span", "tool"}
)

# A second, independent statement of the contract: the registry must say exactly
# this, so a silent type swap (agent -> span) fails a test rather than a dashboard.
_EXPECTED_TYPES: dict[str, set[str]] = {
    "agent": {"run-answer"},
    "chain": {
        "generate-answer",
        "highlight-sources",
        "ingest-documents",
        "plan-retrieval",
        "run-retrieval",
    },
    "embedding": {"embed-text"},
    "generation": {
        "generate-agent-turn",
        "generate-completion",
        "compact-session",
    },
    "retriever": {"retrieve-context"},
    "span": {"call-rerank-model", "rerank-passages"},
    "tool": {"execute-agent-tool"},
}

# Each run root, and the file whose executing worker must open it. A root opened
# anywhere but the code that executes the work produces the dangling-parent
# traces this contract replaced.
_RUN_ROOTS: dict[str, tuple[str, set[str]]] = {
    "run-answer": ("engine/answer/execution/executor.py", {"session_id", "user_id"}),
    "run-retrieval": ("application/retrieval/execution.py", {"user_id"}),
}


def _source_files() -> list[Path]:
    return sorted(_SPAN_VOCABULARY_SOURCE.rglob("*.py"))


def _observe_calls() -> list[tuple[Path, ast.Call]]:
    calls: list[tuple[Path, ast.Call]] = []
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text("utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "observe"
            ):
                calls.append((path, node))
    return calls


def _span_name(call: ast.Call) -> str | None:
    """The literal span name of one observe call, or None when it is computed."""
    argument: ast.expr | None = call.args[0] if call.args else None
    for keyword in call.keywords:
        if keyword.arg == "name":
            argument = keyword.value
    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
        return argument.value
    return None


def _enclosing_function(tree: ast.AST, target: ast.AST) -> ast.AST | None:
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    node: ast.AST | None = target
    while node is not None:
        node = parents.get(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node
    return None


def test_every_observe_call_names_a_registered_span() -> None:
    """Names are an API: a stray name must fail here rather than fragment the data."""
    calls = _observe_calls()
    assert calls, "no observe call sites found; the scanner is reading the wrong tree"

    computed = [f"{path.name}:{call.lineno}" for path, call in calls if _span_name(call) is None]
    assert computed == [], "a span name must be a literal registry entry, never computed"

    observed = {name for _, call in calls if (name := _span_name(call)) is not None}
    assert observed == set(SPAN_TYPES)


def test_observe_is_never_called_through_an_alias_or_a_shadowed_name() -> None:
    """A bound alias would bypass the literal-name scan above."""
    offenders = [
        f"{path.relative_to(_SPAN_VOCABULARY_SOURCE)}:{node.lineno}"
        for path in _source_files()
        for node in ast.walk(ast.parse(path.read_text("utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "observe"
    ]
    assert offenders == []


def test_span_types_match_the_registry_the_docs_and_langfuse() -> None:
    by_type: dict[str, set[str]] = {}
    for name, span_type in SPAN_TYPES.items():
        by_type.setdefault(span_type, set()).add(name)
    assert by_type == _EXPECTED_TYPES
    assert set(SPAN_TYPES.values()) <= _V4_OBSERVATION_TYPES

    documented = set(
        re.findall(
            r"^\| `([a-z-]+)` \| `([a-z]+)` \|",
            _CONTRACT_DOC.read_text("utf-8"),
            re.MULTILINE,
        )
    )
    assert documented == set(SPAN_TYPES.items())


def test_every_run_root_opens_inside_a_trace_attribution_scope() -> None:
    """A root without attribution is a session-less trace nobody can follow later."""
    for root, (relative, keywords) in _RUN_ROOTS.items():
        path = _SPAN_VOCABULARY_SOURCE / relative
        tree = ast.parse(path.read_text("utf-8"))
        root_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "observe"
            and _span_name(node) == root
        ]
        assert len(root_calls) == 1, f"{root} must be opened exactly once, in {relative}"
        root_call = root_calls[0]
        owner = _enclosing_function(tree, root_call)
        assert owner is not None
        trace_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "trace"
            and _enclosing_function(tree, node) is owner
        ]
        assert len(trace_calls) == 1, f"{root} must be opened inside exactly one trace scope"
        attributes = {keyword.arg for keyword in trace_calls[0].keywords}
        assert keywords <= attributes
        assert trace_calls[0].lineno < root_call.lineno
        assert root_call.col_offset > trace_calls[0].col_offset


def test_only_the_observability_adapter_names_observation_types() -> None:
    """The registry is the single owner of name->type; core code must not restate it."""
    offenders = [
        str(path.relative_to(_SPAN_VOCABULARY_SOURCE))
        for path in _source_files()
        if "adapters/observability" not in str(path) and "as_type" in path.read_text("utf-8")
    ]
    assert offenders == []


def test_init_tracing_defaults_the_release_to_the_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every trace carries the running version so releases stay separable."""
    captured: dict[str, Any] = {}

    class FakeLangfuse:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)
    config = SimpleNamespace(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
    )

    observability.init_tracing(config)

    from dlightrag import __version__

    assert captured["release"] == __version__
