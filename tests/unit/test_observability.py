# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the span vocabulary and the seam that publishes it.

The vocabulary is the observability contract: a span name is an API for reviewers,
dashboards, and evaluators, so these tests refuse a name that is not registered, a
registered name no call site uses, a computed name, a kind that disagrees with the
table ``docs/observability.md`` publishes, and a Run root opened outside its
attribution scope. What the exporter renders on the wire is
``test_observability_wire.py``.
"""

import ast
import logging
import re
from pathlib import Path

from dlightrag.engine.ai.telemetry import SPAN_TYPES, NoopTelemetry
from tests.support.source_tree import python_files, syntax_tree


def test_no_log_call_names_a_log_record_attribute_in_extra() -> None:
    """Such a key raises KeyError at the call whenever its level is enabled.

    ``filename`` in one ``extra`` crashed ``read`` adopting an earlier Run's document.
    """
    reserved = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
    source = Path(__file__).resolve().parents[2] / "src" / "dlightrag"
    offenders = [
        f"{path.relative_to(source)}:{key.lineno} {key.value!r}"
        for path in python_files(source)
        for node in ast.walk(syntax_tree(path))
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "extra" and isinstance(keyword.value, ast.Dict)
        for key in keyword.value.keys
        if isinstance(key, ast.Constant) and key.value in reserved
    ]

    assert offenders == []


async def test_noop_telemetry_accepts_updates_without_product_dependencies() -> None:
    async with NoopTelemetry().observe("run-answer", metadata={"source": "test"}) as observation:
        observation.update(output={"ok": True})


def test_answer_trace_output_states_the_answer() -> None:
    """One shaper, one shape: the answer a reviewer reads is always there."""
    from dlightrag.engine.answer.execution.executor import answer_trace_output

    output = answer_trace_output("the answer", [], {})
    assert output["answer"] == "the answer"
    assert output["answer_len"] == len("the answer")


def test_every_numeric_secret_named_setting_is_a_token_count() -> None:
    """The settings dump shows these numbers, so each must be a count, not a credential."""
    from pydantic import BaseModel

    from dlightrag.application.config.sections import DlightragConfig
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


_SPAN_VOCABULARY_SOURCE = Path(__file__).resolve().parents[2] / "src" / "dlightrag"
_CONTRACT_DOC = Path(__file__).resolve().parents[2] / "docs" / "observability.md"
_REGISTRY = _SPAN_VOCABULARY_SOURCE / "engine" / "ai" / "telemetry.py"

_OBSERVATION_KINDS = frozenset(
    {"agent", "chain", "embedding", "generation", "retriever", "span", "tool"}
)

# A second, independent statement of the contract: the registry must say exactly
# this, so a silent kind swap (agent -> span) fails a test rather than a dashboard.
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
    return list(python_files(_SPAN_VOCABULARY_SOURCE))


def _observe_calls() -> list[tuple[Path, ast.Call]]:
    calls: list[tuple[Path, ast.Call]] = []
    for path in _source_files():
        for node in ast.walk(syntax_tree(path)):
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
        for node in ast.walk(syntax_tree(path))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "observe"
    ]
    assert offenders == []


def test_span_types_match_the_registry_and_the_contract_doc() -> None:
    by_type: dict[str, set[str]] = {}
    for name, span_type in SPAN_TYPES.items():
        by_type.setdefault(span_type, set()).add(name)
    assert by_type == _EXPECTED_TYPES
    assert set(SPAN_TYPES.values()) <= _OBSERVATION_KINDS

    documented = set(
        re.findall(
            r"^\| `([a-z-]+)` \| `([a-z]+)` \|",
            _CONTRACT_DOC.read_text("utf-8"),
            re.MULTILINE,
        )
    )
    assert documented == set(SPAN_TYPES.items())


def test_the_registry_is_the_only_place_a_span_name_maps_to_a_kind() -> None:
    """The registry is the single owner of name→kind; core code must not restate it."""
    offenders: list[str] = []
    for path in _source_files():
        if path == _REGISTRY:
            continue
        for node in ast.walk(syntax_tree(path)):
            if isinstance(node, ast.Dict):
                keys = {
                    key.value
                    for key in node.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
                if keys & set(SPAN_TYPES):
                    offenders.append(f"{path.relative_to(_SPAN_VOCABULARY_SOURCE)}:{node.lineno}")
    assert offenders == []


def test_the_composition_root_hands_the_registry_to_the_exporter() -> None:
    """The exporter states no span name: composition passes it the registry."""
    source = (_SPAN_VOCABULARY_SOURCE / "_compose.py").read_text("utf-8")
    assert "span_types=SPAN_TYPES" in source
    assert "from dlightrag_observability.otlp import" in source


def test_every_run_root_opens_inside_a_trace_attribution_scope() -> None:
    """A root without attribution is a session-less trace nobody can follow later."""
    for root, (relative, keywords) in _RUN_ROOTS.items():
        path = _SPAN_VOCABULARY_SOURCE / relative
        tree = syntax_tree(path)
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
