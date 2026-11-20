#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Replay the Session Memory Pass (ADR 0045) over fixed cases, on several models, and score it.

In production the pass has returned `{"writes": []}` on every one of 18 real calls. This harness
asks whether that is the right answer or a judging model (or prompt) that is too conservative. It
runs the pass's own request path -- `consolidation_request`, a `CompletionModel` called with the
pass's `StructuredOutput`, `StructuredOutput.parse`, `plan_consolidation` -- over cases whose gold
memories were written down in advance, for each candidate model in a named matrix, and scores what
comes back.

Two case files, one schema (`scripts/memory_pass_eval/synthetic_cases.json`, committed; and a private
`real_cases.json` of real production calls, gitignored under `docs/research/memory-pass-eval/`).
A case lists the conversation (`turns`, and optionally the `context_turns` just before it, which the
pass is shown as context only), the memories already held, a `must` list of memories the pass should
write, a `may` list it is welcome to, a `must_not` list, and `expect_actions` (an update or fuse the
pass should choose over an add).

An output that plans at least one write is graded by a separate LLM judge (Opus on OpenRouter) that
sees only the gold items and the planned writes: each write is a `match` for a gold item, an
`acceptable` memory nobody listed, or a `violation`. The judge's verdicts are cached by (case, gold,
planned writes) so an identical output is judged once. Scoring is a set of pure functions over the
saved results and judgements, so `--rescore` rebuilds a summary without a model call.

Usage (run from the repository; the project environment provides the packages):
    uv run python scripts/run_memory_pass_eval.py                           # every config, 3 reps
    uv run python scripts/run_memory_pass_eval.py --configs ds-flash-off,sonnet --reps 1 \\
        --case-ids S1,N5,U1,R16 --label smoke
    uv run python scripts/run_memory_pass_eval.py --system-prompt-file p.txt --label p2
    uv run python scripts/run_memory_pass_eval.py --rescore DIR

`--system-prompt-file` replaces only the system message `consolidation_request` produces; the file is
used verbatim, except that the literal `{limit}` becomes the case's `record_limit` (braces are
otherwise literal, so a JSON example needs no doubling). `--print-production-prompt` prints the
production text to start from.

Results append to `results.jsonl` in the output directory and a rerun with the same directory skips
every (config, label, case, rep) that already completed; a run that failed on transport is retried.
Judgements go to `judgements.jsonl`, and `summary.md` / `summary.json` are rebuilt at the end.
API keys come from the environment or the operator's `.env`; they are read, never printed, and never
written to any output file.

Exit status is 0 when every run completed and every non-empty output was judged, 1 otherwise.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast

import openai
from dlightrag_memory import MemoryProvenance, MemoryRecord
from dlightrag_memory.consolidation import (
    ConsolidationRequest,
    ConsolidationVerdict,
    ConversationTurn,
    PlannedWrite,
    consolidation_request,
    plan_consolidation,
    verified_quote,
)
from dlightrag_memory.models import MemoryKind
from dlightrag_memory.policy import MEMORY_FUSE_MAX_SOURCES, MEMORY_FUSE_MIN_SOURCES
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from dlightrag.engine.ai.completion import CompletionModel
from dlightrag.engine.ai.contracts import ApiFamily
from dlightrag.engine.ai.reasoning import ReasoningLevel
from dlightrag.engine.ai.scheduler import ModelScheduler, model_call_scope
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.ai.structured import StructuredOutput
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_CASES = REPO_ROOT / "scripts" / "memory_pass_eval" / "synthetic_cases.json"
CASE_FILE_VERSION = 1

DEFAULT_REPS = 3
DEFAULT_CONCURRENCY = 6
DEFAULT_TIMEOUT_SECONDS = 120.0
#: Retries for a transport failure (timeout, connection, 408/409/429/5xx); a parse error is a result.
MAX_TRANSPORT_RETRIES = 2
RETRY_BACKOFF_SECONDS = 2.0
#: The pass waits as one scheduling owner (session_pass.py); the harness binds the same one.
SCHEDULING_OWNER = "dlightrag-session-memory-pass"
#: The pass's own structured output, copied from session_pass.py so the request carries the same
#: schema name and shape.
VERDICT = StructuredOutput(name="session_memory_verdict", schema=ConsolidationVerdict)

DEEPSEEK_URL = "https://api.deepseek.com"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_KEY_ENV = "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY"
DEEPSEEK_QUERY_KEY_ENV = "DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY"
OPENROUTER_KEY_ENV = "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY"

RESULTS_FILE = "results.jsonl"
JUDGEMENTS_FILE = "judgements.jsonl"
META_FILE = "run_meta.json"
SUMMARY_MD = "summary.md"
SUMMARY_JSON = "summary.json"
LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PRODUCTION_LABEL = "prod"


# --------------------------------------------------------------------------------------- cases


class CaseFileError(ValueError):
    """A case file that does not match the schema, named with its path."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Lenient(BaseModel):
    """For a model's answer, where an extra key is noise and the wire schema is strict anyway."""

    model_config = ConfigDict(extra="ignore")


class CaseTurn(_Strict):
    role: Literal["user", "assistant"]
    text: str


class CaseMemory(_Strict):
    """A memory the owner already holds, shown to the pass as a candidate or session record.

    `updated_at` is when it was last written, shown to the pass to the hour in UTC; a case that
    leaves it out shows the memory with no time. `quote` is the owner's own words behind it, shown
    to the pass on the line under the memory; a case that leaves it out shows none.
    """

    memory_id: str = Field(min_length=1)
    kind: MemoryKind
    body: str = Field(min_length=1)
    updated_at: datetime | None = None
    quote: str | None = Field(default=None, min_length=1)


class MustItem(_Strict):
    """A memory the pass should write. `min_matches` k > 1 asks for k distinct writes."""

    id: str = Field(min_length=1)
    kind: MemoryKind
    description: str = Field(min_length=1)
    min_matches: int = Field(default=1, ge=1)


class MayItem(_Strict):
    """A memory the pass is welcome to write but need not."""

    id: str = Field(min_length=1)
    kind: MemoryKind
    description: str = Field(min_length=1)


class ExpectedAction(_Strict):
    """The write that matches `gold` should be this action on exactly these targets."""

    gold: str
    action: Literal["add", "update", "fuse"]
    target_ids: list[str] = Field(default_factory=list)


class Case(_Strict):
    id: str = Field(min_length=1)
    source: Literal["real", "synthetic"]
    note: str = ""
    turns: list[CaseTurn] = Field(min_length=1)
    #: The turns just before `turns`, oldest first, which the pass shows its model as context
    #: only (it is handed the nearest few, each cut short; a case may list more and see the
    #: pass's own bound applied). An earlier pass is taken to have judged them already.
    context_turns: list[CaseTurn] = Field(default_factory=list)
    candidates: list[CaseMemory] = Field(default_factory=list)
    session_records: list[CaseMemory] = Field(default_factory=list)
    record_limit: int = Field(ge=1)
    must: list[MustItem] = Field(default_factory=list)
    may: list[MayItem] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)
    expect_actions: list[ExpectedAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        gold_ids = [item.id for item in (*self.must, *self.may)]
        if len(set(gold_ids)) != len(gold_ids):
            raise ValueError(f"case {self.id}: must/may ids are not unique")
        for group, records in (
            ("candidates", self.candidates),
            ("session_records", self.session_records),
        ):
            ids = [record.memory_id for record in records]
            if len(set(ids)) != len(ids):
                raise ValueError(f"case {self.id}: {group} memory_ids are not unique")
        shown = {record.memory_id for record in (*self.candidates, *self.session_records)}
        for expected in self.expect_actions:
            if expected.gold not in gold_ids:
                raise ValueError(
                    f"case {self.id}: expect_actions names unknown gold {expected.gold}"
                )
            unknown = [target for target in expected.target_ids if target not in shown]
            if unknown:
                raise ValueError(f"case {self.id}: expect_actions targets unseen ids {unknown}")
            count = len(set(expected.target_ids))
            if expected.action == "add":
                well_targeted = count == 0
            elif expected.action == "update":
                well_targeted = count == 1
            else:
                well_targeted = MEMORY_FUSE_MIN_SOURCES <= count <= MEMORY_FUSE_MAX_SOURCES
            if not well_targeted:
                raise ValueError(
                    f"case {self.id}: {expected.action} cannot have {count} target(s) "
                    "(the plan would drop it)"
                )
        return self

    def gold_ids(self) -> list[str]:
        return [item.id for item in (*self.must, *self.may)]

    def gold_digest(self) -> str:
        """What the judge is told about this case; a change to it invalidates cached verdicts."""
        payload = {
            "must": [item.model_dump() for item in self.must],
            "may": [item.model_dump() for item in self.may],
            "must_not": self.must_not,
        }
        return _digest(payload)


def load_cases(paths: Sequence[Path]) -> list[Case]:
    """Every case in `paths`, validated; ids are unique across the files."""
    cases: list[Case] = []
    seen: dict[str, Path] = {}
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CaseFileError(f"{path}: cannot read case file: {exc}") from exc
        if not isinstance(document, dict) or document.get("version") != CASE_FILE_VERSION:
            raise CaseFileError(f"{path}: expected an object with version {CASE_FILE_VERSION}")
        raw_cases = document.get("cases")
        if not isinstance(raw_cases, list):
            raise CaseFileError(f"{path}: `cases` must be a list")
        for index, raw in enumerate(raw_cases):
            try:
                case = Case.model_validate(raw)
            except ValidationError as exc:
                name = raw.get("id") if isinstance(raw, dict) else index
                raise CaseFileError(f"{path}: case {name}: {_short_validation(exc)}") from exc
            except ValueError as exc:  # the cross-field check
                raise CaseFileError(f"{path}: {exc}") from exc
            if case.id in seen:
                raise CaseFileError(f"{path}: case id {case.id} also appears in {seen[case.id]}")
            seen[case.id] = path
            cases.append(case)
    return cases


def _short_validation(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()
    )


def case_records(case: Case) -> tuple[list[MemoryRecord], list[MemoryRecord]]:
    """The case's candidates and session records as the `MemoryRecord`s the pass is shown."""

    def record(memory: CaseMemory, provenance: MemoryProvenance) -> MemoryRecord:
        return MemoryRecord(
            owner_id="eval-owner",
            memory_id=memory.memory_id,
            kind=memory.kind,
            body=memory.body,
            provenance=provenance,
            updated_at=memory.updated_at,
            quote=memory.quote,
        )

    earlier = MemoryProvenance(origin_kind="answer_run", origin_id=f"{case.id}-earlier-run")
    this_session = MemoryProvenance(
        origin_kind="session_pass", origin_id=f"{case.id}:1-1", session_id=f"{case.id}-session"
    )
    return (
        [record(memory, earlier) for memory in case.candidates],
        [record(memory, this_session) for memory in case.session_records],
    )


def build_request(case: Case, *, system_prompt: str | None = None) -> ConsolidationRequest:
    """The pass's request for `case`; `system_prompt` replaces the system message only."""
    candidates, session_records = case_records(case)
    request = consolidation_request(
        turns=[ConversationTurn(role=turn.role, text=turn.text) for turn in case.turns],
        context_turns=[
            ConversationTurn(role=turn.role, text=turn.text) for turn in case.context_turns
        ],
        candidates=candidates,
        session_records=session_records,
        record_limit=case.record_limit,
    )
    if system_prompt is None:
        return request
    if not request.messages or request.messages[0].get("role") != "system":
        raise RuntimeError("consolidation_request no longer starts with a system message")
    request.messages[0] = {
        "role": "system",
        "content": system_prompt.replace("{limit}", str(case.record_limit)),
    }
    return request


def prompt_digest(messages: Sequence[Mapping[str, str]]) -> str:
    """Short hash of the system message, to tell two prompts apart under one label."""
    return hashlib.sha256(messages[0]["content"].encode("utf-8")).hexdigest()[:12]


def _digest(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------ model matrix


@dataclass(frozen=True, slots=True)
class CandidateConfig:
    """One endpoint the pass could run on, as the `ModelSettings` it would be built from."""

    name: str
    model: str
    base_url: str
    key_env: str
    structured_output: Literal["auto", "json_schema", "json_object"]
    reasoning: ReasoningLevel | None = None
    temperature: float = 0.0
    api_family: ApiFamily = "chat_completion"
    note: str = ""

    def settings(self, api_key: str, *, timeout: float) -> ModelSettings:
        # The SDK's own retries are off: the harness retries transport failures itself, so the
        # count it records is the count that happened.
        return ModelSettings(
            provider="openai",
            model=self.model,
            api_key=api_key,
            base_url=self.base_url,
            structured_output=self.structured_output,
            api_family=self.api_family,
            temperature=self.temperature,
            reasoning=self.reasoning,
            timeout=timeout,
            max_retries=0,
        )


CONFIGS: dict[str, CandidateConfig] = {
    config.name: config
    for config in (
        CandidateConfig(
            "ds-flash-off",
            "deepseek-flash",
            DEEPSEEK_URL,
            DEEPSEEK_KEY_ENV,
            "json_object",
            reasoning="off",
            note="the pass's former model: models.chat.roles.extract in config.yaml",
        ),
        CandidateConfig(
            "ds-flash-on",
            "deepseek-flash",
            DEEPSEEK_URL,
            DEEPSEEK_KEY_ENV,
            "json_object",
            reasoning="high",
            note="the former extract endpoint with thinking enabled (reasoning_effort high)",
        ),
        CandidateConfig(
            "ds-pro",
            "deepseek-v4-pro",
            DEEPSEEK_URL,
            DEEPSEEK_KEY_ENV,
            "json_object",
            reasoning="off",
            note="DeepSeek direct, id from GET /models; not in the model catalogue",
        ),
        CandidateConfig(
            "glm-flash",
            "z-ai/glm-5.3-flash",
            OPENROUTER_URL,
            OPENROUTER_KEY_ENV,
            "auto",
            note="reasoning cannot be disabled on OpenRouter for this model: provider default",
        ),
        CandidateConfig(
            "gemini-flash",
            "google/gemini-3.5-flash",
            OPENROUTER_URL,
            OPENROUTER_KEY_ENV,
            "auto",
            note="reasoning cannot be disabled on OpenRouter for this model: provider default",
        ),
        CandidateConfig(
            "sonnet",
            "anthropic/claude-sonnet-5.5",
            OPENROUTER_URL,
            OPENROUTER_KEY_ENV,
            "auto",
            note="quality ceiling reference; no reasoning parameter sent",
        ),
        CandidateConfig(
            "query-role",
            "deepseek-flash",
            DEEPSEEK_URL,
            DEEPSEEK_QUERY_KEY_ENV,
            "json_object",
            reasoning="off",
            temperature=0.7,
            api_family="response",
            note="models.chat.roles.query in config.yaml as configured: its ordinary reasoning, off",
        ),
        CandidateConfig(
            "query-pass",
            "deepseek-flash",
            DEEPSEEK_URL,
            DEEPSEEK_QUERY_KEY_ENV,
            "json_object",
            reasoning="low",
            temperature=0.7,
            api_family="response",
            note="production: the query role with reasoning set to memory_consolidation.reasoning",
        ),
    )
}

JUDGE_CONFIG = CandidateConfig(
    "judge",
    "anthropic/claude-opus-5.5",
    OPENROUTER_URL,
    OPENROUTER_KEY_ENV,
    "auto",
    note="separate from the candidates; temperature 0",
)


def select_configs(spec: str | None) -> list[CandidateConfig]:
    """The configs named by a comma-separated `--configs`, or all of them."""
    if not spec:
        return list(CONFIGS.values())
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in CONFIGS]
    if unknown:
        raise ValueError(f"unknown config(s) {unknown}; choose from {sorted(CONFIGS)}")
    return [CONFIGS[name] for name in dict.fromkeys(names)]


def load_secrets(env_file: Path, names: Iterable[str]) -> dict[str, str]:
    """The named secrets from the environment, else from `env_file`. Values are never printed."""
    wanted = set(names)
    found = {name: os.environ[name] for name in wanted if os.environ.get(name)}
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            name, separator, value = line.partition("=")
            name = name.strip()
            if not separator or name not in wanted or name in found:
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].strip()
            if value:
                found[name] = value
    return found


def scrub(text: str, secrets: Iterable[str]) -> str:
    """`text` with every secret replaced, for anything that is about to be written down."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


# ----------------------------------------------------------------------------- model calls


@dataclass(frozen=True, slots=True)
class ModelReply:
    text: str
    usage: dict[str, int] | None = None
    cost: float | None = None
    #: What each provider request in this call put on the wire (response_format type, reasoning).
    wire: list[dict[str, Any]] = field(default_factory=list)


class Verdicter(Protocol):
    """A model asked one question with a structured output; closed by its owner."""

    async def __call__(self, messages: list[dict[str, str]]) -> ModelReply: ...

    async def aclose(self) -> None: ...


#: Factory for the model behind one config and one structured output; tests substitute fakes.
ModelFactory = Callable[[CandidateConfig, StructuredOutput], Verdicter]

_wire_log: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "memory_pass_eval_wire", default=None
)
_REASONING_KEYS = ("thinking", "reasoning", "reasoning_effort")


class EvalModel:
    """A `CompletionModel` called the way the pass calls it, plus a record of the request shape."""

    def __init__(
        self, model: CompletionModel, *, structured: StructuredOutput, timeout: float
    ) -> None:
        self._model = model
        self._structured = structured
        self._timeout = timeout
        # Record what each provider request actually carried. CompletionModel owns the transport
        # decision (including a json_schema -> json_object downgrade), and nothing public reports
        # it, so the provider's `complete` is wrapped in place.
        provider: Any = model._provider
        inner = provider.complete

        async def recording(*args: Any, **kwargs: Any) -> Any:
            log = _wire_log.get()
            if log is not None:
                response_format = kwargs.get("response_format") or {}
                model_kwargs = kwargs.get("model_kwargs") or {}
                log.append(
                    {
                        "response_format": response_format.get("type"),
                        "reasoning": {
                            key: model_kwargs[key] for key in _REASONING_KEYS if key in model_kwargs
                        },
                    }
                )
            return await inner(*args, **kwargs)

        provider.complete = recording

    async def __call__(self, messages: list[dict[str, str]]) -> ModelReply:
        wire: list[dict[str, Any]] = []
        token = _wire_log.set(wire)
        try:
            with model_call_scope(SCHEDULING_OWNER):
                raw = await asyncio.wait_for(
                    self._model(list(messages), structured_output=self._structured),
                    timeout=self._timeout,
                )
        finally:
            _wire_log.reset(token)
        cost = getattr(raw, "cost_details", None)
        total = cost.get("total") if isinstance(cost, dict) else None
        return ModelReply(
            text=str(raw),
            usage=getattr(raw, "usage_details", None),
            cost=float(total) if isinstance(total, int | float) else None,
            wire=wire,
        )

    async def aclose(self) -> None:
        await self._model.aclose()


def default_model_factory(
    secrets: Mapping[str, str], *, timeout: float, scheduler: ModelScheduler
) -> ModelFactory:
    """Build real `CompletionModel`s from the operator's keys."""

    def build(config: CandidateConfig, structured: StructuredOutput) -> Verdicter:
        key = secrets.get(config.key_env)
        if not key:
            raise RuntimeError(f"{config.key_env} is not set (environment or .env)")
        model = CompletionModel(
            config.settings(key, timeout=timeout), scheduler=scheduler, telemetry=NOOP_TELEMETRY
        )
        return EvalModel(model, structured=structured, timeout=timeout)

    return build


def is_transport_error(exc: BaseException) -> bool:
    """Whether a failed call is worth repeating: the request may not have been judged at all."""
    if isinstance(exc, TimeoutError | ConnectionError | openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


# ------------------------------------------------------------------------------ run records

type RunKey = tuple[str, str, str, int]


class WriteRecord(_Strict):
    action: Literal["add", "update", "fuse"]
    kind: MemoryKind
    body: str
    target_ids: list[str] = Field(default_factory=list)
    #: The owner's words the host kept behind the write; none when the model gave none or the host
    #: refused what it gave.
    quote: str | None = None

    @classmethod
    def of(cls, write: PlannedWrite) -> WriteRecord:
        return cls(
            action=write.action,
            kind=write.kind,
            body=write.body,
            target_ids=list(write.target_ids),
            quote=write.quote,
        )


class ProposedWrite(_Strict):
    """One write as the model proposed it: `targets` are the numbers it was shown, not ids."""

    action: Literal["add", "update", "fuse"]
    kind: MemoryKind
    body: str
    targets: list[int] = Field(default_factory=list)
    quote: str | None = None


class RunRecord(_Strict):
    """One (config, label, case, rep) call and everything the scorer needs from it."""

    config: str
    label: str
    case_id: str
    rep: int
    status: Literal["ok", "error"]
    prompt_sha: str
    started_at: str
    latency_s: float | None = None
    attempts: int = 1
    raw: str | None = None
    parse_error: str | None = None
    #: What the model proposed (by number), then what `plan_consolidation` kept of it (by id).
    verdict: list[ProposedWrite] = Field(default_factory=list)
    planned: list[WriteRecord] = Field(default_factory=list)
    #: How many quotes the model gave that the host refused (not the owner's words, or not storable).
    quotes_dropped: int = 0
    usage: dict[str, int] | None = None
    cost: float | None = None
    wire: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None

    @property
    def key(self) -> RunKey:
        return (self.config, self.label, self.case_id, self.rep)

    @property
    def total_tokens(self) -> int:
        if not self.usage:
            return 0
        if "total_tokens" in self.usage:
            return self.usage["total_tokens"]
        return self.usage.get("prompt_tokens", 0) + self.usage.get("completion_tokens", 0)


async def run_one(
    model: Verdicter,
    *,
    config: str,
    label: str,
    case: Case,
    rep: int,
    request: ConsolidationRequest,
    secrets: Iterable[str] = (),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> RunRecord:
    """Ask `model` once, as the pass does, and keep whatever happened.

    A transport failure is repeated up to `MAX_TRANSPORT_RETRIES` times; any other failure of the
    call is recorded as an error. Once the model has answered, nothing is repeated: text the
    pass could not parse is a parse error, which is a result.
    """
    secret_values = tuple(secrets)
    sha = prompt_digest(request.messages)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    reply: ModelReply | None = None
    failure: str | None = None
    latency = 0.0
    attempts = 0
    for attempts in range(1, MAX_TRANSPORT_RETRIES + 2):
        began = time.monotonic()
        try:
            reply = await model(list(request.messages))
            latency = time.monotonic() - began
            break
        except Exception as exc:
            failure = scrub(f"{type(exc).__name__}: {exc}", secret_values)[:600]
            if not is_transport_error(exc) or attempts > MAX_TRANSPORT_RETRIES:
                break
            await sleep(RETRY_BACKOFF_SECONDS * attempts)
    if reply is None:
        return RunRecord(
            config=config,
            label=label,
            case_id=case.id,
            rep=rep,
            status="error",
            prompt_sha=sha,
            started_at=started,
            attempts=attempts,
            error=failure,
        )

    verdict: ConsolidationVerdict | None = None
    parse_error: str | None = None
    try:
        verdict = cast(ConsolidationVerdict, VERDICT.parse(reply.text))
    except Exception as exc:
        parse_error = scrub(f"{type(exc).__name__}: {exc}", secret_values)[:600]
    proposed: list[ProposedWrite] = []
    planned: list[WriteRecord] = []
    quotes_dropped = 0
    if verdict is not None:
        proposed = [ProposedWrite(**write.model_dump()) for write in verdict.writes]
        planned = [
            WriteRecord.of(write)
            for write in plan_consolidation(
                verdict,
                handles=request.handles,
                record_limit=case.record_limit,
                owner_words=request.owner_words,
            )
        ]
        quotes_dropped = sum(
            1
            for write in verdict.writes
            if write.quote is not None and verified_quote(write.quote, request.owner_words) is None
        )
    return RunRecord(
        config=config,
        label=label,
        case_id=case.id,
        rep=rep,
        status="ok",
        prompt_sha=sha,
        started_at=started,
        attempts=attempts,
        latency_s=round(latency, 3),
        raw=scrub(reply.text, secret_values),
        parse_error=parse_error,
        verdict=proposed,
        planned=planned,
        quotes_dropped=quotes_dropped,
        usage=reply.usage,
        cost=reply.cost,
        wire=reply.wire,
    )


# ------------------------------------------------------------------------- results storage


def load_results(path: Path) -> dict[RunKey, RunRecord]:
    """Records by key, the last line of a key winning. A torn final line is ignored."""
    records: dict[RunKey, RunRecord] = {}
    if not path.is_file():
        return records
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        try:
            record = RunRecord.model_validate_json(line)
        except ValidationError as exc:
            if number == len(lines):
                continue
            raise ValueError(f"{path}:{number}: not a run record") from exc
        records[record.key] = record
    return records


def append_jsonl(path: Path, model: BaseModel) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(model.model_dump_json() + "\n")


class ResumeConflict(RuntimeError):
    """The output directory holds results for a key made with a different prompt."""


@dataclass(frozen=True, slots=True)
class PendingRun:
    config: CandidateConfig
    case: Case
    rep: int
    request: ConsolidationRequest


def pending_runs(
    configs: Sequence[CandidateConfig],
    cases: Sequence[Case],
    *,
    reps: int,
    label: str,
    system_prompt: str | None,
    done: Mapping[RunKey, RunRecord],
) -> list[PendingRun]:
    """The runs still to make, in rep -> case -> config order so providers interleave.

    A key whose record completed is skipped. A key that ended in a transport error is made again.
    A completed key made with another prompt under the same label is an error: a label names one
    prompt, and mixing two under it would make the summary mean nothing.
    """
    pending: list[PendingRun] = []
    for rep in range(1, reps + 1):
        for case in cases:
            request = build_request(case, system_prompt=system_prompt)
            sha = prompt_digest(request.messages)
            for config in configs:
                previous = done.get((config.name, label, case.id, rep))
                if previous is not None and previous.prompt_sha != sha:
                    raise ResumeConflict(
                        f"{config.name}/{case.id}#{rep} was run under label {label!r} with a "
                        "different prompt; use another --label or --out"
                    )
                if previous is not None and previous.status == "ok":
                    continue
                pending.append(PendingRun(config, case, rep, request))
    return pending


async def run_matrix(
    pending: Sequence[PendingRun],
    *,
    label: str,
    out_dir: Path,
    factory: ModelFactory,
    concurrency: int,
    secrets: Iterable[str] = (),
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> list[RunRecord]:
    """Make `pending` calls with bounded concurrency, appending each record as it lands."""
    results_path = out_dir / RESULTS_FILE
    gate = asyncio.Semaphore(max(1, concurrency))
    write_lock = asyncio.Lock()
    models: dict[str, Verdicter] = {}
    secret_values = tuple(secrets)
    made: list[RunRecord] = []
    total = len(pending)

    def model_for(config: CandidateConfig) -> Verdicter:
        if config.name not in models:
            models[config.name] = factory(config, VERDICT)
        return models[config.name]

    async def one(item: PendingRun) -> None:
        async with gate:
            try:
                model = model_for(item.config)
            except Exception as exc:
                record = RunRecord(
                    config=item.config.name,
                    label=label,
                    case_id=item.case.id,
                    rep=item.rep,
                    status="error",
                    prompt_sha=prompt_digest(item.request.messages),
                    started_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    error=scrub(f"{type(exc).__name__}: {exc}", secret_values)[:600],
                )
            else:
                record = await run_one(
                    model,
                    config=item.config.name,
                    label=label,
                    case=item.case,
                    rep=item.rep,
                    request=item.request,
                    secrets=secret_values,
                )
        async with write_lock:
            append_jsonl(results_path, record)
            made.append(record)
            log(f"[{len(made)}/{total}] {_progress_line(record)}")

    try:
        await asyncio.gather(*(one(item) for item in pending))
    finally:
        await asyncio.gather(*(model.aclose() for model in models.values()), return_exceptions=True)
    return made


def _progress_line(record: RunRecord) -> str:
    head = f"{record.config} {record.case_id}#{record.rep}"
    if record.status == "error":
        return f"{head} ERROR {record.error}"
    outcome = "PARSE-ERROR" if record.parse_error else f"writes={len(record.planned)}"
    return f"{head} {outcome} {record.latency_s:.1f}s"


# ---------------------------------------------------------------------------------- judging

JUDGE_SYSTEM = """\
You grade proposed writes to one person's long-term memory. A memory component read a conversation
between the person (the owner) and an AI Agent and proposed the writes below. You cannot see the
conversation: grade each write only against the reference list in the user message.

A good memory is durable (it keeps applying after the current task), general (it covers a class of
situations, not one file, command or artifact), the owner's own (a working rule or fact about the
owner, or something the Agent learned about doing its work for this owner), and written as one
third-person statement ("The owner prefers ...").

Give every write exactly one status:
- "match": it expresses a MUST or MAY item. Put that item's id in "match". If the same write also
  expresses further items, list their ids in "also_matches". Wording and a different kind do not
  matter; the meaning must be the same, so a write that is much wider or narrower than the item
  matches only if it still carries the item's substance. An item that states min_matches is covered
  by several writes that each carry a different part of it; a write that only repeats a part an
  earlier write already carries is "acceptable".
- "acceptable": not in the reference, but a reasonable durable, general memory of the owner's own,
  phrased as a third-person statement. "match" is null.
- "violation": any of: it hits a MUST NOT entry; it records a task detail, a research finding, a
  claim from a document or the web, a citation or a secret; it states a third party's view or
  preference as the owner's; or it is phrased as an instruction or command instead of a third-person
  statement. A write that carries a MUST or MAY item but is phrased as an instruction is a
  violation; set "match" to that item so the reason is clear.

Answer with one JSON object and nothing else, in exactly this shape, one entry per write and
"index" the write's number:
{"judgements": [{"index": 0, "match": null, "also_matches": [], "status": "acceptable", "reason": "one short sentence"}]}
"""


class JudgeEntry(_Lenient):
    index: int
    match: str | None = None
    also_matches: list[str] = Field(default_factory=list)
    status: Literal["match", "acceptable", "violation"]
    reason: str = ""


class JudgeVerdict(_Lenient):
    judgements: list[JudgeEntry]


JUDGE_OUTPUT = StructuredOutput(name="memory_write_judgement", schema=JudgeVerdict)


class JudgementRecord(_Strict):
    key: str
    case_id: str
    writes: list[WriteRecord]
    judgements: list[JudgeEntry]
    judge_model: str
    judged_at: str


def _judged(write: WriteRecord) -> dict[str, Any]:
    """The write as the judge sees it. A quote is the owner's evidence for a memory, not part of
    the memory, so it neither reaches the judge nor tells two otherwise equal writes apart."""
    return write.model_dump(exclude={"quote"})


def judge_key(case: Case, planned: Sequence[WriteRecord]) -> str:
    """Cache key: the case, what the judge is told about it, and the writes it grades."""
    return _digest(
        {
            "case": case.id,
            "gold": case.gold_digest(),
            "writes": [_judged(write) for write in planned],
        }
    )


def judge_messages(case: Case, planned: Sequence[WriteRecord]) -> list[dict[str, str]]:
    lines = ["MUST items (the memory should contain these):"]
    lines.extend(
        f"- {item.id} [{item.kind}]"
        + (f" (min_matches: {item.min_matches})" if item.min_matches > 1 else "")
        + f": {item.description}"
        for item in case.must
    )
    if not case.must:
        lines.append("(none)")
    lines.append("")
    lines.append("MAY items (welcome, not required):")
    lines.extend(f"- {item.id} [{item.kind}]: {item.description}" for item in case.may)
    if not case.may:
        lines.append("(none)")
    lines.append("")
    lines.append("MUST NOT (a write that does any of this is a violation):")
    lines.extend(f"- {entry}" for entry in case.must_not)
    if not case.must_not:
        lines.append("(none)")
    lines.append("")
    lines.append("Proposed writes:")
    for index, write in enumerate(planned):
        lines.append(json.dumps({"index": index, **_judged(write)}, ensure_ascii=False))
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def validate_judgement(case: Case, planned: Sequence[WriteRecord], verdict: JudgeVerdict) -> None:
    """Raise ValueError when a judge verdict does not cover the writes coherently."""
    indexes = [entry.index for entry in verdict.judgements]
    if sorted(indexes) != list(range(len(planned))):
        raise ValueError(f"expected one entry for each index 0..{len(planned) - 1}, got {indexes}")
    gold = set(case.gold_ids())
    for entry in verdict.judgements:
        named = [entry.match, *entry.also_matches] if entry.match else list(entry.also_matches)
        unknown = [item for item in named if item not in gold]
        if unknown:
            raise ValueError(f"index {entry.index} names unknown gold ids {unknown}")
        if entry.status == "match" and not entry.match:
            raise ValueError(f"index {entry.index} is a match without a gold id")
        if entry.status == "acceptable" and (entry.match or entry.also_matches):
            raise ValueError(f"index {entry.index} is acceptable but names gold ids")
        if entry.status == "violation" and entry.also_matches:
            raise ValueError(f"index {entry.index} is a violation with also_matches")


class JudgeFailure(RuntimeError):
    """The judge could not produce a valid verdict for one set of writes."""


async def judge_writes(
    judge: Verdicter,
    case: Case,
    planned: Sequence[WriteRecord],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> JudgeVerdict:
    """Grade `planned`. An invalid verdict is asked for once more, with the problem named."""
    messages = judge_messages(case, planned)
    problem = ""
    for attempt in range(2):
        asked = messages
        if attempt:
            asked = [
                *messages,
                {
                    "role": "user",
                    "content": f"That answer was invalid ({problem}). Answer again, correctly.",
                },
            ]
        reply = await _call_with_retries(judge, asked, sleep=sleep)
        try:
            verdict = cast(JudgeVerdict, JUDGE_OUTPUT.parse(reply.text))
            validate_judgement(case, planned, verdict)
        except Exception as exc:
            problem = f"{type(exc).__name__}: {str(exc)[:300]}"
            continue
        return verdict
    raise JudgeFailure(problem)


async def _call_with_retries(
    model: Verdicter,
    messages: list[dict[str, str]],
    *,
    sleep: Callable[[float], Awaitable[None]],
) -> ModelReply:
    for attempt in range(1, MAX_TRANSPORT_RETRIES + 2):
        try:
            return await model(list(messages))
        except Exception as exc:
            if not is_transport_error(exc) or attempt > MAX_TRANSPORT_RETRIES:
                raise
            await sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise AssertionError("unreachable")


def load_judgements(path: Path) -> dict[str, JudgementRecord]:
    records: dict[str, JudgementRecord] = {}
    if not path.is_file():
        return records
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for number, line in enumerate(lines, start=1):
        try:
            record = JudgementRecord.model_validate_json(line)
        except ValidationError as exc:
            if number == len(lines):
                continue
            raise ValueError(f"{path}:{number}: not a judgement record") from exc
        records[record.key] = record
    return records


async def judge_missing(
    results: Mapping[RunKey, RunRecord],
    cases: Mapping[str, Case],
    judgements: dict[str, JudgementRecord],
    *,
    out_dir: Path,
    judge_factory: Callable[[], Verdicter],
    concurrency: int,
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> list[str]:
    """Judge every distinct non-empty output not yet judged. Returns the failures, as text."""
    wanted: dict[str, tuple[Case, list[WriteRecord]]] = {}
    for record in results.values():
        case = cases.get(record.case_id)
        if record.status != "ok" or not record.planned or case is None:
            continue
        key = judge_key(case, record.planned)
        if key not in judgements:
            wanted.setdefault(key, (case, record.planned))
    if not wanted:
        return []
    try:
        judge = judge_factory()
    except Exception as exc:
        return [f"judge unavailable: {type(exc).__name__}: {exc}"]
    gate = asyncio.Semaphore(max(1, concurrency))
    lock = asyncio.Lock()
    failures: list[str] = []
    done = 0

    async def one(key: str, case: Case, planned: list[WriteRecord]) -> None:
        nonlocal done
        async with gate:
            try:
                verdict = await judge_writes(judge, case, planned)
            except Exception as exc:
                async with lock:
                    failures.append(f"{case.id}: {type(exc).__name__}: {str(exc)[:300]}")
                return
        record = JudgementRecord(
            key=key,
            case_id=case.id,
            writes=planned,
            judgements=sorted(verdict.judgements, key=lambda entry: entry.index),
            judge_model=JUDGE_CONFIG.model,
            judged_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        async with lock:
            append_jsonl(out_dir / JUDGEMENTS_FILE, record)
            judgements[key] = record
            done += 1
            log(f"judged {done}/{len(wanted)} ({case.id})")

    try:
        await asyncio.gather(*(one(key, case, planned) for key, (case, planned) in wanted.items()))
    finally:
        await judge.aclose()
    return failures


# ---------------------------------------------------------------------------------- scoring


@dataclass(frozen=True, slots=True)
class RunScore:
    """What one run earned. Pure data: built by `score_run` from a record and its judgement."""

    config: str
    label: str
    case_id: str
    source: str
    rep: int
    errored: bool
    judged: bool
    parse_failed: bool
    must_hits: dict[str, bool]
    has_must: bool
    proposed: int
    planned: int
    quoted: int
    quotes_dropped: int
    matches: int
    acceptable: int
    violations: int
    expect_total: int
    expect_matched: int
    expect_correct: int
    latency_s: float | None
    tokens: int
    cost: float

    @property
    def scored(self) -> bool:
        return not self.errored and self.judged


def score_run(case: Case, record: RunRecord, entries: Sequence[JudgeEntry] | None) -> RunScore:
    """Score one run against its case.

    A must item is hit when at least `min_matches` distinct writes carry it (a write carries the
    item its judgement names in `match` or `also_matches`, and only when the status is "match").
    An expected action is correct when a write that carries its gold item has that action on
    exactly those targets. An output with writes but no judgement is `judged=False` and earns
    nothing; an empty output needs no judgement.
    """
    errored = record.status == "error"
    needs_judgement = bool(record.planned) and not errored
    carried: dict[str, set[int]] = defaultdict(set)
    matches = acceptable = violations = 0
    if not errored and entries is not None:
        for entry in entries:
            if entry.status == "violation":
                violations += 1
            elif entry.status == "acceptable":
                acceptable += 1
            else:
                matches += 1
                for gold in {entry.match, *entry.also_matches} - {None}:
                    carried[str(gold)].add(entry.index)
    expect_matched = expect_correct = 0
    for expected in case.expect_actions:
        writes = [
            record.planned[index]
            for index in sorted(carried.get(expected.gold, ()))
            if index < len(record.planned)
        ]
        if writes:
            expect_matched += 1
        if any(
            write.action == expected.action and set(write.target_ids) == set(expected.target_ids)
            for write in writes
        ):
            expect_correct += 1
    return RunScore(
        config=record.config,
        label=record.label,
        case_id=case.id,
        source=case.source,
        rep=record.rep,
        errored=errored,
        judged=entries is not None or not needs_judgement,
        parse_failed=record.parse_error is not None,
        must_hits={
            item.id: len(carried.get(item.id, ())) >= item.min_matches for item in case.must
        },
        has_must=bool(case.must),
        proposed=len(record.verdict),
        planned=len(record.planned),
        quoted=sum(1 for write in record.planned if write.quote is not None),
        quotes_dropped=record.quotes_dropped,
        matches=matches,
        acceptable=acceptable,
        violations=violations,
        expect_total=len(case.expect_actions),
        expect_matched=expect_matched,
        expect_correct=expect_correct,
        latency_s=None if errored else record.latency_s,
        tokens=0 if errored else record.total_tokens,
        cost=0.0 if errored or record.cost is None else record.cost,
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..1); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


@dataclass(slots=True)
class ConfigSummary:
    """Everything the summary table says about one (config, label), summed over its runs."""

    config: str
    label: str
    runs: int = 0
    errored: int = 0
    unjudged: int = 0
    scored: int = 0
    parse_failures: int = 0
    empty_runs: int = 0
    must_total: int = 0
    must_hit: int = 0
    must_total_real: int = 0
    must_hit_real: int = 0
    must_total_synthetic: int = 0
    must_hit_synthetic: int = 0
    violation_runs: int = 0
    scored_no_must: int = 0
    violation_runs_no_must: int = 0
    violations: int = 0
    acceptable: int = 0
    matched_writes: int = 0
    planned_writes: int = 0
    dropped_writes: int = 0
    quoted_writes: int = 0
    quotes_dropped: int = 0
    expect_total: int = 0
    expect_matched: int = 0
    expect_correct: int = 0
    latency_mean: float | None = None
    latency_p95: float | None = None
    tokens: int = 0
    cost: float = 0.0

    @property
    def parse_failure_rate(self) -> float | None:
        completed = self.runs - self.errored
        return self.parse_failures / completed if completed else None

    @property
    def must_recall(self) -> float | None:
        return self.must_hit / self.must_total if self.must_total else None

    @property
    def quote_rate(self) -> float | None:
        """The share of planned writes that kept the owner's words behind them."""
        return self.quoted_writes / self.planned_writes if self.planned_writes else None


def aggregate(scores: Sequence[RunScore]) -> list[ConfigSummary]:
    """One summary per (config, label), in matrix order then label.

    Quality metrics cover scored runs only (not errored, not awaiting a judgement); parse
    failures are scored runs that wrote nothing. Latency covers every run the model answered.
    """
    groups: dict[tuple[str, str], list[RunScore]] = defaultdict(list)
    for score in scores:
        groups[(score.config, score.label)].append(score)
    order = {name: index for index, name in enumerate(CONFIGS)}
    summaries: list[ConfigSummary] = []
    for (config, label), runs in sorted(
        groups.items(), key=lambda item: (item[0][1], order.get(item[0][0], len(order)), item[0][0])
    ):
        summary = ConfigSummary(config=config, label=label, runs=len(runs))
        latencies = [run.latency_s for run in runs if run.latency_s is not None]
        for run in runs:
            summary.errored += run.errored
            summary.unjudged += (not run.errored) and (not run.judged)
            summary.tokens += run.tokens
            summary.cost += run.cost
            if run.errored:
                continue
            summary.parse_failures += run.parse_failed
            if not run.scored:
                continue
            summary.scored += 1
            summary.empty_runs += (not run.parse_failed) and run.planned == 0
            hits = sum(run.must_hits.values())
            summary.must_total += len(run.must_hits)
            summary.must_hit += hits
            if run.source == "real":
                summary.must_total_real += len(run.must_hits)
                summary.must_hit_real += hits
            else:
                summary.must_total_synthetic += len(run.must_hits)
                summary.must_hit_synthetic += hits
            summary.violations += run.violations
            summary.violation_runs += run.violations > 0
            if not run.has_must:
                summary.scored_no_must += 1
                summary.violation_runs_no_must += run.violations > 0
            summary.acceptable += run.acceptable
            summary.matched_writes += run.matches
            summary.planned_writes += run.planned
            summary.dropped_writes += run.proposed - run.planned
            summary.quoted_writes += run.quoted
            summary.quotes_dropped += run.quotes_dropped
            summary.expect_total += run.expect_total
            summary.expect_matched += run.expect_matched
            summary.expect_correct += run.expect_correct
        if latencies:
            summary.latency_mean = sum(latencies) / len(latencies)
            summary.latency_p95 = percentile(latencies, 0.95)
        summaries.append(summary)
    return summaries


def must_item_table(
    scores: Sequence[RunScore],
) -> dict[tuple[str, str], dict[tuple[str, str], tuple[int, int]]]:
    """(case id, must id) -> (config, label) -> (runs that hit it, scored runs)."""
    table: dict[tuple[str, str], dict[tuple[str, str], tuple[int, int]]] = {}
    for score in scores:
        if not score.scored:
            continue
        for must_id, hit in score.must_hits.items():
            cell = table.setdefault((score.case_id, must_id), {})
            hits, runs = cell.get((score.config, score.label), (0, 0))
            cell[(score.config, score.label)] = (hits + hit, runs + 1)
    return table


@dataclass(frozen=True, slots=True)
class ViolationRow:
    config: str
    label: str
    case_id: str
    rep: int
    action: str
    kind: str
    body: str
    target_ids: list[str]
    reason: str


def collect_violations(
    case: Case, record: RunRecord, entries: Sequence[JudgeEntry] | None
) -> list[ViolationRow]:
    rows: list[ViolationRow] = []
    if record.status != "ok" or entries is None:
        return rows
    for entry in entries:
        if entry.status != "violation" or entry.index >= len(record.planned):
            continue
        write = record.planned[entry.index]
        rows.append(
            ViolationRow(
                config=record.config,
                label=record.label,
                case_id=case.id,
                rep=record.rep,
                action=write.action,
                kind=write.kind,
                body=write.body,
                target_ids=write.target_ids,
                reason=entry.reason,
            )
        )
    return rows


# ---------------------------------------------------------------------------------- summary


@dataclass(slots=True)
class Evaluation:
    """Everything `summary.md` and `summary.json` are rendered from."""

    summaries: list[ConfigSummary]
    must_table: dict[tuple[str, str], dict[tuple[str, str], tuple[int, int]]]
    must_items: dict[tuple[str, str], MustItem]
    violations: list[ViolationRow]
    errors: list[RunRecord]
    unjudged: list[RunScore]
    request_shapes: dict[str, list[str]]
    case_counts: dict[str, int]
    generated_at: str
    #: Saved runs on cases outside the case set this evaluation was given.
    left_out: int = 0


def evaluate(
    results: Mapping[RunKey, RunRecord],
    cases: Mapping[str, Case],
    judgements: Mapping[str, JudgementRecord],
) -> Evaluation:
    scores: list[RunScore] = []
    violations: list[ViolationRow] = []
    errors: list[RunRecord] = []
    shapes: dict[str, set[str]] = defaultdict(set)
    left_out = 0
    for key in sorted(results):
        record = results[key]
        case = cases.get(record.case_id)
        if case is None:
            left_out += 1
            continue
        entries: list[JudgeEntry] | None = None
        if record.planned:
            found = judgements.get(judge_key(case, record.planned))
            entries = found.judgements if found else None
        scores.append(score_run(case, record, entries))
        violations.extend(collect_violations(case, record, entries))
        if record.status == "error":
            errors.append(record)
        for wire in record.wire[-1:]:
            shapes[record.config].add(json.dumps(wire, sort_keys=True))
    used = {score.case_id for score in scores}
    return Evaluation(
        summaries=aggregate(scores),
        must_table=must_item_table(scores),
        must_items={(case.id, item.id): item for case in cases.values() for item in case.must},
        violations=violations,
        errors=errors,
        unjudged=[score for score in scores if not score.errored and not score.judged],
        request_shapes={config: sorted(found) for config, found in shapes.items()},
        case_counts={
            "total": len(used),
            "real": sum(1 for case_id in used if cases[case_id].source == "real"),
            "synthetic": sum(1 for case_id in used if cases[case_id].source == "synthetic"),
        },
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        left_out=left_out,
    )


def _ratio(numerator: int, denominator: int) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator:.0%})"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}s"


def _cell(text: str, limit: int = 0) -> str:
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if not limit or len(flat) <= limit else flat[: limit - 1] + "…"


def render_summary(evaluation: Evaluation, *, title: str = "Session Memory Pass eval") -> str:
    multiple_labels = len({summary.label for summary in evaluation.summaries}) > 1

    def column(config: str, label: str) -> str:
        return f"{config} [{label}]" if multiple_labels else config

    counts = evaluation.case_counts
    lines = [
        f"# {title}",
        "",
        f"Generated {evaluation.generated_at}. Cases scored: {counts['total']} "
        f"({counts['real']} real, {counts['synthetic']} synthetic). Each cell counts every "
        "rep; a must item is hit only by writes the judge marks `match`.",
        "",
        "## Configs",
        "",
        "| config | runs | recall | real | synthetic | no writes | parse fail | runs w/ violation "
        "| ... on no-must cases | violations | acceptable | actions ok | quoted | quotes dropped "
        "| lat mean | lat p95 | tokens | cost |",
        "|" + "---|" * 18,
    ]
    for s in evaluation.summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    column(s.config, s.label),
                    f"{s.runs - s.errored}" + (f" (+{s.errored} err)" if s.errored else ""),
                    _ratio(s.must_hit, s.must_total),
                    _ratio(s.must_hit_real, s.must_total_real),
                    _ratio(s.must_hit_synthetic, s.must_total_synthetic),
                    _ratio(s.empty_runs, s.scored),
                    _ratio(s.parse_failures, s.runs - s.errored),
                    _ratio(s.violation_runs, s.scored),
                    _ratio(s.violation_runs_no_must, s.scored_no_must),
                    str(s.violations),
                    str(s.acceptable),
                    _ratio(s.expect_correct, s.expect_total),
                    _ratio(s.quoted_writes, s.planned_writes),
                    str(s.quotes_dropped),
                    _seconds(s.latency_mean),
                    _seconds(s.latency_p95),
                    f"{s.tokens:,}",
                    f"${s.cost:.4f}" if s.cost else "n/a",
                ]
            )
            + " |"
        )
    lines += [
        "",
        "`recall` is must items hit over must items asked (real and synthetic cases split out); "
        "`no writes` is scored runs whose output planned nothing; `actions ok` is expected "
        "update/fuse actions met with the right targets; `quoted` is planned writes that kept the "
        "owner's words behind them, and `quotes dropped` is the quotes the model gave that the host "
        "refused as not the owner's. `err` runs are transport failures and are excluded from every "
        "rate.",
    ]
    if any(s.unjudged for s in evaluation.summaries):
        lines += [
            "",
            f"**{len(evaluation.unjudged)} run(s) with writes have no judgement yet and are not "
            "scored; run `--rescore` on this directory to judge them.**",
        ]

    columns = [(summary.config, summary.label) for summary in evaluation.summaries]
    lines += ["", "## Must items hit, per config (hits / reps)", ""]
    if evaluation.must_table:
        lines.append(
            "| case | must | description | " + " | ".join(column(*c) for c in columns) + " |"
        )
        lines.append("|" + "---|" * (3 + len(columns)))
        for (case_id, must_id), cell in sorted(
            evaluation.must_table.items(), key=lambda item: _natural(item[0])
        ):
            item = evaluation.must_items[(case_id, must_id)]
            need = f" (needs {item.min_matches})" if item.min_matches > 1 else ""
            cells = [f"{cell[c][0]}/{cell[c][1]}" if c in cell else "-" for c in columns]
            lines.append(
                f"| {case_id} | {must_id} | {_cell(item.description, 80)}{need} | "
                + " | ".join(cells)
                + " |"
            )
    else:
        lines.append("No must items were scored.")

    lines += ["", "## Violations", ""]
    if evaluation.violations:
        for row in evaluation.violations:
            targets = f" -> {','.join(row.target_ids)}" if row.target_ids else ""
            lines.append(
                f"- **{column(row.config, row.label)}** {row.case_id}#{row.rep} "
                f"`{row.action}/{row.kind}{targets}`: {_cell(row.body)} -- {_cell(row.reason)}"
            )
    else:
        lines.append("None.")

    if evaluation.errors:
        lines += ["", "## Transport errors (rerun the same command to retry them)", ""]
        lines.extend(
            f"- {column(record.config, record.label)} {record.case_id}#{record.rep}: "
            f"{_cell(record.error or '', 200)}"
            for record in evaluation.errors
        )

    lines += ["", "## Request shape observed", ""]
    for config, shapes in sorted(evaluation.request_shapes.items()):
        note = CONFIGS[config].note if config in CONFIGS else ""
        lines.append(f"- **{config}**: {'; '.join(shapes)}" + (f" ({note})" if note else ""))
    lines.append("")
    return "\n".join(lines)


def _natural(key: tuple[str, str]) -> list[Any]:
    return [
        int(part) if part.isdigit() else part
        for text in key
        for part in re.split(r"(\d+)", text)
        if part
    ]


def summary_payload(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "generated_at": evaluation.generated_at,
        "cases": evaluation.case_counts,
        "configs": [
            {
                **asdict(summary),
                "must_recall": summary.must_recall,
                "parse_failure_rate": summary.parse_failure_rate,
                "quote_rate": summary.quote_rate,
            }
            for summary in evaluation.summaries
        ],
        "must_items": [
            {
                "case": case_id,
                "must": must_id,
                "kind": evaluation.must_items[(case_id, must_id)].kind,
                "description": evaluation.must_items[(case_id, must_id)].description,
                "min_matches": evaluation.must_items[(case_id, must_id)].min_matches,
                "cells": {
                    f"{config}|{label}": {"hits": hits, "runs": runs}
                    for (config, label), (hits, runs) in sorted(cell.items())
                },
            }
            for (case_id, must_id), cell in sorted(
                evaluation.must_table.items(), key=lambda item: _natural(item[0])
            )
        ],
        "violations": [asdict(row) for row in evaluation.violations],
        "errors": [
            {
                "config": record.config,
                "label": record.label,
                "case": record.case_id,
                "rep": record.rep,
                "error": record.error,
            }
            for record in evaluation.errors
        ],
        "unjudged": [
            {"config": score.config, "label": score.label, "case": score.case_id, "rep": score.rep}
            for score in evaluation.unjudged
        ],
        "request_shapes": evaluation.request_shapes,
    }


def write_summary(out_dir: Path, evaluation: Evaluation, *, title: str) -> None:
    (out_dir / SUMMARY_MD).write_text(render_summary(evaluation, title=title), encoding="utf-8")
    (out_dir / SUMMARY_JSON).write_text(
        json.dumps(summary_payload(evaluation), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


async def finalize(
    out_dir: Path,
    cases: Mapping[str, Case],
    *,
    judge_factory: Callable[[], Verdicter],
    concurrency: int,
    title: str,
    log: Callable[[str], None] = lambda line: print(line, flush=True),
) -> tuple[Evaluation, list[str]]:
    """Judge what is missing, score everything saved, and write the summary."""
    results = load_results(out_dir / RESULTS_FILE)
    judgements = load_judgements(out_dir / JUDGEMENTS_FILE)
    failures = await judge_missing(
        results,
        cases,
        judgements,
        out_dir=out_dir,
        judge_factory=judge_factory,
        concurrency=concurrency,
        log=log,
    )
    evaluation = evaluate(results, cases, judgements)
    write_summary(out_dir, evaluation, title=title)
    return evaluation, failures


# ------------------------------------------------------------------------------------- CLI


def main_checkout(repo_root: Path = REPO_ROOT) -> Path:
    """The primary checkout, where the private case file and the operator's `.env` live.

    A git worktree has neither; its common git directory sits in the primary checkout.
    """
    try:
        common = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],  # noqa: S607
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except OSError, subprocess.SubprocessError:
        return repo_root
    path = Path(common)
    return path.parent if path.name == ".git" else repo_root


def private_dir() -> Path:
    return main_checkout() / "docs" / "research" / "memory-pass-eval"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the Session Memory Pass over fixed cases on several models and score it."
    )
    parser.add_argument(
        "--cases",
        action="append",
        type=Path,
        help="case file (repeatable); default: the committed synthetic cases and, when present, "
        "the private real cases",
    )
    parser.add_argument("--case-ids", default="", help="comma-separated case ids to keep")
    parser.add_argument(
        "--configs",
        default="",
        help=f"comma-separated configs; default all of: {', '.join(CONFIGS)}",
    )
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="seconds per model call"
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="replace the pass's system message with this file ({limit} -> the case's limit)",
    )
    parser.add_argument(
        "--label", help="tag for this run's prompt (default: prod, or the file name)"
    )
    parser.add_argument("--out", type=Path, help="output directory (resumable)")
    parser.add_argument("--rescore", type=Path, metavar="DIR", help="rebuild DIR's summary")
    parser.add_argument("--env-file", type=Path, help="dotenv holding the API keys")
    parser.add_argument(
        "--print-production-prompt",
        action="store_true",
        help="print the production system message (limit 5) and exit",
    )
    return parser


def default_case_paths(private: Path) -> list[Path]:
    paths = [SYNTHETIC_CASES]
    real = private / "real_cases.json"
    if real.is_file():
        paths.append(real)
    else:
        print(f"note: {real} not found; running the synthetic cases only", file=sys.stderr)
    return paths


def resolve_label(label: str | None, prompt_file: Path | None) -> str:
    chosen = label or (prompt_file.stem if prompt_file else PRODUCTION_LABEL)
    if not LABEL_PATTERN.match(chosen):
        raise ValueError(f"label {chosen!r} must match {LABEL_PATTERN.pattern}")
    return chosen


def record_invocation(out_dir: Path, entry: dict[str, Any]) -> None:
    """Append what this invocation asked for to `run_meta.json`, for the record."""
    path = out_dir / META_FILE
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        meta = {"invocations": []}
    meta["invocations"].append(entry)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def amain(args: argparse.Namespace) -> int:
    logging.getLogger("dlightrag.engine.ai.catalog").setLevel(logging.ERROR)
    private = private_dir()
    env_file = args.env_file or main_checkout() / ".env"
    case_paths = args.cases or default_case_paths(private)
    cases = load_cases(case_paths)
    wanted_ids = [part.strip() for part in args.case_ids.split(",") if part.strip()]
    if wanted_ids:
        unknown = [case_id for case_id in wanted_ids if case_id not in {c.id for c in cases}]
        if unknown:
            raise ValueError(f"unknown case id(s) {unknown}")
        cases = [case for case in cases if case.id in wanted_ids]
    by_id = {case.id: case for case in cases}

    secrets = load_secrets(env_file, {DEEPSEEK_KEY_ENV, DEEPSEEK_QUERY_KEY_ENV, OPENROUTER_KEY_ENV})
    secret_values = tuple(secrets.values())
    scheduler = ModelScheduler(max_concurrency=max(1, args.concurrency))
    factory = default_model_factory(secrets, timeout=args.timeout, scheduler=scheduler)

    def judge_factory() -> Verdicter:
        return factory(JUDGE_CONFIG, JUDGE_OUTPUT)  # raises when the OpenRouter key is absent

    if args.rescore:
        out_dir: Path = args.rescore
        if not out_dir.is_dir():
            raise ValueError(f"{out_dir} is not a directory")
        title = f"Session Memory Pass eval (rescored {datetime.now(UTC):%Y-%m-%d})"
        evaluation, failures = await finalize(
            out_dir, by_id, judge_factory=judge_factory, concurrency=args.concurrency, title=title
        )
        return _report(out_dir, evaluation, failures)

    configs = select_configs(args.configs)
    missing = sorted({c.key_env for c in (*configs, JUDGE_CONFIG)} - set(secrets))
    if missing:
        raise ValueError(f"missing API key(s) {missing}: set them in the environment or {env_file}")
    system_prompt: str | None = None
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
    label = resolve_label(args.label, args.system_prompt_file)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or private / "runs" / f"{label}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if system_prompt is not None:
        (out_dir / f"system_prompt.{label}.txt").write_text(system_prompt, encoding="utf-8")
    record_invocation(
        out_dir,
        {
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "label": label,
            "configs": [config.name for config in configs],
            "cases": [case.id for case in cases],
            "case_files": [str(path) for path in case_paths],
            "reps": args.reps,
            "concurrency": args.concurrency,
            "timeout": args.timeout,
            "system_prompt_file": str(args.system_prompt_file) if args.system_prompt_file else None,
            "judge": JUDGE_CONFIG.model,
        },
    )

    done = load_results(out_dir / RESULTS_FILE)
    pending = pending_runs(
        configs, cases, reps=args.reps, label=label, system_prompt=system_prompt, done=done
    )
    print(
        f"{len(pending)} run(s) to make, {len(configs) * len(cases) * args.reps - len(pending)} "
        f"already done; output {out_dir}",
        flush=True,
    )
    await run_matrix(
        pending,
        label=label,
        out_dir=out_dir,
        factory=factory,
        concurrency=args.concurrency,
        secrets=secret_values,
    )
    evaluation, failures = await finalize(
        out_dir,
        by_id,
        judge_factory=judge_factory,
        concurrency=args.concurrency,
        title=f"Session Memory Pass eval: {label}",
    )
    return _report(out_dir, evaluation, failures)


def _report(out_dir: Path, evaluation: Evaluation, failures: Sequence[str]) -> int:
    for failure in failures:
        print(f"judge failed: {failure}", file=sys.stderr)
    if evaluation.left_out:
        print(
            f"note: {evaluation.left_out} saved run(s) are on cases outside this command's case "
            "set and are left out of the summary",
            file=sys.stderr,
        )
    print(f"summary: {out_dir / SUMMARY_MD}", flush=True)
    return 1 if (failures or evaluation.errors or evaluation.unjudged) else 0


def production_prompt() -> str:
    """The production system message, rendered for a limit of 5."""
    case = Case(
        id="prompt",
        source="synthetic",
        turns=[CaseTurn(role="user", text="-")],
        record_limit=5,
    )
    return build_request(case).messages[0]["content"]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_production_prompt:
        sys.stdout.write(production_prompt())
        return 0
    try:
        return asyncio.run(amain(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
