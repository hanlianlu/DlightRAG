# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The two retrieval tools a run offers, and where their evidence lands.

Both tools answer one narrow question per call and return only how much new
evidence arrived; the passages themselves land in the run's ledger, and the
Runtime then freezes that run's newly admitted evidence into the Tool result the
model reads back, so a passage reaches the model once, where it arrived.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import date
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolDeclaration,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.web_search import web_context_rows
from dlightrag.engine.answer.web_sources.contracts import (
    WebSearchHit,
    WebSearchRequest,
    WebSearchResult,
    WebSourceUnavailable,
)
from dlightrag.engine.rag.retrieval.results import (
    RetrievalResult,
)

KnowledgeRetrieval = Callable[[str], Awaitable[RetrievalResult]]
WebSearch = Callable[[WebSearchRequest], Awaitable[WebSearchResult]]
RegisterWebSource = Callable[[str], str | None]


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(
        min_length=1,
        description=(
            "One concrete unresolved question or fact to search for, in natural "
            "language. Search one angle per call rather than combining several."
        ),
    )


class WebSearchInput(SearchInput):
    """Provider-neutral controls accepted by every configured search adapter."""

    # Overrides the knowledge base's wording: a hosted engine takes a short run of key terms,
    # and one adapter caps the whole query at 128 characters.
    query: str = Field(
        min_length=1,
        description=(
            "A short search query of key terms, under about 100 characters. Search one "
            "angle per call rather than combining several."
        ),
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of results; each adds its page's passages to your context.",
    )
    include_domains: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description="Limits results to these hostnames, such as who.int; no scheme or path.",
    )
    exclude_domains: tuple[str, ...] = Field(
        default=(),
        max_length=20,
        description="Leaves out results from these hostnames; no scheme or path.",
    )
    start_date: str | None = Field(
        default=None,
        description="Limits results to pages published on or after this date (YYYY-MM-DD).",
    )
    end_date: str | None = Field(
        default=None,
        description="Limits results to pages published on or before this date (YYYY-MM-DD).",
    )
    effort: Literal["fast", "balanced"] = Field(
        default="balanced",
        description="Search depth: fast is quickest; balanced trades some latency for relevance.",
    )

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def _domains_only(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            domain = value.lower().strip().rstrip(".")
            if not domain or "://" in domain or "/" in domain or any(ch.isspace() for ch in domain):
                raise ValueError("domains must be hostnames without scheme or path")
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @field_validator("start_date", "end_date")
    @classmethod
    def _iso_date(cls, value: str | None) -> str | None:
        if value is not None:
            date.fromisoformat(value)
        return value

    @model_validator(mode="after")
    def _coherent_filters(self) -> WebSearchInput:
        overlap = set(self.include_domains) & set(self.exclude_domains)
        if overlap:
            raise ValueError("the same domain cannot be both included and excluded")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date cannot be after end_date")
        return self

    def request(self) -> WebSearchRequest:
        return WebSearchRequest(**self.model_dump())


def knowledge_base_search_declaration() -> ToolDeclaration:
    # Mechanics only. Workspaces are created at runtime and each request selects its
    # own, while a tool definition is provider prefix-cache input the Run pins at
    # acceptance, so no workspace name or content claim belongs in this text.
    return ToolDeclaration(
        "search_knowledge_base",
        "Search the knowledge-base workspaces selected for this conversation. Each "
        "passage it returns names the document it came from.",
        SearchInput,
        read_only=True,
    )


def knowledge_base_search_tool(
    *,
    retrieve: KnowledgeRetrieval,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = _as(raw, SearchInput)
        await runtime.emit_update(ToolResult.text("", subject=args.query))
        return await _search_corpus(retrieve, args.query, evidence, trace, runtime)

    return knowledge_base_search_declaration().bind(execute)


def web_search_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        "search_web",
        "Search the open web for one concrete unresolved or current fact, source page, "
        "document, image, or file. Results are page excerpts. Use it before claiming "
        "open-web search is unavailable.",
        WebSearchInput,
        contract_version=2,
        read_only=True,
    )


def web_search_tool(
    *,
    search: WebSearch,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    register_web_source: RegisterWebSource | None,
    browser_available: bool = False,
) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = _as(raw, WebSearchInput)
        await runtime.emit_update(ToolResult.text("", subject=args.query))
        return await _search_open_web(
            search,
            args.request(),
            evidence,
            trace,
            register_web_source,
            runtime,
            browser_available=browser_available,
        )

    return web_search_declaration().bind(execute)


async def _search_corpus(
    retrieve: KnowledgeRetrieval,
    query: str,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    runtime: ToolRuntime,
) -> ToolResult:
    try:
        result = await retrieve(query)
    except Exception as exc:
        raise RuntimeError("knowledge-base search failed") from exc
    await runtime.in_source_order()
    delta = evidence.add_contexts(result.contexts)
    await evidence.aflush_images()
    retrievals = trace.setdefault("knowledge_base_retrievals", [])
    if isinstance(retrievals, list):
        retrievals.append({**result.trace, "query": query})
    content = f"Knowledge base added {delta.new_chunks} new passages."
    if delta.dropped_rows:
        content += f" Dropped {delta.dropped_rows} unusable evidence item(s)."
    searched = _searched_workspaces(result.trace, result.contexts.get("chunks") or ())
    return ToolResult.text(
        f"{content} {searched}" if searched else content,
        # The query is already the row's bounded subject; repeating it here would put the
        # model's own words on a reader surface through a second carrier (ADR 0061).
        facts={
            "new_chunks": delta.new_chunks,
            "dropped_rows": delta.dropped_rows,
        },
    )


def _searched_workspaces(trace: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]) -> str:
    """Each workspace the search covered and what it gave, as the retrieval trace has it.

    The tool's description names no workspace (its text is cached and pinned while
    workspaces come and go), so the scope reaches the model in the results instead,
    with the workspaces that hold nothing and the ones whose search failed.
    """
    workspaces = [str(workspace) for workspace in trace.get("workspaces") or ()]
    if not workspaces:
        return ""
    per_workspace = trace.get("per_workspace")
    if isinstance(per_workspace, Mapping):
        traces: Mapping[str, Any] = per_workspace
        counts = trace.get("per_workspace_chunk_count") or {}
    else:
        traces, counts = {workspaces[0]: trace}, {workspaces[0]: len(chunks)}
    parts = [
        f"{workspace} (no published documents)"
        if isinstance(traces.get(workspace), Mapping) and traces[workspace].get("workspace_empty")
        else f"{workspace} ({counts.get(workspace, 0)} passages)"
        for workspace in workspaces
    ]
    parts.extend(
        f"{workspace} (search failed)" for workspace in trace.get("failed_workspaces") or ()
    )
    return f"Searched {', '.join(parts)}."


#: How many Web search hits a result states as structured sources. Enough for a step card to
#: list what the search found without repeating the passages the result already carries.
_SEARCH_FACT_SOURCES = 10


def _search_source_fact(hit: WebSearchHit) -> dict[str, Any]:
    """One Web search hit as the bounded fact a step card lists: where it came from, not its text."""
    fact: dict[str, Any] = {}
    if hit.url:
        fact["url"] = hit.url
        domain = urlparse(hit.url).hostname
        if domain:
            fact["domain"] = domain
    # The title is the page's words, not a structural fact about where the hit came from: a
    # reader gets it from the passage the result already carries, through the content gate.
    if hit.published_date:
        fact["published_at"] = hit.published_date
    return fact


def _web_search_facts(result: WebSearchResult) -> dict[str, Any]:
    """The structure a settled Web search states: who answered, what it cost, and what it found."""
    facts: dict[str, Any] = {
        "provider": result.provider,
        "hits": len(result.hits),
        "degraded": result.degradation is not None,
        "cost": round(result.cost_dollars, 6),
    }
    # A provider may cut several passages from one page; a card states the page once.
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in result.hits:
        if len(sources) >= _SEARCH_FACT_SOURCES:
            break
        if hit.url in seen:
            continue
        seen.add(hit.url)
        sources.append(_search_source_fact(hit))
    if sources:
        facts["sources"] = sources
    return facts


async def _search_open_web(
    search: WebSearch,
    request: WebSearchRequest,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    register_web_source: RegisterWebSource | None,
    runtime: ToolRuntime,
    *,
    browser_available: bool,
) -> ToolResult:
    try:
        result = await search(request)
    except WebSourceUnavailable as exc:
        return _open_web_unavailable(exc, browser_available=browser_available)
    except Exception as exc:
        raise RuntimeError("open-web search failed") from exc
    await runtime.in_source_order()
    rows = web_context_rows(result.hits)
    if register_web_source is not None:
        resources_by_url: dict[str, str | None] = {}
        for row in rows:
            metadata = row.get("metadata") or {}
            url = str(metadata.get("source_uri") or "")
            if url not in resources_by_url:
                resources_by_url[url] = register_web_source(url)
            resource_id = resources_by_url[url]
            if resource_id is not None:
                metadata["resource_id"] = resource_id
    delta = evidence.add_rows(rows)
    await evidence.aflush_images()
    trace["web_search_cost_dollars"] += result.cost_dollars
    content = f"Open web added {delta.new_chunks} new passages."
    if delta.dropped_rows:
        content += f" Dropped {delta.dropped_rows} unusable evidence item(s)."
    if result.dropped_results:
        content += f" Dropped {result.dropped_results} malformed result(s)."
    if result.degradation:
        content += f"\n{result.degradation}"
    return ToolResult.text(
        content,
        telemetry={
            "web_search_provider": result.provider,
            "web_search_hits": len(result.hits),
            "web_search_degraded": result.degradation is not None,
            "web_search_cost_dollars": round(result.cost_dollars, 6),
        },
        facts=_web_search_facts(result),
    )


def _open_web_unavailable(
    failure: WebSourceUnavailable,
    *,
    browser_available: bool,
) -> ToolResult:
    """What the model is told when no provider of the Search chain answered.

    The failure is reported, not raised: the Run should read it as one failed call with something
    to do next, and it is the caller here — not the provider layer — that knows what else this Run
    can reach. A Run with the Agent Browser is pointed at it, because a search engine's own page is
    a search this product can still make.
    """
    next_step = (
        " The Agent Browser is available in this Run: drive a search engine's page with the browser "
        "tool, or read a URL you already know with read."
        if browser_available
        else " Answer from the evidence you already have, or read a URL you already know with read."
    )
    return ToolResult.text(
        f"Open-web search is unavailable: {failure}.{next_step}",
        is_error=True,
        telemetry={
            "web_search_provider": "",
            "web_search_hits": 0,
            "web_search_degraded": True,
            "web_search_cost_dollars": 0.0,
        },
        facts={"provider": "", "hits": 0, "degraded": True, "cost": 0.0},
    )


def _as[T: BaseModel](value: BaseModel, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


__all__ = [
    "KnowledgeRetrieval",
    "RegisterWebSource",
    "SearchInput",
    "WebSearchInput",
    "WebSearch",
    "knowledge_base_search_tool",
    "web_search_tool",
]
