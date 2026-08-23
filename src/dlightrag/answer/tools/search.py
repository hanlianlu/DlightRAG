# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The two retrieval tools a run offers, and where their evidence lands.

Both tools answer one narrow question per call and return only how much new
evidence arrived; the passages themselves land in the run's ledger, never in the
reply the model reads back.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.agent.tools import AgentTool, ToolResult, ToolRuntime
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.tools.web import (
    WebSearchResult,
    WebSearchUnavailable,
    web_context_rows,
)
from dlightrag.rag.retrieval import RetrievalResult

KnowledgeRetrieval = Callable[[str], Awaitable[RetrievalResult]]
WebSearch = Callable[[str], Awaitable[WebSearchResult]]
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


def knowledge_base_search_tool(
    *,
    retrieve: KnowledgeRetrieval,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
) -> AgentTool:
    async def execute(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = _as(raw, SearchInput)
        return await _search_corpus(retrieve, args.query, evidence, trace)

    return AgentTool(
        "search_knowledge_base",
        "Search the indexed knowledge base for one concrete unresolved fact.",
        SearchInput,
        execute,
    )


def web_search_tool(
    *,
    search: WebSearch,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    register_web_source: RegisterWebSource | None,
) -> AgentTool:
    async def execute(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = _as(raw, SearchInput)
        return await _search_open_web(search, args.query, evidence, trace, register_web_source)

    return AgentTool(
        "search_web",
        "Search the open web for one concrete unresolved or current fact.",
        SearchInput,
        execute,
    )


async def _search_corpus(
    retrieve: KnowledgeRetrieval,
    query: str,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
) -> ToolResult:
    try:
        result = await retrieve(query)
    except Exception as exc:
        raise RuntimeError("knowledge-base search failed") from exc
    delta = evidence.add_contexts(result.contexts)
    await evidence.aflush_images()
    retrievals = trace.setdefault("knowledge_base_retrievals", [])
    if isinstance(retrievals, list):
        retrievals.append({**result.trace, "query": query})
    return ToolResult.text(f"Knowledge base added {delta.new_chunks} new passages.")


async def _search_open_web(
    search: WebSearch,
    query: str,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    register_web_source: RegisterWebSource | None,
) -> ToolResult:
    try:
        result = await search(query)
    except WebSearchUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError("open-web search failed") from exc
    rows = web_context_rows(result.hits)
    readable_sources: dict[str, str] = {}
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
                readable_sources.setdefault(resource_id, str(metadata.get("title") or "Source"))
    delta = evidence.add_rows(rows)
    await evidence.aflush_images()
    trace["web_search_cost_dollars"] += result.cost_dollars
    content = f"Open web added {delta.new_chunks} new passages."
    if delta.new_chunks and readable_sources:
        content += "\nResource handles:\n" + "\n".join(
            f"- {title} [resource: {resource_id}]"
            for resource_id, title in readable_sources.items()
        )
    return ToolResult.text(content)


def _as[T: BaseModel](value: BaseModel, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


__all__ = [
    "KnowledgeRetrieval",
    "RegisterWebSource",
    "SearchInput",
    "WebSearch",
    "knowledge_base_search_tool",
    "web_search_tool",
]
