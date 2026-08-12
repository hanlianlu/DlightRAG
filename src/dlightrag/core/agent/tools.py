# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, and where their observations land."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.core.agent.tool_loop import AgentTool, ToolResult
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.retrieval.web_search import (
    WebSearchResult,
    WebSearchUnavailable,
    web_context_rows,
)

KnowledgeRetrieval = Callable[[str], Awaitable[RetrievalResult]]
WebSearch = Callable[[str], Awaitable[WebSearchResult]]
RegisterWebSource = Callable[[str], str | None]


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1)


def build_run_tools(
    *,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    retrieve_knowledge_base: KnowledgeRetrieval,
    search_web: WebSearch | None,
    resource_tools: list[AgentTool],
    register_web_source: RegisterWebSource | None,
) -> tuple[list[AgentTool], _ToolCallCache]:
    """Bind one run's tools to its ledger; every observation lands there, not in a reply."""
    cache = _ToolCallCache()

    async def run_corpus_search(raw: BaseModel) -> ToolResult:
        args = _as(raw, SearchInput)
        return await cache.run(
            _call_key("knowledge_base", args.query),
            lambda: _search_corpus(retrieve_knowledge_base, args.query, evidence, trace),
        )

    tools = [
        AgentTool(
            "search_knowledge_base",
            "Search the indexed knowledge base for one concrete unresolved fact.",
            SearchInput,
            run_corpus_search,
        ),
    ]
    if search_web is not None:
        web = search_web

        async def run_web_search(raw: BaseModel) -> ToolResult:
            args = _as(raw, SearchInput)
            return await cache.run(
                _call_key("web", args.query),
                lambda: _search_open_web(web, args.query, evidence, trace, register_web_source),
            )

        tools.append(
            AgentTool(
                "search_web",
                "Search the open web for one concrete unresolved or current fact.",
                SearchInput,
                run_web_search,
            )
        )
    tools.extend(_ledger_backed(tool, evidence, cache) for tool in resource_tools)
    return tools, cache


def _ledger_backed(tool: AgentTool, evidence: EvidenceLedger, cache: _ToolCallCache) -> AgentTool:
    """Cache equivalent resource calls and land each observation in the ledger."""

    async def execute(raw: BaseModel) -> ToolResult:
        async def run_once() -> ToolResult:
            result = await tool.execute(raw)
            row = _resource_row(tool.name, result)
            if row is not None:
                evidence.add_rows([row])
                await evidence.aflush_images()
            return result

        return await cache.run(_resource_call_key(tool.name, raw), run_once)

    return AgentTool(tool.name, tool.description, tool.input_model, execute)


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
    return ToolResult(content=f"Knowledge base added {delta.new_chunks} new passages.")


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
    return ToolResult(content=content)


class _ToolCallCache:
    """Run each distinct tool call once, so a repeat costs a turn and not a search.

    This is execution bookkeeping, not memory the model reads: it keys on exact
    arguments, and the episode is what shows the model which angles are spent.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Future[ToolResult]] = {}
        self._closed = False

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks, self._tasks = list(self._tasks.values()), {}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(
        self,
        key: str,
        operation: Callable[[], Awaitable[ToolResult]],
    ) -> ToolResult:
        async with self._lock:
            if self._closed:
                raise RuntimeError("tool-call cache is closed")
            task = self._tasks.get(key)
            repeated = task is not None
            if task is None:
                task = asyncio.ensure_future(operation())
                self._tasks[key] = task
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                async with self._lock:
                    if self._tasks.get(key) is task:
                        self._tasks.pop(key, None)
            raise
        except BaseException:
            async with self._lock:
                if self._tasks.get(key) is task:
                    self._tasks.pop(key, None)
            raise
        if repeated:
            return ToolResult(
                content="Equivalent tool call already executed; no new evidence was added.",
                details=result.details,
            )
        return result


def _resource_row(tool_name: str, result: ToolResult) -> dict[str, Any] | None:
    """Project a resource observation into a citable, re-readable evidence row."""
    details = result.details or {}
    resource_id = str(details.get("resource_id") or "")
    if not resource_id or not result.content.strip():
        return None
    source_type = str(details.get("source_type") or "web_attachment")
    source_uri = str(details.get("source_uri") or resource_id)
    metadata = {
        "source_type": source_type,
        "source_uri": source_uri,
        "source_download_locator": str(details.get("source_download_locator") or source_uri),
        "title": str(details.get("title") or resource_id),
    }
    evidence_key = result.content
    if tool_name == "read_resource":
        content, marker, cursor = evidence_key.rpartition("\n[more text available; cursor=")
        if marker and cursor.endswith("]"):
            evidence_key = content
    identity = hashlib.sha256(f"{tool_name}\0{evidence_key}".encode()).hexdigest()[:16]
    return {
        "chunk_id": f"{resource_id}::{tool_name}::{identity}",
        "reference_id": resource_id,
        "full_doc_id": resource_id,
        "file_path": str(metadata.get("title") or resource_id),
        "content": evidence_key if tool_name == "read_resource" else result.content,
        "page_number": None,
        "_workspace": "__web_search__" if source_type == "web_search" else "__attachment__",
        "_evidence_key": f"{tool_name}:{identity}",
        "metadata": metadata,
    }


def _call_key(name: str, query: str) -> str:
    return f"{name}:{json.dumps(query.strip(), ensure_ascii=False)}"


def _resource_call_key(name: str, raw: BaseModel) -> str:
    payload = json.dumps(raw.model_dump(), ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{payload}"


def _as[T: BaseModel](value: BaseModel, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise TypeError(f"Expected {expected.__name__}, got {type(value).__name__}")
    return value


__all__ = ["KnowledgeRetrieval", "SearchInput", "WebSearch", "build_run_tools"]
