# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, composed per run and never globally."""

import hashlib
import json
from collections import Counter
from typing import Any

from pydantic import BaseModel

from dlightrag.core.answer.errors import InvalidToolConfigurationError
from dlightrag.core.memory.evidence import EvidenceLedger
from dlightrag.core.tools.cache import ExactCallCache
from dlightrag.core.tools.models import AgentTool, ToolResult
from dlightrag.core.tools.search import (
    KnowledgeRetrieval,
    RegisterWebSource,
    WebSearch,
    knowledge_base_search_tool,
    web_search_tool,
)


def compose_research_tools(
    *,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    retrieve_knowledge_base: KnowledgeRetrieval,
    search_web: WebSearch | None,
    resource_tools: list[AgentTool],
    register_web_source: RegisterWebSource | None,
) -> tuple[list[AgentTool], ExactCallCache]:
    """Bind one run's tools to its ledger; every observation lands there, not in a reply."""
    cache = ExactCallCache()
    tools = [
        knowledge_base_search_tool(
            retrieve=retrieve_knowledge_base,
            evidence=evidence,
            trace=trace,
            cache=cache,
        )
    ]
    if search_web is not None:
        tools.append(
            web_search_tool(
                search=search_web,
                evidence=evidence,
                trace=trace,
                register_web_source=register_web_source,
                cache=cache,
            )
        )
    tools.extend(_ledger_backed(tool, evidence, cache) for tool in resource_tools)
    _reject_duplicate_names(tools)
    return tools, cache


def _reject_duplicate_names(tools: list[AgentTool]) -> None:
    """Fail the run before any model call when two peer tools share a name.

    A tool name is the model's only handle on a tool, so a collision silently
    hides one of them behind the other.
    """
    counts = Counter(tool.name for tool in tools)
    duplicates = tuple(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise InvalidToolConfigurationError(duplicates)


def _ledger_backed(tool: AgentTool, evidence: EvidenceLedger, cache: ExactCallCache) -> AgentTool:
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


def _resource_call_key(name: str, raw: BaseModel) -> str:
    payload = json.dumps(raw.model_dump(), ensure_ascii=False, sort_keys=True, default=str)
    return f"{name}:{payload}"


__all__ = ["compose_research_tools"]
