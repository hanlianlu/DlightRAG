# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, composed per run and never globally."""

import hashlib
from collections import Counter
from typing import Any

from pydantic import BaseModel

from dlightrag.agent.environment import AccessScheduler
from dlightrag.agent.environment.local import LocalExecutionEnvironment
from dlightrag.agent.tools import AgentTool, ToolResult
from dlightrag.agent.tools.files import path_tools, read_tool
from dlightrag.answer.errors import InvalidToolConfigurationError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.tools.delegate import DelegateHost, delegate_research_tool
from dlightrag.answer.tools.memory import MemoryHost, forget_tool, recall_memory_tool, remember_tool
from dlightrag.answer.tools.search import (
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
    resource_reader: Any | None = None,
    environment: LocalExecutionEnvironment | None = None,
    scheduler: AccessScheduler | None = None,
    spill: Any | None = None,
    ripgrep: str = "rg",
    delegate_host: DelegateHost | None = None,
    memory_host: MemoryHost | None = None,
    child: bool = False,
) -> list[AgentTool]:
    """Bind one run's tools to its ledger. Path tools appear only with an environment."""
    access = scheduler or AccessScheduler()
    tools = [
        knowledge_base_search_tool(
            retrieve=retrieve_knowledge_base,
            evidence=evidence,
            trace=trace,
        )
    ]
    if search_web is not None:
        tools.append(
            web_search_tool(
                search=search_web,
                evidence=evidence,
                trace=trace,
                register_web_source=register_web_source,
            )
        )
    if resource_reader is not None:
        tools.append(read_tool(environment, access, resource_reader=resource_reader, spill=spill))
    else:
        tools.extend(
            _ledger_backed(tool, evidence) for tool in resource_tools if tool.name == "read"
        )
    tools.extend(
        _ledger_backed(tool, evidence) for tool in resource_tools if tool.name == "inspect"
    )
    if environment is not None:
        path = path_tools(environment, scheduler=access, ripgrep=ripgrep, spill=spill)
        if child:
            keep = {"read", "grep"} if resource_reader is None else {"grep"}
            extras = [tool for tool in path if tool.name in keep]
        else:
            extras = [tool for tool in path if tool.name != "read"]
        tools.extend(extras)
    if delegate_host is not None and not child:
        tools.append(delegate_research_tool(host=delegate_host))
    if memory_host is not None and not child:
        tools.extend(
            (
                remember_tool(host=memory_host),
                forget_tool(host=memory_host),
                recall_memory_tool(host=memory_host),
            )
        )
    _reject_duplicate_names(tools)
    return tools


def _reject_duplicate_names(tools: list[AgentTool]) -> None:
    counts = Counter(tool.name for tool in tools)
    duplicates = tuple(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise InvalidToolConfigurationError(duplicates)


def _ledger_backed(tool: AgentTool, evidence: EvidenceLedger) -> AgentTool:
    async def execute(raw: BaseModel) -> ToolResult:
        result = await tool.execute(raw)
        row = _resource_row(tool.name, result)
        if row is not None:
            evidence.add_rows([row])
            await evidence.aflush_images()
        return result

    return AgentTool(
        tool.name,
        tool.description,
        tool.input_model,
        execute,
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
    )


def _resource_row(tool_name: str, result: ToolResult) -> dict[str, Any] | None:
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
    if tool_name == "read":
        content, marker, cursor = evidence_key.rpartition("\n[more text available; cursor=")
        if marker and cursor.endswith("]"):
            evidence_key = content
    identity = hashlib.sha256(f"{tool_name}\0{evidence_key}".encode()).hexdigest()[:16]
    return {
        "chunk_id": f"{resource_id}::{tool_name}::{identity}",
        "reference_id": resource_id,
        "full_doc_id": resource_id,
        "file_path": str(metadata.get("title") or resource_id),
        "content": evidence_key if tool_name == "read" else result.content,
        "page_number": None,
        "_workspace": "__web_search__" if source_type == "web_search" else "__attachment__",
        "_evidence_key": f"{tool_name}:{identity}",
        "metadata": metadata,
    }


__all__ = ["compose_research_tools"]
