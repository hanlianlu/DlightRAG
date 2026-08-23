# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, composed per run and never globally."""

import hashlib
from typing import Any

from pydantic import BaseModel

from dlightrag.agent.environment import AccessScheduler
from dlightrag.agent.environment.execution import ExecutionEnvironment
from dlightrag.agent.skills import SkillCatalog, load_skill_tool
from dlightrag.agent.tools import AgentTool, ToolResult
from dlightrag.agent.tools.files import path_tools, read_tool
from dlightrag.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.answer.errors import InvalidToolConfigurationError
from dlightrag.answer.evidence import EvidenceLedger
from dlightrag.answer.tools.memory import MemoryHost, forget_tool, recall_memory_tool, remember_tool
from dlightrag.answer.tools.search import (
    KnowledgeRetrieval,
    RegisterWebSource,
    WebSearch,
    knowledge_base_search_tool,
    web_search_tool,
)
from dlightrag.answer.tools.subagents import SubagentHost, subagent_tools


def compose_research_tools(
    *,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    retrieve_knowledge_base: KnowledgeRetrieval,
    search_web: WebSearch | None,
    resource_tools: list[AgentTool],
    register_web_source: RegisterWebSource | None,
    resource_reader: Any | None = None,
    environment: ExecutionEnvironment | None = None,
    scheduler: AccessScheduler | None = None,
    spill: Any | None = None,
    ripgrep: str = "rg",
    subagent_host: SubagentHost | None = None,
    memory_host: MemoryHost | None = None,
    skill_catalog: SkillCatalog | None = None,
    child: bool = False,
    tool_names: tuple[str, ...] | None = None,
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
        tools.append(
            _ledger_backed(
                read_tool(environment, access, resource_reader=resource_reader, spill=spill),
                evidence,
            )
        )
    else:
        tools.extend(
            _ledger_backed(tool, evidence) for tool in resource_tools if tool.name == "read"
        )
    tools.extend(
        _ledger_backed(tool, evidence) for tool in resource_tools if tool.name == "inspect"
    )
    tools.extend(tool for tool in resource_tools if tool.name not in {"read", "inspect"})
    if environment is not None:
        path = path_tools(environment, scheduler=access, ripgrep=ripgrep, spill=spill)
        existing_names = {tool.name for tool in tools}
        tools.extend(tool for tool in path if tool.name not in existing_names)
    if subagent_host is not None:
        tools.extend(subagent_tools(host=subagent_host))
    if memory_host is not None:
        tools.extend(
            (
                remember_tool(host=memory_host),
                forget_tool(host=memory_host),
                recall_memory_tool(host=memory_host),
            )
        )
    if skill_catalog is not None and skill_catalog.metadata:
        tools.append(load_skill_tool(skill_catalog))
    try:
        registry = ToolRegistry(tools)
        return list(
            registry.resolve(
                tool_names,
                exclude={"spawn_agent"} if child else (),
            )
        )
    except DuplicateToolError as exc:
        raise InvalidToolConfigurationError(exc.names) from exc


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
