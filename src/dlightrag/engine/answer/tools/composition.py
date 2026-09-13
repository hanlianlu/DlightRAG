# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The peer tools one research run offers, composed per run and never globally."""

import hashlib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.environment.errors import FullOutputUnavailable
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.environment.toolchain import SearchToolchain
from dlightrag.engine.agent.tool_content import ToolTextPart, tool_content_attachments
from dlightrag.engine.agent.tools import AgentTool, ToolResult, ToolRuntime
from dlightrag.engine.agent.tools.files import (
    ImagePreparer,
    ResourceViewer,
    SpillWriter,
    path_tools,
    preview_or_spill,
    read_tool,
    view_tool,
)
from dlightrag.engine.agent.tools.registry import DuplicateToolError, ToolRegistry
from dlightrag.engine.answer.errors import InvalidToolConfigurationError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.tools.artifacts import attach_artifact_tool
from dlightrag.engine.answer.tools.memory import (
    MemoryHost,
    forget_tool,
    recall_memory_tool,
    remember_tool,
)
from dlightrag.engine.answer.tools.search import (
    KnowledgeRetrieval,
    RegisterWebSource,
    WebSearch,
    knowledge_base_search_tool,
    web_search_tool,
)
from dlightrag.engine.answer.tools.subagents import (
    SubagentHost,
    child_guidance_tools,
    subagent_tools,
)


def compose_research_tools(
    *,
    evidence: EvidenceLedger,
    trace: dict[str, Any],
    retrieve_knowledge_base: KnowledgeRetrieval,
    search_web: WebSearch | None,
    injected_tools: list[AgentTool],
    register_web_source: RegisterWebSource | None,
    resource_reader: Any | None = None,
    resource_viewer: ResourceViewer | None = None,
    environment: ExecutionEnvironment | None = None,
    scheduler: AccessScheduler | None = None,
    spill: Any | None = None,
    output_stage_factory: Any | None = None,
    artifacts_root: Path | None = None,
    publication_limits: PublicationLimits | None = None,
    ripgrep: str = "rg",
    search_toolchain: SearchToolchain | None = None,
    image_preparer: ImagePreparer | None = None,
    subagent_host: SubagentHost | None = None,
    memory_host: MemoryHost | None = None,
    skill_tools: list[AgentTool] | None = None,
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
                read_tool(
                    environment,
                    access,
                    resource_reader=resource_reader,
                    spill=spill,
                    image_preparer=image_preparer,
                ),
                evidence,
            )
        )
    if resource_viewer is not None or environment is not None:
        tools.append(
            _ledger_backed(
                view_tool(
                    environment,
                    access,
                    resource_viewer=resource_viewer,
                    image_preparer=image_preparer,
                ),
                evidence,
            )
        )
    tools.extend(_bounded_injected_result(tool, spill) for tool in injected_tools)
    if environment is not None:
        path = path_tools(
            environment,
            scheduler=access,
            ripgrep=ripgrep,
            search_toolchain=search_toolchain,
            image_preparer=image_preparer,
            spill=spill,
            output_stage_factory=output_stage_factory,
        )
        existing_names = {tool.name for tool in tools}
        tools.extend(tool for tool in path if tool.name not in existing_names)
        if artifacts_root is not None and not child:
            tools.append(
                attach_artifact_tool(
                    artifacts_root,
                    scheduler=access,
                    limits=publication_limits or PublicationLimits(),
                )
            )
    if subagent_host is not None:
        if child:
            if subagent_host.async_lifecycle and subagent_host.interactive_controls:
                tools.extend(child_guidance_tools(host=subagent_host))
        else:
            tools.extend(subagent_tools(host=subagent_host))
    if memory_host is not None:
        tools.extend(
            (
                remember_tool(host=memory_host),
                forget_tool(host=memory_host),
                recall_memory_tool(host=memory_host),
            )
        )
    if skill_tools:
        # Tool membership is pinned before a run workspace exists. Keep the
        # contract stable even when this particular catalog is empty; execution
        # then returns an ordinary not-found result rather than changing the Plan.
        tools.extend(skill_tools)
    try:
        registry = ToolRegistry(tools)
        selected_names = tool_names
        if (
            child
            and selected_names is None
            and (subagent_host is None or subagent_host.async_lifecycle)
        ):
            read_only = {
                "search_knowledge_base",
                "search_web",
                "read",
                "view",
                "grep",
                "find",
                "ls",
                "recall_memory",
                "load_skill",
                "ask_parent",
            }
            selected_names = tuple(tool.name for tool in tools if tool.name in read_only)
        elif child and any(tool.name == "ask_parent" for tool in tools):
            selected_names = tuple(dict.fromkeys((*(selected_names or ()), "ask_parent")))
        return list(
            registry.resolve(
                selected_names,
                exclude={
                    "spawn_agent",
                    "subagent_status",
                    "wait_subagent",
                    "cancel_subagent",
                    "steer_subagent",
                    "continue_subagent",
                    "reply_subagent",
                    "remember",
                    "forget",
                }
                if child
                else (),
            )
        )
    except DuplicateToolError as exc:
        raise InvalidToolConfigurationError(exc.names) from exc


def _bounded_injected_result(tool: AgentTool, spill: SpillWriter | None) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        result = await tool.execute(raw, runtime)
        # Existing cursor/spill results already own their continuation. Runtime
        # fitting retains protected text and attachments without duplicating it.
        if result.protected_text or result.effects.committed_outputs:
            return result
        try:
            text, receipt = await preview_or_spill(result.text_content, spill=spill, tool=tool.name)
        except FullOutputUnavailable, OSError:
            return replace(
                result,
                parts=(
                    ToolTextPart(
                        f"Tool {tool.name} completed but its full output is unavailable. Do not retry automatically; identify the unavailable part in the final Answer."
                    ),
                    *tool_content_attachments(result.parts),
                ),
                is_error=True,
            )
        if receipt is None:
            return result
        return replace(
            result,
            parts=(ToolTextPart(text), *tool_content_attachments(result.parts)),
            protected_text=f"Full output: read(resource_id={receipt.resource_id!r}, cursor=...)",
            effects=replace(
                result.effects, committed_outputs=(*result.effects.committed_outputs, receipt)
            ),
        )

    return replace(tool, execute=execute)


def _ledger_backed(tool: AgentTool, evidence: EvidenceLedger) -> AgentTool:
    async def execute(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        result = await tool.execute(raw, runtime)
        rows = _resource_rows(tool.name, result)
        if rows:
            evidence.add_rows(rows)
            await evidence.aflush_images()
        return result

    return AgentTool(
        tool.name,
        tool.description,
        tool.input_model,
        execute,
        replay_policy=tool.replay_policy,
        contract_version=tool.contract_version,
        guidance=tool.guidance,
    )


def _resource_rows(tool_name: str, result: ToolResult) -> list[dict[str, Any]]:
    if not result.effects.evidence_sources or not result.text_content.strip():
        return []
    source = result.effects.evidence_sources[0]
    resource_id = source.resource_id
    source_type = source.source_type
    source_uri = source.source_uri
    metadata = {
        "source_type": source_type,
        "source_uri": source_uri,
        "source_download_locator": source_uri,
        "title": source.title,
        **dict(source.attributes),
    }
    evidence_key = result.text_content
    if tool_name == "read":
        content, marker, cursor = evidence_key.rpartition("\n[more text available; cursor=")
        if marker and cursor.endswith("]"):
            evidence_key = content
    identity = hashlib.sha256(f"{tool_name}\0{evidence_key}".encode()).hexdigest()[:16]
    row = {
        "chunk_id": f"{resource_id}::{tool_name}::{identity}",
        "reference_id": resource_id,
        "full_doc_id": resource_id,
        "file_path": str(metadata.get("title") or resource_id),
        "content": evidence_key if tool_name == "read" else result.text_content,
        "page_number": None,
        "_workspace": "__web_search__" if source_type == "web_search" else "__attachment__",
        "_evidence_key": f"{tool_name}:{identity}",
        "metadata": metadata,
    }
    if tool_name != "view":
        return [row]
    rows = []
    for attachment in tool_content_attachments(result.parts):
        source = attachment.source
        if source is None:
            raise ValueError("visual evidence requires exact source provenance")
        rows.append(
            {
                **row,
                "chunk_id": f"{source.resource_id}::view::{attachment.resource_id}",
                "_evidence_key": attachment.resource_id,
                "content": f"Viewed pixels: {attachment.safe_name}",
                "page_number": source.page,
                "metadata": {
                    **metadata,
                    "visual_source": asdict(source),
                    "content_digest": attachment.content_digest,
                },
            }
        )
    return rows


__all__ = ["compose_research_tools"]
