# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the read_resource / inspect_resource peer tools."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from dlightrag.core.agent.tool_loop import AgentTool
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.resources.registry import ResourceRegistry
from dlightrag.core.resources.tools import build_resource_tools
from dlightrag.core.resources.visual import ResourceInspectionError, ResourceInspector


class _RecordingVLM:
    def __init__(self, reply: str = "A chart of quarterly revenue.") -> None:
        self.reply = reply

    async def __call__(self, *, messages: list[dict], **_kwargs) -> str:
        return self.reply


class _FailingVLM:
    async def __call__(self, *, messages: list[dict], **_kwargs) -> str:
        raise RuntimeError("vlm upstream 503")


def _png(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), color).save(buffer, "PNG")
    return buffer.getvalue()


def _tools_by_name(tools: list[AgentTool]) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in tools}


def test_read_resource_registered_without_inspector() -> None:
    registry = ResourceRegistry()
    names = {tool.name for tool in build_resource_tools(registry)}
    assert names == {"read_resource"}


def test_inspect_resource_absent_when_capability_unverified() -> None:
    registry = ResourceRegistry()
    inspector = ResourceInspector(registry, vlm_func=_RecordingVLM())
    names = {
        tool.name
        for tool in build_resource_tools(registry, inspector=inspector, visual_supported=False)
    }
    assert names == {"read_resource"}


def test_inspect_resource_registered_only_for_verified_capability() -> None:
    registry = ResourceRegistry()
    inspector = ResourceInspector(registry, vlm_func=_RecordingVLM())
    names = {
        tool.name
        for tool in build_resource_tools(registry, inspector=inspector, visual_supported=True)
    }
    assert names == {"read_resource", "inspect_resource"}


def test_read_resource_tool_schema_is_exact() -> None:
    registry = ResourceRegistry()
    (read_tool,) = build_resource_tools(registry)
    fields = read_tool.input_model.model_fields
    assert set(fields) == {"resource_id", "focus", "cursor"}
    assert fields["resource_id"].is_required()
    assert not fields["focus"].is_required()
    assert not fields["cursor"].is_required()


def test_inspect_resource_tool_schema_is_exact() -> None:
    registry = ResourceRegistry()
    inspector = ResourceInspector(registry, vlm_func=_RecordingVLM())
    tools = _tools_by_name(
        build_resource_tools(registry, inspector=inspector, visual_supported=True)
    )
    fields = tools["inspect_resource"].input_model.model_fields
    assert set(fields) == {"resource_id", "focus", "locator", "cursor"}
    assert fields["resource_id"].is_required()
    assert fields["focus"].is_required()
    assert not fields["locator"].is_required()
    assert not fields["cursor"].is_required()


async def test_read_resource_tool_returns_text_and_handles() -> None:
    async with ResourceRegistry() as registry:
        resource_id = registry.register(
            ResourceInput(filename="notes.txt", content=b"alpha\nbeta\ngamma")
        )
        (read_tool,) = build_resource_tools(registry)
        args = read_tool.input_model.model_validate({"resource_id": resource_id})

        result = await read_tool.execute(args)

    assert "alpha" in result.content
    assert "gamma" in result.content


async def test_inspect_resource_tool_returns_derived_evidence() -> None:
    async with ResourceRegistry() as registry:
        resource_id = registry.register(
            ResourceInput(
                filename="chart.png", content=_png((200, 30, 30)), declared_mime="image/png"
            )
        )
        inspector = ResourceInspector(registry, vlm_func=_RecordingVLM("Ascending bars."))
        tools = _tools_by_name(
            build_resource_tools(registry, inspector=inspector, visual_supported=True)
        )
        inspect_tool = tools["inspect_resource"]
        args = inspect_tool.input_model.model_validate(
            {"resource_id": resource_id, "focus": "describe"}
        )

        result = await inspect_tool.execute(args)

    assert "Ascending bars." in result.content
    assert "derived_by_vlm" in result.content


async def test_inspect_resource_tool_propagates_vlm_failure() -> None:
    async with ResourceRegistry() as registry:
        resource_id = registry.register(
            ResourceInput(filename="chart.png", content=_png((1, 2, 3)), declared_mime="image/png")
        )
        inspector = ResourceInspector(registry, vlm_func=_FailingVLM())
        tools = _tools_by_name(
            build_resource_tools(registry, inspector=inspector, visual_supported=True)
        )
        inspect_tool = tools["inspect_resource"]
        args = inspect_tool.input_model.model_validate(
            {"resource_id": resource_id, "focus": "describe"}
        )

        with pytest.raises(ResourceInspectionError):
            await inspect_tool.execute(args)
