# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structured root Artifact attachment through the parent Research tool."""

import hashlib
from pathlib import Path

import pytest

from dlightrag.engine.agent.environment import AccessScheduler
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.answer.publication import PublicationLimits
from dlightrag.engine.answer.tools.artifacts import AttachArtifactArgs, attach_artifact_tool
from tests.tool_helpers import recording_tool_runtime, tool_runtime


@pytest.mark.asyncio
async def test_attach_artifact_validates_and_returns_a_structured_receipt(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    )

    result = await tool.execute(
        AttachArtifactArgs(path="analysis.md", label="Open analysis"),
        tool_runtime(),
    )

    assert tool.replay_policy == "replayable"
    assert "genuinely benefits from a separate reading or download surface" in (tool.guidance or "")
    assert "merely because the tool is available" in (tool.guidance or "")
    assert result.is_error is False
    assert "[Open analysis](artifact:artifact-431b1900963e6cd2f4a1)" in result.text_content
    assert result.details is not None
    # The receipt names the address a later turn reads the published version with:
    # it is the same deterministic id publication binds, so it is known now.
    assert result.details["artifact_attachment"] == {
        "relative_path": "analysis.md",
        "label": "Open analysis",
        "content_digest": hashlib.sha256(b"Grounded analysis.").hexdigest(),
        "size_bytes": len(b"Grounded analysis."),
        "presentation": "markdown",
        "resource_id": "artifact-431b1900963e6cd2f4a1",
    }


@pytest.mark.asyncio
async def test_attach_artifact_rejects_missing_and_unsafe_paths(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    )

    missing = await tool.execute(AttachArtifactArgs(path="missing.md"), tool_runtime())
    unsafe = await tool.execute(AttachArtifactArgs(path="../secret.md"), tool_runtime())

    assert missing.is_error is True
    assert missing.text_content.startswith("missing_file:")
    assert unsafe.is_error is True
    assert unsafe.text_content.startswith("invalid_reference:")


@pytest.mark.asyncio
async def test_attach_artifact_reports_its_path_as_the_live_subject(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    updates: list[ToolResult] = []

    await attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    ).execute(
        AttachArtifactArgs(path="analysis.md"),
        recording_tool_runtime(updates, tool_name="attach_artifact"),
    )

    assert [update.subject for update in updates] == ["analysis.md"]
