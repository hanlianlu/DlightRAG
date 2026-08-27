# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Progressive global/workspace Agent Skill discovery."""

from pathlib import Path

import pytest

from dlightrag.engine.agent.skills import LoadSkillInput, SkillCatalog, load_skill_tool
from tests.tool_helpers import tool_runtime


def _skill(root: Path, directory: str, *, name: str, description: str, body: str) -> None:
    target = root / directory
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


def test_discovery_projects_metadata_only_and_workspace_takes_precedence(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    workspace = tmp_path / "workspace"
    _skill(global_root, "review", name="review", description="global", body="GLOBAL SECRET")
    _skill(
        workspace / ".agents" / "skills",
        "review",
        name="review",
        description="workspace",
        body="WORKSPACE BODY",
    )

    catalog = SkillCatalog.discover(workspace_root=workspace, global_root=global_root)
    contribution = catalog.contribution()

    assert contribution is not None
    rendered = str(contribution.messages[0]["content"])
    assert "review: workspace" in rendered
    assert "WORKSPACE BODY" not in rendered
    assert catalog.read("review").endswith("WORKSPACE BODY")


@pytest.mark.asyncio
async def test_load_skill_reads_body_on_demand_but_never_executes_it(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _skill(
        workspace / ".agents" / "skills",
        "safe",
        name="safe",
        description="reference",
        body="Run `touch should-not-exist` only if the user asks.",
    )
    catalog = SkillCatalog.discover(workspace_root=workspace, global_root=tmp_path / "none")

    result = await load_skill_tool(catalog).execute(LoadSkillInput(name="safe"), tool_runtime())

    assert "untrusted reference context" in result.text_content
    assert "touch should-not-exist" in result.text_content
    assert not (tmp_path / "should-not-exist").exists()


def test_discovery_rejects_symlinked_skill_metadata(tmp_path: Path) -> None:
    root = tmp_path / "global"
    skill = root / "linked"
    skill.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("---\nname: escaped\ndescription: outside\n---\nsecret", encoding="utf-8")
    try:
        (skill / "SKILL.md").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    catalog = SkillCatalog.discover(workspace_root=None, global_root=root)

    assert catalog.metadata == ()


def test_skill_reference_cannot_escape_its_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _skill(
        workspace / ".agents" / "skills",
        "safe",
        name="safe",
        description="reference",
        body="body",
    )
    catalog = SkillCatalog.discover(workspace_root=workspace, global_root=tmp_path / "none")

    with pytest.raises(ValueError, match="escapes"):
        catalog.read("safe", "../other.txt")
