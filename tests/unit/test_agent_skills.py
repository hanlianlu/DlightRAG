# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Progressive global/owner Agent Skill discovery and owner publication."""

from pathlib import Path

import pytest

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.skills import (
    DeleteSkillInput,
    LoadSkillInput,
    PublishSkillInput,
    SkillCatalog,
    delete_skill_tool,
    load_skill_tool,
    owner_skill_root,
    publish_skill_tool,
)
from dlightrag.engine.agent.tools import ToolResult, ToolRuntime
from tests.tool_helpers import tool_runtime


def _skill(root: Path, directory: str, *, name: str, description: str, body: str) -> None:
    target = root / directory
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


def _skill_files(
    *, name: str, description: str, body: str, extra: dict[str, str] | None = None
) -> dict[str, str]:
    files = {
        "SKILL.md": f"---\nname: {name}\ndescription: {description}\n---\n{body}",
    }
    files.update(extra or {})
    return files


def test_discovery_projects_metadata_only_and_owner_takes_precedence(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    owner_root = tmp_path / "owner"
    _skill(global_root, "review", name="review", description="global", body="GLOBAL SECRET")
    _skill(owner_root, "review", name="review", description="owner", body="OWNER BODY")

    catalog = SkillCatalog.discover(global_root=global_root, owner_root=owner_root)
    contribution = catalog.contribution()

    assert contribution is not None
    rendered = str(contribution.messages[0]["content"])
    assert "review: owner (owner)" in rendered
    assert "OWNER BODY" not in rendered
    assert catalog.read("review").endswith("OWNER BODY")


def test_discovery_merges_distinct_names_across_tiers(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    owner_root = tmp_path / "owner"
    _skill(global_root, "review", name="review", description="global", body="g")
    _skill(owner_root, "tdd", name="tdd", description="owner", body="o")

    catalog = SkillCatalog.discover(global_root=global_root, owner_root=owner_root)

    assert {skill.name for skill in catalog.metadata} == {"review", "tdd"}
    sources = {skill.name: skill.source for skill in catalog.metadata}
    assert sources == {"review": "global", "tdd": "owner"}


@pytest.mark.asyncio
async def test_load_skill_reads_body_on_demand_but_never_executes_it(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _skill(
        owner_root,
        "safe",
        name="safe",
        description="reference",
        body="Run `touch should-not-exist` only if the user asks.",
    )
    catalog = SkillCatalog.discover(global_root=tmp_path / "none", owner_root=owner_root)

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

    catalog = SkillCatalog.discover(global_root=root)

    assert catalog.metadata == ()


def test_skill_reference_cannot_escape_its_directory(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    _skill(owner_root, "safe", name="safe", description="reference", body="body")
    catalog = SkillCatalog.discover(global_root=tmp_path / "none", owner_root=owner_root)

    with pytest.raises(ValueError, match="escapes"):
        catalog.read("safe", "../other.txt")


def test_owner_skill_root_shards_owners_apart(tmp_path: Path) -> None:
    root_a = owner_skill_root(tmp_path, "owner-a")
    root_b = owner_skill_root(tmp_path, "owner-b")

    assert root_a != root_b
    assert root_a.name == "owner-a"
    assert root_a.is_relative_to(tmp_path)


@pytest.mark.asyncio
async def test_publish_installs_a_multifile_skill_atomically(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)
    files = _skill_files(
        name="weekly-report",
        description="Generate weekly reports.",
        body="Follow references/template.md.",
        extra={"references/template.md": "# Weekly template"},
    )

    result = await tool.execute(
        PublishSkillInput(name="weekly-report", files=files), tool_runtime()
    )

    assert "Published Agent Skill 'weekly-report'" in result.text_content
    assert not result.is_error
    assert (owner_root / "weekly-report" / "SKILL.md").is_file()
    assert (owner_root / "weekly-report" / "references" / "template.md").is_file()
    assert not list(owner_root.glob(".staging-*"))
    assert not list(owner_root.glob(".backup-*"))


@pytest.mark.asyncio
async def test_publish_validates_name_frontmatter_and_paths(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)

    async def publish(**kwargs: object) -> str:
        result = await tool.execute(
            PublishSkillInput.model_validate(kwargs),
            tool_runtime(),  # type: ignore[arg-type]
        )
        assert result.is_error
        return result.text_content

    assert "kebab-case" in await publish(
        name="Bad_Name",
        files=_skill_files(name="bad-name", description="d", body="b"),
    )
    assert "must contain a 'SKILL.md'" in await publish(name="review", files={"notes.md": "x"})
    assert "frontmatter name" in await publish(
        name="review",
        files=_skill_files(name="other", description="d", body="b"),
    )
    assert "description" in await publish(
        name="review",
        files={"SKILL.md": "---\nname: review\n---\nbody"},
    )
    assert "plain relative path" in await publish(
        name="review",
        files={
            "SKILL.md": "---\nname: review\ndescription: d\n---\nb",
            "../escape.md": "x",
        },
    )
    assert not (owner_root / "review").exists()


@pytest.mark.asyncio
async def test_publish_updates_an_existing_owner_skill(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)
    await tool.execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="first", body="v1"),
        ),
        tool_runtime(),
    )

    result = await tool.execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="second", body="v2"),
        ),
        tool_runtime(),
    )

    assert not result.is_error
    catalog = SkillCatalog.discover(owner_root=owner_root)
    assert catalog.read("review").endswith("v2")


@pytest.mark.asyncio
async def test_publish_enforces_skill_count_quota(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    tool = publish_skill_tool(owner_root)
    for index in range(20):
        result = await tool.execute(
            PublishSkillInput(
                name=f"skill-{index}",
                files=_skill_files(name=f"skill-{index}", description="d", body="b"),
            ),
            tool_runtime(),
        )
        assert not result.is_error

    result = await tool.execute(
        PublishSkillInput(
            name="overflow",
            files=_skill_files(name="overflow", description="d", body="b"),
        ),
        tool_runtime(),
    )

    assert result.is_error
    assert "quota" in result.text_content
    assert not (owner_root / "overflow").exists()


@pytest.mark.asyncio
async def test_delete_skill_is_idempotent(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    publish = publish_skill_tool(owner_root)
    await publish.execute(
        PublishSkillInput(
            name="review",
            files=_skill_files(name="review", description="d", body="b"),
        ),
        tool_runtime(),
    )

    delete = delete_skill_tool(owner_root)
    removed = await delete.execute(DeleteSkillInput(name="review"), tool_runtime())
    missing = await delete.execute(DeleteSkillInput(name="review"), tool_runtime())

    assert "Deleted Agent Skill 'review'" in removed.text_content
    assert not (owner_root / "review").exists()
    assert "does not exist" in missing.text_content
    assert not missing.is_error


@pytest.mark.asyncio
async def test_skill_tools_report_their_object_label_live(tmp_path: Path) -> None:
    updates: list[ToolResult] = []

    async def sink(result: ToolResult) -> None:
        updates.append(result)

    def runtime(tool_name: str) -> ToolRuntime:
        return ToolRuntime(
            call_id="test-call",
            tool_name=tool_name,
            intent_id=IntentId.new(),
            execution_scope="test-scope",
            _update_sink=sink,
        )

    owner_root = tmp_path / "owner"
    _skill(owner_root, "review", name="review", description="reference", body="body")
    catalog = SkillCatalog.discover(owner_root=owner_root)
    await load_skill_tool(catalog).execute(LoadSkillInput(name="review"), runtime("load_skill"))
    await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name="weekly-report",
            files=_skill_files(name="weekly-report", description="d", body="b"),
        ),
        runtime("publish_skill"),
    )
    await delete_skill_tool(owner_root).execute(
        DeleteSkillInput(name="weekly-report"), runtime("delete_skill")
    )

    labels = [update.details["object_label"] for update in updates if update.details]
    assert labels == ["review", "weekly-report", "weekly-report"]
