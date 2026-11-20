# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one-load Skill kit (ADR 0066 phase 2): a declared kit becomes workspace files in one call.

The contract is a declaration, not a destination: a Skill's frontmatter names one kit directory,
and ``materialize_skill`` writes exactly that directory's shipped files into the same workspace
path. These tests pin what the safety claims mean: the whole kit or a named refusal (nothing
partial), the declaration as the only destination (no arbitrary write), idempotence, discovery
refusing a kit it cannot honour, path confinement, and a packaged Skill's own kit.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
)
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.skills import (
    MaterializeSkillInput,
    PublishSkillInput,
    SkillCatalog,
    SkillsBundle,
    materialize_skill_tool,
    publish_skill_tool,
)
from tests.support.skills import SEALED_SKILL, sealed_skills_root
from tests.support.tool_helpers import tool_runtime


def _write_skill(
    root: Path,
    name: str,
    *,
    frontmatter: str,
    files: dict[str, str],
) -> Path:
    """One Skill directory whose SKILL.md carries ``frontmatter`` and whose files are ``files``."""
    target = root / name
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: reference\n{frontmatter}---\nbody\n",
        encoding="utf-8",
    )
    for relative, content in files.items():
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        (target / relative).write_text(content, encoding="utf-8")
    return target


def _kit_skill(root: Path, *, name: str = "kit-skill") -> Path:
    """A well-formed Skill holding two plain files in its declared ``scripts`` kit."""
    return _write_skill(
        root,
        name,
        frontmatter="kit: scripts\n",
        files={"scripts/one.py": "print('one')\n", "scripts/two.py": "print('two')\n"},
    )


def _materialize_tool(catalog: SkillCatalog, workspace: Path):
    return materialize_skill_tool(
        lambda: catalog,
        environment=LocalExecutionEnvironment(workspace),
        scheduler=AccessScheduler(),
    )


@pytest.mark.asyncio
async def test_one_call_materializes_the_whole_declared_kit(tmp_path: Path) -> None:
    """The declaration is the destination and the receipt is every file with its sha256."""
    global_root = tmp_path / "global"
    skill = _kit_skill(global_root)
    one, two = (
        (skill / "scripts" / "one.py").read_bytes(),
        (skill / "scripts" / "two.py").read_bytes(),
    )
    catalog = SkillCatalog.discover(global_root=global_root)
    workspace = tmp_path / "workspace"
    tool = _materialize_tool(catalog, workspace)

    result = await tool.execute(
        MaterializeSkillInput(name="kit-skill"), tool_runtime(tool_name="materialize_skill")
    )

    assert not result.is_error, result.text_content
    assert (workspace / "scripts" / "one.py").read_bytes() == one
    assert (workspace / "scripts" / "two.py").read_bytes() == two
    kit = catalog.read_kit("kit-skill")
    assert kit.directory == "scripts"
    assert [file.relative for file in kit.files] == ["scripts/one.py", "scripts/two.py"]
    assert kit.digest in result.text_content
    for file in kit.files:
        assert f"{file.relative} {len(file.content)} bytes sha256 {file.digest}" in (
            result.text_content
        )
    inventory = result.effects.workspace_inventory
    assert inventory is not None
    assert [record.relative_path for record in inventory.upserts] == [
        "scripts/one.py",
        "scripts/two.py",
    ]


@pytest.mark.asyncio
async def test_a_second_call_reports_what_is_already_current_and_writes_nothing(
    tmp_path: Path,
) -> None:
    """Idempotence is per file: the shipped bytes are the test, not a timestamp."""
    global_root = tmp_path / "global"
    _kit_skill(global_root)
    catalog = SkillCatalog.discover(global_root=global_root)
    workspace = tmp_path / "workspace"
    tool = _materialize_tool(catalog, workspace)
    runtime = tool_runtime(tool_name="materialize_skill")

    first = await tool.execute(MaterializeSkillInput(name="kit-skill"), runtime)
    second = await tool.execute(MaterializeSkillInput(name="kit-skill"), runtime)

    assert not first.is_error and not second.is_error
    assert "2 written, 0 already current" in first.text_content
    assert "0 written, 2 already current" in second.text_content
    assert second.effects.workspace_inventory is None


@pytest.mark.asyncio
async def test_a_kit_whose_sibling_cannot_be_read_is_refused_whole(
    tmp_path: Path, monkeypatch
) -> None:
    """A listed sibling that is not readable fails the kit before anything is written."""
    global_root = tmp_path / "global"
    _kit_skill(global_root)
    catalog = SkillCatalog.discover(global_root=global_root)
    workspace = tmp_path / "workspace"
    tool = _materialize_tool(catalog, workspace)
    original = SkillCatalog.read_bytes

    def fail_on_second(self, name: str, relative_path: str = "SKILL.md") -> bytes:
        if relative_path.endswith("two.py"):
            raise FileNotFoundError("two.py was removed after listing")
        return original(self, name, relative_path)

    monkeypatch.setattr(SkillCatalog, "read_bytes", fail_on_second)

    result = await tool.execute(
        MaterializeSkillInput(name="kit-skill"), tool_runtime(tool_name="materialize_skill")
    )

    assert result.is_error
    assert "kit_missing" in result.text_content and "two.py" in result.text_content
    assert not (workspace / "scripts" / "one.py").exists()
    assert not (workspace / "scripts" / "two.py").exists()


@pytest.mark.asyncio
async def test_a_write_failure_rolls_the_kit_back_and_names_the_entry(tmp_path: Path) -> None:
    """Complete kit or nothing: a write that fails after its siblings leaves none of them."""

    class FailingSecondWrite(LocalExecutionEnvironment):
        def write_bytes(self, path: Path, data: bytes) -> None:
            if path.name == "two.py":
                raise OSError("no space left on device")
            super().write_bytes(path, data)

    global_root = tmp_path / "global"
    _kit_skill(global_root)
    catalog = SkillCatalog.discover(global_root=global_root)
    workspace = tmp_path / "workspace"
    tool = materialize_skill_tool(
        lambda: catalog,
        environment=FailingSecondWrite(workspace),
        scheduler=AccessScheduler(),
    )

    result = await tool.execute(
        MaterializeSkillInput(name="kit-skill"), tool_runtime(tool_name="materialize_skill")
    )

    assert result.is_error
    assert "scripts/two.py" in result.text_content
    assert not (workspace / "scripts" / "one.py").exists()
    assert not (workspace / "scripts" / "two.py").exists()


@pytest.mark.asyncio
async def test_a_symlinked_workspace_destination_is_refused_and_writes_nothing_outside(
    tmp_path: Path,
) -> None:
    """The declaration is a workspace path and meets the same no-link rule the file tools do."""
    global_root = tmp_path / "global"
    _kit_skill(global_root)
    catalog = SkillCatalog.discover(global_root=global_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    tool = _materialize_tool(catalog, workspace)
    # The environment scanned an empty workspace; the link appears after it, so the refusal is
    # the path rule (a link is never followed) and not the envelope latch.
    os.symlink(outside, workspace / "scripts")

    result = await tool.execute(
        MaterializeSkillInput(name="kit-skill"), tool_runtime(tool_name="materialize_skill")
    )

    assert result.is_error
    assert "symbolic link" in result.text_content
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_a_skill_that_declares_no_kit_says_so(tmp_path: Path) -> None:
    """Absent kit and broken kit are different answers."""
    global_root = tmp_path / "global"
    _write_skill(global_root, "plain", frontmatter="", files={"ref.md": "text\n"})
    catalog = SkillCatalog.discover(global_root=global_root)
    tool = _materialize_tool(catalog, tmp_path / "workspace")

    result = await tool.execute(
        MaterializeSkillInput(name="plain"), tool_runtime(tool_name="materialize_skill")
    )

    assert result.is_error
    assert "declares no kit" in result.text_content


def test_a_kit_cannot_be_told_where_to_write() -> None:
    """The only argument is the Skill name: the declaration fixes the destination."""
    assert set(MaterializeSkillInput.model_json_schema()["properties"]) == {"name"}
    with pytest.raises(ValidationError):
        MaterializeSkillInput(name="kit-skill", path="tmp/elsewhere")  # type: ignore[call-arg]


def test_a_kit_path_that_escapes_the_skill_is_not_served(tmp_path: Path) -> None:
    """A traversal in the declaration is refused where the declaration is read."""
    global_root = tmp_path / "global"
    _write_skill(
        global_root,
        "escaping",
        frontmatter="kit: ../elsewhere\n",
        files={"scripts/one.py": "x\n"},
    )
    catalog = SkillCatalog.discover(global_root=global_root)
    assert [skill.name for skill in catalog.metadata] == []


def test_a_kit_entry_that_is_not_a_plain_file_is_not_served(tmp_path: Path) -> None:
    """A nested directory or a symlink is not a sibling script, so the Skill is refused."""
    global_root = tmp_path / "global"
    skill = _kit_skill(global_root)
    (skill / "scripts" / "nested").mkdir()
    catalog = SkillCatalog.discover(global_root=global_root)
    assert [entry.name for entry in catalog.metadata] == []


def test_a_symlinked_kit_entry_is_not_served(tmp_path: Path) -> None:
    global_root = tmp_path / "global"
    skill = _kit_skill(global_root)
    os.symlink(skill / "scripts" / "one.py", skill / "scripts" / "link.py")
    catalog = SkillCatalog.discover(global_root=global_root)
    assert [entry.name for entry in catalog.metadata] == []


@pytest.mark.asyncio
async def test_a_published_skill_kit_is_validated_with_its_payload(tmp_path: Path) -> None:
    """Publishing checks the kit it declares: sibling files only, and a valid one is served."""
    owner_root = tmp_path / "owners"

    def hub(kit: str) -> str:
        return f"---\nname: review\ndescription: Review plans.\n{kit}---\nbody\n"

    valid = publish_skill_tool(owner_root)
    published = await valid.execute(
        PublishSkillInput(
            name="review",
            files={
                "SKILL.md": hub("kit: scripts\n"),
                "scripts/one.py": "print('one')\n",
            },
        ),
        tool_runtime(tool_name="publish_skill"),
    )
    assert not published.is_error, published.text_content

    nested = publish_skill_tool(tmp_path / "nested-owners")
    refused_nested = await nested.execute(
        PublishSkillInput(
            name="review",
            files={
                "SKILL.md": hub("kit: scripts\n"),
                "scripts/one.py": "print('one')\n",
                "scripts/deep/two.py": "print('two')\n",
            },
        ),
        tool_runtime(tool_name="publish_skill"),
    )
    assert refused_nested.is_error
    assert "nested" in refused_nested.text_content

    empty = publish_skill_tool(tmp_path / "empty-owners")
    refused_empty = await empty.execute(
        PublishSkillInput(
            name="review",
            files={"SKILL.md": hub("kit: scripts\n"), "references/one.md": "text\n"},
        ),
        tool_runtime(tool_name="publish_skill"),
    )
    assert refused_empty.is_error
    assert "no published sibling file" in refused_empty.text_content


@pytest.mark.asyncio
async def test_a_packaged_kit_materializes_every_script(tmp_path: Path) -> None:
    """A packaged kit: every shipped script, byte for byte, in one call (ADR 0066 phase 2)."""
    catalog = SkillCatalog.discover(builtin_root=sealed_skills_root())
    scripts = sealed_skills_root().joinpath(SEALED_SKILL, "scripts")
    shipped = {
        path.name: path.read_bytes()
        for path in sorted(scripts.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    workspace = tmp_path / "workspace"
    tool = _materialize_tool(catalog, workspace)

    result = await tool.execute(
        MaterializeSkillInput(name=SEALED_SKILL),
        tool_runtime(tool_name="materialize_skill"),
    )

    assert not result.is_error, result.text_content
    for name, content in shipped.items():
        written = (workspace / "scripts" / name).read_bytes()
        assert written == content, name
        assert hashlib.sha256(written).hexdigest() in result.text_content
    kit = catalog.read_kit(SEALED_SKILL)
    assert {file.name for file in kit.files} == set(shipped)
    assert kit.digest in result.text_content


def test_a_child_gets_the_skill_read_but_not_the_kit_write(tmp_path: Path) -> None:
    """The kit is the Run's construction work: the parent materializes, a Child only loads."""
    global_root = tmp_path / "global"
    _kit_skill(global_root)
    bundle = SkillsBundle(global_root=global_root)
    parent = {tool.name for tool in bundle.tools(child=False)}
    child = {tool.name for tool in bundle.tools(child=True)}
    assert "materialize_skill" in parent
    assert "materialize_skill" not in child
