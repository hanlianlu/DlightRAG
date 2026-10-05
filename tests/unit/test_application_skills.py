# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application composition for packaged, global, and owner Agent Skills."""

from pathlib import Path

from dlightrag.application.config import DlightragConfig
from dlightrag.application.skills import (
    owner_skills_directory,
    skill_read_layers,
    skills_bundle_factory,
)
from dlightrag.engine.agent.skills import SkillCatalog, builtin_skills_root, owner_skill_root


def _config(
    tmp_path: Path,
    *,
    disabled_builtin_skills: tuple[str, ...] = (),
) -> DlightragConfig:
    return DlightragConfig.model_validate(
        {
            "answer": {
                "agent": {
                    "skills_root": str(tmp_path / "global"),
                    "owner_skills_root": str(tmp_path / "owners"),
                    "disabled_builtin_skills": disabled_builtin_skills,
                }
            }
        }
    )


def _skill(root: Path, *, description: str) -> None:
    root.mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: skill-creator\ndescription: {description}\n---\nOVERRIDE",
        encoding="utf-8",
    )


def test_fresh_roots_discover_and_load_packaged_skill_without_copying(tmp_path: Path) -> None:
    config = _config(tmp_path)
    build = skills_bundle_factory(config, ensure_dirs=True)

    catalog = build("fresh-owner").catalog()

    assert catalog is not None
    names = {skill.name: skill.source for skill in catalog.metadata}
    assert names["skill-creator"] == "builtin"
    assert names["council"] == "builtin"
    contribution = catalog.contribution()
    assert contribution is not None
    rendered = str(contribution.messages[0]["content"])
    assert "# Skill Creator" not in rendered
    assert "# Council" not in rendered
    assert "# Skill Creator" in catalog.read("skill-creator")
    assert "# Council" in catalog.read("council")
    assert list((tmp_path / "global").iterdir()) == []
    assert list((tmp_path / "owners").iterdir()) == []


def test_disabled_builtin_filter_does_not_hide_global_or_owner_overrides(tmp_path: Path) -> None:
    config = _config(tmp_path, disabled_builtin_skills=("skill-creator",))
    global_root = tmp_path / "global" / "skill-creator"
    _skill(global_root, description="Global creator.")
    build = skills_bundle_factory(config)

    global_catalog = build("owner-without-override").catalog()

    assert global_catalog is not None
    sources = {skill.name: skill.source for skill in global_catalog.metadata}
    assert sources["council"] == "builtin"
    assert sources["skill-creator"] == "global"

    owner_root = owner_skill_root(tmp_path / "owners", "owner-with-override")
    _skill(owner_root / "skill-creator", description="Owner creator.")
    owner_catalog = build("owner-with-override").catalog()

    assert owner_catalog is not None
    sources = {skill.name: skill.source for skill in owner_catalog.metadata}
    assert sources["council"] == "builtin"
    assert sources["skill-creator"] == "owner"


def test_disabled_builtin_filter_removes_unoverridden_builtin(tmp_path: Path) -> None:
    every = {
        skill.name for skill in SkillCatalog.discover(builtin_root=builtin_skills_root()).metadata
    }
    catalog = skills_bundle_factory(_config(tmp_path, disabled_builtin_skills=("skill-creator",)))(
        "owner"
    ).catalog()

    assert catalog is not None
    assert {skill.name for skill in catalog.metadata} == every - {"skill-creator"}

    hidden = skills_bundle_factory(_config(tmp_path, disabled_builtin_skills=tuple(sorted(every))))(
        "owner"
    ).catalog()

    assert hidden is not None
    assert hidden.metadata == ()


def test_the_directory_an_owner_manages_is_the_shard_their_agent_reads(tmp_path: Path) -> None:
    config = _config(tmp_path)

    managed = owner_skills_directory(config, owner_id="alice")
    [_global_layer, owner_layer] = skill_read_layers(config, owner_id="alice")

    # What Settings lists and changes is what the owner's Agent processes read, and nobody else's.
    assert managed == owner_layer.path == owner_skill_root(tmp_path / "owners", "alice")
    assert managed != owner_skills_directory(config, owner_id="bob")
