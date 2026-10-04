# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Built-in council Skill: autonomous recipe metadata that cannot widen child tools."""

from pathlib import Path
from unittest.mock import MagicMock

from dlightrag.engine.agent.environment import SearchToolchain
from dlightrag.engine.agent.skills import SkillCatalog, SkillsBundle, builtin_skills_root
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.composition import compose_research_tools


async def _retrieve(_query: str) -> object:
    return object()


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=builtin_skills_root())


def test_council_skill_is_packaged_builtin_with_autonomous_metadata() -> None:
    catalog = _catalog()
    council = next(skill for skill in catalog.metadata if skill.name == "council")
    contribution = catalog.contribution()

    assert council.source == "builtin"
    assert council.description.startswith("Use when")
    assert "only when the user" not in council.description.lower()
    assert "permission" not in council.description.lower()
    assert contribution is not None
    rendered = str(contribution.messages[0]["content"])
    assert f"council: {council.description} (builtin)" in rendered
    assert "# Council" not in rendered


def test_council_catalog_presence_does_not_widen_child_tools() -> None:
    empty = SkillsBundle()
    bundled = SkillsBundle(builtin_root=builtin_skills_root())
    without_skills = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        injected_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        child=True,
        search_toolchain=SearchToolchain(),
    )
    with_council = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        injected_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        skill_tools=list(bundled.tools(child=True)),
        child=True,
        search_toolchain=SearchToolchain(),
    )
    without_names = {tool.name for tool in without_skills}
    with_names = {tool.name for tool in with_council}

    assert {tool.name for tool in empty.tools(child=True)} == set()
    assert {tool.name for tool in bundled.tools(child=True)} == {"load_skill"}
    assert with_names - without_names == {"load_skill"}
    # A catalog widens only how Skills are read. What a Child never holds is the Run's
    # authority — its roster, durable memory, and publication — rather than side
    # effects, which its own workspace tools provide (ADR 0025).
    assert (
        not {
            "spawn_agent",
            "remember",
            "forget",
            "attach_artifact",
            "publish_skill",
            "delete_skill",
        }
        & with_names
    )
