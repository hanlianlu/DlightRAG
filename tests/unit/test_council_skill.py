# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Built-in council Skill: autonomous recipe metadata that cannot widen child tools."""

from pathlib import Path
from unittest.mock import MagicMock

from dlightrag.engine.agent.skills import SkillCatalog, SkillsBundle, builtin_skills_root
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.tools.composition import compose_research_tools

_COUNCIL_DESCRIPTION = (
    "Council recipe for independent Child Session investigations and one curated "
    "cross-examination. Load when independent scrutiny would materially improve a "
    "contested, high-stakes, or multi-source answer, or when the user asks for "
    "independent critique. Skip ordinary factual or trivial questions. User veto, "
    "cancellation, and scope constraints win."
)
_SIDE_EFFECT_TOOLS = frozenset({"spawn_agent", "attach_artifact", "write", "edit", "bash"})


async def _retrieve(_query: str) -> object:
    return object()


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=builtin_skills_root())


def test_council_skill_is_packaged_builtin_with_autonomous_metadata() -> None:
    catalog = _catalog()
    council = next(skill for skill in catalog.metadata if skill.name == "council")
    contribution = catalog.contribution()

    assert council.source == "builtin"
    assert council.description == _COUNCIL_DESCRIPTION
    assert "only when the user" not in council.description.lower()
    assert "permission" not in council.description.lower()
    assert "materially improve" in council.description
    assert contribution is not None
    rendered = str(contribution.messages[0]["content"])
    assert f"council: {_COUNCIL_DESCRIPTION} (builtin)" in rendered
    assert "# Council" not in rendered


def test_council_skill_body_is_a_bounded_read_only_recipe() -> None:
    text = _catalog().read("council")
    lowered = text.lower()

    assert text.startswith("---\nname: council\n")
    assert "# Council" in text
    assert "spawn_agent" in text
    assert "continue_subagent" in text
    assert "read-only" in lowered
    assert "user veto" in lowered
    assert "dissent" in lowered
    assert "loading it grants no tools" in lowered
    assert "at most one focused cross-examination" in lowered
    assert "council entity" in lowered
    assert "supervisor" not in lowered
    assert "budget" not in lowered


def test_council_catalog_presence_does_not_widen_child_tools() -> None:
    empty = SkillsBundle()
    bundled = SkillsBundle(builtin_root=builtin_skills_root())
    without_skills = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        child=True,
    )
    with_council = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=_retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        environment=MagicMock(),
        artifacts_root=Path("/unused/artifacts"),
        skill_tools=list(bundled.tools(child=True)),
        child=True,
    )
    without_names = {tool.name for tool in without_skills}
    with_names = {tool.name for tool in with_council}

    assert {tool.name for tool in empty.tools(child=True)} == set()
    assert {tool.name for tool in bundled.tools(child=True)} == {"load_skill"}
    assert with_names - without_names == {"load_skill"}
    assert not _SIDE_EFFECT_TOOLS & with_names
    assert "publish_skill" not in with_names
    assert "delete_skill" not in with_names
