# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Built-in interactive-html Skill: it is discovered, it stays small, and what it points at exists."""

from dlightrag.engine.agent.skills import SkillCatalog, builtin_skills_root

# The hub is paid for by every Run that loads it, so it stays a hub: the long material is in references.
_HUB_WORDS = 1_000
_DESCRIPTION_WORDS = 70


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=builtin_skills_root())


def _skill(name: str):
    return next(skill for skill in _catalog().metadata if skill.name == name)


def test_it_is_a_builtin_that_routes_by_when_to_use_it() -> None:
    skill = _skill("interactive-html")

    assert skill.source == "builtin"
    assert skill.description.startswith("Use when")
    assert len(skill.description.split()) <= _DESCRIPTION_WORDS
    assert "(charts)" in skill.description and "(office-documents)" in skill.description


def test_the_hub_stays_a_hub_and_every_reference_it_names_loads() -> None:
    catalog = _catalog()
    hub = catalog.read("interactive-html")
    assert len(hub.split()) <= _HUB_WORDS

    references = (
        "fragment.md",
        "example-brief.html",
        "example-dashboard.html",
        "example-multipage.html",
        "example-scenario.html",
    )
    for name in references:
        assert name in hub
        assert catalog.read("interactive-html", f"references/{name}")


def test_the_two_chart_skills_point_at_each_other() -> None:
    catalog = _catalog()

    assert "`charts`" in catalog.read("interactive-html")
    assert "interactive-html" in catalog.read("charts")


def test_the_hub_states_the_rules_a_model_cannot_infer() -> None:
    hub = _catalog().read("interactive-html")

    # Each line below caused a real failure when a model was not told it.
    assert "load_skill" in hub and "`path`" in hub
    assert "html-report build" in hub
    assert "Mineral" in hub
    assert "AI generated" in hub
