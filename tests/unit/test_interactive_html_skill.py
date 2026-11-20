# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Built-in interactive-html Skill: it is discovered, and what it points at exists.

Hub size is enforced for every built-in in one place, `test_builtin_skill_hubs.py`, so the
budget cannot drift between files. What follows is what is particular to this Skill.
"""

from dlightrag.engine.agent.skills import SkillCatalog, builtin_skills_root


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=builtin_skills_root())


def _skill(name: str):
    return next(skill for skill in _catalog().metadata if skill.name == name)


def test_it_is_a_builtin_that_routes_by_when_to_use_it() -> None:
    skill = _skill("interactive-html")

    assert skill.source == "builtin"
    assert skill.description.startswith("Use when")
    # It names the Skills it must not be confused with, which is what keeps a report request
    # from landing on the chart Skill and vice versa.
    assert "(charts)" in skill.description and "(office-documents)" in skill.description


def test_every_example_it_names_loads() -> None:
    catalog = _catalog()
    hub = catalog.read("interactive-html")

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
