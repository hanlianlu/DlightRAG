# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The routing eval suite's machine half: the anchors, and the matrix's own consistency.

Routing across built-in Skills is joint — one description edit moves where the others load — so
the suite lives in one place (`docs/skills-routing-eval.md`) rather than inside any one Skill. What
runs here is the cheap half: every anchor phrase still appears in the description it is anchored to,
and the matrix still covers every built-in Skill. The half that needs orchestrator model families is
the run log in that file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dlightrag.engine.agent.skills import SkillCatalog, builtin_skills_root

_SUITE = Path(__file__).resolve().parents[2] / "docs" / "skills-routing-eval.md"


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=builtin_skills_root())


def _rows(heading: str) -> list[list[str]]:
    """The data rows of the first table under a `## heading`, header and rule dropped."""
    text = _SUITE.read_text(encoding="utf-8")
    assert f"## {heading}" in text, f"the suite has no '## {heading}' section"
    section = text.split(f"## {heading}", 1)[1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(set(cell) <= set("-: ") for cell in cells):
            continue
        rows.append(cells)
    return rows[1:]  # the header


def _bare(cell: str) -> str:
    return cell.replace("`", "").strip()


@pytest.fixture(scope="module")
def suite() -> Path:
    assert _SUITE.is_file(), "the routing eval suite is part of every checkout, the core's too"
    return _SUITE


def test_every_anchor_is_still_in_the_description_it_is_anchored_to(suite: Path) -> None:
    """This is the half a description edit breaks, and the commonest way a working router dies."""
    descriptions = {
        skill.name: " ".join(skill.description.lower().split()) for skill in _catalog().metadata
    }

    for skill_name, anchor in (
        (_bare(row[0]), _bare(row[1])) for row in _rows("Anchors — the CI half")
    ):
        assert skill_name in descriptions, f"{skill_name} is not a built-in Skill"
        needle = " ".join(anchor.lower().split())
        assert needle in descriptions[skill_name], (
            f"{skill_name}: its description no longer contains the anchor {anchor!r}. Either the "
            "description lost a trigger the suite depends on, or the case changed and this file must "
            "change with it."
        )


def test_the_matrix_covers_every_built_in_skill(suite: Path) -> None:
    """A Skill added later must be given anchors and cases, not left out of the joint eval."""
    anchored = {_bare(row[0]) for row in _rows("Anchors — the CI half")}
    catalogued = {skill.name for skill in _catalog().metadata}
    assert catalogued == anchored, (
        f"missing from the suite: {sorted(catalogued - anchored)}; "
        f"named but not shipped: {sorted(anchored - catalogued)}"
    )


def test_positives_name_a_real_owner_and_are_identified_once(suite: Path) -> None:
    names = {skill.name for skill in _catalog().metadata}
    identifiers: set[str] = set()
    rows = _rows("Positives — the Skill must load")
    for row in rows:
        case_id, query, owner = _bare(row[0]), row[1], _bare(row[2])
        assert re.fullmatch(r"P\d+", case_id), f"{case_id!r} is not a positive-case id"
        assert case_id not in identifiers, f"{case_id} appears twice"
        identifiers.add(case_id)
        assert query.strip(), f"{case_id} has no query"
        assert owner in names, f"{case_id}: {owner} is not a built-in Skill"


def test_forbidden_cases_name_both_skills_and_never_the_same_one(suite: Path) -> None:
    names = {skill.name for skill in _catalog().metadata}
    identifiers: set[str] = set()
    for row in _rows("Forbidden loads — the query must not load that Skill"):
        case_id, query, must_not, owner = _bare(row[0]), row[1], _bare(row[2]), _bare(row[3])
        assert re.fullmatch(r"F\d+", case_id), f"{case_id!r} is not a forbidden-case id"
        assert case_id not in identifiers, f"{case_id} appears twice"
        identifiers.add(case_id)
        assert query.strip(), f"{case_id} has no query"
        assert must_not in names, f"{case_id}: {must_not} is not a built-in Skill"
        assert owner == "none" or owner in names, (
            f"{case_id}: {owner} is neither a Skill nor 'none'"
        )
        assert owner != must_not, f"{case_id}: the query cannot both load and not load {owner}"
