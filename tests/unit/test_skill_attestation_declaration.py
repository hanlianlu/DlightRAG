# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Skill's declarations about its own work: parsed, checked at discovery, refused when false.

``attestation:`` names the Skill-relative script that verifies a document the Skill produces, and
``required_reads:`` (ADR 0057) names the documents a Run that loads the Skill must read. Each is a
claim the Skill makes about itself, so this layer does two things with it and nothing else: it
reads the claim, and it refuses a Skill whose claim the Skill cannot honour. It never runs the
script and never checks a read -- running the verifier and making the reads are the Run's business,
and whether the Run did is derived from the Run's own records in
``dlightrag.application.answer.provenance``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pytest

from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.agent.skills import (
    PublishSkillInput,
    RequiredRead,
    SkillCatalog,
    builtin_skills_root,
    publish_skill_tool,
)
from dlightrag.engine.agent.tools.contracts import ToolResult
from tests.support.skills import SEALED_ENGINE_LABEL, SEALED_SKILL, sealed_skills_root
from tests.support.tool_helpers import tool_runtime

_VERIFIER = "scripts/check.py"


def _skill(
    root: Path,
    directory: str,
    *,
    name: str,
    attestation: str | None = None,
    files: dict[str, str] | None = None,
    description: str = "Use when a test needs one Skill.",
) -> None:
    """Write one Skill directory, declaring ``attestation`` only when the test asks for it."""
    declaration = f"attestation: {attestation}\n" if attestation is not None else ""
    target = root / directory
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{declaration}---\nbody\n",
        encoding="utf-8",
    )
    for relative, content in (files or {}).items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _catalog(root: Path) -> SkillCatalog:
    return SkillCatalog.discover(global_root=root)


def test_a_declaration_is_parsed_and_the_file_it_names_is_readable(tmp_path: Path) -> None:
    root = tmp_path / "global"
    _skill(root, "review", name="review", attestation=_VERIFIER, files={_VERIFIER: "print('ok')\n"})

    catalog = _catalog(root)

    (skill,) = catalog.metadata
    assert skill.attestation == _VERIFIER
    assert catalog.read("review", _VERIFIER) == "print('ok')\n"


def test_a_declaration_naming_a_file_the_skill_does_not_carry_refuses_the_skill(
    tmp_path: Path,
) -> None:
    """A claim that cannot be honoured is a false claim, and discovery is where it is cheapest."""
    root = tmp_path / "global"
    _skill(root, "review", name="review", attestation=_VERIFIER)

    catalog = _catalog(root)

    assert catalog.metadata == ()
    assert ("review", "SKILL.md") in catalog.protected_documents()
    try:
        catalog.read("review")
    except ValueError as exc:
        assert "no Agent Skill is named 'review'" in str(exc)
    else:  # pragma: no cover - the refusal above is the assertion
        raise AssertionError("a Skill with an unhonourable declaration was still loadable")


def test_a_skill_without_a_declaration_is_valid_and_states_the_absence(tmp_path: Path) -> None:
    """No declaration is a legitimate Skill that simply cannot attest anything -- said as None."""
    root = tmp_path / "global"
    _skill(root, "review", name="review")

    (skill,) = _catalog(root).metadata

    assert skill.attestation is None
    assert skill.attestation != ""


def test_a_declaration_that_would_leave_the_skill_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "global"
    _skill(root, "escape", name="escape", attestation="../outside.py")
    _skill(root, "absolute", name="absolute", attestation="/usr/bin/python")
    _skill(root, "empty", name="empty", attestation='""')

    catalog = _catalog(root)

    assert catalog.metadata == ()


def test_a_declaration_naming_a_link_is_not_the_file_it_points_at(tmp_path: Path) -> None:
    """``load_skill`` refuses a link, so a verifier reached through one is not carried either."""
    root = tmp_path / "global"
    _skill(root, "review", name="review", attestation="scripts/linked.py")
    real = tmp_path / "outside.py"
    real.write_text("print('outside')\n", encoding="utf-8")
    (root / "review" / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "review" / "scripts" / "linked.py").symlink_to(real)

    assert _catalog(root).metadata == ()


async def test_publishing_a_declaration_the_payload_carries_is_accepted(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    result = await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name="review",
            files={
                "SKILL.md": (
                    "---\nname: review\ndescription: Use when reviewing.\n"
                    f"attestation: {_VERIFIER}\n---\nbody\n"
                ),
                _VERIFIER: "print('ok')\n",
            },
        ),
        tool_runtime(tool_name="publish_skill"),
    )

    assert not result.is_error, result.text_content
    (skill,) = SkillCatalog.discover(owner_root=owner_root).metadata
    assert skill.attestation == _VERIFIER


async def test_publishing_a_declaration_the_payload_lacks_is_refused(tmp_path: Path) -> None:
    """The install boundary applies the same rule discovery does, so no Run reads a false claim."""
    owner_root = tmp_path / "owner"
    result = await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name="review",
            files={
                "SKILL.md": (
                    "---\nname: review\ndescription: Use when reviewing.\n"
                    f"attestation: {_VERIFIER}\n---\nbody\n"
                )
            },
        ),
        tool_runtime(tool_name="publish_skill"),
    )

    assert isinstance(result, ToolResult)
    assert f"declares attestation '{_VERIFIER}'" in result.text_content
    assert not (owner_root / "review").exists()


def test_a_packaged_hub_declares_the_delivery_entry_point_it_ships() -> None:
    """A built-in with a mechanical gate: its hub names the command that runs and seals it."""
    catalog = SkillCatalog.discover(builtin_root=sealed_skills_root())

    skill = next(item for item in catalog.metadata if item.name == SEALED_SKILL)

    assert skill.attestation == "scripts/case_validate.py"
    assert catalog.read(skill.name, skill.attestation).strip()


# ---------------------------------------------------------------------------
# required_reads
# ---------------------------------------------------------------------------

_ONE = "references/one.md"
_TWO = "references/two.md"
_ALTERNATIVES = ("references/a.html", "references/b.html")


def _declaring(root: Path, directory: str, *, name: str, required_reads: str, files=()) -> None:
    """Write one Skill whose frontmatter carries ``required_reads`` exactly as the test spells it."""
    target = root / directory
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use when a test needs one Skill.\n"
        f"{required_reads}---\nbody\n",
        encoding="utf-8",
    )
    for relative in files:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")


async def _publish_declaring(owner_root: Path, required_reads: str, *, files=(_ONE, _TWO)):
    """Publish a Skill declaring ``required_reads`` with ``files`` beside its SKILL.md."""
    return await publish_skill_tool(owner_root).execute(
        PublishSkillInput(
            name="review",
            files={
                "SKILL.md": (
                    "---\nname: review\ndescription: Use when reviewing.\n"
                    f"{required_reads}---\nbody\n"
                ),
                **{relative: f"# {relative}\n" for relative in files},
            },
        ),
        tool_runtime(tool_name="publish_skill"),
    )


def test_a_skill_declares_one_file_or_a_choice_of_files_with_the_step_for_reading_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "global"
    _declaring(
        root,
        "review",
        name="review",
        required_reads=(
            "required_reads:\n"
            f"  - path: {_ONE}\n"
            "    when: before writing the fragment\n"
            "  - any_of:\n"
            f"      - {_ALTERNATIVES[0]}\n"
            f"      - {_ALTERNATIVES[1]}\n"
            "    when: before writing, the example nearest the request\n"
        ),
        files=(_ONE, *_ALTERNATIVES),
    )

    (skill,) = _catalog(root).metadata

    assert skill.required_reads == (
        RequiredRead(paths=(_ONE,), when="before writing the fragment"),
        RequiredRead(paths=_ALTERNATIVES, when="before writing, the example nearest the request"),
    )


def test_a_skill_that_declares_no_reads_states_the_absence_as_an_empty_tuple(
    tmp_path: Path,
) -> None:
    root = tmp_path / "global"
    _skill(root, "review", name="review")
    _declaring(root, "listed", name="listed", required_reads="required_reads: []\n")

    assert [skill.required_reads for skill in _catalog(root).metadata] == [(), ()]


def test_a_path_is_normalized_and_when_is_collapsed_to_one_line(tmp_path: Path) -> None:
    """Discovery checks the walk's own spelling, so ``./x`` must name ``x`` and not nothing."""
    root = tmp_path / "global"
    _declaring(
        root,
        "review",
        name="review",
        required_reads=(
            "required_reads:\n"
            f"  - path: ./{_ONE}\n"
            "    when: >\n"
            "      before the build,\n"
            "      every time\n"
        ),
        files=(_ONE,),
    )

    (skill,) = _catalog(root).metadata

    assert skill.required_reads == (
        RequiredRead(paths=(_ONE,), when="before the build, every time"),
    )


def test_the_bounds_are_inclusive(tmp_path: Path) -> None:
    """Sixteen entries, eight alternatives and a hundred and twenty characters are all allowed."""
    root = tmp_path / "global"
    paths = tuple(f"references/r{n}.md" for n in range(16))
    entries = "".join(f"  - path: {path}\n    when: {'w' * 120}\n" for path in paths)
    _declaring(root, "many", name="many", required_reads=f"required_reads:\n{entries}", files=paths)
    alternatives = tuple(f"references/a{n}.md" for n in range(8))
    choice = "".join(f"      - {path}\n" for path in alternatives)
    _declaring(
        root,
        "choice",
        name="choice",
        required_reads=f"required_reads:\n  - any_of:\n{choice}    when: x\n",
        files=alternatives,
    )

    served = {skill.name: skill for skill in _catalog(root).metadata}

    assert len(served["many"].required_reads) == 16
    assert all(len(entry.when) == 120 for entry in served["many"].required_reads)
    assert served["choice"].required_reads[0].paths == alternatives


_REFUSALS = {
    "an unknown key": (
        f"  - path: {_ONE}\n    when: x\n    paths: y\n",
        "unknown key(s) paths",
    ),
    "a misspelled when": (f"  - path: {_ONE}\n    whn: x\n", "unknown key(s) whn"),
    "both path and any_of": (
        f"  - path: {_ONE}\n    any_of:\n      - {_TWO}\n    when: x\n",
        "exactly one of 'path' or 'any_of'",
    ),
    "neither path nor any_of": ("  - when: x\n", "exactly one of 'path' or 'any_of'"),
    "no when": (f"  - path: {_ONE}\n", "needs 'when'"),
    "an empty when": (f"  - path: {_ONE}\n    when: '   '\n", "needs 'when'"),
    "a when that is not text": (f"  - path: {_ONE}\n    when: 3\n", "needs 'when'"),
    "a when over 120 characters": (
        f"  - path: {_ONE}\n    when: {'w' * 121}\n",
        "'when' is 121 characters; the limit is 120",
    ),
    "an empty any_of": ("  - any_of: []\n    when: x\n", "non-empty list"),
    "an any_of that is not a list": (
        f"  - any_of: {_ONE}\n    when: x\n",
        "non-empty list",
    ),
    "nine any_of paths": (
        "  - any_of:\n"
        + "".join(f"      - references/n{n}.md\n" for n in range(9))
        + "    when: x\n",
        "'any_of' names 9 paths; the limit is 8",
    ),
    "a duplicate in any_of": (
        f"  - any_of:\n      - {_ONE}\n      - ./{_ONE}\n    when: x\n",
        "the same path twice",
    ),
    "SKILL.md": ("  - path: SKILL.md\n    when: x\n", "SKILL.md is read by loading the Skill"),
    "SKILL.md spelled ./SKILL.md": (
        "  - path: ./SKILL.md\n    when: x\n",
        "SKILL.md is read by loading the Skill",
    ),
    "a path that leaves the Skill": (
        "  - path: ../outside.md\n    when: x\n",
        "is not Skill-relative",
    ),
    "an absolute path": ("  - path: /etc/passwd\n    when: x\n", "is not Skill-relative"),
    "a path that is not text": ("  - path: 3\n    when: x\n", "must be a Skill-relative file path"),
    "an entry that is not a mapping": (f"  - {_ONE}\n", "must be a mapping"),
}


@pytest.mark.parametrize(("entries", "message"), _REFUSALS.values(), ids=_REFUSALS.keys())
async def test_a_declaration_with_a_fault_is_refused_where_it_is_made(
    tmp_path: Path, entries: str, message: str
) -> None:
    result = await _publish_declaring(tmp_path / "owner", f"required_reads:\n{entries}")

    assert result.is_error, result.text_content
    assert "Skill publication rejected: SKILL.md" in result.text_content
    assert message in result.text_content
    assert not (tmp_path / "owner" / "review").exists()


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (f"required_reads: {_ONE}\n", "must be a list of entries"),
        (
            "required_reads:\n"
            + "".join(f"  - path: references/r{n}.md\n    when: x\n" for n in range(17)),
            "names 17 entries; the limit is 16",
        ),
    ],
)
async def test_the_declaration_as_a_whole_is_checked_too(
    tmp_path: Path, declaration: str, message: str
) -> None:
    result = await _publish_declaring(tmp_path / "owner", declaration)

    assert result.is_error and message in result.text_content


def test_discovery_does_not_serve_a_skill_whose_required_read_is_not_carried(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A read that cannot be made is a false claim, so the Skill is held and stays protected."""
    root = tmp_path / "global"
    _declaring(
        root,
        "review",
        name="review",
        required_reads=f"required_reads:\n  - path: {_ONE}\n    when: first\n",
        files=(),
    )

    with caplog.at_level(logging.WARNING):
        catalog = _catalog(root)

    assert catalog.metadata == ()
    assert ("review", "SKILL.md") in catalog.protected_documents()
    assert "required_reads names 'references/one.md'" in caplog.text


def test_one_missing_path_of_an_any_of_holds_the_skill(tmp_path: Path) -> None:
    """Every path declared must be a file, not just the one a Run happens to pick."""
    root = tmp_path / "global"
    _declaring(
        root,
        "review",
        name="review",
        required_reads=(
            f"required_reads:\n  - any_of:\n      - {_ALTERNATIVES[0]}\n"
            f"      - {_ALTERNATIVES[1]}\n    when: x\n"
        ),
        files=(_ALTERNATIVES[0],),
    )

    assert _catalog(root).metadata == ()


def test_a_required_read_naming_a_link_is_not_the_file_it_points_at(tmp_path: Path) -> None:
    root = tmp_path / "global"
    _declaring(
        root,
        "review",
        name="review",
        required_reads=f"required_reads:\n  - path: {_ONE}\n    when: x\n",
    )
    real = tmp_path / "outside.md"
    real.write_text("outside\n", encoding="utf-8")
    (root / "review" / "references").mkdir(parents=True, exist_ok=True)
    (root / "review" / "references" / "one.md").symlink_to(real)

    assert _catalog(root).metadata == ()


async def test_publishing_the_reads_the_payload_carries_is_accepted(tmp_path: Path) -> None:
    owner_root = tmp_path / "owner"
    result = await _publish_declaring(
        owner_root, f"required_reads:\n  - path: ./{_ONE}\n    when: before writing\n"
    )

    assert not result.is_error, result.text_content
    (skill,) = SkillCatalog.discover(owner_root=owner_root).metadata
    assert skill.required_reads == (RequiredRead(paths=(_ONE,), when="before writing"),)


async def test_publishing_a_read_the_payload_lacks_is_refused(tmp_path: Path) -> None:
    """The install boundary applies the rule discovery does, to every path of an any_of."""
    owner_root = tmp_path / "owner"
    result = await _publish_declaring(
        owner_root,
        f"required_reads:\n  - any_of:\n      - {_ONE}\n      - references/missing.md\n"
        "    when: x\n",
        files=(_ONE,),
    )

    assert result.is_error
    assert "declares required_reads 'references/missing.md'" in result.text_content
    assert not (owner_root / "review").exists()


def test_interactive_html_declares_its_attribute_reference_before_it_writes() -> None:
    """The starter carries the example's shape; the attribute reference is the read before writing."""
    catalog = SkillCatalog.discover(builtin_root=builtin_skills_root())

    skill = next(item for item in catalog.metadata if item.name == "interactive-html")

    (entry,) = skill.required_reads
    assert entry.paths == ("references/fragment.md",)
    assert entry.when == "before writing or restructuring the fragment"
    assert catalog.read(skill.name, "references/fragment.md").strip()


def test_engine_scripts_are_exact_protected_but_not_in_the_publication_prose_index() -> None:
    """Reader masking still covers executable Skill code; the artifact text gate does not.

    Engine/verifier constants appear in the case and page those scripts produce, so the
    publication prose index excludes the Skill's declared kit; a reader answer quoting the
    script is still masked by the full protected-text index.
    """
    root = sealed_skills_root()
    catalog = SkillCatalog.discover(builtin_root=root)
    reader_texts = {(name, relative) for name, relative, _ in catalog.protected_texts()}
    publication_texts = {(name, relative) for name, relative, _ in catalog.publication_texts()}
    script = "scripts/engine.py"
    reference = "references/gate-audit.md"
    skill_name = SEALED_SKILL
    assert (skill_name, script) in reader_texts
    assert (skill_name, script) not in publication_texts
    assert (skill_name, reference) in reader_texts
    assert (skill_name, reference) in publication_texts

    reader_index = ProtectedTextIndex.build(catalog.protected_texts())
    publication_index = ProtectedTextIndex.build(catalog.publication_texts())
    # A label written by the engine, not copied from a prose reference.
    engine_text = SEALED_ENGINE_LABEL
    reference_text = catalog.read(skill_name, reference)
    assert reader_index.scan(engine_text)
    assert publication_index.scan(engine_text) == ()
    assert publication_index.scan(reference_text)

    digests = catalog.protected_digests()
    engine = root / skill_name / "scripts" / "engine.py"
    assert hashlib.sha256(engine.read_bytes()).hexdigest() in digests
