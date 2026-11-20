# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Every built-in Skill hub: it routes, and it carries what it says it carries.

A loaded `SKILL.md` body costs context in every Run that loads it, so a hub is a centre that
dispatches long material to spokes rather than a document that contains it. What these tests
assert is that structure and the product's own published limits: the 500-line formatting rule,
the 4,000-token hub budget, the 1,200-character step rule for a hub with a `## Steps` section,
and the 20,000-token budget on what a Skill declares as required reading (ADR 0048 / ADR 0057
amendments, 2026-10-19). The numbers are published limits now, not probes: a hub over them is
refused at publication and red here.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import tiktoken

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
)
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.skills import (
    _MAX_DESCRIPTION_CHARS,
    _MAX_HUB_TOKENS,
    _MAX_REQUIRED_READING_TOKENS,
    LoadSkillInput,
    SkillCatalog,
    builtin_skills_root,
    load_skill_tool,
)
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.tokens import estimate_tokens
from tests.support.skills import SEALED_ENGINE, SEALED_SKILL, sealed_skills_root
from tests.support.tool_helpers import tool_runtime

#: The published guidance for a Skill's router trigger; see the test that uses it.
_MAX_DESCRIPTION_WORDS = 50

# A hub names what it dispatches to as a backticked path relative to the Skill directory, which
# is how `load_skill(name, path=...)` takes it. Directories of the Agent's workspace rather than
# of a Skill are written in this same shape so the two kinds cannot be confused.
_DISPATCHED_PATH = re.compile(r"`([a-zA-Z0-9_-]+/[a-zA-Z0-9_./-]+\.(?:md|html|py|json|css|js))`")
_WORKSPACE_ROOTS = frozenset({"tmp", "artifacts", "report"})

#: A section heading's letter, and a reference to one — `§D` and `§D1` alike, since an item
#: reference names its section too, while `§10` (another standard's paragraph) has no letter.
_SECTION = re.compile(r"^##+ ([A-Z0-9]+)[.\s—]", re.M)
_SECTION_BLOCKS = re.compile(r"^##+ ", re.M)
_BARE_SECTION_REF = re.compile(r"§([A-Z])(?![A-Za-z])")


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=builtin_skills_root())


def _dispatched(catalog: SkillCatalog, name: str) -> list[str]:
    hub = catalog.read(name)
    return sorted(
        relative
        for relative in set(_DISPATCHED_PATH.findall(hub))
        if relative.split("/", 1)[0] not in _WORKSPACE_ROOTS
    )


def test_every_builtin_routes_by_a_bounded_description() -> None:
    skills = _catalog().metadata
    assert skills, "no built-in Skill was discovered"

    for skill in skills:
        assert skill.source == "builtin"
        # The description is the router trigger and the only part of a Skill always in context.
        # Its bound is the product's own, imported rather than restated: a copy here would drift
        # from the value `publish_skill` enforces.
        assert "Use when" in skill.description, skill.name
        assert len(skill.description) <= _MAX_DESCRIPTION_CHARS, skill.name


def test_every_description_fits_the_index_budget() -> None:
    """The description is paid for by every session and every user, so it has a word budget.

    The published guidance for a Skill's router trigger is to keep it under 50 words and to
    write it as a trigger rather than a summary of the work; the hard character cap the
    publisher enforces (`_MAX_DESCRIPTION_CHARS`) is a different and much looser limit. A
    description that opens with what the Skill does, or that summarises its workflow, spends
    the budget on words that cannot help the model decide whether to load it.
    """
    for skill in _catalog().metadata:
        words = len(skill.description.split())
        assert words <= _MAX_DESCRIPTION_WORDS, f"{skill.name}: description is {words} words"
        assert skill.description.startswith("Use when"), (
            f"{skill.name}: the description opens with {skill.description.split('.')[0][:40]!r} "
            "rather than the trigger"
        )


def test_a_hub_carries_every_path_it_names() -> None:
    """The centre-and-spokes rule, asserted as reachability rather than as a size.

    A hub that names a Skill path the catalog cannot serve is worse than a hub with no
    references at all: the Agent is told to load something that is not there. Whether a spoke is
    worth its bytes is a judgment about content — an example a hub dispatches to can legitimately
    be shorter than the hub itself — so this test does not pretend a size answers it.
    """
    catalog = _catalog()

    for skill in catalog.metadata:
        for relative in _dispatched(catalog, skill.name):
            try:
                loaded = catalog.read(skill.name, relative)
            except FileNotFoundError:
                raise AssertionError(
                    f"{skill.name}: the hub names {relative!r} but the Skill does not carry it. "
                    "Write it as a workspace path (tmp/…, artifacts/…) or ship the file."
                ) from None
            assert loaded.strip(), f"{skill.name}: {relative} is empty"


def test_a_hub_that_splits_its_material_dispatches_all_of_it() -> None:
    """A hub with spokes names them, because an unnamed spoke costs bytes and reaches nobody."""
    catalog = _catalog()

    for skill in catalog.metadata:
        dispatched = set(_dispatched(catalog, skill.name))
        if not dispatched:
            continue
        hub = catalog.read(skill.name)
        for relative in dispatched:
            assert relative in hub, f"{skill.name}: {relative} is dispatched but not named"


def test_no_builtin_hub_exceeds_the_industry_line_rule() -> None:
    """ADR 0048: the only published limit is on the hub — 500 lines — and it is a hard rule here.

    The previous uniform 50,000-character cap was simultaneously looser than this rule for hubs
    and stricter than the industry for references and scripts, which have no size limit at all.
    """
    for skill in _catalog().metadata:
        hub = _catalog().read(skill.name)
        lines = hub.count("\n") + 1
        assert lines <= 500, f"{skill.name}: hub is {lines} lines (guidance caps it at 500)"


def test_every_reference_is_reachable_from_its_hub() -> None:
    """A spoke no path reaches is a document nobody will read, and the hub is the only map.

    The reverse direction is checked above — every path a hub names exists. This is the direction
    that failed in practice: a reference was created, three documents cited it, and the hub's Files
    table never listed it, so an agent reading the index could not know it was there or which step
    it belonged to.
    """
    root = Path(str(builtin_skills_root()))
    for skill in _catalog().metadata:
        hub = _catalog().read(skill.name)
        references = root / skill.name / "references"
        if not references.is_dir():
            continue
        for document in sorted(references.glob("*.md")):
            relative = f"references/{document.name}"
            assert relative in hub, f"{skill.name}: {relative} is never named by its hub"


def test_a_bare_section_reference_resolves_somewhere() -> None:
    """`§D` with no file named beside it must exist in the file that says it, or in one named there.

    A section letter is a pointer, and a pointer that resolves nowhere costs a reader the document
    it was meant to open. The defect this catches lived in a spoke for a whole release: the letter
    had survived a renumbering that moved the section it named, and the same file still labelled
    its items with the old letter, so the fossil proved what had happened.
    """
    root = Path(str(builtin_skills_root()))
    for skill in _catalog().metadata:
        documents = sorted((root / skill.name).rglob("*.md"))
        sections = {
            document.name: set(_SECTION.findall(document.read_text())) for document in documents
        }
        for document in documents:
            text = document.read_text()
            for chunk in _SECTION_BLOCKS.split(text)[1:]:
                named = {name for name in sections if name in chunk}
                for reference in _BARE_SECTION_REF.finditer(chunk):
                    letter = reference.group(1)
                    if letter in sections[document.name]:
                        continue
                    assert any(letter in sections[name] for name in named), (
                        f"{skill.name}/{document.name}: §{letter} resolves neither in this file nor "
                        f"in one it names here ({sorted(named) or 'none named'})"
                    )


def test_hub_sizes_are_reported() -> None:
    """Print what each hub costs. Deliberately not an assertion — it is a reviewer's input.

    A hub's growth is worth a look when it starts carrying what a spoke should hold, and only a
    reader can tell that. The measured figure is here so the question is asked with the real
    number in hand instead of a threshold someone invented.
    """
    catalog = _catalog()
    encoder = tiktoken.get_encoding("o200k_base")

    rows = []
    for skill in catalog.metadata:
        hub = catalog.read(skill.name)
        spoke_words = sum(
            len(catalog.read(skill.name, path).split()) for path in _dispatched(catalog, skill.name)
        )
        rows.append((skill.name, len(encoder.encode(hub)), spoke_words))

    for name, tokens, spoke_words in sorted(rows, key=lambda row: -row[1]):
        print(f"  {name:<24}{tokens:>6} tokens in the hub{spoke_words:>8} words in its spokes")


def _markdown_files(node, *, prefix: str) -> list[tuple[str, str]]:
    """Every markdown document under a Skill, as a Skill-relative path and its text.

    Written against the ``Traversable`` protocol rather than ``Path``, because a packaged
    built-in is not a directory on disk and has no ``rglob``.
    """
    found: list[tuple[str, str]] = []
    for child in sorted(node.iterdir(), key=lambda item: item.name):
        if child.name.startswith("."):
            continue
        relative = f"{prefix}{child.name}"
        if child.is_dir():
            found.extend(_markdown_files(child, prefix=f"{relative}/"))
        elif child.name.endswith(".md"):
            found.append((relative, child.read_text(encoding="utf-8")))
    return found


def test_a_skill_never_points_at_a_document_that_is_not_there() -> None:
    """A spoke is reached by reading it, so a path that does not resolve is a dead instruction.

    A Skill is written by hand and revised in pieces, and the failure this catches is the one a
    reviewer cannot see: prose that cites a file or a section by name after the name changed. The
    cost of the citation being wrong is paid by a model that follows it and finds nothing.
    """
    catalog = _catalog()
    missing: list[str] = []

    for skill in catalog.metadata:
        root = builtin_skills_root().joinpath(skill.name)
        documents = dict(_markdown_files(root, prefix=""))
        for relative, text in documents.items():
            for cited in set(re.findall(r"`((?:references|scripts)/[a-zA-Z0-9_.-]+\.md)`", text)):
                if cited not in documents:
                    missing.append(f"{skill.name}/{relative} cites {cited}, which does not exist")

    assert not missing, "\n".join(missing)


# ---------------------------------------------------------------------------
# Handing a packaged script over: followable pages, or one exact copy
# ---------------------------------------------------------------------------

#: A 128k-window profile: its per-result ceiling is tens of thousands of tokens, so the
#: half-megabyte engine arrives over several pages rather than as a silently cut prefix.
_PAGING_PROFILE = ModelProfile(context_window_tokens=128_000, max_output_tokens=16_384)
_ENGINE = (SEALED_SKILL, SEALED_ENGINE)
_CONTINUATION = re.compile(r"\[more; load_skill\(name='([^']+)', path='([^']+)', offset=(\d+)\)\]")


@pytest.mark.asyncio
async def test_a_packaged_script_arrives_in_pages_that_reconstruct_it_or_as_one_exact_copy(
    tmp_path: Path,
) -> None:
    """The loader owns the fit for a built-in script, and its bytes are the contract.

    `sealed-report` tells the model to run `scripts/engine.py` and its verifier compares the
    engine's sha256, so every byte has to reach the workspace: either as
    followable pages whose concatenation is the file, or copied whole by ``write_to``. A model
    that was handed a prefix instead would run a script that was never audited.
    """
    packaged = sealed_skills_root().joinpath(*_ENGINE).read_bytes()
    ceiling = CONTEXT_POLICY.tool_result_ceiling(_PAGING_PROFILE)
    catalog = SkillCatalog.discover(builtin_root=sealed_skills_root())
    tool = load_skill_tool(
        lambda: catalog,
        model_profile=_PAGING_PROFILE,
        environment=LocalExecutionEnvironment(tmp_path / "workspace"),
        scheduler=AccessScheduler(),
    )
    marker = f"--- {_ENGINE[0]}/{_ENGINE[1]} ---\n"

    pages: list[str] = []
    offset = 0
    while True:
        result = await tool.execute(
            LoadSkillInput(name=_ENGINE[0], path=_ENGINE[1], offset=offset),
            tool_runtime(tool_name="load_skill"),
        )
        assert not result.is_error, result.text_content
        assert estimate_tokens(result.text_content) <= ceiling
        body = result.text_content.split(marker, 1)[1]
        window, separator, last_line = body.rpartition("\n")
        match = _CONTINUATION.fullmatch(last_line) if separator else None
        if match is None:
            pages.append(body)
            break
        pages.append(window)
        offset = int(match.group(3))
        assert len(pages) < 100

    assert len(pages) > 1
    assert (
        hashlib.sha256("".join(pages).encode("utf-8")).hexdigest()
        == hashlib.sha256(packaged).hexdigest()
    )

    copied = await tool.execute(
        LoadSkillInput(name=_ENGINE[0], path=_ENGINE[1], write_to="tmp/engine.py"),
        tool_runtime(tool_name="load_skill"),
    )

    assert not copied.is_error, copied.text_content
    assert (
        hashlib.sha256((tmp_path / "workspace" / "tmp" / "engine.py").read_bytes()).hexdigest()
        == hashlib.sha256(packaged).hexdigest()
    )


def test_no_builtin_hub_exceeds_the_token_budget() -> None:
    """ADR 0048 amendment: the hub costs what a reader pays; 500 lines is only the formatting rule."""
    catalog = _catalog()
    for skill in catalog.metadata:
        tokens = estimate_tokens(catalog.read(skill.name))
        assert tokens <= _MAX_HUB_TOKENS, f"{skill.name}: hub is {tokens} tokens"


def test_no_builtin_step_runs_to_a_section() -> None:
    """A style guideline, not a publication rule: a step over 3,000 characters reads as a section."""
    maximum = 3_000
    catalog = _catalog()
    for skill in catalog.metadata:
        hub = catalog.read(skill.name)
        if "## Steps" not in hub:
            continue
        for line in hub.splitlines():
            if re.match(r"^\d+\.\s", line):
                assert len(line) <= maximum, (skill.name, len(line))


def test_a_builtin_skill_declares_only_affordable_reading() -> None:
    """ADR 0057 amendment: required reading is short gate cards, not the deep references."""
    catalog = _catalog()
    for skill in catalog.metadata:
        if not skill.required_reads:
            continue
        total = sum(
            max(estimate_tokens(catalog.read(skill.name, path)) for path in entry.paths)
            for entry in skill.required_reads
        )
        assert total <= _MAX_REQUIRED_READING_TOKENS, f"{skill.name} declares {total} tokens"
