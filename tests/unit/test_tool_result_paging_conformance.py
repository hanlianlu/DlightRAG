# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Independent conformance checks: loader paging, the result ceiling, and ``write_to``.

Task-4 (S2-D) is adversarial verification, not implementation.  Every assertion is made
against the shipped Skill document, its real sha256 and the real ``ContextPolicy``, so a
change that only *looks* complete cannot pass: reconstruction is byte equality and a
``write_to`` copy is a digest comparison, never a substring check.

The decisive facts are:

* the pages a loader returns concatenate to the packaged document, byte for byte, and the
  offset each page names is exactly where the previous page stopped;
* every page is inside ``ContextPolicy.tool_result_ceiling`` and the runtime's own fit
  leaves it untouched (``observed_truncated`` absent);
* a ``write_to`` copy hashes to the packaged file's sha256 and its model-visible text
  carries no line of the script.

Paging is a guard for documents longer than the model's window, not a forced chunker: on a
1M-token profile the shipped document fits one result and must come back whole.  The
offset machinery is therefore exercised on a profile small enough to force several pages.

The run that produced the task report's raw evidence is this file plus
``uv run pytest tests/unit/test_tool_result_paging_conformance.py -q``.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from importlib.resources.abc import Traversable
from pathlib import Path

import pytest

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
)
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.skills import (
    LoadSkillInput,
    SkillCatalog,
    SkillsBundleFactory,
    load_skill_document_pair,
    load_skill_tool,
)
from dlightrag.engine.agent.tools.capacity import fit_tool_result
from dlightrag.engine.agent.tools.contracts import ToolResult, ToolResultCapacityError
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.ai.tokens import estimate_tokens
from tests.support.skills import SEALED_ENGINE, SEALED_SKILL, sealed_skills_root
from tests.support.tool_helpers import tool_runtime

SKILL = SEALED_SKILL
DOCUMENT = SEALED_ENGINE

#: One million tokens of window, as the task names it.  ``max_output_tokens`` is left
#: unstated so the policy's own requested output reserve applies; this is the profile
#: shape the ceiling is specified against.
PROFILE_1M = ModelProfile(context_window_tokens=1_000_000)

#: A profile small enough that the shipped 130,000-token document cannot fit one page
#: (hard input 114,688, ceiling 40,000), used to prove the offset machinery actually
#: pages rather than only fitting by luck.
PROFILE_SMALL = ModelProfile(context_window_tokens=131_072, max_output_tokens=16_384)

#: How many pages a runaway loader may produce before the test calls it a loop.
_MAX_PAGES = 64

#: The continuation the loader appends, e.g. ``[more; load_skill(name='x', path='y',
#: offset=1234)]``.  The exact next offset is what makes it followable.
_CONTINUATION = re.compile(r"\[more;\s*load_skill\((?P<args>[^)]*)\)\]\s*\Z")
_ARGUMENT = re.compile(r"(?P<key>[A-Za-z_]\w*)\s*=\s*(?P<value>'[^']*'|\"[^\"]*\"|-?\d+)")


def _packaged_path() -> Traversable:
    """The document as a file on disk: the bytes ``write_to`` must reproduce exactly."""
    return sealed_skills_root() / SKILL / DOCUMENT


def _packaged_bytes() -> bytes:
    return _packaged_path().read_bytes()


def _catalog() -> SkillCatalog:
    return SkillCatalog.discover(builtin_root=sealed_skills_root())


def _loader(
    catalog: SkillCatalog, *, profile: ModelProfile, workspace: Path | None = None
) -> object:
    """Build the packaged loader for one model profile and one workspace.

    ``model_profile`` is the seam the loader work line added to ``load_skill_tool``; the
    environment plus scheduler is the Agent Workspace surface ``write_to`` copies through,
    the same one ``write`` uses.
    """
    if workspace is None:
        return load_skill_tool(lambda: catalog, model_profile=profile)
    return load_skill_tool(
        lambda: catalog,
        model_profile=profile,
        environment=LocalExecutionEnvironment(workspace),
        scheduler=AccessScheduler(),
    )


async def _call(
    args: LoadSkillInput, *, profile: ModelProfile, workspace: Path | None = None
) -> object:
    tool = _loader(_catalog(), profile=profile, workspace=workspace)
    return await tool.execute(args, tool_runtime(tool_name="load_skill"))  # type: ignore[attr-defined]


def _arguments(match: re.Match[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in _ARGUMENT.finditer(match.group("args")):
        value = item.group("value")
        parsed[item.group("key")] = value[1:-1] if value[0] in "'\"" else value
    return parsed


def _page_bytes(
    text: str, *, offset: int, document: str, path: str = DOCUMENT
) -> tuple[str, int | None]:
    """The document bytes one result carries, and the offset its continuation names.

    The result is ``banner + --- name/path --- + slice [+ continuation]``.  An intermediate
    page is pinned to the exact length its own continuation names, so framing (a separator
    line, a blank line before the marker) can never be mistaken for document bytes; the
    final page is trimmed to the longest suffix that really is the document at that offset.
    """
    pair = load_skill_document_pair(text)
    assert pair == (SKILL, path), f"result does not name the requested document: {pair!r}"
    marker = f"--- {SKILL}/{path} ---\n"
    body = text[text.index(marker) + len(marker) :]
    match = _CONTINUATION.search(body)
    if match is None:
        raw, next_offset = body, None
    else:
        raw, next_offset = body[: match.start()], int(_arguments(match).get("offset", "-1"))
        assert next_offset > offset, f"continuation did not advance past offset {offset}"
    if next_offset is None:
        candidate = raw
        while candidate and not document.startswith(candidate, offset):
            candidate = candidate[:-1]
    else:
        length = next_offset - offset
        assert length <= len(raw), (
            f"continuation names offset {next_offset} (length {length}) but only "
            f"{len(raw)} characters followed the marker"
        )
        candidate = raw[:length]
    assert candidate, f"page at offset {offset} carried no document bytes"
    assert document.startswith(candidate, offset), (
        f"page at offset {offset} is not the document's bytes; it starts {candidate[:60]!r}"
    )
    return candidate, next_offset


async def _pages(
    *, profile: ModelProfile, workspace: Path | None = None
) -> AsyncIterator[tuple[object, str, int, int | None]]:
    """Follow the continuations from offset 0, yielding each real page.

    Yields ``(result, slice, offset, next_offset)``.  Every page's named offset must equal
    the byte the previous page stopped at, which is what makes the paging resumable.
    """
    document = _packaged_path().read_text(encoding="utf-8")
    offset = 0
    for _ in range(_MAX_PAGES):
        result = await _call(
            LoadSkillInput(name=SKILL, path=DOCUMENT, offset=offset),
            profile=profile,
            workspace=workspace,
        )
        text = result.text_content  # type: ignore[attr-defined]
        slice_, next_offset = _page_bytes(text, offset=offset, document=document)
        yield result, slice_, offset, next_offset
        if next_offset is None:
            return
        assert next_offset == offset + len(slice_), (
            f"continuation named offset {next_offset}, but the page ended at {offset + len(slice_)}"
        )
        offset = next_offset
    raise AssertionError(f"paging did not terminate within {_MAX_PAGES} pages")


async def _collect(*, profile: ModelProfile) -> list[tuple[object, str, int, int | None]]:
    return [page async for page in _pages(profile=profile)]


@pytest.fixture(scope="module")
def document() -> str:
    return _packaged_path().read_text(encoding="utf-8")


def test_the_packaged_script_is_large_enough_to_have_been_cut_before(document: str) -> None:
    """The pre-change defect's precondition: the document dwarfs the old 40K reserve.

    Recorded as the baseline the fix must flip: at ``agent-v5`` a loader result was fitted
    at the 40,000-token observation capacity and came back as 159,944 characters ending in
    ``[tool result truncated to the shared model residual]``. The exact figures track the
    script the Skill ships today; the precondition is what this test holds, not the digits.
    """
    assert len(_packaged_bytes()) >= 500_000
    # The exact byte/line/token counts move whenever the Skill ships a legitimate edit; the
    # precondition is that the document is far larger than the old 40K reserve, not those digits.
    assert estimate_tokens(document) > 40_000


async def test_a_document_that_fits_comes_back_whole_with_no_continuation() -> None:
    """Check 1(a)+(c) on a 1M profile: whole, byte-identical, and inside the ceiling.

    The loader is a guard, not a forced chunker: a document inside the profile's ceiling comes
    back as the entire document in one result and no continuation at all. The shipped
    130,000-token engine is no longer such a document — the ceiling is the reserve, 60,000 —
    so this uses a reference that fits, and the engine's own cases are the paging and
    ``write_to`` checks below.
    """
    ceiling = CONTEXT_POLICY.tool_result_ceiling(PROFILE_1M)
    reference = "references/gotchas.md"
    packaged = sealed_skills_root().joinpath(SKILL, reference).read_text(encoding="utf-8")
    assert estimate_tokens(packaged) <= ceiling

    result = await _call(LoadSkillInput(name=SKILL, path=reference, offset=0), profile=PROFILE_1M)
    text = result.text_content  # type: ignore[attr-defined]
    slice_, next_offset = _page_bytes(text, offset=0, document=packaged, path=reference)
    assert next_offset is None, (
        f"a document that fits was paged anyway: continuation names offset {next_offset}"
    )
    assert slice_ == packaged
    assert estimate_tokens(text) <= ceiling


async def test_paging_is_lossless_and_advances_on_a_small_profile(document: str) -> None:
    """Check 1(b)+(c) forced: several pages, each continuation exactly at the next byte.

    On a profile whose ceiling is 40,000 tokens the 130,000-token document cannot be one
    page, so this is the check that the offset mechanism — not the document's size —
    reconstructs it.
    """
    pages = await _collect(profile=PROFILE_SMALL)
    assert len(pages) > 1, "a profile too small for one page returned the whole document"
    for _, _, _, next_offset in pages[:-1]:
        assert next_offset is not None, "an intermediate page carried no continuation"
    rebuilt = "".join(slice_ for _, slice_, _, _ in pages)
    assert rebuilt == document, (
        f"reconstruction differs: {len(rebuilt)} chars vs {len(document)}; pages={len(pages)}"
    )
    assert (
        hashlib.sha256(rebuilt.encode("utf-8")).hexdigest()
        == hashlib.sha256(_packaged_bytes()).hexdigest()
    )
    # The pages tile the document: strictly advancing, no gap and no overlap.
    cursor = 0
    for _, slice_, offset, next_offset in pages:
        assert offset == cursor, f"gap or overlap at offset {offset}, expected {cursor}"
        assert slice_ == document[offset : offset + len(slice_)]
        cursor += len(slice_)
        if next_offset is not None:
            assert next_offset == cursor
    assert cursor == len(document), f"the pages cover {cursor} of {len(document)} characters"
    assert pages[-1][3] is None


async def test_every_page_fits_the_ceiling_and_survives_the_runtime_fit() -> None:
    """Check 2: no loader page can be cut by the runtime, so the defect is impossible.

    Fitting each page at the ceiling replays the runtime's own operation: the text must
    come back unchanged and the runtime's cut fact must stay unset.
    """
    for profile in (PROFILE_1M, PROFILE_SMALL):
        ceiling = CONTEXT_POLICY.tool_result_ceiling(profile)
        for result, _, offset, _ in await _collect(profile=profile):
            text = result.text_content  # type: ignore[attr-defined]
            assert estimate_tokens(text) <= ceiling, (
                f"page at offset {offset} estimates {estimate_tokens(text)} > ceiling {ceiling}"
            )
            fitted = fit_tool_result(result, max_tokens=ceiling)  # type: ignore[arg-type]
            assert fitted is result, (
                f"the runtime replaced the page at offset {offset} instead of keeping it whole"
            )
            assert fitted.text_content == text, f"page at offset {offset} would be cut"
            assert fitted.observed_truncated is False
            assert "observed_truncated" not in (fitted.details or {})


def test_the_cut_fact_is_recorded_only_when_the_fit_actually_cuts() -> None:
    """The runtime's own truncation is an observable fact, and nothing else is disturbed.

    A result that fits keeps its details exactly and gains nothing; a result that is cut
    gains ``observed_truncated`` without losing the keys the tool itself set.
    """
    untouched = ToolResult.text("short enough", details={"tool_note": "kept"})
    passed = fit_tool_result(untouched, max_tokens=1_000)
    assert passed.details == {"tool_note": "kept"}
    assert passed.observed_truncated is False

    long = ToolResult.text("x" * 40_000, details={"tool_note": "kept"})
    cut = fit_tool_result(long, max_tokens=100)
    assert cut.observed_truncated is True
    assert cut.details is not None and cut.details["tool_note"] == "kept"
    assert cut.text_content.endswith("[tool result truncated to the shared model residual]")


async def test_a_profile_with_no_room_fails_loudly_instead_of_cutting_silently() -> None:
    """The degenerate boundary: a window that cannot hold a request has no ceiling at all.

    The loader cannot hand over a page that fits a zero ceiling, so the runtime's fit
    refuses with ``ToolResultCapacityError`` rather than showing the model a prefix. Loud,
    not silent — which is what keeps the old defect impossible even where the policy leaves
    no room.  No shipped catalog profile is anywhere near this: the smallest hard input
    limit in the catalogue is about 483,616 tokens.
    """
    profile = ModelProfile(context_window_tokens=8_192)
    assert CONTEXT_POLICY.tool_result_ceiling(profile) == 0
    result = await _call(LoadSkillInput(name=SKILL, path=DOCUMENT, offset=0), profile=profile)
    with pytest.raises(ToolResultCapacityError):
        fit_tool_result(result, max_tokens=0)  # type: ignore[arg-type]


async def test_write_to_copies_the_packaged_bytes_exactly(tmp_path: Path) -> None:
    """Check 1(d): the workspace file's sha256 equals the packaged file's, byte for byte."""
    packaged = _packaged_bytes()
    result = await _call(
        LoadSkillInput(name=SKILL, path=DOCUMENT, write_to="scripts/engine.py"),
        profile=PROFILE_1M,
        workspace=tmp_path,
    )
    written = tmp_path / "scripts" / "engine.py"
    assert written.is_file(), (
        f"write_to did not create the workspace file: {result.text_content!r}"  # type: ignore[attr-defined]
    )
    assert written.read_bytes() == packaged
    assert hashlib.sha256(written.read_bytes()).hexdigest() == hashlib.sha256(packaged).hexdigest()
    # The copy is a workspace write the Run can account for, like the `write` tool's, not a
    # raw write beside the workspace surface.
    inventory = result.effects.workspace_inventory  # type: ignore[attr-defined]
    assert inventory is not None, "the write_to copy was not reported as a workspace mutation"
    assert any(
        fact.relative_path.endswith("scripts/engine.py")
        and fact.content_digest == hashlib.sha256(packaged).hexdigest()
        for fact in inventory.upserts
    ), f"the inventory does not name the copied file with its digest: {inventory!r}"


async def test_write_to_result_shows_no_line_of_the_script(tmp_path: Path) -> None:
    """Check 1(e): the receipt is a receipt — no line of the script reaches the model."""
    document = _packaged_path().read_text(encoding="utf-8")
    result = await _call(
        LoadSkillInput(name=SKILL, path=DOCUMENT, write_to="engine.py"),
        profile=PROFILE_1M,
        workspace=tmp_path,
    )
    text = result.text_content  # type: ignore[attr-defined]
    assert estimate_tokens(text) < 1_000, "a write_to receipt is not a document"
    leaks = [
        line for line in document.splitlines() if len(line.strip()) >= 12 and line.strip() in text
    ]
    assert not leaks, f"the receipt carries script lines: {leaks[:3]!r}"
    assert hashlib.sha256(_packaged_bytes()).hexdigest() in text, (
        "the receipt does not carry the document's content digest"
    )


async def test_write_to_refuses_a_workspace_escape(tmp_path: Path) -> None:
    """The new file channel is confined: an escaping path must be refused, not written."""
    result = await _call(
        LoadSkillInput(name=SKILL, path=DOCUMENT, write_to="../escaped.py"),
        profile=PROFILE_1M,
        workspace=tmp_path,
    )
    text = result.text_content  # type: ignore[attr-defined]
    assert result.is_error is True, "an escaping write_to was accepted"  # type: ignore[attr-defined]
    assert "Skill load failed" in text or "reject" in text.lower()
    assert not (tmp_path.parent / "escaped.py").exists()


async def test_write_to_refuses_an_absolute_path(tmp_path: Path) -> None:
    """A destination outside the workspace may not be named at all, even absolutely."""
    outside = tmp_path / "absolute.py"
    result = await _call(
        LoadSkillInput(name=SKILL, path=DOCUMENT, write_to=str(outside)),
        profile=PROFILE_1M,
        workspace=tmp_path,
    )
    assert result.is_error is True, "an absolute write_to was accepted"  # type: ignore[attr-defined]
    assert not outside.exists(), "an absolute write_to created a file"


async def test_write_to_refuses_a_symlink_that_leaves_the_workspace(tmp_path: Path) -> None:
    """A symlink is neither followed nor overwritten: the write is refused outright."""
    outside = tmp_path / "outside.py"
    outside.write_text("secret\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link.py").symlink_to(outside)
    result = await _call(
        LoadSkillInput(name=SKILL, path=DOCUMENT, write_to="link.py"),
        profile=PROFILE_1M,
        workspace=workspace,
    )
    assert result.is_error is True, "a symlink destination was accepted"  # type: ignore[attr-defined]
    assert outside.read_text(encoding="utf-8") == "secret\n"
    assert (workspace / "link.py").is_symlink(), "the symlink itself was replaced"


async def test_write_to_without_a_workspace_refuses(document: str) -> None:
    """The copy is an Agent Workspace write or nothing: with no workspace it is refused."""
    result = await _call(
        LoadSkillInput(name=SKILL, path=DOCUMENT, write_to="engine.py"),
        profile=PROFILE_1M,
    )
    assert result.is_error is True  # type: ignore[attr-defined]
    assert "no Agent Workspace" in result.text_content  # type: ignore[attr-defined]


async def test_no_profile_keeps_the_whole_document_behaviour(document: str) -> None:
    """A caller that declared no model still gets the whole document: no paging, no change.

    The seam's fallback is what keeps every caller that has not stated a model's capacity
    on the behaviour this tool always had.
    """
    tool = load_skill_tool(lambda: _catalog(), model_profile=None)
    result = await tool.execute(
        LoadSkillInput(name=SKILL, path=DOCUMENT, offset=0), tool_runtime(tool_name="load_skill")
    )
    slice_, next_offset = _page_bytes(result.text_content, offset=0, document=document)
    assert next_offset is None, "a profile-less call was paged"
    assert slice_ == document
    # An offset without a profile still slices the document, and still does not page.
    later = await tool.execute(
        LoadSkillInput(name=SKILL, path=DOCUMENT, offset=1_000),
        tool_runtime(tool_name="load_skill"),
    )
    later_slice, later_next = _page_bytes(later.text_content, offset=1_000, document=document)
    assert later_next is None
    assert later_slice == document[1_000:]


async def test_the_child_call_site_pages_as_well_as_the_parent(document: str) -> None:
    """Both bundle call sites bind a profile: a Child pages against its own model's ceiling.

    The child seam is a separate code path from the parent's, so a profile that reached only
    one of them would leave a Child receiving a whole document it cannot read.
    """
    factory = SkillsBundleFactory(builtin_root=sealed_skills_root())
    bundle = factory("owner-1")
    for child in (False, True):
        tools = bundle.tools(child=child, model_profile=PROFILE_SMALL)
        load = next(tool for tool in tools if tool.name == "load_skill")
        first = await load.execute(
            LoadSkillInput(name=SKILL, path=DOCUMENT, offset=0),
            tool_runtime(tool_name="load_skill"),
        )
        text = first.text_content
        slice_, next_offset = _page_bytes(text, offset=0, document=document)
        assert next_offset is not None, f"child={child} returned the whole document"
        assert next_offset == len(slice_) > 0
        assert estimate_tokens(text) <= CONTEXT_POLICY.tool_result_ceiling(PROFILE_SMALL)
