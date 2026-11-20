# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The documentation's machine-checked contracts.

Three things a reader trusts and nobody re-reads: that the decision-record index still
matches the decision records, that the configuration reference names every setting the code
ships, and that a documented default is the default the code ships. All are generated or
verified here rather than maintained by hand.
"""

import re
from pathlib import Path
from types import UnionType
from typing import Annotated, Union, get_args, get_origin

import pytest
from pydantic import BaseModel

from scripts.adr_index import ADR_DIRECTORY, INDEX_PATH, entries, entry_for, render
from scripts.verify_release_contract import (
    _CONFIGURATION_DOC,
    _SETTINGS_SOURCE,
    _verify_documented_parser_defaults,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_adr_index_is_current() -> None:
    """`make platform-check` runs the same comparison; this is the fast half of it."""
    assert INDEX_PATH.read_text(encoding="utf-8") == render(entries())


def test_every_decision_record_appears_once() -> None:
    listed = [entry.number for entry in entries()]
    files = sorted(path.name[:4] for path in ADR_DIRECTORY.glob("[0-9][0-9][0-9][0-9]-*.md"))
    assert listed == files
    assert len(listed) == len(set(listed))


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("# ADR 0099 — a thing is decided\n\n## Status\n\nAccepted.\n", "Accepted."),
        ("# a thing is decided\n\nStatus: accepted.\n", "accepted."),
        (
            "# a thing is decided\n\n**Status**: accepted, with a\nsecond line.\n",
            "accepted, with a second line.",
        ),
    ],
)
def test_the_index_reads_every_status_shape(tmp_path: Path, body: str, expected: str) -> None:
    """Most records use a section; some state it on one labelled line, plain or bold; all are quoted."""
    path = tmp_path / "0099-a-thing-is-decided.md"
    path.write_text(body, encoding="utf-8")
    entry = entry_for(path)
    assert entry.status == expected
    assert entry.title == "a thing is decided"


#: Configuration settings deliberately absent from the reference, each with its reason. Empty is
#: the goal: a knob that exists and is documented nowhere is the inverse of a reference that
#: claims provision the code does not have, and nothing caught the first one -- an Answer
#: generation setting shipped in the code and in no document at all. A secret that is documented
#: only where it belongs, a value derived from other settings, or an internal field goes here with
#: its reason, so the absence is a decision rather than an oversight.
_UNDOCUMENTED_SETTINGS: dict[str, str] = {}


def _nested_models(annotation: object) -> tuple[type[BaseModel], ...]:
    """The models a field's annotation can hold, through `Annotated`, `Optional`, and unions."""
    if (origin := get_origin(annotation)) is Annotated:
        return _nested_models(get_args(annotation)[0])
    if origin is Union or origin is UnionType:
        return tuple(model for arg in get_args(annotation) for model in _nested_models(arg))
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (annotation,)
    return ()


def _configuration_leaves() -> list[str]:
    """Every leaf setting of `DlightragConfig`, as its full `section.…field` path.

    A field that holds a model is a section and is walked; everything else is a setting an
    operator can set. Every top-level section is walked, so one added later is measured here
    without editing this test.
    """
    from dlightrag.application.config.sections import DlightragConfig

    leaves: list[str] = []

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            if nested := _nested_models(field.annotation):
                for child in nested:
                    walk(child, path)
            else:
                leaves.append(path)

    walk(DlightragConfig, "")
    return list(dict.fromkeys(leaves))


def test_every_configuration_setting_appears_in_the_configuration_reference() -> None:
    """The reference documents every section; a field it omits is a gap, not a style choice."""
    documented = _CONFIGURATION_DOC.read_text(encoding="utf-8")
    missing = sorted(
        path
        for path in _configuration_leaves()
        # Word boundaries, not a substring test: a setting renamed to something containing this
        # name would otherwise read as documented.
        if re.search(rf"\b{re.escape(path.rsplit('.', 1)[-1])}\b", documented) is None
        and path not in _UNDOCUMENTED_SETTINGS
    )

    assert missing == [], f"settings the reference never mentions: {missing}"


def test_documented_parser_defaults_match_the_settings() -> None:
    _verify_documented_parser_defaults(ROOT)


def test_a_drifted_documented_default_is_refused(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / _CONFIGURATION_DOC.name).write_text(
        "### MinerU (default)\n\n"
        "| Field | Default | Notes |\n|---|---|---|\n"
        "| `max_polls` | `999` | Two-hour default polling window |\n\n"
        "### Docling\n\n"
        "| Field | Default | Notes |\n|---|---|---|\n"
        "| `force_ocr` | `true` | |\n",
        encoding="utf-8",
    )
    settings = tmp_path / _SETTINGS_SOURCE
    settings.parent.mkdir(parents=True)
    settings.write_text(
        "class MinerUSidecarSettings:\n"
        "    max_polls: int = Field(default=1440, ge=1)\n"
        "\n"
        "class DoclingSidecarSettings:\n"
        "    force_ocr: bool = True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="documents max_polls=999"):
        _verify_documented_parser_defaults(tmp_path)


def test_a_field_documented_but_absent_from_the_settings_is_refused(tmp_path: Path) -> None:
    """A renamed or invented field is drift in the other direction, and it is caught."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / _CONFIGURATION_DOC.name).write_text(
        "### MinerU (default)\n\n"
        "| Field | Default | Notes |\n|---|---|---|\n"
        "| `language` | `ch` | OCR hint |\n"
        "| `something_else` | `x` | invented |\n\n"
        "### Docling\n\n| Field | Default | Notes |\n|---|---|---|\n",
        encoding="utf-8",
    )
    settings = tmp_path / _SETTINGS_SOURCE
    settings.parent.mkdir(parents=True)
    settings.write_text(
        "class MinerUSidecarSettings:\n"
        '    language: str = "ch"\n'
        "\n"
        "class DoclingSidecarSettings:\n"
        "    force_ocr: bool = True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="documents .* and omits"):
        _verify_documented_parser_defaults(tmp_path)


def test_the_index_quotes_each_record_without_asking_the_index() -> None:
    """An independent read of every ADR, so the check above cannot be a tautology."""
    index = INDEX_PATH.read_text(encoding="utf-8")
    for path in sorted(ADR_DIRECTORY.glob("[0-9][0-9][0-9][0-9]-*.md")):
        text = path.read_text(encoding="utf-8")
        heading = next(line for line in text.splitlines() if line.startswith("# "))
        title = re.sub(r"^ADR\s+\d{4}\s*[—–-]\s*", "", heading[2:].strip())
        # The status the index shows must be text the ADR actually contains, read here
        # without the generator's own sentence rule: a prefix of the whole section (or,
        # for the records that state it on one line, of that line). Inline emphasis is
        # dropped on both sides, since the index is prose and not a Markdown copy.
        # Three records state their status outside a `## Status` section, so the whole
        # record is the haystack: the invariant is that the index quotes the ADR rather
        # than inventing or truncating a status, not which heading carried it.
        document = re.sub(r"[*_`]+", "", " ".join(text.split()))
        # The row that links this record, not a row whose status happens to mention it.
        row = next(line for line in index.splitlines() if line.startswith(f"| [{path.name[:4]}]("))
        shown = row.strip().strip("|").split("|")[-1].strip()
        assert title[:40] in row, (path.name, title)
        if shown and shown != "—":
            assert shown.removesuffix("…") in document, (path.name, shown)


def test_a_misnamed_record_is_an_error(tmp_path: Path) -> None:
    loose = tmp_path / "0055.md"
    loose.write_text("# a thing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="four-digit ADR number"):
        entry_for(loose)
