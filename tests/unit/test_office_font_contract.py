# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The fonts a PDF embeds are the image's, and the instruction that says so stays true.

fpdf2 embeds no Unicode font by default, raises on Chinese in a built-in one, drops a glyph its
font lacks without raising, and reads no environment of its own. So the things an Agent cannot
discover from the code it writes are which families the machine already carries, where they are,
which one belongs to the document's language, and what none of them can draw. They are prose in a
Skill, and prose drifts from the Dockerfile's pins; this pins the two together.
"""

import fnmatch
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE = _ROOT / "Dockerfile"
_FONT_INSTALLER = _ROOT / "scripts/install-fonts.sh"
_CHART_INSTALLER = _ROOT / "scripts/install-chart-tools.sh"
_SKILL = _ROOT / "src/dlightrag/engine/agent/builtin_skills/office-documents/SKILL.md"
_THEME = _ROOT / "chart-render/theme.json"

#: The one directory the image puts the font set in, and the path every consumer defaults to.
_IMAGE_FONT_DIR = "/usr/local/share/fonts/dlightrag"

_FONTS_STAGE = re.compile(r"^FROM \S+ AS fonts$(.*?)^FROM ", re.MULTILINE | re.DOTALL)
_ADDED = re.compile(r"^ADD --checksum=sha256:([0-9a-f]{64}) \S+ /fonts/(\S+)$", re.MULTILINE)

#: family name in fpdf2 -> the file stem the image ships it as, and its styles.
_LANGUAGES = {"latin": "SourceSans3", "sc": "NotoSansSC", "jp": "NotoSansJP", "kr": "NotoSansKR"}


def _fonts_stage() -> str:
    match = _FONTS_STAGE.search(_DOCKERFILE.read_text(encoding="utf-8"))
    assert match, "the Dockerfile has no fonts stage"
    return match.group(1)


def _pinned_files() -> dict[str, str]:
    """Every file the font stage puts in the image, by name, with its checksum."""
    return {name: digest for digest, name in _ADDED.findall(_fonts_stage())}


def _pdf_instruction() -> str:
    return next(
        line
        for line in _SKILL.read_text(encoding="utf-8").splitlines()
        if line.startswith("- **PDF**")
    )


def _instruction(marker: str) -> str:
    """The one Dockerfile instruction holding ``marker``, its line continuations joined."""
    joined = re.sub(r"\\\n\s*", " ", _DOCKERFILE.read_text(encoding="utf-8"))
    return next(line for line in joined.splitlines() if marker in line)


def test_the_font_stage_pins_a_face_for_every_language_the_skill_names() -> None:
    """Each family the Skill tells an Agent to register is a file the image actually carries."""
    pinned = _pinned_files()
    assert pinned, "the Dockerfile's fonts stage pins no font"

    for family, stem in _LANGUAGES.items():
        assert f"{stem}-Regular" in " ".join(pinned), family
        assert f"{stem}-Bold" in " ".join(pinned), family
    assert any(name.startswith("LICENSE") for name in pinned), "no license ships with the fonts"


def test_the_installer_installs_what_the_stage_pins_and_pins_nothing_itself() -> None:
    """A development machine installs the image's bytes; a second pin is one that can disagree."""
    script = _FONT_INSTALLER.read_text(encoding="utf-8")

    assert "ADD --checksum=sha256:" in script, "the installer must read the stage's own ADD lines"
    assert not re.search(r"\b[0-9a-f]{64}\b", script), "the checksums belong to the Dockerfile"

    # Every pinned name has to match one of the installer's URL patterns, or it cannot be fetched.
    patterns = re.findall(r"^\s{4}([A-Za-z0-9_.*-]+)\)", script, re.MULTILINE)
    assert patterns, "the installer builds no source URL"
    for name, stem in _LANGUAGES.items():
        unprefixed = [name for name in _pinned_files() if name.startswith(stem)]
        assert unprefixed, stem
        for pinned_name in unprefixed:
            assert any(fnmatch.fnmatchcase(pinned_name, p) for p in patterns), pinned_name


def test_the_chart_installer_takes_the_fonts_from_the_one_installer() -> None:
    """Two copies of the font fetch would be two sets of pins and two licence checks."""
    assert "install-fonts.sh" in _CHART_INSTALLER.read_text(encoding="utf-8")


def test_the_theme_offers_resvg_every_family_the_image_ships() -> None:
    """A chart label in Korean or Turkish has to reach a family that has the glyph."""
    family = re.search(r'"fontFamily": "([^"]+)"', _THEME.read_text(encoding="utf-8"))
    assert family, "the theme names no font family"
    offered = [name.strip() for name in family.group(1).split(",")]

    for name in ("Noto Sans SC", "Noto Sans JP", "Noto Sans KR", "Source Sans 3"):
        assert name in offered, name


def test_the_pdf_instruction_names_the_families_where_they_are_and_what_they_lack() -> None:
    """What an Agent cannot read out of the image: the path, the families, and the dropped glyphs."""
    instruction = _pdf_instruction()

    assert _IMAGE_FONT_DIR in instruction
    assert "FONTS_DIR" in instruction
    for family, stem in _LANGUAGES.items():
        assert family in instruction, family
        assert stem in instruction, stem
    assert "set_fallback_fonts" in instruction
    # The CJK families have no italic face, and fpdf2 raises rather than substituting one.
    assert "italic" in instruction
    # Scripts no shipped font has: fpdf2 drops them without raising, so the Skill has to say so.
    for script in ("Emoji", "Arabic", "Hebrew", "Thai", "Devanagari"):
        assert script in instruction, script


def test_the_build_fails_when_a_shipped_font_drops_a_glyph() -> None:
    """The smoke test's assertion is fpdf2's own warning: a silent drop must not reach a Run."""
    run = _instruction("missing the following glyphs")

    # The page it writes is the Skill's own rule: every family registered, each language's own set,
    # the rest as fallbacks.
    for family in _LANGUAGES:
        assert f"'{family}'" in run, family
    assert "add_font" in run
    assert "set_fallback_fonts" in run


def test_the_build_fails_when_a_chart_drops_a_character() -> None:
    """resvg boxes what no font has and `echarts-render` still exits 0, so its note is the gate."""
    run = _instruction("chart-fonts-smoke")

    # The scripts the image only recently gained, one per field: resvg gives a text element one
    # family and does not walk on for the characters that family lacks.
    for script in ("한국어", "Türkçe", "東京"):
        assert script in run, script
    assert "grep -q" in run and "no font has" in run
