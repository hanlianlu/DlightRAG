# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Content fingerprints of protected Skill documents, and the mask a reader sees."""

import time

import pytest

from dlightrag.engine.agent.skill_protection import (
    FINGERPRINT_CHARS,
    MASK_PLACEHOLDER,
    MIN_MASK_CHARS,
    ProtectedTextIndex,
    _normalize,
)

#: A paragraph long enough to be a run a reader surface replaces, written in prose a
#: document could plausibly carry and an answer could plausibly quote.
PARAGRAPH = (
    "The run ledger is the first thing a calculation states: every input carries the period "
    "it was observed in, the source that stated it, and the mechanism by which it reaches "
    "the conclusion. An input nobody can date is an input nobody can defend."
)


def _index(*documents: tuple[str, str, str]) -> ProtectedTextIndex:
    return ProtectedTextIndex.build(
        (name, path, text.encode("utf-8")) for name, path, text in documents
    )


def test_normalization_agrees_across_presentation() -> None:
    """Both sides of the comparison read the same words the same way."""
    written = "Café \u201cquoted\u201d — spaced"
    reshaped = 'CAFÉ "QUOTED" - SPACED'
    assert _normalize(reshaped)[0] == _normalize(written)[0] == 'café "quoted" - spaced'


def test_normalization_collapses_whitespace_and_drops_invisibles() -> None:
    """A copier's line wrapping and zero-width characters do not change the fingerprint."""
    assert _normalize("a  b\n\nc\u200b d")[0] == "a b c d"


def test_mask_replaces_a_whole_paragraph() -> None:
    index = _index(("skill", "references/method.md", PARAGRAPH))
    text = f"Before.\n{PARAGRAPH}\nAfter."
    masked = index.mask(text)
    assert masked == f"Before.\n{MASK_PLACEHOLDER}\nAfter."


def test_mask_keeps_the_banner_that_names_the_document() -> None:
    """The result of load_skill names its document; only the author's words go."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    result = f"Skill document skill/references/method.md\n--- skill/references/method.md ---\n{PARAGRAPH}"
    masked = index.mask(result)
    assert "--- skill/references/method.md ---" in masked
    assert PARAGRAPH not in masked
    assert masked.endswith(MASK_PLACEHOLDER)


def test_a_short_quotation_is_left_alone() -> None:
    """An answer may quote a term or a line; a reader surface masks a reproduction."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    short = PARAGRAPH[: MIN_MASK_CHARS - 10]
    assert index.mask(f"Method says: {short}") == f"Method says: {short}"


def test_ordinary_prose_and_the_skill_name_are_not_masked() -> None:
    """The routing surface — a name, a description — is public and must survive."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    normal = (
        "我用 sealed-report 技能算了一遍，结论是买入；它的方法是先建运行台账，再估市场隐含的假设。"
    )
    assert index.mask(normal) == normal


def test_an_invisible_character_between_letters_does_not_evade() -> None:
    """Inserting a zero-width space is the cheapest bypass; it must not work."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    injected = PARAGRAPH.replace("ledger", "led\u200bger")
    masked = index.mask(injected)
    assert "ledger" not in masked
    assert MASK_PLACEHOLDER in masked
    # The replacement covers the injected character too, so nothing of the run survives.
    assert masked.count(MASK_PLACEHOLDER) == 1


def test_reflowed_line_breaks_do_not_evade() -> None:
    """A model that rewraps the paragraph still reproduces it."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    reflowed = PARAGRAPH.replace(" ", "\n", 6)
    assert index.mask(reflowed) == MASK_PLACEHOLDER


def test_a_reference_document_is_protected_from_its_first_character() -> None:
    """Only SKILL.md has a routing surface; a reference is the author's from the start."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    assert index.mask(PARAGRAPH) == MASK_PLACEHOLDER


def test_two_documents_do_not_bleed_into_each_other() -> None:
    other = (
        "A different document states a different procedure: it ranks drivers by the swing "
        "each one causes in the value, and it names the observation that would falsify it."
    )
    index = _index(("skill", "references/a.md", PARAGRAPH), ("skill", "references/b.md", other))
    findings = index.scan(other)
    assert [finding.path for finding in findings] == ["references/b.md"]


def test_scan_reports_the_raw_span_it_will_replace() -> None:
    """A reader surface replaces exactly the run, not its neighbourhood."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    text = f"lead {PARAGRAPH} trail"
    (finding,) = index.scan(text)
    assert text[finding.text_start : finding.text_end] == PARAGRAPH
    assert finding.name == "skill"
    assert finding.path == "references/method.md"


def test_empty_index_is_the_identity() -> None:
    """No protected Skill composes to a working object, not to a branch at every call site."""
    index = ProtectedTextIndex.empty()
    assert not index
    assert index.scan(PARAGRAPH) == ()
    assert index.mask(PARAGRAPH) is PARAGRAPH


def test_mask_returns_the_same_string_when_nothing_matches() -> None:
    """A hot path may skip the copy."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    text = "An answer that copied nothing."
    assert index.mask(text) is text


def test_mask_is_bounded_by_a_minimum() -> None:
    """The caller may raise the bar for a surface where a shorter quotation is acceptable."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    assert index.mask(PARAGRAPH, minimum=len(PARAGRAPH) + 1) == PARAGRAPH


@pytest.mark.parametrize("boundary", [1, 2, 3, 7, FINGERPRINT_CHARS - 1, FINGERPRINT_CHARS])
def test_a_run_is_found_however_a_reader_chunks_the_call(boundary: int) -> None:
    """Every call is independent: the index does not depend on how a caller split its text."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    chunk = PARAGRAPH[boundary : boundary + MIN_MASK_CHARS + 20]
    assert index.mask(chunk) == MASK_PLACEHOLDER


def test_overlapping_documents_merge_into_one_union() -> None:
    """Two documents can each own part of one passage; neither may uncover the other's words."""
    first = (
        "The first document states the opening procedure: it dates every input to the period "
        "the filing covers and refuses a figure whose period nobody can name."
    )
    shared = (
        " The shared middle paragraph states the same mechanism twice, once for the operating "
        "side and once for the capital side, and names the observation that would refute it."
    )
    last = (
        " The second document closes on a different conclusion and names the evidence that "
        "would overturn it, which the first document does not mention anywhere at all."
    )
    index = _index(
        ("skill", "references/first.md", first + shared),
        ("skill", "references/second.md", shared + last),
    )
    text = first + shared + last
    findings = index.scan(text)
    assert len(findings) == 1
    assert (findings[0].text_start, findings[0].text_end) == (0, len(text))
    assert index.mask(text) == MASK_PLACEHOLDER


def test_a_presentation_choice_does_not_evade() -> None:
    """The same document, spelled the way a renderer would show it, is the same document."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    encoded = "".join(
        f"&#x{ord(character):x};" if character.isalnum() else character for character in PARAGRAPH
    )
    assert index.mask(encoded) == MASK_PLACEHOLDER


def test_a_combining_grapheme_joiner_does_not_evade() -> None:
    """An invisible character between every thirty-nine letters is a reproduction still."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    injected = "\u034f".join(
        PARAGRAPH[offset : offset + 39] for offset in range(0, len(PARAGRAPH), 39)
    )
    assert index.mask(injected) == MASK_PLACEHOLDER


def test_a_long_run_of_one_character_is_scanned_in_bounded_time() -> None:
    """A text that repeats one window everywhere must not turn a projection into a walk.

    A shipped Skill carries separator rules hundreds of characters long, so a document can hold
    hundreds of identical fingerprints; without the posting bound, a query of the same character
    builds their cartesian product on every projection.
    """
    index = _index(("skill", "scripts/check.py", "=" * 600 + "\n" + PARAGRAPH))
    started = time.perf_counter()
    masked = index.mask("=" * 20_000)
    assert time.perf_counter() - started < 1.0
    assert masked.startswith(MASK_PLACEHOLDER)


def test_named_character_references_do_not_evade() -> None:
    """A reference is folded to the character it names, exactly as a typed one is."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    assert index.mask(PARAGRAPH.replace(" ", "&nbsp;")) == MASK_PLACEHOLDER
    # An invisible reference has no raw character to replace, so whatever follows the run
    # stays; it renders as nothing and the run itself is one placeholder.
    zero_width = "".join(f"{character}&#x200B;" for character in PARAGRAPH)
    assert index.mask(zero_width).replace("&#x200B;", "") == MASK_PLACEHOLDER


def test_a_named_reference_to_a_dash_is_the_dash() -> None:
    """``&mdash;`` must fold the way the typed dash does, or one form escapes."""
    document = (
        "The export is a well—known procedure and the house—style guide names it on the "
        "first page, where the period and the source are stated together."
    )
    index = _index(("skill", "references/style.md", document))
    assert index.mask(document.replace("—", "&mdash;")) == MASK_PLACEHOLDER


def test_format_characters_a_copier_can_insert_do_not_evade() -> None:
    """Every Unicode format character is invisible; one inserted inside a run must not split it."""
    index = _index(("skill", "references/method.md", PARAGRAPH))
    for inserted in ("\ufff9", "\u0890", "\U00013430"):
        injected = inserted.join(
            PARAGRAPH[offset : offset + 30] for offset in range(0, len(PARAGRAPH), 30)
        )
        assert index.mask(injected) == MASK_PLACEHOLDER
