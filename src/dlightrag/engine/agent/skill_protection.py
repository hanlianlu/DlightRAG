# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fingerprints of the Skill documents a reader surface must never be handed.

A protected Skill document is its author's text (ADR 0046): the Run's own model keeps the
full copy, and every surface a user reads replaces the passages that came from it. The
key is the text itself, not the tool that carried it: the same bytes reach a reader
through ``load_skill``, through ``bash cat``, through a copy that was read back, through a
child agent's conclusion, and through the model's own answer, and a rule keyed on a tool
name only catches the first.

The index is a set of fingerprints of every protected document's own text: one rolling hash
per window of :data:`FINGERPRINT_CHARS` characters, taken over a normalization both sides
share. A query is scanned by taking the same windows and asking which of them a document
contains; a candidate is then *verified* by comparing the two normalized texts character by
character, so a match is exact and its length and span are the real ones. The fingerprint
proposes and the comparison disposes — a hash collision can never mask a passage that is
not there.

Nothing here reads a file, chooses a document, or knows what a Skill is: the composition
root hands the bytes in and decides where the result is applied. The module is therefore
free of the answer pipeline and testable with two strings.
"""

from __future__ import annotations

import html
import re
import unicodedata
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

MASK_PLACEHOLDER = "[...]"
"""What one masked passage becomes.

Three characters and no sentence. A reader surface does not explain the product's
authorization policy to its reader; it states what happened and, where the text is not the
reader's to see, says nothing at all (contrast the retired placeholder, which said
"its author's text, not shown in the trace" — a developer's sentence in a reader's pane).
"""

MIN_MASK_CHARS = 80
"""Shortest verified run, in normalized characters, that a reader surface masks.

Eighty characters is every meaningful reproduction of a document — a paragraph or a table
row copied whole — and short of the quotations an answer may legitimately make of a term,
a name or a one-line definition. The cost of masking too little is a leak; the cost of
masking too much is a reader seeing ``[...]`` where a sentence should be, which is why the
number is a calibration parameter rather than a guarantee.
"""

FINGERPRINT_CHARS = 40
"""Characters in one fingerprint window.

Long enough that a window of ordinary prose is very unlikely to occur twice by accident,
short enough that a copied sentence still lands on the same window. With exact verification
behind it, the window only decides *where to look*, never *whether it matches*.
"""

_MAX_DOCUMENT_CHARS = 4 * 1024 * 1024
"""Longest document this index fingerprints, in characters.

A Skill ships a document a person reads, not a data file; a file past this bound is either
not text or not a document, and fingerprinting it would cost memory in every process for a
passage no reader would be shown anyway.
"""

_MAX_POSTINGS_PER_WINDOW = 64
"""How many document positions one query window may be compared against.

A protected document that repeats one window hundreds of times — a separator rule in a script,
a table of dashes — would otherwise make every matching query window cost that many candidates.
Sixty-four alignments are far more than a real passage needs to prove itself.
"""

_MAX_CANDIDATES = 200_000
"""Hard bound on the candidate list one scan may build, for a text that matches everywhere."""

_SCAN_WINDOW_CHARS = 64 * 1024
"""Characters one :meth:`ProtectedTextIndex.contains` window reads before it looks at the next.

A deliverable may be tens of megabytes; verifying it through one normalized string would cost
memory proportional to the file for an answer that is usually "no". Windows overlap by one
fingerprint so a run that straddles a boundary is still verified.
"""

_MAX_RAW_EXPANSION = 10
"""Most raw characters one normalized character can cost: ``&#x10FFFF;`` is ten."""

_BASE = 0x9E3779B97F4A7C15
"""The rolling hash's multiplier. Odd, so it has an inverse modulo 2**64."""

_BASE_INVERSE = pow(_BASE, -1, 1 << 64)

_CHARACTER_REFERENCE = r"&(?:#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});"
"""One HTML character reference, which a renderer resolves back into the character it names.

An answer is shown as rendered Markdown, so ``&#x54;`` reaches a reader as ``T``. Reading the
reference as its character is what keeps an encoded reproduction from walking past the gate
while looking identical in the browser.
"""

_FINGERPRINT_TOKEN = re.compile(
    _CHARACTER_REFERENCE + "|"  # a character reference, resolved to the character it names
    r"[A-Za-z0-9\x21-\x25\x27-\x7E]+"  # printable ASCII: lowercased as a run ('&' excluded)
    r"|[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uAC00-\uD7AF]+"  # CJK, kana
    r"|\s+"  # a whitespace run collapses to one space
    r"|.",  # everything else, one character at a time
    re.S,
)

_UNIFY = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2007": " ",
        "\u202f": " ",
        "\u3000": " ",
    }
)
"""One-to-one character equivalences applied before anything else.

One-to-one so the raw offset of every character is preserved, and applied to both sides so
a curly quote in a document and a straight one in an answer are the same character.
"""

_DROPPED = frozenset(
    [
        *range(0x00, 0x09),  # control characters that are not whitespace
        *range(0x0E, 0x20),
        0x7F,  # delete
        *range(0x80, 0x85),  # C1 controls, except NEL (0x85), which Python reads as whitespace
        *range(0x86, 0xA0),
        0xAD,  # soft hyphen, which a copier can insert between any two letters
        0x034F,  # combining grapheme joiner: invisible, and it splits a run in two
        0x061C,  # Arabic letter mark
        0x115F,
        0x1160,
        0x17B4,
        0x17B5,  # Hangul fillers
        0x180B,
        0x180C,
        0x180D,
        0x180E,
        0x180F,  # Mongolian free variation selectors
        0x200B,
        0x200C,
        0x200D,
        0x200E,
        0x200F,  # zero-width and direction marks
        *range(0x202A, 0x202F),  # bidi embedding and override controls
        *range(0x2060, 0x2070),  # invisible operators and bidi isolates
        0x3164,
        0xFFA0,  # Hangul filler halves
        *range(0xFE00, 0xFE10),  # variation selectors
        0xFEFF,  # byte order mark
        *range(0x1BCA0, 0x1BCA4),  # shorthand format controls
        *range(0x1D173, 0x1D17B),  # musical symbol format controls
        *range(0xE0000, 0xE1000),  # Tags and variation selectors supplement
        *range(0xE0100, 0xE01F0),
    ]
)
"""Code points normalization removes rather than keeps.

Insertion of an invisible character between two letters is the cheapest way to walk past a
fingerprint, so the pipeline removes them on both sides instead of transliterating them.
"""


@dataclass(frozen=True, slots=True)
class LeakFinding:
    """One verified run of a protected document's own text inside a queried string."""

    name: str
    """The Skill that owns the document."""

    path: str
    """The document, Skill-relative."""

    text_start: int
    """Where the run begins in the queried string, as a raw character offset."""

    text_end: int
    """Where the run ends in the queried string, exclusive, as a raw character offset."""

    @property
    def length(self) -> int:
        return self.text_end - self.text_start


@dataclass(frozen=True, slots=True)
class _Document:
    """One protected document, normalized, ready to be compared against a query."""

    name: str
    path: str
    normalized: str
    """The text every query is compared against, produced by the shared pipeline.

    A ``SKILL.md``'s frontmatter is not here: the catalog hands in the document's body, so
    the rule for where its routing surface ends lives beside the parser that owns it.
    """


def _code_points(text: str) -> np.ndarray:
    """One string as its code points, for the vectorized hash.

    ``utf-32-le`` is the conversion the standard library does in C; decoding it back with
    ``np.frombuffer`` reads one code point per element without a Python loop.
    """
    return np.frombuffer(text.encode("utf-32-le"), dtype="<u4").astype(np.uint64)


def _window_hashes(code_points: np.ndarray, width: int) -> np.ndarray:
    """The rolling hash of every ``width``-character window of one string.

    ``hash(i) = sum_j code_points[i + j] * BASE**j`` modulo 2**64, computed for all ``i`` at
    once: a prefix sum of ``code_points[j] * BASE**j`` gives every window sum, and one
    multiplication by ``BASE**-i`` moves it to the window's own scale. numpy's unsigned
    integers wrap, which is the modulus.
    """
    size = code_points.size
    windows = size - width + 1
    if windows <= 0:
        return np.empty(0, dtype=np.uint64)
    powers = np.ones(size, dtype=np.uint64)
    np.cumprod(np.full(size - 1, _BASE, dtype=np.uint64), out=powers[1:])
    weighted = code_points * powers
    prefix = np.zeros(size + 1, dtype=np.uint64)
    np.cumsum(weighted, out=prefix[1:])
    inverse_powers = np.ones(windows, dtype=np.uint64)
    np.cumprod(np.full(windows - 1, _BASE_INVERSE, dtype=np.uint64), out=inverse_powers[1:])
    return (prefix[width:] - prefix[:-width]) * inverse_powers


def _fold_character(character: str) -> str | None:
    """One character through the normalization, or None when it is removed.

    A character reference can name any of the characters the fast paths already fold —
    ``&nbsp;`` a space, ``&#x200B;`` an invisible character, ``&mdash;`` a dash — and a
    reference resolved after those paths would otherwise produce a character they never see,
    so an encoded reproduction would be the document in a browser and not here.
    """
    unified = character.translate(_UNIFY)
    if not unified:
        return None
    if ord(unified[0]) in _DROPPED or unicodedata.category(unified[0]) == "Cf":
        return None
    if unified.isspace():
        return " "
    folded = unicodedata.normalize("NFKC", unified).casefold()
    return folded or None


def _normalize(text: str, *, with_map: bool = False) -> tuple[str, array | None]:
    """The normalization both the index and a query are read through.

    One function, called by both sides. A second implementation that agrees today is what
    makes a fingerprint index silently stop matching after a change, so the pipeline has
    exactly one home.

    The returned map, when asked for, gives the raw offset each normalized character came
    from; a reader surface needs it to replace the run in the string it actually holds.
    """
    if not text:
        return "", (array("q") if with_map else None)
    text = text.translate(_UNIFY)
    pieces: list[str] = []
    mapping: array | None = array("q") if with_map else None
    previous_was_space = False
    for match in _FINGERPRINT_TOKEN.finditer(text):
        token = match.group()
        start = match.start()
        first = ord(token[0])
        if token[0] == "&" and len(token) > 2 and token.endswith(";"):
            resolved = html.unescape(token)
            if resolved == token:
                # A reference this build does not know stays the characters it spells: the
                # reader sees them literally, so the comparison reads them literally too.
                pieces.append(token.lower())
                if mapping is not None:
                    mapping.extend(range(start, start + len(token)))
            else:
                for character in resolved:
                    folded_character = _fold_character(character)
                    if folded_character is None:
                        continue
                    if folded_character == " ":
                        if not previous_was_space:
                            pieces.append(" ")
                            if mapping is not None:
                                mapping.append(start)
                        previous_was_space = True
                        continue
                    pieces.append(folded_character)
                    if mapping is not None:
                        mapping.extend([start] * len(folded_character))
                    previous_was_space = False
                continue
        elif 0x21 <= first <= 0x7E:
            # A run of printable ASCII: lowercasing is one-to-one here, so the raw offsets
            # stay contiguous and the whole run is handled by the C implementation.
            pieces.append(token.lower())
            if mapping is not None:
                mapping.extend(range(start, start + len(token)))
        elif (
            0x3040 <= first <= 0x30FF
            or 0x3400 <= first <= 0x4DBF
            or 0x4E00 <= first <= 0x9FFF
            or 0xF900 <= first <= 0xFAFF
            or 0xAC00 <= first <= 0xD7AF
        ):
            pieces.append(token)
            if mapping is not None:
                mapping.extend(range(start, start + len(token)))
        elif first in _DROPPED:
            continue
        elif token.isspace():
            if not previous_was_space:
                pieces.append(" ")
                if mapping is not None:
                    mapping.append(start)
            previous_was_space = True
            continue
        else:
            folded = _fold_character(token)
            if folded is None:
                continue
            pieces.append(folded)
            if mapping is not None:
                mapping.extend([start] * len(folded))
        previous_was_space = False
    return "".join(pieces), mapping


def mask_reader_strings(value: Any, mask: Callable[[str], str]) -> Any:
    """Every string inside one JSON-shaped value, passed through a reader's content gate.

    A structured fact is not prose, but its leaves can be: a search query is the model's own
    words, a page title is the page's. A reader surface masks those exactly as it masks the
    result they came from, so the shape keeps and only the text inside it changes.
    """
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, Mapping):
        return {key: mask_reader_strings(item, mask) for key, item in value.items()}
    if isinstance(value, list):
        return [mask_reader_strings(item, mask) for item in value]
    if isinstance(value, tuple):
        return tuple(mask_reader_strings(item, mask) for item in value)
    return value


def _raw_character_end(text: str, start: int) -> int:
    """Where the raw character that produced one normalized character ends.

    A character reference is one normalized character produced by six or eight raw ones, and
    the map points at its ``&``. One past that ``&`` would cut the reference in half, so a
    masked span that ends on one has to end after its ``;`` — a half-reference is not the
    character, and a carry that starts inside one no longer matches the document it came from.
    """
    if text[start] == "&":
        semicolon = text.find(";", start, start + 64)
        if semicolon >= 0:
            return semicolon + 1
    return start + 1


class ProtectedTextIndex:
    """Every protected document's fingerprints, and the scan that reads them.

    An empty index — no built-in Skill, no operator Skill, a deployment that disabled them
    all — is a first-class value: :meth:`empty` returns a working object whose every method
    is the identity, so a call site never branches on whether protection exists.
    """

    def __init__(
        self,
        documents: Sequence[_Document],
        hashes: np.ndarray,
        document_indexes: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        self._documents = tuple(documents)
        self._hashes = hashes
        self._document_indexes = document_indexes
        self._positions = positions

    @classmethod
    def empty(cls) -> ProtectedTextIndex:
        return cls(
            (),
            np.empty(0, dtype=np.uint64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )

    def __bool__(self) -> bool:
        return bool(self._documents)

    @property
    def document_count(self) -> int:
        """How many documents this index covers, for the one log line that says it built."""
        return len(self._documents)

    @property
    def character_count(self) -> int:
        """How many normalized characters it indexes."""
        return sum(len(document.normalized) for document in self._documents)

    @classmethod
    def build(cls, documents: Iterable[tuple[str, str, bytes]]) -> ProtectedTextIndex:
        """Index one ``(name, path, bytes)`` per document.

        A file that is not UTF-8 text, is empty, or is past the size bound is skipped: it is
        not a document a reader would be handed as prose, and the digest gate is what covers
        a file's bytes.
        """
        normalized: list[_Document] = []
        for name, path, content in documents:
            if not content:
                continue
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if len(text) > _MAX_DOCUMENT_CHARS:
                continue
            document_text, _ = _normalize(text)
            if len(document_text) < FINGERPRINT_CHARS:
                continue
            normalized.append(_Document(name, path, document_text))
        if not normalized:
            return cls.empty()
        hash_blocks: list[np.ndarray] = []
        document_blocks: list[np.ndarray] = []
        position_blocks: list[np.ndarray] = []
        for index, document in enumerate(normalized):
            windows = _window_hashes(_code_points(document.normalized), FINGERPRINT_CHARS)
            if windows.size == 0:
                continue
            hash_blocks.append(windows)
            document_blocks.append(np.full(windows.size, index, dtype=np.int32))
            position_blocks.append(np.arange(windows.size, dtype=np.int32))
        if not hash_blocks:
            return cls.empty()
        hashes = np.concatenate(hash_blocks)
        # Order among equal hashes does not matter: the scan groups candidates by document
        # and diagonal itself, so a stable sort would only cost time.
        order = np.argsort(hashes)
        return cls(
            normalized,
            hashes[order],
            np.concatenate(document_blocks)[order],
            np.concatenate(position_blocks)[order],
        )

    def scan(self, text: str, *, minimum: int = MIN_MASK_CHARS) -> tuple[LeakFinding, ...]:
        """Every verified run of a protected document's text in ``text``.

        Sorted by position and non-overlapping; a run shorter than ``minimum`` normalized
        characters is not reported, because the caller masks what this returns.
        """
        if not self._documents or not text:
            return ()
        normalized, mapping = _normalize(text, with_map=True)
        if mapping is None or len(normalized) < max(FINGERPRINT_CHARS, minimum):
            return ()
        query_hashes = _window_hashes(_code_points(normalized), FINGERPRINT_CHARS)
        # A run of identical windows — a separator rule, a repeated character — is one question
        # to the index, not one per character: a query of two thousand "=" is a single posting
        # list, not two million candidates. The document side keeps every position, so a
        # periodic passage still finds an alignment that verifies.
        positions_to_ask = np.arange(query_hashes.size)
        if query_hashes.size > 1:
            changed = np.empty(query_hashes.size, dtype=bool)
            changed[0] = True
            np.not_equal(query_hashes[1:], query_hashes[:-1], out=changed[1:])
            positions_to_ask = np.flatnonzero(changed)
        lookup = query_hashes[positions_to_ask]
        left = np.searchsorted(self._hashes, lookup, side="left")
        right = np.searchsorted(self._hashes, lookup, side="right")
        present = np.flatnonzero(right > left)
        if present.size == 0:
            return ()
        candidates: list[tuple[int, int, int]] = []
        for slot in present.tolist():
            query_position = int(positions_to_ask[slot])
            first_entry = int(left[slot])
            last_entry = min(int(right[slot]), first_entry + _MAX_POSTINGS_PER_WINDOW)
            for entry in range(first_entry, last_entry):
                document = int(self._document_indexes[entry])
                candidates.append(
                    (document, int(self._positions[entry]) - query_position, query_position)
                )
            if len(candidates) >= _MAX_CANDIDATES:
                # A posting list longer than the bound is a document full of the same window;
                # no reader's text is a sequence of them, and the first postings carry the
                # alignment a genuine run needs. The bound is what keeps a hostile text from
                # turning a projection into a quadratic walk.
                break
        candidates.sort()
        found: list[LeakFinding] = []
        index = 0
        while index < len(candidates):
            document, delta, query_start = candidates[index]
            query_end = query_start
            following = index + 1
            while (
                following < len(candidates)
                and candidates[following][0] == document
                and candidates[following][1] == delta
                and candidates[following][2] == query_end + 1
            ):
                query_end = candidates[following][2]
                following += 1
            start = query_start
            end = query_end + FINGERPRINT_CHARS
            document_text = self._documents[document].normalized
            document_start = start + delta
            if (
                normalized[start:end]
                == document_text[document_start : document_start + (end - start)]
            ):
                # Every window matched on one diagonal, so the run is real; the final window
                # is only fingerprint-wide, so walk the tail forward until it stops matching.
                while (
                    end < len(normalized)
                    and document_start + (end - start) < len(document_text)
                    and normalized[end] == document_text[document_start + (end - start)]
                ):
                    end += 1
                raw_start = int(mapping[start])
                raw_end = _raw_character_end(text, int(mapping[end - 1]))
                if end - start >= minimum:
                    found.append(
                        LeakFinding(
                            self._documents[document].name,
                            self._documents[document].path,
                            raw_start,
                            raw_end,
                        )
                    )
            index = following
        if not found:
            return ()
        found.sort(key=lambda finding: (finding.text_start, -finding.length))
        merged: list[LeakFinding] = []
        for finding in found:
            if merged and finding.text_start < merged[-1].text_end:
                # Two documents can each own part of one stretch of the query — a shared
                # paragraph, a quoted passage — so an overlap is a union, not a contest: the
                # longer finding does not cover the shorter one's leading characters, and
                # replacing it with either alone would leave the other's words in the clear.
                previous = merged[-1]
                longest = previous if previous.length >= finding.length else finding
                merged[-1] = LeakFinding(
                    longest.name,
                    longest.path,
                    previous.text_start,
                    max(previous.text_end, finding.text_end),
                )
                continue
            merged.append(finding)
        return tuple(merged)

    def contains(self, text: str, *, minimum: int = MIN_MASK_CHARS) -> bool:
        """Whether ``text`` holds one verified run of at least ``minimum`` characters.

        The same rule as :meth:`scan`, asked for one answer and stopped at the first: a caller
        that only needs "may this be published?" must not build the whole finding list for a
        file it is about to refuse. Long text is read through a *normalized* carry: each window
        keeps the last ``minimum`` normalized characters of the previous one, so a run that
        straddles the boundary is whole in the next window however the raw length is inflated
        (a whitespace run collapses, an invisible character is dropped, a character reference
        costs ten). A raw-character overlap cannot bound that inflation.
        """
        if not self._documents or not text:
            return False
        if len(text) <= _SCAN_WINDOW_CHARS:
            return bool(self.scan(text, minimum=minimum))
        carry = ""
        for start in range(0, len(text), _SCAN_WINDOW_CHARS):
            buffer = carry + text[start : start + _SCAN_WINDOW_CHARS]
            if self.scan(buffer, minimum=minimum):
                return True
            normalized, mapping = _normalize(buffer, with_map=True)
            if mapping is None or len(normalized) <= minimum:
                carry = buffer
                continue
            carry = buffer[int(mapping[len(normalized) - minimum]) :]
        return False

    def mask(self, text: str, *, minimum: int = MIN_MASK_CHARS) -> str:
        """Replace every verified run with :data:`MASK_PLACEHOLDER`.

        A text with no run comes back as the same object, so a hot path can skip the copy.
        """
        findings = self.scan(text, minimum=minimum)
        if not findings:
            return text
        pieces: list[str] = []
        cursor = 0
        for finding in findings:
            if finding.text_start < cursor:
                continue
            pieces.append(text[cursor : finding.text_start])
            pieces.append(MASK_PLACEHOLDER)
            cursor = finding.text_end
        pieces.append(text[cursor:])
        return "".join(pieces)
