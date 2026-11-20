# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which of the documents a Skill says a Run must read did the Run read, from its own records.

A Skill can name a document it must be read with (``required_reads`` in its frontmatter, ADR
0057). Whether a Run read it is not something the model can be asked: it is a fact about what
``load_skill`` handed back, and ``load_skill`` records that fact on the durable ``tool_progress``
event of every call it answers -- the document's path, the window of characters it returned, and
the document's length. This module reads those records back and answers one question with them:
for each Skill a Run loaded, which declared reads were never read to the end.

Two callers have to agree on the answer: the attestation derivation that publishes it beside an
Artifact, and the in-loop delivery ledger that previews it. Both read it from here; a rule that two
modules each implemented would be the copy that drifts.

**The read rule.** A document counts as read iff the windows ``load_skill`` returned for one
version of it (one ``document_chars`` value) cover ``[0, document_chars)`` with no gap. Read to
the end, and nothing less. The rule is the whole document rather than "the first window" because
the size of a window is not the document's but the reading model's: it is fitted to that model's
tool-result ceiling, so the first window is a few thousand characters for one model and the whole
file for another. A rule that counted the first window would credit a reader that stopped after
the opening with a different amount of text depending on which model it was, and could not say
what "read" means across the models the product runs. Whole-document coverage is the one rule
that does not depend on the reader. Section-level rules were considered and rejected: a section has
no stable address in the windowing -- offsets are characters, not headings -- and the references a
Skill requires are meant to be short enough to read whole.

**What a read is not.** A ``write_to`` copy hands a document's bytes to the Agent Workspace and
the model never sees them, so it records the path and no window and never satisfies a required
read. A window the Runtime later cut to the model's residual is no read either: the window
``load_skill`` recorded is what it returned, and the settlement's ``observed_truncated`` says the
model was given less, so that call keeps its path and loses its window. And a recorded window is the evidence that the text was put in front of the model, not proof
that the model attended to it: like every record in the attestation, it removes the silent case --
a Skill that says "read this" and a Run that never opened it -- and no more.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: The Tool whose call carries a Skill load. Its subject is the Skill name the call named.
LOAD_SKILL_TOOL = "load_skill"

#: The Tool whose call reports the gate's sealed attestation (``FACT_SEALED_SKILL``).
ATTACH_ARTIFACT_TOOL = "attach_artifact"

#: The keys a Tool reports on a ``tool_progress`` update, which the Runtime forwards onto the
#: durable event (``TOOL_PROGRESS_FACT_KEYS`` in ``tools/contracts.py`` is the forwarding allow-list
#: and a test holds the two together). ``load_skill`` reports the first four: ``skill_path`` alone is
#: a ``write_to`` copy; the other three together are the window one read returned.
#: ``attach_artifact`` reports ``sealed_skill``: the Skills whose gate attested what it attached.
FACT_PATH = "skill_path"
FACT_WINDOW_START = "skill_window_start"
FACT_WINDOW_END = "skill_window_end"
FACT_DOCUMENT_CHARS = "skill_document_chars"
FACT_SEALED_SKILL = "sealed_skill"
FACT_KEYS = frozenset(
    {FACT_PATH, FACT_WINDOW_START, FACT_WINDOW_END, FACT_DOCUMENT_CHARS, FACT_SEALED_SKILL}
)

#: The metadata bool the Runtime puts on a call's settlement (``tool_end``) when it cut the result to
#: the model's residual. It is not a ``load_skill`` fact, so it is not in ``FACT_KEYS``; it is read
#: from the same payloads and voids the credit of the window the call recorded.
FACT_OBSERVED_TRUNCATED = "observed_truncated"


@dataclass(frozen=True, slots=True)
class RequiredRead:
    """One document, or one of several, that a Skill says a Run must read.

    ``paths`` are Skill-relative and normalized. One path is the entry ``path:`` in the
    frontmatter; several are ``any_of:``, where reading any one of them satisfies the entry --
    "the example nearest the request" is a choice among files, and requiring all of them would
    require reading what the request does not call for. ``when`` is the step or condition the
    Skill gives, in the Skill author's words, shown to a reader beside the path.
    """

    paths: tuple[str, ...]
    when: str


@dataclass(frozen=True, slots=True)
class SkillRead:
    """The fact one ``load_skill`` call recorded about the document it answered.

    ``start``, ``end`` and ``document_chars`` are all ``None`` for a ``write_to`` copy: the
    record says which document was copied and says nothing of text the model saw. A fact whose
    window keys are missing, mistyped or contradictory reads the same way -- present, so the
    record is known to postdate the fact, but never a read.
    """

    skill: str
    path: str
    start: int | None = None
    end: int | None = None
    document_chars: int | None = None


@dataclass(frozen=True, slots=True)
class LoadSkillCall:
    """One ``load_skill`` call as the durable events describe it.

    ``read`` is ``None`` when the record carries no fact for the call -- a Run recorded before
    the fact existed, or a call that failed and so handed over nothing. A call whose result the
    Runtime cut has a ``read`` with a path and no window.
    """

    call_id: str
    skill: str
    outcome: str | None
    read: SkillRead | None


def fact_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The keys one durable event payload carries that :func:`skill_read` reads, and nothing else.

    The read-fact keys, ``sealed_skill``, and ``observed_truncated`` where the settlement carries
    it. Both merges of a Run's events -- ``skill_reads_from_events`` here and the attestation's --
    collect these per call and hand them to :func:`skill_read`, so the rule for what a call read
    lives in that one function and the two cannot diverge.
    """
    return {
        key: value
        for key, value in payload.items()
        if key in FACT_KEYS or key == FACT_OBSERVED_TRUNCATED
    }


def _whole_number(value: object) -> int | None:
    """``value`` when it is an integer a window can be made of, else ``None``.

    A bool is an ``int`` to Python and not to a fact: ``True`` is not a character offset.
    """
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def skill_read(skill: str, facts: Mapping[str, Any]) -> SkillRead | None:
    """The :class:`SkillRead` the merged fact keys of one call describe, or ``None`` for none.

    ``None`` means the call recorded no fact at all (no ``skill_path``). A path with a complete,
    consistent window is a read; a path with anything less is a copy, or a fact too damaged to
    credit, and either way carries no window. So does a call the Runtime cut to the model's
    residual (``observed_truncated``): the window it recorded is what ``load_skill`` returned,
    not what the model was given, so the record is still a post-fact one and never a read.
    """
    path = facts.get(FACT_PATH)
    if not isinstance(path, str) or not path or not skill:
        return None
    if facts.get(FACT_OBSERVED_TRUNCATED) is True:
        return SkillRead(skill=skill, path=path)
    start = _whole_number(facts.get(FACT_WINDOW_START))
    end = _whole_number(facts.get(FACT_WINDOW_END))
    chars = _whole_number(facts.get(FACT_DOCUMENT_CHARS))
    if start is None or end is None or chars is None or not start <= end <= chars:
        return SkillRead(skill=skill, path=path)
    return SkillRead(skill=skill, path=path, start=start, end=end, document_chars=chars)


def _calls_of(
    tool: str, events: Iterable[tuple[str, Mapping[str, Any]]]
) -> tuple[tuple[str, str, str | None, dict[str, Any]], ...]:
    """Merge a Run's durable events into one ``(call_id, label, outcome, facts)`` per ``tool`` call.

    ``events`` are ``(event_type, payload)`` pairs as the Run's event stream holds them. The label
    is the call's ``object_label`` (the Skill name, for ``load_skill``), the outcome is the
    ``tool_end``'s, and the fact keys are taken from whichever event carries them, first one
    winning: a call recovery re-dispatched reports the same fact twice, and reading it once is the
    same fact. Calls come in call order.
    """
    order: dict[str, int] = {}
    tools: dict[str, str] = {}
    skills: dict[str, str] = {}
    outcomes: dict[str, str] = {}
    facts: dict[str, dict[str, Any]] = {}
    for event_type, payload in events:
        call_id = str(payload.get("call_id") or "")
        if not call_id:
            continue
        order.setdefault(call_id, len(order))
        if name := str(payload.get("tool_name") or ""):
            tools.setdefault(call_id, name)
        if label := str(payload.get("object_label") or ""):
            skills.setdefault(call_id, label)
        if event_type == "tool_end" and (outcome := str(payload.get("outcome") or "")):
            outcomes[call_id] = outcome
        for key, value in fact_values(payload).items():
            facts.setdefault(call_id, {}).setdefault(key, value)
    return tuple(
        (call_id, skills.get(call_id, ""), outcomes.get(call_id), facts.get(call_id, {}))
        for call_id in sorted(order, key=lambda value: order[value])
        if tools.get(call_id) == tool
    )


def skill_reads_from_events(
    events: Iterable[tuple[str, Mapping[str, Any]]],
) -> tuple[LoadSkillCall, ...]:
    """The Run's ``load_skill`` calls, one record each, in call order."""
    return tuple(
        LoadSkillCall(call_id=call_id, skill=skill, outcome=outcome, read=skill_read(skill, facts))
        for call_id, skill, outcome, facts in _calls_of(LOAD_SKILL_TOOL, events)
    )


def sealed_skills_from_events(events: Iterable[tuple[str, Mapping[str, Any]]]) -> tuple[str, ...]:
    """The Skills whose gate attested a sealed artifact in the Run's successful attach calls.

    A failed call is skipped the way a failed ``load_skill`` is, by the ``tool_end`` outcome.
    """
    skills: dict[str, None] = {}
    for _, _, outcome, facts in _calls_of(ATTACH_ARTIFACT_TOOL, events):
        joined = facts.get(FACT_SEALED_SKILL)
        if outcome != "failed" and isinstance(joined, str):
            skills.update(dict.fromkeys(name for name in joined.split(",") if name))
    return tuple(skills)


def documents_read_to_end(reads: Iterable[SkillRead]) -> frozenset[tuple[str, str]]:
    """The ``(skill, path)`` of every document the recorded windows cover from first to last.

    Windows are grouped by ``(skill, path, document_chars)``: two lengths are two versions of
    the file (a Skill republished between reads), and windows of different versions do not
    combine into a read of either. A group covers its document when its windows, taken by start,
    run from ``0`` to ``document_chars`` with no gap; overlap is harmless, because a model that
    re-reads a page has still read it. A document of zero characters is read by any window.
    A record with no window -- a ``write_to`` copy -- adds nothing.
    """
    windows: dict[tuple[str, str, int], list[tuple[int, int]]] = {}
    for read in reads:
        if read.start is None or read.end is None or read.document_chars is None:
            continue
        windows.setdefault((read.skill, read.path, read.document_chars), []).append(
            (read.start, read.end)
        )
    covered: set[tuple[str, str]] = set()
    for (skill, path, chars), spans in windows.items():
        position = 0
        for start, end in sorted(spans):
            if start > position:
                break
            position = max(position, end)
        if position >= chars:
            covered.add((skill, path))
    return frozenset(covered)


def missing_required_reads(
    declared: Mapping[str, Sequence[RequiredRead]],
    loaded: Sequence[str],
    read: frozenset[tuple[str, str]],
) -> tuple[tuple[str, RequiredRead], ...]:
    """Every ``(skill, entry)`` a loaded Skill declares and the Run did not read to the end.

    ``declared`` maps a Skill to its entries, ``loaded`` is the Skills the Run loaded in the order
    it loaded them, and ``read`` is :func:`documents_read_to_end`. An entry is satisfied when any
    one of its paths was read. A Skill that is loaded and not in ``declared`` declares nothing.
    """
    return tuple(
        (skill, entry)
        for skill in dict.fromkeys(loaded)
        for entry in declared.get(skill, ())
        if not any((skill, path) in read for path in entry.paths)
    )


__all__ = [
    "ATTACH_ARTIFACT_TOOL",
    "FACT_DOCUMENT_CHARS",
    "FACT_KEYS",
    "FACT_OBSERVED_TRUNCATED",
    "FACT_PATH",
    "FACT_SEALED_SKILL",
    "FACT_WINDOW_END",
    "FACT_WINDOW_START",
    "LOAD_SKILL_TOOL",
    "LoadSkillCall",
    "RequiredRead",
    "SkillRead",
    "documents_read_to_end",
    "fact_values",
    "missing_required_reads",
    "sealed_skills_from_events",
    "skill_read",
    "skill_reads_from_events",
]
