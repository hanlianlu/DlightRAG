# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Whether a Run read what its Skill says it must read: the rule, on recorded windows.

``engine/agent/skill_reads.py`` is the one place that says a document was read. These tests pin the
rule against hand-built facts in the shape ``load_skill`` records them, and pin that its fact keys
are the ones the Runtime forwards.
"""

from __future__ import annotations

from dlightrag.engine.agent.skill_reads import (
    FACT_DOCUMENT_CHARS,
    FACT_KEYS,
    FACT_OBSERVED_TRUNCATED,
    FACT_PATH,
    FACT_WINDOW_END,
    FACT_WINDOW_START,
    RequiredRead,
    SkillRead,
    documents_read_to_end,
    missing_required_reads,
    skill_read,
    skill_reads_from_events,
)
from dlightrag.engine.agent.tools.contracts import TOOL_PROGRESS_FACT_KEYS

_SKILL = "sealed-report"
_DOC = "references/gotchas.md"


def _window(
    start: int, end: int, chars: int, *, path: str = _DOC, skill: str = _SKILL
) -> SkillRead:
    return SkillRead(skill=skill, path=path, start=start, end=end, document_chars=chars)


def _fact(start: int, end: int, chars: int, *, path: str = _DOC) -> dict[str, object]:
    return {
        FACT_PATH: path,
        FACT_WINDOW_START: start,
        FACT_WINDOW_END: end,
        FACT_DOCUMENT_CHARS: chars,
    }


# --- the fact keys -------------------------------------------------------------------------------


def test_the_fact_keys_are_the_ones_the_runtime_forwards() -> None:
    """Two lists of strings, in two layers that cannot import each other, held together."""
    assert FACT_KEYS == TOOL_PROGRESS_FACT_KEYS


# --- the read rule ------------------------------------------------------------------------------


def test_contiguous_windows_that_reach_the_end_are_a_read() -> None:
    reads = [_window(0, 400, 1000), _window(400, 900, 1000), _window(900, 1000, 1000)]
    assert documents_read_to_end(reads) == {(_SKILL, _DOC)}


def test_the_order_the_windows_arrive_in_does_not_matter_and_overlap_is_harmless() -> None:
    reads = [_window(500, 1000, 1000), _window(0, 600, 1000)]
    assert documents_read_to_end(reads) == {(_SKILL, _DOC)}


def test_a_gap_between_windows_is_not_a_read() -> None:
    reads = [_window(0, 400, 1000), _window(500, 1000, 1000)]
    assert documents_read_to_end(reads) == frozenset()


def test_stopping_after_the_first_window_is_not_a_read() -> None:
    """Whatever the first window held, a reader that stopped there has not read the document."""
    assert documents_read_to_end([_window(0, 400, 1000)]) == frozenset()


def test_a_document_that_fits_one_window_is_read_by_that_window() -> None:
    assert documents_read_to_end([_window(0, 800, 800)]) == {(_SKILL, _DOC)}


def test_a_later_window_without_the_start_is_not_a_read() -> None:
    assert documents_read_to_end([_window(400, 1000, 1000)]) == frozenset()


def test_windows_of_two_versions_of_a_document_do_not_combine() -> None:
    """A Skill republished between two reads is two documents; neither was read to the end."""
    first_half_of_old = _window(0, 500, 1000)
    second_half_of_new = _window(500, 1200, 1200)
    assert documents_read_to_end([first_half_of_old, second_half_of_new]) == frozenset()


def test_one_complete_version_is_a_read_whatever_the_other_version_holds() -> None:
    reads = [_window(0, 500, 1000), _window(0, 1200, 1200)]
    assert documents_read_to_end(reads) == {(_SKILL, _DOC)}


def test_a_copy_records_no_window_and_never_reads() -> None:
    assert documents_read_to_end([SkillRead(skill=_SKILL, path=_DOC)]) == frozenset()


def test_an_empty_document_is_read_by_any_window_of_it() -> None:
    assert documents_read_to_end([_window(0, 0, 0)]) == {(_SKILL, _DOC)}


def test_the_same_path_in_two_skills_is_two_documents() -> None:
    reads = [_window(0, 100, 100, skill="a"), _window(0, 50, 100, skill="b")]
    assert documents_read_to_end(reads) == {("a", _DOC)}


# --- one call's fact ----------------------------------------------------------------------------


def test_a_complete_window_fact_is_a_read_with_its_window() -> None:
    assert skill_read(_SKILL, _fact(0, 400, 1000)) == _window(0, 400, 1000)


def test_a_path_alone_is_a_copy() -> None:
    assert skill_read(_SKILL, {FACT_PATH: _DOC}) == SkillRead(skill=_SKILL, path=_DOC)


def test_no_path_or_no_skill_is_no_fact() -> None:
    assert skill_read(_SKILL, {}) is None
    assert skill_read(_SKILL, {FACT_PATH: ""}) is None
    assert skill_read("", {FACT_PATH: _DOC}) is None


def test_a_window_that_is_damaged_is_never_credited() -> None:
    """Present, so the record is known to postdate the fact; but never a read."""
    damaged = [
        {**_fact(0, 400, 1000), FACT_WINDOW_END: 1500},  # past the document
        {**_fact(0, 400, 1000), FACT_WINDOW_START: 500},  # starts after it ends
        {**_fact(0, 400, 1000), FACT_WINDOW_START: -1},
        {**_fact(0, 400, 1000), FACT_WINDOW_END: True},  # a bool is not an offset
        {**_fact(0, 400, 1000), FACT_DOCUMENT_CHARS: "1000"},
        {FACT_PATH: _DOC, FACT_WINDOW_START: 0},  # a window missing its other keys
    ]
    for facts in damaged:
        assert skill_read(_SKILL, facts) == SkillRead(skill=_SKILL, path=_DOC), facts


# --- a Run's events -----------------------------------------------------------------------------


def _call(
    call_id: str,
    skill: str,
    *facts: dict[str, object],
    outcome: str = "succeeded",
    cut: bool = False,
):
    """One ``load_skill`` call's events; ``cut`` is the Runtime having cut the result on settling."""
    base = {"call_id": call_id, "tool_name": "load_skill"}
    events: list[tuple[str, dict[str, object]]] = [
        ("tool_start", dict(base)),
        ("tool_progress", {**base, "object_label": skill}),
    ]
    events += [("tool_progress", {**base, "object_label": skill, **fact}) for fact in facts]
    settled: dict[str, object] = {**base, "object_label": skill, "outcome": outcome}
    if cut:
        settled["observed_truncated"] = True
    events.append(("tool_end", settled))
    return events


def test_events_merge_into_one_record_per_load_skill_call_in_call_order() -> None:
    events = [
        *_call("c1", _SKILL, _fact(0, 400, 1000)),
        ("tool_start", {"call_id": "c2", "tool_name": "bash"}),
        ("tool_progress", {"call_id": "c2", "tool_name": "bash", "object_label": "ls"}),
        *_call("c3", "interactive-html", {FACT_PATH: "references/x.html"}),
        *_call("c4", _SKILL, outcome="failed"),
    ]

    calls = skill_reads_from_events(events)

    assert [(call.call_id, call.skill, call.outcome) for call in calls] == [
        ("c1", _SKILL, "succeeded"),
        ("c3", "interactive-html", "succeeded"),
        ("c4", _SKILL, "failed"),
    ]
    assert calls[0].read == _window(0, 400, 1000)
    assert calls[1].read == SkillRead(skill="interactive-html", path="references/x.html")
    assert calls[2].read is None


def test_a_call_a_recovery_reported_twice_is_one_fact() -> None:
    events = _call("c1", _SKILL, _fact(0, 400, 1000), _fact(0, 400, 1000))
    (call,) = skill_reads_from_events(events)
    assert call.read == _window(0, 400, 1000)


def test_a_call_that_recorded_no_fact_has_none() -> None:
    (call,) = skill_reads_from_events(_call("c1", _SKILL))
    assert call.read is None


def test_a_payload_key_outside_the_fact_is_not_part_of_one() -> None:
    events = _call("c1", _SKILL, {**_fact(0, 10, 10), "text_chars": 99, "extra": "x"})
    (call,) = skill_reads_from_events(events)
    assert call.read == _window(0, 10, 10)


# --- what was declared and not read --------------------------------------------------------------


def test_an_any_of_entry_is_satisfied_by_reading_any_one_of_its_paths() -> None:
    entry = RequiredRead(paths=("references/a.html", "references/b.html"), when="before writing")
    read = frozenset({(_SKILL, "references/b.html")})

    assert missing_required_reads({_SKILL: [entry]}, [_SKILL], read) == ()
    assert missing_required_reads({_SKILL: [entry]}, [_SKILL], frozenset()) == ((_SKILL, entry),)


def test_only_loaded_skills_are_held_to_their_declarations_once_each_in_load_order() -> None:
    one = RequiredRead(paths=("references/one.md",), when="first")
    two = RequiredRead(paths=("references/two.md",), when="second")
    declared = {"alpha": [one], "beta": [two], "never-loaded": [one]}

    assert missing_required_reads(
        declared, ["beta", "alpha", "beta", "undeclared"], frozenset()
    ) == (
        ("beta", two),
        ("alpha", one),
    )


def test_a_path_read_in_another_skill_does_not_satisfy_this_one() -> None:
    entry = RequiredRead(paths=(_DOC,), when="always")
    assert missing_required_reads({_SKILL: [entry]}, [_SKILL], frozenset({("other", _DOC)})) == (
        (_SKILL, entry),
    )


# --- a window the Runtime cut ---------------------------------------------------------------------


def test_a_call_the_runtime_cut_keeps_its_path_and_loses_its_window() -> None:
    """The window ``load_skill`` recorded is what it returned, not what the model was given."""
    cut = skill_read(_SKILL, {**_fact(0, 400, 1000), FACT_OBSERVED_TRUNCATED: True})

    assert cut is not None and cut == SkillRead(skill=_SKILL, path=_DOC)
    assert documents_read_to_end([cut]) == frozenset()
    # Only a true bool is the Runtime's cut: anything else leaves the window as recorded.
    for not_a_cut in (False, None, 1, "true"):
        assert skill_read(_SKILL, {**_fact(0, 400, 1000), FACT_OBSERVED_TRUNCATED: not_a_cut}) == (
            _window(0, 400, 1000)
        )


def test_the_cut_flag_is_read_from_the_settlement_and_not_counted_as_a_fact_key() -> None:
    assert FACT_OBSERVED_TRUNCATED not in FACT_KEYS
    assert FACT_OBSERVED_TRUNCATED not in TOOL_PROGRESS_FACT_KEYS


def test_a_cut_window_leaves_a_gap_the_next_windows_cannot_close() -> None:
    """A cut page never showed the model the continuation it ends with, so the document is not read."""
    events = [
        *_call("c1", _SKILL, _fact(0, 400, 1000), cut=True),
        *_call("c2", _SKILL, _fact(400, 1000, 1000)),
    ]

    first, second = skill_reads_from_events(events)

    assert first.read == SkillRead(skill=_SKILL, path=_DOC)
    assert second.read == _window(400, 1000, 1000)
    assert (
        documents_read_to_end([call.read for call in (first, second) if call.read]) == frozenset()
    )


def test_a_cut_page_does_not_void_the_pages_that_were_not_cut() -> None:
    """Credit is per call: the same document read whole in another call is still a read."""
    events = [
        *_call("c1", _SKILL, _fact(0, 400, 1000), cut=True),
        *_call("c2", _SKILL, _fact(0, 1000, 1000)),
    ]

    reads = [call.read for call in skill_reads_from_events(events) if call.read]

    assert documents_read_to_end(reads) == {(_SKILL, _DOC)}
