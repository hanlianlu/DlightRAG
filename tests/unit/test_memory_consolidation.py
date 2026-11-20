# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The host-neutral half of a Session Memory Pass: prompt, verdict and plan."""

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from dlightrag_memory import MemoryProvenance, MemoryRecord
from dlightrag_memory.consolidation import (
    CONTEXT_ASSISTANT_CHARS,
    CONTEXT_ROUNDS,
    PASS_MUTATION_LIMIT,
    ConsolidationVerdict,
    ConversationTurn,
    PlannedWrite,
    VerdictWrite,
    consolidation_request,
    plan_consolidation,
    planned_operation,
    verified_quote,
)
from dlightrag_memory.policy import MEMORY_QUOTE_LIMIT, evaluate_memory_operation

_CONTEXT = (
    "Earlier in this conversation (context only — remember nothing from it unless the "
    "newest part restates or relies on it):"
)
_NEWEST = "Newest part of this conversation — the part to judge (oldest first):"


def _write(action: str, body: str = "The owner prefers Chinese.", *targets: int) -> VerdictWrite:
    return VerdictWrite(action=action, kind="preference", body=body, targets=list(targets))  # type: ignore[arg-type]


def _verdict(*writes: VerdictWrite) -> ConsolidationVerdict:
    return ConsolidationVerdict(writes=list(writes))


def _record(
    memory_id: str, body: str = "Answer in Chinese.", *, updated_at: datetime | None = None
) -> MemoryRecord:
    return MemoryRecord(
        owner_id="o",
        memory_id=memory_id,
        kind="preference",
        body=body,
        provenance=MemoryProvenance(origin_kind="session_pass", origin_id="p", session_id="s"),
        updated_at=updated_at,
    )


def _plan(
    verdict: ConsolidationVerdict,
    *,
    handles: Mapping[int, str],
    record_limit: int = 5,
    owner_words: tuple[str, ...] = (),
) -> tuple[PlannedWrite, ...]:
    return plan_consolidation(
        verdict, handles=handles, record_limit=record_limit, owner_words=owner_words
    )


def test_an_empty_verdict_plans_nothing() -> None:
    assert _plan(_verdict(), handles={}) == ()


def test_a_verdict_names_memories_by_the_numbers_it_was_shown() -> None:
    planned = _plan(
        _verdict(_write("update", "refined", 2), _write("fuse", "merged", 3, 1)),
        handles={1: "mem-a", 2: "mem-b", 3: "mem-c"},
        record_limit=5,
    )
    assert [(write.action, write.target_ids) for write in planned] == [
        ("update", ("mem-b",)),
        ("fuse", ("mem-c", "mem-a")),
    ]


def test_a_write_naming_a_number_the_model_was_not_shown_is_left_out() -> None:
    for number in (4, 0, -1):
        planned = _plan(
            _verdict(
                _write("update", "x", number), _write("fuse", "z", 1, number), _write("add", "y")
            ),
            handles={1: "real", 2: "other", 3: "third"},
            record_limit=5,
        )
        assert [write.body for write in planned] == ["y"], number


def test_targets_must_fit_the_action_and_be_claimed_once() -> None:
    planned = _plan(
        _verdict(
            _write("add", "has a target", 1),
            _write("update", "two targets", 1, 2),
            _write("fuse", "one source", 1),
            _write("fuse", "merged", 1, 2),
            _write("update", "a again", 1),
        ),
        handles={1: "a", 2: "b"},
        record_limit=5,
    )
    assert [write.body for write in planned] == ["merged"]


def test_a_fuse_naming_one_number_twice_has_one_source_and_is_left_out() -> None:
    planned = _plan(
        _verdict(_write("fuse", "merged", 1, 1)), handles={1: "a", 2: "b"}, record_limit=5
    )
    assert planned == ()


def test_a_pass_plans_no_more_adds_than_its_limit() -> None:
    """The bound is per pass and counts adds, counted from zero inside one verdict."""
    planned = _plan(
        _verdict(*(_write("add", f"memory {index}") for index in range(7))),
        handles={},
        record_limit=5,
    )
    assert [write.body for write in planned] == [f"memory {index}" for index in range(5)]


def test_maintenance_never_spends_the_add_budget_and_survives_a_full_verdict() -> None:
    """Memory is the owner's across conversations: the cap counts adds, not maintenance.

    Updating or fusing records the owner already holds — this Session's own or an earlier
    conversation's — never counts, so a verdict that spent its five adds still updates and
    fuses, whatever conversation wrote the targets.
    """
    planned = _plan(
        _verdict(
            *(_write("add", f"memory {index}") for index in range(5)),
            _write("update", "refined", 1),
            _write("fuse", "merged", 2, 3),
            _write("add", "one too many"),
        ),
        handles={1: "own-0", 2: "older-1", 3: "older-2"},
        record_limit=5,
    )
    assert [write.body for write in planned] == [
        "memory 0",
        "memory 1",
        "memory 2",
        "memory 3",
        "memory 4",
        "refined",
        "merged",
    ]


def test_a_pass_fuses_records_an_earlier_session_contributed() -> None:
    planned = _plan(
        _verdict(_write("fuse", "merged", 1, 2)),
        handles={1: "older-1", 2: "older-2"},
        record_limit=5,
    )
    (write,) = planned
    assert (write.action, write.body, write.target_ids) == (
        "fuse",
        "merged",
        ("older-1", "older-2"),
    )
    provenance = MemoryProvenance(origin_kind="session_pass", origin_id="s:1-9", session_id="s")
    operation = planned_operation(
        write, index=0, owner_id="o", provenance=provenance, pass_key="pass:s:1-9"
    )
    assert operation.action == "fuse" and operation.source_ids == ("older-1", "older-2")


def test_planned_writes_round_trip_and_become_valid_operations() -> None:
    planned = (
        PlannedWrite(action="add", kind="experience", body="The corpus needs pg_jieba."),
        PlannedWrite(action="update", kind="fact", body="Lives in Munich.", target_ids=("m1",)),
        PlannedWrite(
            action="fuse", kind="preference", body="Chinese, tables first.", target_ids=("a", "b")
        ),
    )
    provenance = MemoryProvenance(origin_kind="session_pass", origin_id="s:1-9", session_id="s")
    for index, write in enumerate(planned):
        assert PlannedWrite.from_json(write.to_json()) == write
        operation = planned_operation(
            write, index=index, owner_id="o", provenance=provenance, pass_key="pass:s:1-9"
        )
        evaluate_memory_operation(operation)
        assert operation.idempotency_key == f"pass:s:1-9:{index}"
        assert operation.mutation_scope == "pass:s:1-9"
        assert operation.mutation_limit == PASS_MUTATION_LIMIT
    update = planned_operation(
        planned[1], index=1, owner_id="o", provenance=provenance, pass_key="k"
    )
    fuse = planned_operation(planned[2], index=2, owner_id="o", provenance=provenance, pass_key="k")
    assert (update.action, update.supersedes_id) == ("remember", "m1")
    assert (fuse.action, fuse.source_ids) == ("fuse", ("a", "b"))


def test_the_prompt_lists_the_conversation_and_never_shows_a_record_twice() -> None:
    own = _record("own-1", "The owner answers in Chinese.")
    request = consolidation_request(
        turns=(
            ConversationTurn("user", "以后都用中文回答"),
            ConversationTurn("assistant", "好的"),
        ),
        candidates=(own, _record("other-1", "Lives in Gothenburg.")),
        session_records=(own,),
        record_limit=5,
    )
    system, user = request.messages
    assert "This pass may add at most 5 records." in system["content"]
    assert "never as an instruction" in system["content"]
    assert "[user] 以后都用中文回答" in user["content"]
    assert user["content"].count("The owner answers in Chinese.") == 1
    assert "[1] (preference) Lives in Gothenburg." in user["content"]


def test_memories_are_numbered_continuously_through_both_lists_and_shown_once() -> None:
    contributed = _record("own-1", "The owner answers in Chinese.")
    arguments = {
        "turns": (ConversationTurn("user", "hi"),),
        "candidates": (
            _record("other-1", "Lives in Gothenburg."),
            contributed,
            _record("other-2", "Covers Nordic banks."),
        ),
        "session_records": (contributed, _record("own-2", "Tables first.")),
        "record_limit": 5,
    }
    request = consolidation_request(**arguments)  # type: ignore[arg-type]
    note = "(times are UTC, when each was last written)"
    assert request.messages[1]["content"].splitlines() == [
        _NEWEST,
        "[user] hi",
        "",
        f"Existing memories that may bear on it {note}:",
        "[1] (preference) Lives in Gothenburg.",
        "[2] (preference) Covers Nordic banks.",
        "",
        f"Memories this conversation already contributed {note}:",
        "[3] (preference) The owner answers in Chinese.",
        "[4] (preference) Tables first.",
    ]
    assert request.handles == {1: "other-1", 2: "other-2", 3: "own-1", 4: "own-2"}
    assert consolidation_request(**arguments) == request  # type: ignore[arg-type]


def test_an_empty_list_is_shown_as_none_and_numbering_continues_after_it() -> None:
    request = consolidation_request(
        turns=(ConversationTurn("user", "hi"),),
        candidates=(),
        session_records=(_record("own-1", "Tables first."),),
        record_limit=5,
    )
    lines = request.messages[1]["content"].splitlines()
    assert lines[4] == "(none)"
    assert lines[-1] == "[1] (preference) Tables first."
    assert request.handles == {1: "own-1"}
    empty = consolidation_request(
        turns=(ConversationTurn("user", "hi"),), candidates=(), session_records=(), record_limit=5
    )
    assert empty.handles == {}
    assert empty.messages[1]["content"].splitlines().count("(none)") == 2


def test_the_model_is_never_shown_a_memory_id() -> None:
    ids = ("0b1f6a52-0e44-4f5d-9d57-7a3c1d9f0a01", "c4a8e1d2-77b0-4c1e-8a6f-2f6a9b3d5e02")
    request = consolidation_request(
        turns=(ConversationTurn("user", "hi"),),
        candidates=(_record(ids[0], "Lives in Gothenburg."),),
        session_records=(_record(ids[1], "Tables first."),),
        record_limit=5,
    )
    for message in request.messages:
        for memory_id in ids:
            assert memory_id not in message["content"]
    assert [request.handles[number] for number in (1, 2)] == list(ids)


def test_a_listed_memory_carries_when_it_was_last_written_to_the_hour_in_utc() -> None:
    request = consolidation_request(
        turns=(ConversationTurn("user", "hi"),),
        candidates=(
            _record("a", "Dated.", updated_at=datetime(2026, 10, 17, 14, 59, tzinfo=UTC)),
            _record("b", "Undated."),
        ),
        session_records=(),
        record_limit=5,
    )
    lines = request.messages[1]["content"].splitlines()
    assert "[1] (preference, 2026-10-17 14:00) Dated." in lines
    assert "[2] (preference) Undated." in lines


def _shown(
    turns: tuple[ConversationTurn, ...], context_turns: tuple[ConversationTurn, ...] = ()
) -> list[str]:
    """The turns part of a request's user message: everything before the memories."""
    request = consolidation_request(
        turns=turns,
        context_turns=context_turns,
        candidates=(),
        session_records=(),
        record_limit=5,
    )
    return request.messages[1]["content"].split("\n\nExisting memories")[0].splitlines()


def test_earlier_turns_come_first_as_context_and_the_newest_part_is_the_one_to_judge() -> None:
    shown = _shown(
        (ConversationTurn("user", "对，以后报告都这样做"),),
        (
            ConversationTurn("user", "你的报告几乎全是表格，我看着累。这份能不能用图？"),
            ConversationTurn("assistant", "可以，我改成图表为主。"),
        ),
    )

    assert shown == [
        _CONTEXT,
        "[user] 你的报告几乎全是表格，我看着累。这份能不能用图？",
        "[assistant] 可以，我改成图表为主。",
        "",
        _NEWEST,
        "[user] 对，以后报告都这样做",
    ]


def test_with_no_earlier_turns_the_context_section_is_omitted() -> None:
    shown = _shown((ConversationTurn("user", "hi"), ConversationTurn("assistant", "hello")))

    assert shown == [_NEWEST, "[user] hi", "[assistant] hello"]
    assert "Earlier in this conversation" not in "\n".join(shown)


def test_a_user_context_turn_is_whole_and_an_assistant_one_keeps_its_first_characters() -> None:
    whole = "x" * CONTEXT_ASSISTANT_CHARS
    long = "HEAD" + "m" * (CONTEXT_ASSISTANT_CHARS + 50) + "TAIL"

    shown = _shown(
        (ConversationTurn("user", long),),
        (
            ConversationTurn("user", long),
            ConversationTurn("assistant", long),
            ConversationTurn("assistant", whole),
        ),
    )

    # A user turn is shown whole however long; an assistant turn over the bound keeps its first
    # characters before a trailing ellipsis, one at the bound is whole, and the newest part always is.
    assert shown[1] == "[user] " + long
    assert shown[2] == "[assistant] " + long[:CONTEXT_ASSISTANT_CHARS] + "…"
    assert shown[2].startswith("[assistant] HEAD") and "TAIL" not in shown[2]
    assert shown[3] == "[assistant] " + whole
    assert shown[-1] == "[user] " + long


def test_the_context_bounds_are_five_rounds_and_five_hundred_assistant_characters() -> None:
    assert (CONTEXT_ROUNDS, CONTEXT_ASSISTANT_CHARS) == (5, 500)


def test_the_context_is_only_a_section_ahead_of_the_rest_of_the_request() -> None:
    arguments = {
        "turns": (ConversationTurn("user", "hi"),),
        "candidates": (_record("other-1", "Lives in Gothenburg."),),
        "session_records": (_record("own-1", "Tables first."),),
        "record_limit": 5,
    }
    plain = consolidation_request(**arguments)  # type: ignore[arg-type]
    with_context = consolidation_request(
        **arguments,  # type: ignore[arg-type]
        context_turns=(ConversationTurn("assistant", "earlier"),),
    )

    # Same rules, same numbered memories and handles: the earlier turns add a section, nothing more.
    assert with_context.messages[0] == plain.messages[0]
    assert with_context.handles == plain.handles == {1: "other-1", 2: "own-1"}
    assert with_context.messages[1]["content"] == (
        f"{_CONTEXT}\n[assistant] earlier\n\n" + plain.messages[1]["content"]
    )


# ------------------------------------------------------------------------------------ quotes


def _quoting(
    quote: str | None, body: str = "The owner prefers charts in a dark theme."
) -> VerdictWrite:
    return VerdictWrite(action="add", kind="preference", body=body, quote=quote)


def _planned_quote(quote: str | None, *owner_words: str) -> str | None:
    (write,) = _plan(_verdict(_quoting(quote)), handles={}, owner_words=owner_words)
    return write.quote


def test_a_quote_the_owner_said_is_kept_exactly() -> None:
    said = "以后图表都用深色主题，别用饼图。"

    assert _planned_quote("以后图表都用深色主题", said, "好的") == "以后图表都用深色主题"
    assert _planned_quote(said, said) == said
    # Any one of the owner's turns will do.
    assert _planned_quote("别用饼图", "先看数据", said) == "别用饼图"


def test_a_quote_differs_from_the_owners_words_by_whitespace_alone() -> None:
    said = "Please   keep every\n\tconclusion to three sentences."

    # Runs of whitespace, line breaks included, are one space on both sides; the host keeps the
    # squeezed form so that it renders as the one line it is charged as.
    assert _planned_quote("keep every conclusion to three sentences.", said) == (
        "keep every conclusion to three sentences."
    )
    assert _planned_quote("every\nconclusion", said) == "every conclusion"
    assert _planned_quote("  keep every  ", said) == "keep every"
    # Nothing else is forgiven: a space where the owner wrote none is not their words.
    assert _planned_quote("以后 图表", "以后图表都用深色主题") is None


@pytest.mark.parametrize(
    ("quote", "owner_words"),
    [
        pytest.param("always cite the filing", ("Always cite the filing.",), id="case"),
        pytest.param("，", ("以后图表都用深色主题,",), id="full-width-punctuation"),
        pytest.param("The owner prefers dark charts.", ("以后图表都用深色主题。",), id="invented"),
        pytest.param("keep it short", ("keep it", "short"), id="across-two-turns"),
        pytest.param("keep it short", (), id="no-owner-turn-at-all"),
        pytest.param("", ("keep it short",), id="empty"),
        pytest.param("   ", ("keep it short",), id="blank"),
        pytest.param("see [1] for the filing", ("see [1] for the filing",), id="citation-marker"),
        pytest.param(
            "my key is sk-abcdefghijklmnopqrstuvwxyz",
            ("my key is sk-abcdefghijklmnopqrstuvwxyz",),
            id="credential",
        ),
    ],
)
def test_a_quote_that_is_not_the_owners_words_or_cannot_be_stored_is_dropped(
    quote: str, owner_words: tuple[str, ...]
) -> None:
    # The write stands; only the quote goes.
    (write,) = _plan(_verdict(_quoting(quote)), handles={}, owner_words=owner_words)

    assert write.quote is None
    assert write.body == "The owner prefers charts in a dark theme."


def test_a_quote_over_the_bound_is_dropped_and_one_at_it_is_kept() -> None:
    said = "字" * (MEMORY_QUOTE_LIMIT + 30)

    assert MEMORY_QUOTE_LIMIT == 120
    assert _planned_quote("字" * MEMORY_QUOTE_LIMIT, said) == "字" * MEMORY_QUOTE_LIMIT
    assert _planned_quote("字" * (MEMORY_QUOTE_LIMIT + 1), said) is None


def test_the_owners_words_are_the_user_turns_of_the_newest_part_alone() -> None:
    request = consolidation_request(
        turns=(
            ConversationTurn("user", "以后图表都用深色主题"),
            ConversationTurn("assistant", "好的，我记住了深色主题。"),
            ConversationTurn("user", "结论短一点"),
        ),
        context_turns=(
            ConversationTurn("user", "上一轮：别用饼图"),
            ConversationTurn("assistant", "上一轮的回复"),
        ),
        candidates=(),
        session_records=(),
        record_limit=5,
    )

    assert request.owner_words == ("以后图表都用深色主题", "结论短一点")

    def planned(quote: str) -> str | None:
        (write,) = _plan(
            _verdict(_quoting(quote)),
            handles=request.handles,
            owner_words=request.owner_words,
        )
        return write.quote

    assert planned("以后图表都用深色主题") == "以后图表都用深色主题"
    assert planned("结论短一点") == "结论短一点"
    # The Agent's reply and the earlier turns are not the owner's words in this part.
    assert planned("我记住了深色主题") is None
    assert planned("别用饼图") is None
    assert planned("上一轮的回复") is None


def test_verified_quote_is_the_one_rule_the_plan_and_the_eval_share() -> None:
    assert verified_quote(None, ("anything",)) is None
    assert verified_quote("a  b", ("a b c",)) == "a b"
    assert verified_quote("a b", ("a  b c",)) == "a b"
    assert verified_quote("x", ()) is None


def test_a_verdict_parses_with_a_quote_without_one_and_with_one_too_long_to_keep() -> None:
    plain = ConsolidationVerdict.model_validate_json(
        '{"writes": [{"action": "add", "kind": "fact", "body": "Lives in Oslo."}]}'
    )
    null = ConsolidationVerdict.model_validate_json(
        '{"writes": [{"action": "add", "kind": "fact", "body": "Lives in Oslo.", "quote": null}]}'
    )
    long = ConsolidationVerdict.model_validate_json(
        '{"writes": [{"action": "add", "kind": "fact", "body": "Lives in Oslo.", "quote": "'
        + "x" * 500
        + '"}]}'
    )

    assert plain.writes[0].quote is None and null.writes[0].quote is None
    # The parser keeps a quote it will not store, so that the host can drop it and keep the write.
    assert long.writes[0].quote == "x" * 500
    assert [write.quote for write in _plan(long, handles={}, owner_words=("x" * 500,))] == [None]


def test_a_quote_goes_into_the_stored_plan_and_the_operation() -> None:
    write = PlannedWrite(
        action="add", kind="preference", body="Dark charts.", quote="图表都用深色主题"
    )
    provenance = MemoryProvenance(origin_kind="session_pass", origin_id="s:1-9", session_id="s")

    assert PlannedWrite.from_json(write.to_json()) == write
    assert write.to_json()["quote"] == "图表都用深色主题"
    operation = planned_operation(
        write, index=0, owner_id="o", provenance=provenance, pass_key="pass:s:1-9"
    )
    evaluate_memory_operation(operation)
    assert operation.quote == "图表都用深色主题"
    fuse = planned_operation(
        PlannedWrite(action="fuse", kind="fact", body="Merged.", target_ids=("a", "b"), quote="q"),
        index=1,
        owner_id="o",
        provenance=provenance,
        pass_key="pass:s:1-9",
    )
    assert (fuse.action, fuse.quote) == ("fuse", "q")


def test_the_prompt_asks_for_the_owners_words_and_the_shape_shows_where_they_go() -> None:
    prompt = " ".join(
        consolidation_request(
            turns=(ConversationTurn("user", "one"),),
            candidates=(),
            session_records=(),
            record_limit=5,
        )
        .messages[0]["content"]
        .split()
    )

    assert "Give a write a quote when the owner said it" in prompt
    assert "at most 120 characters" in prompt
    assert "copied exactly from a [user] turn of the newest part" in prompt
    assert "never from the earlier turns, never from the Agent" in prompt
    assert "Leave quote out when the owner did not say it" in prompt
    assert "an experience the Agent learned on its own" in prompt
    assert (
        '{"writes": [{"action": "add", "kind": "preference", "body": "The owner prefers ...", '
        '"targets": [], "quote": "the owner\'s own words"}]}'
    ) in prompt
    assert "{quote_limit}" not in prompt


def test_a_listed_memory_shows_the_owners_words_behind_it_under_it() -> None:
    def memory(memory_id: str, body: str, quote: str | None) -> MemoryRecord:
        return MemoryRecord(
            owner_id="o",
            memory_id=memory_id,
            kind="preference",
            body=body,
            provenance=MemoryProvenance(origin_kind="session_pass", origin_id="p", session_id="s"),
            quote=quote,
        )

    request = consolidation_request(
        turns=(ConversationTurn("user", "hi"),),
        candidates=(
            memory("a", "Dark charts.", "以后图表都用深色主题"),
            memory("b", "Short conclusions.", None),
        ),
        session_records=(memory("c", "Tables first.", "表格放在最前面"),),
        record_limit=5,
    )

    lines = request.messages[1]["content"].splitlines()
    first = lines.index("[1] (preference) Dark charts.")
    assert lines[first + 1] == '  owner said: "以后图表都用深色主题"'
    # A memory with no quote is followed by whatever comes next (here the end of its list), and
    # the numbers go on counting memories, not lines.
    second = lines.index("[2] (preference) Short conclusions.")
    assert second == first + 2 and lines[second + 1] == ""
    third = lines.index("[3] (preference) Tables first.")
    assert lines[third + 1] == '  owner said: "表格放在最前面"'
    assert request.handles == {1: "a", 2: "b", 3: "c"}
