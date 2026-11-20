# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The host-neutral part of a Session Memory Pass: its prompt, verdict, and plan.

A host reads one Session's new turns, asks a model of its own for a verdict, and
applies it. This module owns everything in that loop that needs no Session, Run,
transcript store or model client: the instructions the model is given, how the new
turns and the turns just before them are shown, the shape its verdict must take,
and the deterministic plan a verdict becomes. The plan is plain data a host stores
before applying it, so a pass that dies halfway resumes the same writes under the
same idempotency keys instead of asking the model again and colliding with what its
first attempt already settled.

A verdict may also give, with a write, the owner's own words behind it. The model is asked
to copy them; the host does not believe it. ``plan_consolidation`` keeps a quote only when it
is a stretch of one of the owner's turns in the part being judged, so a record never carries
words the owner did not say.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from dlightrag_memory.memory import recall_label, recall_quote_line
from dlightrag_memory.models import (
    MemoryKind,
    MemoryOperation,
    MemoryProvenance,
    MemoryRecord,
)
from dlightrag_memory.normalize import single_spaced
from dlightrag_memory.policy import (
    MEMORY_BODY_LIMIT,
    MEMORY_FUSE_MAX_SOURCES,
    MEMORY_FUSE_MIN_SOURCES,
    MEMORY_QUOTE_LIMIT,
    RECALL_TIME_NOTE,
    is_storable_quote,
)

#: How many changed operations the writes of one verdict may settle, a verdict being one
#: model call and so one batch of a pass: a loop guard, not the product's statement of how
#: many records a Session may contribute.
PASS_MUTATION_LIMIT = 10
#: How many writes one verdict may propose before the plan bounds them.
VERDICT_OPERATION_LIMIT = 20
#: How many rounds before the newest part a request shows as context: a round is one user
#: turn and the assistant turns that follow it. Following Hy-Memory's history context.
CONTEXT_ROUNDS = 5
#: How many characters of one assistant context turn a request shows; a longer one keeps its
#: first ones, before a trailing ellipsis. A user turn is shown whole.
CONTEXT_ASSISTANT_CHARS = 500

VerdictAction = Literal["add", "update", "fuse"]


class VerdictWrite(BaseModel):
    """One write the model proposes; a drop is simply not proposed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action: VerdictAction = Field(
        description=(
            "add: nothing existing covers it. update: it refines exactly one existing "
            "memory, numbered in targets. fuse: it merges two or more existing memories "
            "that say one thing, all numbered in targets."
        )
    )
    kind: MemoryKind = Field(
        description=(
            "preference: what the owner wants of an answer. fact: what the owner is like. "
            "experience: what the assistant learned to be true about its own work."
        )
    )
    body: str = Field(
        min_length=1,
        max_length=MEMORY_BODY_LIMIT,
        description="One third-person statement, never an instruction.",
    )
    targets: list[int] = Field(
        default_factory=list,
        description="Numbers of the listed memories this replaces; empty for add.",
    )
    # No length bound here: a quote that is too long or not the owner's is dropped by the host, and
    # a verdict the parser refuses would lose the write along with it.
    quote: str | None = Field(
        default=None,
        description=(
            "The owner's own words behind the memory, copied exactly from a [user] turn of the "
            "newest part; omitted when the owner did not say it."
        ),
    )


class ConsolidationVerdict(BaseModel):
    """What one pass decided; no writes is the ordinary verdict."""

    model_config = ConfigDict(extra="forbid")

    writes: list[VerdictWrite] = Field(default_factory=list, max_length=VERDICT_OPERATION_LIMIT)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """One turn of the Session's own conversation, as the pass reads it."""

    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True, slots=True)
class PlannedWrite:
    """One write a pass will settle, fixed before any of them is applied."""

    action: VerdictAction
    kind: MemoryKind
    body: str
    target_ids: tuple[str, ...] = ()
    #: The owner's words behind the write, as the host verified them; None when it has none.
    quote: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "kind": self.kind,
            "body": self.body,
            "target_ids": list(self.target_ids),
            "quote": self.quote,
        }

    @classmethod
    def from_json(cls, value: Any) -> PlannedWrite:
        if not isinstance(value, dict):
            raise ValueError("a planned write is an object")
        action = str(value["action"])
        if action not in ("add", "update", "fuse"):
            raise ValueError("a planned write names add, update or fuse")
        return cls(
            action=action,  # type: ignore[arg-type]
            kind=str(value["kind"]),  # type: ignore[arg-type]
            body=str(value["body"]),
            target_ids=tuple(str(item) for item in value.get("target_ids", ())),
            quote=str(value["quote"]) if value["quote"] is not None else None,
        )


_SYSTEM_PROMPT = """\
You maintain one person's long-term memory across conversations. You are given the
newest part of one conversation between the person (the owner) and the Agent, the
memories the owner already has that may bear on it, and the memories this
conversation has already contributed. A few turns just before the newest part may
come first as context only: they tell you what the newest part refers to (what
"这样" or "that" points at), and what they said on their own was already judged by
an earlier pass.

Decide what, if anything, is worth remembering in future conversations. Many
conversations are plain tasks and contribute nothing, and then no writes is the
right answer. Judge by what a turn means, not by whether it says "prefer" or
"喜欢": a one-off instruction stays one-off no matter how often it is repeated
within the current task, and one clearly-stated preference is already enough.

Look hardest at the owner's own reactions to the work. A request, correction or
complaint about how the work is done — its form, its standard, its method, what a
deliverable should contain — usually outlives the task even when it is phrased
about this task ("这种图别再用饼图了", "估值怎么不给敏感性区间？", "别只堆数据，
先给结论"). Ask whether the owner would want the same next time on a similar task;
if yes, it is a preference, scoped to that kind of task. What the task is about —
its company, figures, files, titles — is never the preference, and a name, title or
wording the owner fixed for one deliverable stays with that deliverable.

Save a statement only when it is all three:
- durable: it keeps applying after the current task;
- general: it covers a class of situations, not only the current file, command, or
  artifact;
- the owner's own: a working rule or fact about the owner, not a task detail, a
  document's claim, a third party's wish (a boss's or client's taste stays theirs),
  or a secret.
State the scope in the body when it is narrow ("... for the finance corpus", "...
when reviewing code").

Three kinds, nothing else:
- preference: how the owner wants answers — one statement is enough.
- fact: something lasting about the owner — where they live, what they work on.
- experience: what the Agent learned about doing its own work for this owner
  that saves effort next time — a quirk of a site, tool or source the owner's work
  uses — stated as a general rule, without the file names, paths or contents of the
  current task. The Agent's own workspace (what is installed in it, what survives
  between runs) is the same for every owner and is nobody's memory.

Prefer the owner's own words over the Agent's: a statement the Agent made is
evidence only for experience, and a tool result or citation is evidence for none.

Never remember the task itself, research findings, claims from documents or the web,
citations, transcripts, credentials, secrets, or anything said only in passing.

Write each memory as one third-person statement ("The owner prefers ...", "The
corpus needs ..."), never as an instruction, even if the conversation phrased it as
one.

Prefer changing what exists over adding:
- update when a new statement refines exactly one existing memory;
- fuse when two or more existing memories say one thing, and write the merged
  statement;
- add only when nothing existing covers it.
The memories shown span this and earlier conversations alike; the owner's memory is
one profile. Update or fuse any of them by the numbers they are listed with, whatever
conversation wrote them, and a fuse retires every source it names.

One write holds one statement: split what a turn said about several things into
several writes, and never merge unrelated statements into one.

No change is not a write. A statement that repeats an existing memory in other
words, or says less than one, returns nothing for it; update only when the new
statement adds information the existing one does not already say, and a
contradiction replaces it the same way.

When memories conflict, the more recently written one is likelier current; and never
put a date or time in a memory's body, since the store keeps the time.

This pass may add at most {limit} records. Updating or merging existing records
never counts against it.

Give a write a quote when the owner said it: a short excerpt of what the owner said, at most
{quote_limit} characters, copied exactly from a [user] turn of the newest part (never from the
earlier turns, never from the Agent). Copy the shortest stretch that carries the point, and
leave out material the owner pasted in and any profanity around it. Leave quote out when the
owner did not say it, as with an experience the Agent learned on its own.

Answer with one JSON object and nothing else, in exactly this shape:
{{"writes": [{{"action": "add", "kind": "preference", "body": "The owner prefers ...", "targets": [], "quote": "the owner's own words"}}]}}

- action is "add", "update" or "fuse"; targets holds the numbers the memories are listed
  with: an add leaves it empty, an update puts in it the one memory it refines, a fuse
  the two or more it merges.
- kind is one of "preference", "fact", "experience".
- body is one third-person statement.
- With nothing to remember, answer {{"writes": []}}.

Examples.

Conversation: "以后图表都用深色主题，结论短一点。"
Existing: (none)
Answer: {{"writes": [{{"action": "add", "kind": "preference", "body": "The owner prefers charts in a dark theme.", "targets": []}}, {{"action": "add", "kind": "preference", "body": "The owner prefers short conclusions.", "targets": []}}]}}

Conversation: "结论再短一点。"
Existing: [1] (preference, 2026-03-02 10:00) The owner prefers short conclusions.
Answer: {{"writes": []}}

Conversation: "帮我把这个 PDF 总结一下。"
Existing: (none)
Answer: {{"writes": []}}

Conversation: "这次先别跑测试，直接部署。"
Existing: (none)
Answer: {{"writes": []}}
"""


@dataclass(frozen=True, slots=True)
class ConsolidationRequest:
    """One model request, and the numbers its memories were listed with.

    The model never sees a memory's id: the memories shown are numbered from 1 and the
    verdict names them by number. ``handles`` maps each number back to its memory id
    for exactly this request, so it is read once, by ``plan_consolidation``, and never
    stored: a stored plan holds the real ids.
    """

    messages: list[dict[str, str]]
    handles: Mapping[int, str]
    #: What the owner said in the part being judged, one entry per [user] turn: the only text a
    #: quote may be a stretch of. Like ``handles``, it is read by ``plan_consolidation`` and not stored.
    owner_words: tuple[str, ...]


_CONTEXT_HEADING = (
    "Earlier in this conversation (context only — remember nothing from it unless the "
    "newest part restates or relies on it):"
)
_NEWEST_HEADING = "Newest part of this conversation — the part to judge (oldest first):"


def consolidation_request(
    *,
    turns: Sequence[ConversationTurn],
    candidates: Sequence[MemoryRecord],
    session_records: Sequence[MemoryRecord],
    record_limit: int,
    context_turns: Sequence[ConversationTurn] = (),
) -> ConsolidationRequest:
    """The model request for one pass: its rules, then the material and the profile.

    ``turns`` are the newest part of the conversation, the part to judge.
    ``context_turns`` are the turns just before it, oldest first (the last ``CONTEXT_ROUNDS``
    rounds, which the host fetches), shown ahead of it as context only: a user turn
    whole, an assistant turn cut to its first ``CONTEXT_ASSISTANT_CHARS`` characters under
    a trailing ellipsis. That cut lives here and only here; with no context turns, the
    section is omitted.

    The memories are numbered continuously through both lists, in the order shown; a
    record in both is listed once, with the conversation's own.
    """
    contributed = {record.memory_id for record in session_records}
    others = [record for record in candidates if record.memory_id not in contributed]
    handles = {
        number: record.memory_id
        for number, record in enumerate((*others, *session_records), start=1)
    }
    lines: list[str] = []
    if context_turns:
        lines.append(_CONTEXT_HEADING)
        lines.extend(_context_line(turn) for turn in context_turns)
        lines.append("")
    lines.append(_NEWEST_HEADING)
    lines.extend(f"[{turn.role}] {turn.text}" for turn in turns)
    lines.append("")
    lines.append(f"Existing memories that may bear on it ({RECALL_TIME_NOTE}):")
    lines.extend(_listed(others, first=1))
    lines.append("")
    lines.append(f"Memories this conversation already contributed ({RECALL_TIME_NOTE}):")
    lines.extend(_listed(session_records, first=len(others) + 1))
    return ConsolidationRequest(
        messages=[
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(
                    limit=record_limit, quote_limit=MEMORY_QUOTE_LIMIT
                ),
            },
            {"role": "user", "content": "\n".join(lines)},
        ],
        handles=handles,
        owner_words=tuple(turn.text for turn in turns if turn.role == "user"),
    )


def _context_line(turn: ConversationTurn) -> str:
    shown = turn.text
    if turn.role == "assistant" and len(shown) > CONTEXT_ASSISTANT_CHARS:
        shown = shown[:CONTEXT_ASSISTANT_CHARS] + "…"
    return f"[{turn.role}] {shown}"


def _listed(records: Sequence[MemoryRecord], *, first: int) -> list[str]:
    """The memories as numbered lines, each followed by the owner's words behind it, if it has any."""
    lines: list[str] = []
    for number, record in enumerate(records, start=first):
        lines.append(f"[{number}] {recall_label(record)} {record.body}")
        quote = recall_quote_line(record)
        if quote is not None:
            lines.append(quote)
    return lines or ["(none)"]


def verified_quote(quote: str | None, owner_words: Sequence[str]) -> str | None:
    """The quote as the host keeps it, or None when it cannot stand behind a memory.

    Whitespace is squeezed to single spaces on both sides and nothing else is forgiven: the quote
    must be, character for character, a stretch of one of the owner's turns (``owner_words``,
    the [user] turns of the part being judged: never a turn before it, never the Agent's), and
    pass the store's own check of what a quote may be. A model told to copy cannot be taken at
    its word, and a quote of words the owner never said would be shown to every later Run as theirs.
    """
    if quote is None:
        return None
    candidate = single_spaced(quote)
    if not is_storable_quote(candidate):
        return None
    if not any(candidate in single_spaced(words) for words in owner_words):
        return None
    return candidate


def plan_consolidation(
    verdict: ConsolidationVerdict,
    *,
    handles: Mapping[int, str],
    record_limit: int,
    owner_words: Sequence[str],
) -> tuple[PlannedWrite, ...]:
    """The writes a verdict becomes, in its order, within what a pass may do.

    ``handles`` are the numbers the model was shown, mapped to the memory ids the
    plan keeps. A write is left out when it names a number the model was not shown,
    names a record an earlier write already replaces, has the wrong number of targets
    for its action, or is an add past the ``record_limit`` adds this verdict may
    hold. The bound lives here and only here: it is per verdict, one model call and so
    one batch of a pass, and counts adds, because memory is the owner's across
    conversations and replacing or merging records the owner already holds, whichever
    conversation wrote them, is maintenance rather than contribution. A fuse retires
    every source it names.

    A write keeps its quote only if ``verified_quote`` accepts it against ``owner_words``; one
    it refuses costs the write its quote and nothing more.
    """
    adds = 0
    claimed: set[str] = set()
    planned: list[PlannedWrite] = []
    for write in verdict.writes:
        if any(number not in handles for number in write.targets):
            continue
        targets = tuple(dict.fromkeys(handles[number] for number in write.targets))
        if not _well_targeted(write.action, targets):
            continue
        if any(target in claimed for target in targets):
            continue
        if write.action == "add":
            if adds + 1 > record_limit:
                continue
            adds += 1
        claimed.update(targets)
        planned.append(
            PlannedWrite(
                action=write.action,
                kind=write.kind,
                body=write.body,
                target_ids=targets,
                quote=verified_quote(write.quote, owner_words),
            )
        )
    return tuple(planned)


def _well_targeted(action: VerdictAction, targets: tuple[str, ...]) -> bool:
    if action == "add":
        return not targets
    if action == "update":
        return len(targets) == 1
    return MEMORY_FUSE_MIN_SOURCES <= len(targets) <= MEMORY_FUSE_MAX_SOURCES


def planned_operation(
    write: PlannedWrite,
    *,
    index: int,
    owner_id: str,
    provenance: MemoryProvenance,
    pass_key: str,
) -> MemoryOperation:
    """One planned write as an idempotent operation, keyed by the pass and its position."""
    common: dict[str, Any] = {
        "owner_id": owner_id,
        "idempotency_key": f"{pass_key}:{index}",
        "provenance": provenance,
        "kind": write.kind,
        "body": write.body,
        "mutation_scope": pass_key,
        "mutation_limit": PASS_MUTATION_LIMIT,
        "quote": write.quote,
    }
    if write.action == "fuse":
        return MemoryOperation(action="fuse", source_ids=write.target_ids, **common)
    supersedes_id = write.target_ids[0] if write.action == "update" else None
    return MemoryOperation(action="remember", supersedes_id=supersedes_id, **common)


__all__ = [
    "CONTEXT_ASSISTANT_CHARS",
    "CONTEXT_ROUNDS",
    "PASS_MUTATION_LIMIT",
    "ConsolidationRequest",
    "ConsolidationVerdict",
    "ConversationTurn",
    "PlannedWrite",
    "VerdictWrite",
    "consolidation_request",
    "plan_consolidation",
    "planned_operation",
    "verified_quote",
]
