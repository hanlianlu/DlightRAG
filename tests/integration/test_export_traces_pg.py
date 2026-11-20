# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The trace export reads only what owners contributed (ADR 0041, Conversation Contribution).

The export is where conversations leave the product for training, so these tests drive it as an
operator does, against a scratch database holding Runs as the product leaves them, and read what
it writes. They hold it to the promise Settings makes: with the switch off nothing is read, with it
on only the Sessions first accepted after it was turned on are, and nothing an operator passes
widens that. A Session is judged by the first Run it accepted, so a conversation begun before the
permission stays out however it is continued, forked or followed up. And a Run's record holds the
entries that Run wrote, whatever else its Session holds: a fork shares its parent's Session, and
deleting a conversation deletes its Runs but not the entries they wrote.

The dataset (ADR 0044) is read the way a trainer would find it: `main()` run as `make
export-traces ARGS="--dataset DIR"` runs it, against a `.env` in a scratch repository root, and the
files it leaves are parsed back. What such a dataset must hold is stated by the design, so the
expectations here are written from it (the HMAC of an id, the hash a Session's side comes from)
and not taken from the exporter's own functions.

Every test owns a scratch database; skipped without PostgreSQL.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import hashlib
import hmac
import itertools
import json
import os
import re
import stat
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from scripts import export_traces
from scripts.export_traces import as_jsonl, export
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.support.pg import PG_CONN_KWARGS, delete_runs, skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

MILLISECOND = datetime.timedelta(milliseconds=1)
SECOND = datetime.timedelta(seconds=1)
MINUTE = datetime.timedelta(minutes=1)
HOUR = datetime.timedelta(hours=1)

#: What the answer of a Run cites: the sources and the passages cited, as the product stores them.
SOURCES = [
    {
        "id": "1",
        "title": "A paper",
        "source_uri": "file:///a.pdf",
        "cited_chunk_ids": ["c1"],
        "chunks": [{"chunk_id": "c1", "content": "the-cited-passage"}],
    }
]

#: An owner as the product names a person: a SHA-256 digest of the issuer and the user.
DAVE = hashlib.sha256(b"jwt\0https://issuer.example\0dave@example.com").hexdigest()

#: The operator's key: at least 32 characters, and a string no scenario text can contain.
KEY = "key-for-the-pseudonyms-of-this-test-" + "0123456789abcdef" * 2

_INSERT_RUN = """
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind, submission_key,
    accepted_input_json, prepared_input_json, request_fingerprint, status, result_json,
    error_kind, error_message, retention_seconds, created_at, started_at, finished_at
)
VALUES ($1, $2, $3, 'query', $1, 'owner', $4, $5::jsonb, $8::jsonb, $4, $7, $9::jsonb,
        CASE WHEN $7 = 'failed' THEN 'run_execution_failed' END,
        CASE WHEN $7 = 'failed' THEN $10::text END,
        31536000, $6::timestamptz, $6::timestamptz,
        CASE WHEN $7 IN ('succeeded', 'failed', 'cancelled')
             THEN $6::timestamptz + interval '2 seconds' END)
"""

_INSERT_ROUTING = """
INSERT INTO dlightrag_answer_run_routing (
    owner_id, run_id, requested_mode, valid_modes, resolved_mode, context_policy_revision,
    agent_session_id, agent_lane_id, source_lane_id
)
VALUES ($1, $2, 'auto', ARRAY['fast', 'research'], $6, 'test', $3, $4, $5)
"""

_INSERT_SESSION = """
INSERT INTO dlightrag_agent_sessions (owner_id, session_id, lease_run_id, commit_sequence,
                                        fencing_epoch, created_at)
VALUES ($1, $2, $3, 0, 1, $4)
ON CONFLICT DO NOTHING
"""

_INSERT_ENTRY = """
INSERT INTO dlightrag_agent_session_entries (
    owner_id, session_id, sequence, entry_id, parent_entry_id, entry_type, schema_version,
    timestamp, payload_json, run_id
)
SELECT $1, $2, coalesce(max(sequence), 0) + 1, $3, $8, $7, 2, $4, $5::jsonb, $6
FROM dlightrag_agent_session_entries
WHERE owner_id = $1 AND session_id = $2
"""

# A note outlives the Run that wrote it: it hangs off its Session, and names its writer.
_INSERT_NOTE = """
INSERT INTO dlightrag_answer_session_notes (
    owner_id, session_id, relative_path, size_bytes, content_digest, content, revision,
    written_by_run_id, updated_at
)
VALUES ($1, $2, $3, octet_length(convert_to($4, 'UTF8')), repeat('2', 64),
        convert_to($4, 'UTF8'), 1, $5, $6)
"""

_INSERT_CHILD = """
INSERT INTO dlightrag_answer_child_sessions (
    owner_id, run_id, child_session_id, parent_session_id, parent_call_id, status, depth,
    objective, context_snapshot_json, created_at, updated_at, parent_intent_id
)
VALUES ($1, $2, $3, $4, $7, $8, 1, $5, '{}'::jsonb, $6, $9, $10)
"""

# The events a tool call leaves in its Run's stream, as the Runtime writes them. The trigger that
# lets only a live Run's writer add one is lifted here, because the Run is over by the time a
# scenario is written: what is under test is what the export makes of what that writer left.
_INSERT_EVENT = """
INSERT INTO dlightrag_run_events (owner_id, run_id, event_sequence, event_type, payload, created_at)
SELECT $1, $2, coalesce(max(event_sequence), 0) + 1, $3, $4::jsonb, $5
FROM dlightrag_run_events
WHERE owner_id = $1 AND run_id = $2
"""

_INSERT_EVIDENCE = """
INSERT INTO dlightrag_answer_evidence (
    owner_id, run_id, session_id, intent_id, result_ordinal, content_digest, locator_digest,
    content, locator
)
VALUES ($1, $2, $3, $4, 0, repeat('0', 64), repeat('1', 64),
        convert_to($5, 'UTF8'), convert_to('{}', 'UTF8'))
"""


class Written:
    """A dataset directory as the operator finds it, and as a trainer would read it."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def text(self, name: str) -> str:
        return (self.directory / name).read_text(encoding="utf-8")

    def lines(self, name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.text(name).splitlines()]

    @property
    def train(self) -> list[dict[str, Any]]:
        return self.lines("train.jsonl")

    @property
    def held_out(self) -> list[dict[str, Any]]:
        return self.lines("eval.jsonl")

    @property
    def records(self) -> list[dict[str, Any]]:
        return [*self.train, *self.held_out]

    @property
    def sheet(self) -> dict[str, Any]:
        return json.loads(self.text("datasheet.json"))

    @property
    def everything(self) -> str:
        """Every word the directory holds."""
        return "".join(self.text(name) for name in export_traces.DATASET_FILES)


def pseudonym(raw: object) -> str:
    """What the design says an id becomes: the HMAC-SHA256 of its stored text under the key."""
    return hmac.new(KEY.encode(), str(raw).encode(), hashlib.sha256).hexdigest()


def skill_error(owner: str) -> str:
    """What a Skill publication that fails writes into its tool result: the operating system's
    error, which names the path of the owner's skill directory."""
    return (
        "Skill publication failed: [Errno 28] No space left on device: "
        f"'/srv/dlightrag/skills/9f/{owner}/notes/SKILL.md'"
    )


def entries_a_run_can_reach(
    records: list[dict[str, Any]], record: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """The entries a consumer can reach for a Run from the files alone: the steps of the records
    of its Session, and the Run's own `carried`."""
    reach = {
        step["entry_id"]: step
        for other in records
        if other["session"]["id"] == record["session"]["id"]
        for step in other["steps"]
    }
    return reach | {step["entry_id"]: step for step in record["session"]["carried"]}


def rebuilt_chain(records: list[dict[str, Any]], record: dict[str, Any]) -> list[str]:
    """The context a Run ran on, root first, rebuilt as a consumer does: follow `parent_entry_id`
    from the Run's first step up to the root. A parent no file holds fails the walk."""
    reach = entries_a_run_can_reach(records, record)
    chain: list[str] = []
    cursor = record["steps"][0]["parent_entry_id"] if record["steps"] else None
    while cursor is not None:
        assert cursor in reach, "a parent entry that no file holds"
        chain.append(cursor)
        cursor = reach[cursor]["parent_entry_id"]
    return chain[::-1]


def bucket(session: uuid.UUID) -> float:
    """What the design says decides a Session's side: u, from a hash of its stored id alone."""
    digest = hashlib.sha256(b"dlightrag-split-v1\0" + str(session).encode()).hexdigest()
    return int(digest[:16], 16) / 2**64


def side_of(session: uuid.UUID, eval_fraction: float) -> str:
    return "eval" if bucket(session) < eval_fraction else "train"


def session_where(low: float, high: float, name: str) -> uuid.UUID:
    """A Session id whose u is in [low, high), found by trying names."""
    for number in itertools.count():
        session = uuid.uuid5(uuid.NAMESPACE_URL, f"export-split/{name}/{number}")
        if low <= bucket(session) < high:
            return session
    raise AssertionError("unreachable")


class Scratch:
    """A scratch deployment: the store that keeps consent, and Runs written as the product writes."""

    def __init__(self, store: PGRunStore, pool: Any, settings: dict[str, Any], root: Path) -> None:
        self.store, self.pool, self.settings, self.root = store, pool, settings, root
        self.markers: dict[str, str] = {}
        # The entries each Run wrote into its own Session, oldest first.
        self.entries: dict[str, list[str]] = {}
        self._events_free = False
        self.configure()

    def configure(self, *, key: str | None = KEY, port: int | None = None) -> None:
        """The repository's `.env` as an operator keeps it: the database to read, and the key."""
        values = {
            f"DLIGHTRAG_STORAGE__POSTGRES__{name.upper()}": str(value)
            for name, value in self.settings.items()
        }
        if port is not None:
            values["DLIGHTRAG_STORAGE__POSTGRES__PORT"] = str(port)
        if key is not None:
            values[export_traces.KEY_NAME] = key
        (self.root / ".env").write_text(
            "".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8"
        )

    async def dataset(self, directory: Path, *options: str) -> int:
        """`make export-traces ARGS="--dataset DIRECTORY ..."`: the tool's own entry point, which
        reads the `.env` above, run off the event loop as a command runs outside it. The exit
        status is what a shell would see, an option the command refuses included."""

        def run() -> int:
            try:
                return export_traces.main(["--dataset", str(directory), *options])
            except SystemExit as stopped:
                return int(stopped.code or 0)

        return await asyncio.to_thread(run)

    async def turn(self, owner: str, contribute: bool) -> datetime.datetime | None:
        """The owner's switch, through the product's own write; the moment it began, if on."""
        kept = await self.store.write_privacy_settings(owner_id=owner, contribute=contribute)
        return kept.since

    async def accept(
        self,
        owner: str,
        *,
        at: datetime.datetime,
        session: uuid.UUID,
        text: str,
        lane: str = "main",
        source_lane: str | None = None,
        kind: str = "answer",
        status: str = "succeeded",
        after: str | None = None,
        accepted: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        mode: str | None = None,
        evidence: str | None = None,
        error_message: str | None = None,
        acceptance: bool = False,
    ) -> str:
        """One Run accepted at ``at``, with the words ``text`` in everything it holds.

        An Answer Run is routed to ``session`` and leaves a message and Evidence there, the
        message following the entry ``after`` as a turn follows the one before it in its lane. A
        Retrieval Run has no Session: it is not a conversation. A Run that is still going holds
        the prepared input the product keeps for it, execution secrets included, until it ends.
        """
        run_id = uuid.uuid4()
        self.markers[str(run_id)] = text
        going = status in ("queued", "running")
        prepared = {"query": text, "resource_identity": f"secret-{run_id}"} if going else None
        stored = result if result is not None else {"answer": "done"}
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(
                _INSERT_RUN,
                owner,
                run_id,
                kind,
                f"key-{run_id}",
                json.dumps({"query": text, **(accepted or {})}),
                at,
                status,
                None if prepared is None else json.dumps(prepared),
                json.dumps(stored) if status == "succeeded" else None,
                error_message,
            )
            if kind != "answer":
                return str(run_id)
            await conn.execute(_INSERT_ROUTING, owner, run_id, session, lane, source_lane, mode)
            # The row is written when the Run first commits to its Session, a moment after the
            # Run was accepted, and the Session keeps that time as later Runs join it.
            await conn.execute(
                _INSERT_SESSION, owner, session, run_id, at + datetime.timedelta(seconds=1)
            )
            entry_id = uuid.uuid4()
            await conn.execute(
                _INSERT_ENTRY,
                owner,
                session,
                entry_id,
                at + datetime.timedelta(seconds=1),
                json.dumps(
                    {"content": text, **({"acceptance_id": str(run_id)} if acceptance else {})}
                ),
                run_id,
                "user_message",
                uuid.UUID(after) if after else None,
            )
            self.entries[str(run_id)] = [str(entry_id)]
            await conn.execute(
                _INSERT_EVIDENCE, owner, run_id, session, uuid.uuid4(), evidence or text
            )
        return str(run_id)

    async def entry(
        self,
        owner: str,
        session: uuid.UUID | str,
        run_id: str,
        *,
        at: datetime.datetime,
        kind: str,
        payload: dict[str, Any],
        after: str | None,
    ) -> str:
        """One entry a Run committed to a Session, under the entry ``after`` it follows."""
        entry_id = uuid.uuid4()
        await self.pool.execute(
            _INSERT_ENTRY,
            owner,
            uuid.UUID(str(session)),
            entry_id,
            at,
            json.dumps(payload),
            uuid.UUID(run_id),
            kind,
            uuid.UUID(after) if after else None,
        )
        return str(entry_id)

    async def chat(
        self,
        owner: str,
        session: uuid.UUID,
        run_id: str,
        *,
        at: datetime.datetime,
        text: str,
        kind: str = "assistant_message",
        payload: dict[str, Any] | None = None,
    ) -> str:
        """One more message committed to a Session by a Run that is still going, under the one
        before it."""
        entry_id = await self.entry(
            owner,
            session,
            run_id,
            at=at,
            kind=kind,
            payload=payload if payload is not None else {"content": text},
            after=self.entries[run_id][-1],
        )
        self.entries[run_id].append(entry_id)
        return entry_id

    async def head(self, run_id: str) -> None:
        """Record where the Run ended, which a fork of it starts from."""
        await self.pool.execute(
            "UPDATE dlightrag_answer_run_routing SET fork_point_entry_id = $2 WHERE run_id = $1",
            uuid.UUID(run_id),
            self.entries[run_id][-1],
        )

    async def events(
        self, owner: str, run_id: str, *events: tuple[str, str, datetime.datetime]
    ) -> None:
        """The events of a Run's stream, `(type, call id or text, when)`: the start and end of a
        tool call as the Runtime writes them, and the text it streamed."""
        if not self._events_free:
            await self.pool.execute(
                "ALTER TABLE dlightrag_run_events DISABLE TRIGGER trg_dlightrag_run_events_enforce"
            )
            self._events_free = True
        for kind, call, when in events:
            payload: dict[str, Any] = {"text": call}
            if kind in ("tool_start", "tool_end"):
                payload = {"tool_name": "bash", "call_id": call, "source_index": 0}
                payload |= (
                    {"attempt_id": str(uuid.uuid4())}
                    if kind == "tool_start"
                    else {"outcome": "succeeded"}
                )
            await self.pool.execute(
                _INSERT_EVENT, owner, uuid.UUID(run_id), kind, json.dumps(payload), when
            )

    async def note(
        self,
        owner: str,
        session: uuid.UUID,
        run_id: str,
        *,
        path: str,
        at: datetime.datetime,
        text: str,
    ) -> None:
        """A note the Run wrote into its Session's memory."""
        await self.pool.execute(_INSERT_NOTE, owner, session, path, text, uuid.UUID(run_id), at)

    async def child(
        self,
        owner: str,
        session: uuid.UUID,
        run_id: str,
        *,
        at: datetime.datetime,
        text: str,
        status: str = "succeeded",
        call: str = "call-1",
        updated: datetime.datetime | None = None,
        wrote: bool = True,
        also: tuple[tuple[str, dict[str, Any], datetime.datetime], ...] = (),
        child_session: uuid.UUID | None = None,
        intent: uuid.UUID | None = None,
    ) -> tuple[str, str | None]:
        """A Child the Run spawned and the message its own Session holds, stamped with the Run as
        the product stamps a Child's, and the entries that follow it, `(type, payload, when)`.

        Returns the Child's Session id and its first message's id, which is None for a Child that
        wrote nothing. A Child's row last changed when it ended, ``updated``, and names the
        ``intent`` of the tool call that spawned it.
        """
        child_session = child_session or uuid.uuid4()
        written: list[str] = []
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute(_INSERT_SESSION, owner, child_session, uuid.UUID(run_id), at)
            await conn.execute(
                _INSERT_CHILD,
                owner,
                uuid.UUID(run_id),
                child_session,
                session,
                text,
                at,
                call,
                status,
                updated or at,
                intent,
            )
        if wrote:
            written.append(
                await self.entry(
                    owner,
                    child_session,
                    run_id,
                    at=at,
                    kind="assistant_message",
                    payload={"content": text},
                    after=None,
                )
            )
        for kind, payload, when in also:
            written.append(
                await self.entry(
                    owner,
                    child_session,
                    run_id,
                    at=when,
                    kind=kind,
                    payload=payload,
                    after=written[-1] if written else None,
                )
            )
        return str(child_session), (written[0] if wrote else None)

    async def exported(self, **narrowing: Any) -> list[dict[str, Any]]:
        """What the export writes for these options, as the operator would run it."""
        options: dict[str, Any] = {
            "owner": None,
            "since": None,
            "until": None,
            "run_ids": None,
            "limit": 1000,
            "dedupe": False,
        }
        return (await export(self.settings, **{**options, **narrowing})).records

    async def run_ids(self, **narrowing: Any) -> set[str]:
        return {record["run"]["run_id"] for record in await self.exported(**narrowing)}

    def words_in(self, written: list[dict[str, Any]] | str) -> set[str]:
        """Which Runs' words appear anywhere in what was written, entries and Evidence included."""
        text = written if isinstance(written, str) else as_jsonl(written)
        return {run_id for run_id, words in self.markers.items() if words in text}


@pytest.fixture
async def scratch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> AsyncIterator[Scratch]:
    await skip_without_postgres()
    for name in [name for name in os.environ if name.startswith("DLIGHTRAG_")]:
        monkeypatch.delenv(name)
    async with isolated_run_runtime("export_traces") as (store, pool):
        database = await pool.fetchval("SELECT current_database()")
        root = tmp_path / "repository"
        root.mkdir()
        # The tool reads its `.env` from the repository root; here that is a directory of its own.
        monkeypatch.setattr(export_traces, "ROOT", root)
        yield Scratch(store, pool, {**PG_CONN_KWARGS, "database": database}, root)


class Interposed:
    """The exporter's connection, letting the world move on once the Runs have been chosen."""

    def __init__(self, real: Any, after_the_runs_are_chosen: Any) -> None:
        self._real = real
        self._after = after_the_runs_are_chosen
        self._fetches = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    async def fetch(self, *args: Any, **kwargs: Any) -> Any:
        rows = await self._real.fetch(*args, **kwargs)
        self._fetches += 1
        if self._fetches == 1:
            await self._after()
        return rows


async def test_an_owner_who_never_chose_has_nothing_exported(scratch: Scratch) -> None:
    await scratch.accept(
        "alice", at=datetime.datetime.now(datetime.UTC), session=uuid.uuid4(), text="never-chose"
    )

    assert await scratch.exported() == []


async def test_an_export_of_nothing_prints_its_summary_instead_of_crashing(
    scratch: Scratch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing exported is the usual case until someone opts in, and the command still reports it."""
    await scratch.accept(
        "alice", at=datetime.datetime.now(datetime.UTC), session=uuid.uuid4(), text="never-chose"
    )

    assert await asyncio.to_thread(export_traces.main, []) == 0

    written = capsys.readouterr()
    assert written.out == ""
    assert "0 run(s), 0 step(s)" in written.err


async def test_only_sessions_first_accepted_since_the_owner_contributed_are_exported(
    scratch: Scratch,
) -> None:
    since = await scratch.turn("alice", True)
    assert since is not None
    begun_before, begun_a_moment_before = uuid.uuid4(), uuid.uuid4()
    begun_after, begun_at_the_moment = uuid.uuid4(), uuid.uuid4()

    # A conversation begun before the permission, then continued, followed up and forked after it.
    # Each of those Runs was accepted after the moment, and each is in the earlier Session.
    await scratch.accept("alice", at=since - HOUR, session=begun_before, text="early-first")
    await scratch.accept("alice", at=since + HOUR, session=begun_before, text="early-continued")
    await scratch.accept("alice", at=since + 2 * HOUR, session=begun_before, text="early-followup")
    await scratch.accept(
        "alice",
        at=since + 3 * HOUR,
        session=begun_before,
        lane="fork",
        source_lane="main",
        text="early-fork",
    )
    # A Session begun a microsecond before the moment is still before it.
    just_before = datetime.timedelta(microseconds=1)
    await scratch.accept(
        "alice", at=since - just_before, session=begun_a_moment_before, text="just-before-first"
    )
    await scratch.accept(
        "alice", at=since + HOUR, session=begun_a_moment_before, text="just-before-later"
    )
    # A Session begun after the permission is exported whole, its follow-up and its fork with it.
    kept = {
        await scratch.accept("alice", at=since + 4 * HOUR, session=begun_after, text="new-first"),
        await scratch.accept(
            "alice", at=since + 5 * HOUR, session=begun_after, text="new-followup"
        ),
        await scratch.accept(
            "alice",
            at=since + 6 * HOUR,
            session=begun_after,
            lane="fork",
            source_lane="main",
            text="new-fork",
        ),
        # And one begun at the very moment is after it.
        await scratch.accept("alice", at=since, session=begun_at_the_moment, text="at-the-moment"),
    }
    # A Retrieval Run is no conversation: it has no Session to have been begun after anything.
    await scratch.accept(
        "alice",
        at=since + 7 * HOUR,
        session=uuid.uuid4(),
        text="retrieval-no-session",
        kind="retrieval",
    )

    records = await scratch.exported()

    assert {record["run"]["run_id"] for record in records} == kept
    # What an excluded Run said is nowhere in what was written: not its input, not its message in
    # its Session, not its Evidence.
    assert scratch.words_in(records) == kept


async def test_the_words_of_a_fork_the_owner_deleted_are_in_no_record(scratch: Scratch) -> None:
    """A fork shares its parent's Session, and deleting it deletes its Runs and not the entries
    they wrote: those stay in the Session for as long as another Run routes there, and so does a
    note a deleted Run wrote. They are words the owner deleted, they sit between and after the
    surviving Runs' own, and no record may carry them."""
    since = await scratch.turn("alice", True)
    assert since is not None
    session, start = uuid.uuid4(), since + HOUR

    first = await scratch.accept("alice", at=start, session=session, text="run-a")
    await scratch.chat("alice", session, first, at=start + 2 * SECOND, text="run-a answer")
    await scratch.note(
        "alice", session, first, path="a.md", at=start + 3 * SECOND, text="run-a note"
    )
    child, child_entry = await scratch.child(
        "alice", session, first, at=start + 4 * SECOND, text="run-a child"
    )
    # Two forks open lanes of the same Session: one between the survivors, one after the last. Each
    # branches from the turn it forks, and the main lane goes on from the first without them.
    fork_between = await scratch.accept(
        "alice",
        at=start + 5 * MINUTE,
        session=session,
        lane="lane-fork-1",
        source_lane="main",
        text="fork-b",
        after=scratch.entries[first][-1],
    )
    await scratch.chat(
        "alice", session, fork_between, at=start + 5 * MINUTE + 2 * SECOND, text="fork-b answer"
    )
    await scratch.note(
        "alice",
        session,
        fork_between,
        path="b.md",
        at=start + 5 * MINUTE + 3 * SECOND,
        text="fork-b note",
    )
    last = await scratch.accept(
        "alice",
        at=start + 10 * MINUTE,
        session=session,
        text="run-c",
        after=scratch.entries[first][-1],
    )
    await scratch.chat(
        "alice", session, last, at=start + 10 * MINUTE + 2 * SECOND, text="run-c answer"
    )
    await scratch.note(
        "alice", session, last, path="c.md", at=start + 10 * MINUTE + 3 * SECOND, text="run-c note"
    )
    fork_after = await scratch.accept(
        "alice",
        at=start + 15 * MINUTE,
        session=session,
        lane="lane-fork-2",
        source_lane="main",
        text="fork-e",
        after=scratch.entries[last][-1],
    )
    await scratch.chat(
        "alice", session, fork_after, at=start + 15 * MINUTE + 2 * SECOND, text="fork-e answer"
    )
    await scratch.note(
        "alice",
        session,
        fork_after,
        path="e.md",
        at=start + 15 * MINUTE + 3 * SECOND,
        text="fork-e note",
    )

    gone = [uuid.UUID(fork_between), uuid.UUID(fork_after)]
    await delete_runs(
        scratch.pool, scratch.store, owner_id="alice", run_ids=[fork_between, fork_after]
    )

    # What the product leaves: the forks' Runs are gone, and the words they wrote are not.
    assert not await scratch.pool.fetchval(
        "SELECT count(*) FROM dlightrag_runs WHERE run_id = ANY($1::uuid[])", gone
    )
    assert (
        await scratch.pool.fetchval(
            "SELECT count(*) FROM dlightrag_agent_session_entries WHERE run_id = ANY($1::uuid[])",
            gone,
        )
        == 4
    )
    assert (
        await scratch.pool.fetchval(
            "SELECT count(*) FROM dlightrag_answer_session_notes"
            " WHERE written_by_run_id = ANY($1::uuid[])",
            gone,
        )
        == 2
    )

    records = {record["run"]["run_id"]: record for record in await scratch.exported()}

    assert set(records) == {first, last}
    assert scratch.words_in(list(records.values())) == {first, last}
    # Each survivor holds exactly the entries it wrote, in its own Session and in its Child's.
    for run_id, record in records.items():
        assert [step["entry_id"] for step in record["steps"]] == scratch.entries[run_id]
    assert [note["path"] for note in records[first]["session_notes"]] == ["a.md"]
    assert [note["path"] for note in records[last]["session_notes"]] == ["c.md"]
    (kept_child,) = records[first]["children"]
    assert kept_child["child_session_id"] == child
    assert [step["entry_id"] for step in kept_child["steps"]] == [child_entry]
    assert records[last]["children"] == []
    # The last Run ran on the first's entries, which are exported with the first, so none is carried.
    assert rebuilt_chain(list(records.values()), records[last]) == scratch.entries[first]
    assert records[last]["session"]["carried"] == []

    # The dataset holds the same two Runs, and not a word of either fork in any file it wrote.
    directory = scratch.root.parent / "dataset"
    assert await scratch.dataset(directory) == 0
    written = Written(directory)
    assert {record["run"]["run_id"] for record in written.records} == {
        pseudonym(first),
        pseudonym(last),
    }
    assert scratch.words_in(written.everything) == {first, last}
    (record_of_last,) = (r for r in written.records if r["run"]["run_id"] == pseudonym(last))
    assert rebuilt_chain(written.records, record_of_last) == [
        pseudonym(entry) for entry in scratch.entries[first]
    ]
    assert record_of_last["session"]["carried"] == []


async def test_runs_that_write_into_one_session_by_turns_export_exactly_their_own_entries(
    scratch: Scratch,
) -> None:
    """The Runs of a Session do not take turns neatly. A Run reclaimed after its lease lapsed writes
    after the next Run of its Session has begun, so the Session's entries alternate between them."""
    since = await scratch.turn("alice", True)
    assert since is not None
    session, start = uuid.uuid4(), since + HOUR
    resumed = await scratch.accept("alice", at=start, session=session, text="run-x")
    other = await scratch.accept(
        "alice",
        at=start + MINUTE,
        session=session,
        lane="lane-fork",
        source_lane="main",
        text="run-y",
    )
    await scratch.chat("alice", session, resumed, at=start + 90 * SECOND, text="run-x resumed")
    await scratch.chat("alice", session, other, at=start + 2 * MINUTE, text="run-y answer")
    await scratch.chat("alice", session, resumed, at=start + 150 * SECOND, text="run-x answer")

    records = {record["run"]["run_id"]: record for record in await scratch.exported()}

    assert {
        run_id: [step["entry_id"] for step in record["steps"]] for run_id, record in records.items()
    } == scratch.entries
    # Asking for one Run narrows the export and never widens its record.
    (alone,) = await scratch.exported(run_ids=[resumed])
    assert [step["entry_id"] for step in alone["steps"]] == scratch.entries[resumed]


async def test_a_runs_steps_stay_in_the_order_of_their_sequence(scratch: Scratch) -> None:
    """A message carries the clock of its writer, and a Run reclaimed on a host whose clock is behind
    writes times that step back. Sequence is the Session's own order, so it is what orders steps."""
    since = await scratch.turn("alice", True)
    assert since is not None
    session, start = uuid.uuid4(), since + HOUR
    run = await scratch.accept("alice", at=start, session=session, text="run-clock")
    await scratch.chat("alice", session, run, at=start + 5 * SECOND, text="run-clock later")
    await scratch.chat("alice", session, run, at=start + 3 * SECOND, text="run-clock stepped back")
    await scratch.chat("alice", session, run, at=start + 4 * SECOND, text="run-clock last")

    (record,) = await scratch.exported()

    assert [step["sequence"] for step in record["steps"]] == [1, 2, 3, 4]
    assert [step["entry_id"] for step in record["steps"]] == scratch.entries[run]


async def test_a_session_begun_before_the_grant_stays_out_when_retention_prunes_its_oldest_runs(
    scratch: Scratch,
) -> None:
    """Retention deletes a long conversation's oldest Runs one by one while the Session lives on in
    its newer ones, so the oldest Run left is no guide to when the Session began."""
    since = await scratch.turn("alice", True)
    assert since is not None
    session = uuid.uuid4()
    oldest = await scratch.accept("alice", at=since - 3 * HOUR, session=session, text="oldest")
    older = await scratch.accept("alice", at=since - 2 * HOUR, session=session, text="older")
    await scratch.accept("alice", at=since + HOUR, session=session, text="after-the-grant")
    await scratch.pool.execute(
        "UPDATE dlightrag_runs SET purge_after = NOW() - interval '1 day'"
        " WHERE run_id = ANY($1::uuid[])",
        [uuid.UUID(oldest), uuid.UUID(older)],
    )

    pruned = await scratch.store.prune_expired_runs()

    assert pruned.runs == 2
    assert (
        await scratch.pool.fetchval(
            "SELECT count(*) FROM dlightrag_agent_sessions WHERE session_id = $1", session
        )
        == 1
    ), "the Session lives on in the Run retention left"
    assert await scratch.exported() == []


async def test_the_export_reads_one_snapshot_so_what_happens_after_it_chose_its_runs_is_not_written(
    scratch: Scratch, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An owner who turns the switch off and keeps chatting while the export runs.

    A Run's record holds every entry it wrote, so a message it commits after the Runs were chosen
    would land in its record if the reads were not one snapshot. The connection here lets that
    happen between the choosing of the Runs and the reading of entries.
    """
    since = await scratch.turn("alice", True)
    assert since is not None
    session = uuid.uuid4()
    run = await scratch.accept("alice", at=since + HOUR, session=session, text="before-the-export")

    async def the_owner_withdraws_and_chats_on() -> None:
        await scratch.turn("alice", False)
        await scratch.chat(
            "alice", session, run, at=since + 2 * HOUR, text="after-the-export-began"
        )

    real_connect = asyncpg.connect

    async def connect(**settings: Any) -> Interposed:
        return Interposed(await real_connect(**settings), the_owner_withdraws_and_chats_on)

    with monkeypatch.context() as patch:
        patch.setattr(export_traces.asyncpg, "connect", connect)
        records = await scratch.exported()

    written = as_jsonl(records)
    assert [record["run"]["run_id"] for record in records] == [run]
    assert "before-the-export" in written
    assert "after-the-export-began" not in written
    # The message is really there, and the switch really is off, for the next export.
    assert (
        await scratch.pool.fetchval(
            "SELECT count(*) FROM dlightrag_agent_session_entries WHERE session_id = $1", session
        )
        == 2
    )
    assert await scratch.exported() == []


async def test_turning_it_off_exports_nothing_and_turning_it_on_again_starts_over(
    scratch: Scratch,
) -> None:
    first = await scratch.turn("alice", True)
    assert first is not None
    first_window = await scratch.accept(
        "alice",
        at=first + datetime.timedelta(milliseconds=1),
        session=uuid.uuid4(),
        text="first-window",
    )
    assert await scratch.run_ids() == {first_window}

    await scratch.turn("alice", False)
    assert await scratch.exported() == []

    await asyncio.sleep(0.05)
    second = await scratch.turn("alice", True)
    assert second is not None and second > first
    # Neither the conversation begun in the first window nor one begun while the switch was off
    # is read again: turning it on covers no conversation begun earlier.
    await scratch.accept(
        "alice",
        at=second - datetime.timedelta(milliseconds=1),
        session=uuid.uuid4(),
        text="while-off",
    )
    assert await scratch.exported() == []
    after = await scratch.accept(
        "alice", at=second + HOUR, session=uuid.uuid4(), text="second-window"
    )
    assert await scratch.run_ids() == {after}


async def test_one_owners_choice_never_exports_another_owners_runs(scratch: Scratch) -> None:
    alice_since = await scratch.turn("alice", True)
    assert alice_since is not None
    await scratch.turn("carol", True)
    await scratch.turn("carol", False)
    at = alice_since + HOUR
    alices = await scratch.accept("alice", at=at, session=uuid.uuid4(), text="alice")
    bobs = await scratch.accept("bob", at=at, session=uuid.uuid4(), text="bob never chose")
    carols = await scratch.accept("carol", at=at, session=uuid.uuid4(), text="carol withdrew")

    assert await scratch.run_ids() == {alices}
    # Asking for an owner, or for a Run by name, narrows what is exported and never widens it.
    assert await scratch.run_ids(owner="bob") == set()
    assert await scratch.run_ids(owner="carol") == set()
    assert await scratch.run_ids(run_ids=[bobs]) == set()
    assert await scratch.run_ids(run_ids=[bobs, carols, alices]) == {alices}


async def test_the_other_selections_still_narrow_what_is_exported(scratch: Scratch) -> None:
    alice, dave = await scratch.turn("alice", True), await scratch.turn("dave", True)
    assert alice is not None and dave is not None
    base = max(alice, dave) + HOUR
    runs = {
        name: await scratch.accept(owner, at=base + number * HOUR, session=uuid.uuid4(), text=name)
        for number, (name, owner) in enumerate(
            [("a1", "alice"), ("d1", "dave"), ("a2", "alice"), ("d2", "dave")]
        )
    }

    assert await scratch.run_ids() == set(runs.values())
    assert await scratch.run_ids(owner="alice") == {runs["a1"], runs["a2"]}
    assert await scratch.run_ids(run_ids=[runs["d1"], runs["a2"]]) == {runs["d1"], runs["a2"]}
    assert [r["run"]["run_id"] for r in await scratch.exported(limit=2)] == [
        runs["a1"],
        runs["d1"],
    ]
    # The bounds are ISO timestamps, as the command line gives them.
    since, until = (base + HOUR).isoformat(), (base + 3 * HOUR).isoformat()
    assert await scratch.run_ids(since=since, until=until) == {runs["d1"], runs["a2"]}


async def test_a_row_that_says_off_exports_nothing_even_if_it_still_holds_a_moment(
    scratch: Scratch,
) -> None:
    """The table keeps a flag and a moment in agreement, but the export does not lean on that:
    only an owner who says on is read. The table's check is lifted here to write the row it
    would never keep."""
    async with scratch.pool.acquire() as conn:
        await conn.execute(
            "ALTER TABLE dlightrag_privacy_settings"
            " DROP CONSTRAINT dlightrag_privacy_settings_since_check"
        )
        await conn.execute(
            "INSERT INTO dlightrag_privacy_settings (owner_id, contribute, contribute_since)"
            " VALUES ('alice', FALSE, '2020-01-01T00:00:00Z')"
        )
    await scratch.accept(
        "alice", at=datetime.datetime.now(datetime.UTC), session=uuid.uuid4(), text="off"
    )

    assert await scratch.exported() == []


# ── the dataset ──────────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class World:
    """Three conversations of two owners who contribute, and one of an owner who does not.

    ``s1`` began with ``r0``, which retention has since pruned, and goes on with ``r1``, a research
    turn that spawned a Child, passing its id on and reading a page of someone else's, and whose
    answer the owner judged twice, and ``r2``, a fast follow-up that follows both, compacted, and
    answers under the id of its own Run. ``s2`` holds a failed Run and ``s3`` a cancelled one,
    which is another owner's.
    """

    s1: uuid.UUID
    s2: uuid.UUID
    s3: uuid.UUID
    r0: str
    r1: str
    r2: str
    r3: str
    r4: str
    stranger: str
    child: str
    foreign: uuid.UUID
    intent: uuid.UUID
    projection: uuid.UUID
    lane: uuid.UUID
    answer: int
    child_answer: int
    given: dict[str, datetime.datetime]


async def _sequence(scratch: Scratch, entry: str) -> int:
    return await scratch.pool.fetchval(
        "SELECT sequence FROM dlightrag_agent_session_entries WHERE entry_id = $1",
        uuid.UUID(entry),
    )


async def build_world(scratch: Scratch) -> World:
    alice, dave = await scratch.turn("alice", True), await scratch.turn(DAVE, True)
    assert alice is not None and dave is not None
    start = max(alice, dave) + HOUR
    # Two conversations on the train side, interleaved in time, and one held out.
    s1, s2 = session_where(0.1, 1.0, "world-1"), session_where(0.1, 1.0, "world-2")
    s3 = session_where(0.0, 0.1, "world-3")
    child_id, foreign, intent, projection, lane = (uuid.uuid4() for _ in range(5))
    r0 = await scratch.accept(
        "alice", at=start, session=s1, text="pruned-parent-words", mode="research"
    )
    await scratch.chat("alice", s1, r0, at=start + 2 * SECOND, text="pruned-parent-answer")
    r1 = await scratch.accept(
        "alice",
        at=start + MINUTE,
        session=s1,
        text="research-question",
        mode="research",
        after=scratch.entries[r0][-1],
        accepted={"agent_session_id": str(s1), "parent_run_id": r0},
        evidence="evidence-snapshot-marker",
        result={
            "answer": "result-answer-marker",
            "contexts": {"chunks": [{"content": "result-contexts-marker"}]},
            "trace": {"note": "result-trace-marker"},
            "sources": SOURCES,
        },
    )
    at = start + MINUTE
    await scratch.chat(
        "alice",
        s1,
        r1,
        at=at + 2 * SECOND,
        text="",
        payload={
            "content": "",
            "stop_reason": "tool_use",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "read_subagent",
                    "arguments": {"child_session_id": str(child_id), "page_id": str(foreign)},
                    "argument_error": None,
                }
            ],
            "provider_state": {"blob": "provider-state-marker"},
        },
    )
    await scratch.events(
        "alice",
        r1,
        ("token", "event-text-marker", at + 2 * SECOND),
        ("tool_start", "call-1", at + 2 * SECOND + 100 * MILLISECOND),
        ("tool_end", "call-1", at + 3 * SECOND),
    )
    await scratch.chat(
        "alice",
        s1,
        r1,
        at=at + 3 * SECOND,
        text="",
        kind="tool_result",
        payload={
            "call_id": "call-1",
            "tool_name": "read_subagent",
            "outcome": "succeeded",
            "intent_id": str(intent),
            "attempt_id": str(uuid.uuid4()),
            "content": [
                {"type": "text", "text": f"Child session {child_id} finished, page {foreign}"}
            ],
        },
    )
    answer = await scratch.chat("alice", s1, r1, at=at + 4 * SECOND, text="research-answer")
    child, child_answer = await scratch.child(
        "alice",
        s1,
        r1,
        at=at + 5 * SECOND,
        text="child-answer",
        call="call-1",
        child_session=child_id,
        intent=intent,
    )
    assert child_answer is not None
    await scratch.note("alice", s1, r1, path="notes.md", at=at + 6 * SECOND, text="a-note")
    await scratch.head(r1)
    at = start + 5 * MINUTE
    r2 = await scratch.accept(
        "alice",
        at=at,
        session=s1,
        text="fast-follow-up",
        mode="fast",
        lane=str(lane),
        source_lane="main",
        after=scratch.entries[r1][-1],
        acceptance=True,
        accepted={
            "agent_session_id": str(s1),
            "parent_run_id": r1,
            "agent_lane_id": str(lane),
            "source_lane_id": "main",
        },
    )
    await scratch.chat(
        "alice",
        s1,
        r2,
        at=at + SECOND,
        text="",
        kind="compaction",
        payload={
            "projection_id": str(projection),
            "summary": "the earlier turns",
            "covered_through_sequence": 1,
            "first_retained_sequence": 2,
            "covered_through_entry_id": scratch.entries[r0][0],
            "first_retained_entry_id": scratch.entries[r1][-1],
            "source_digest": "d" * 64,
        },
    )
    await scratch.chat("alice", s1, r2, at=at + 2 * SECOND, text="fast-answer")
    await scratch.head(r2)
    r3 = await scratch.accept(
        "alice",
        at=start + 2 * MINUTE,
        session=s2,
        text="failed-question",
        status="failed",
        error_message="error-message-marker",
    )
    # It died while spawning: its Child names an intent whose result was never written.
    await scratch.child(
        "alice",
        s2,
        r3,
        at=start + 2 * MINUTE + SECOND,
        text="never-settled",
        status="cancelled",
        call="call-dead",
        wrote=False,
        intent=uuid.uuid4(),
    )
    r4 = await scratch.accept(
        DAVE, at=start + 3 * MINUTE, session=s3, text="cancelled-question", status="cancelled"
    )
    # It was cancelled while publishing a Skill: the tool failed, and its result carries the path
    # of the owner's skill directory, which names the owner.
    await scratch.chat(
        DAVE,
        s3,
        r4,
        at=start + 3 * MINUTE + 2 * SECOND,
        text="",
        kind="tool_result",
        payload={
            "call_id": "call-skill",
            "tool_name": "publish_skill",
            "outcome": "failed",
            "content": [
                {"type": "text", "text": skill_error(DAVE)},
                {"type": "text", "text": f"checksum {hashlib.sha256(b'a file').hexdigest()}"},
            ],
        },
    )
    # Another owner never chose, and judged an answer all the same.
    strangers = uuid.uuid4()
    stranger = await scratch.accept("bob", at=start, session=strangers, text="stranger-question")
    strangers_answer = await scratch.chat(
        "bob", strangers, stranger, at=start + SECOND, text="stranger-answer"
    )

    # Retention prunes the oldest Run of s1, and the Session lives on in the others.
    await scratch.pool.execute(
        "UPDATE dlightrag_runs SET purge_after = NOW() - interval '1 day' WHERE run_id = $1",
        uuid.UUID(r0),
    )
    assert (await scratch.store.prune_expired_runs()).runs == 1

    # The owner judges the answer, changes their mind a moment later, and judges the Child's.
    answer_sequence, child_sequence = (
        await _sequence(scratch, answer),
        await _sequence(scratch, child_answer),
    )
    put = scratch.store.put_answer_feedback
    assert await put(
        owner_id="alice",
        run_id=r1,
        agent="main",
        sequence=answer_sequence,
        value="up",
        reasons=[],
        note="first-thought",
    )
    await asyncio.sleep(0.02)
    assert await put(
        owner_id="alice",
        run_id=r1,
        agent="main",
        sequence=answer_sequence,
        value="down",
        reasons=["unsupported", "factual_error"],
        note="second-thought",
    )
    assert await put(
        owner_id="alice",
        run_id=r1,
        agent=child,
        sequence=child_sequence,
        value="up",
        reasons=[],
        note=None,
    )
    assert await put(
        owner_id="bob",
        run_id=stranger,
        agent="main",
        sequence=await _sequence(scratch, strangers_answer),
        value="down",
        reasons=["off_topic"],
        note="strangers-label-note",
    )
    rows = await scratch.pool.fetch(
        "SELECT agent, created_at FROM dlightrag_answer_feedback WHERE run_id = $1",
        uuid.UUID(r1),
    )
    return World(
        s1, s2, s3, r0, r1, r2, r3, r4, stranger, child, foreign, intent, projection, lane,
        answer_sequence, child_sequence, {row["agent"]: row["created_at"] for row in rows},
    )  # fmt: skip


async def test_labels_are_the_owners_latest_judgements_and_only_exported_runs_carry_them(
    scratch: Scratch,
) -> None:
    """The reward is kept as the owner gave it, and never reduced to a number: each judgement is
    its vote, its reasons, its note and when it was first given, on the agent that wrote the
    answer it judges. A changed mind replaces the first thought, and a Run that is not exported
    brings its judgements nowhere."""
    world = await build_world(scratch)
    directory = scratch.root.parent / "dataset"
    assert await scratch.dataset(directory) == 0
    written = Written(directory)
    records = {record["run"]["run_id"]: record for record in written.records}

    assert set(records) == {pseudonym(run) for run in (world.r1, world.r2, world.r3, world.r4)}
    assert records[pseudonym(world.r1)]["labels"] == [
        {
            "agent": "main",
            "sequence": world.answer,
            "value": "down",
            "reasons": ["unsupported", "factual_error"],
            "note": "second-thought",
            "given_at": world.given["main"].isoformat(),
        },
        {
            "agent": pseudonym(world.child),
            "sequence": world.child_answer,
            "value": "up",
            "reasons": [],
            "note": None,
            "given_at": world.given[world.child].isoformat(),
        },
    ]
    for unjudged in (world.r2, world.r3, world.r4):
        assert records[pseudonym(unjudged)]["labels"] == []
    # A Child's judgement names the Child, and the step in the Child's own steps that it judges.
    (child,) = records[pseudonym(world.r1)]["children"]
    assert child["child_session_id"] == pseudonym(world.child)
    judged = next(step for step in child["steps"] if step["sequence"] == world.child_answer)
    assert judged["type"] == "assistant_message"
    # Neither the first thought nor an owner who never chose reaches a file.
    assert "first-thought" not in written.everything
    assert "strangers-label-note" not in written.everything
    assert written.sheet["labels"] == {
        "count": 2,
        "by_value": {"down": 1, "up": 1},
        "by_agent": {"child": 1, "main": 1},
        "by_reason": {"factual_error": 1, "unsupported": 1},
        "with_note": 1,
        "runs_without_labels": 3,
        "unlabelled_share": 0.75,
    }


async def test_a_dataset_is_reproducible_and_its_datasheet_describes_exactly_its_files(
    scratch: Scratch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two exports of one database under one key are the same files byte for byte, so datasets
    made over time can be compared; the datasheet's counts are of what is in the files and its
    hashes are of the files; and the files, a raw `--out` file too, are private to the operator,
    and the dataset never holds the key."""
    await build_world(scratch)
    first, second, third = (scratch.root.parent / name for name in ("first", "second", "third"))
    first.mkdir(
        mode=0o755
    )  # an empty directory that exists is a place to write, and is made private

    assert await scratch.dataset(first) == 0
    assert await scratch.dataset(second) == 0
    one, two = Written(first), Written(second)
    for name in ("train.jsonl", "eval.jsonl"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert {**one.sheet, "generated_at": None} == {**two.sheet, "generated_at": None}

    # One line a Run, in the order of Session, then when the Run was made, then the Run.
    records = one.records
    assert len(records) == 4
    for part in (one.train, one.held_out):
        order = [
            (
                record["session"]["id"],
                datetime.datetime.fromisoformat(record["run"]["created_at"]),
                record["run"]["run_id"],
            )
            for record in part
        ]
        assert order == sorted(order)
    assert all(record["schema_version"] == 5 for record in records)

    # What the datasheet counts is what the files hold.
    counts = one.sheet["counts"]
    assert counts["runs"] == len(records)
    assert counts["owners"] == len({r["run"]["owner_id"] for r in records}) == 2
    assert counts["sessions"] == len({r["session"]["id"] for r in records}) == 3
    assert counts["children"] == sum(len(r["children"]) for r in records) == 2
    assert counts["steps"] == sum(len(r["steps"]) for r in records)
    assert counts["child_steps"] == sum(len(c["steps"]) for r in records for c in r["children"])
    assert counts["carried_entries"] == sum(len(r["session"]["carried"]) for r in records) == 4
    assert counts["evidence_snapshots"] == sum(len(r["evidence"]) for r in records)
    assert counts["session_notes"] == sum(len(r["session_notes"]) for r in records) == 1
    assert counts["by_status"] == {"cancelled": 1, "failed": 1, "succeeded": 2}
    assert counts["by_mode"] == {"fast": 1, "research": 1, "unresolved": 2}
    for side, part in (("train", one.train), ("eval", one.held_out)):
        assert {record["split"] for record in part} <= {side}
        assert counts["by_split"][side] == {
            "runs": len(part),
            "sessions": len({record["session"]["id"] for record in part}),
        }
    assert one.sheet["time_window"] == {
        "first_run_created_at": min(r["run"]["created_at"] for r in records),
        "last_run_created_at": max(r["run"]["created_at"] for r in records),
    }
    assert one.sheet["consent"]["contributing_owners"] == 2
    assert (
        one.sheet["consent"]["earliest_contribute_since"]
        == (
            await scratch.pool.fetchval(
                "SELECT min(contribute_since) FROM dlightrag_privacy_settings"
            )
        ).isoformat()
    )
    assert one.sheet["exporter"]["commit"] is None or re.fullmatch(
        r"[0-9a-f]{40}|[0-9a-f]{64}", one.sheet["exporter"]["commit"]
    )
    # What it says of the files is true of the files.
    assert set(one.sheet["files"]) == {"train.jsonl", "eval.jsonl"}
    for name, described in one.sheet["files"].items():
        data = (first / name).read_bytes()
        assert described == {
            "bytes": len(data),
            "records": data.count(b"\n"),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    # Private: files for their owner alone in a directory only they can enter.
    for directory in (first, second):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        for name in export_traces.DATASET_FILES:
            assert stat.S_IMODE((directory / name).stat().st_mode) == 0o600
    # The raw export's `--out` is as private, whether the file is new or was there with wider
    # rights and is replaced, and the directory it is in is left as it is.
    shared, replaced = scratch.root.parent / "shared", scratch.root.parent / "elsewhere.jsonl"
    shared.mkdir()
    shared.chmod(0o755)
    replaced.write_text("stale\n", encoding="utf-8")
    replaced.chmod(0o644)
    for out in (shared / "records.jsonl", replaced):
        assert await asyncio.to_thread(_status, ["--out", str(out)]) == 0
        assert stat.S_IMODE(out.stat().st_mode) == 0o600
        assert len(out.read_text(encoding="utf-8").splitlines()) == 4
    assert stat.S_IMODE(shared.stat().st_mode) == 0o755
    # The key is nowhere: not in a file, not on the terminal. Its fingerprint says which key it was.
    streams = capsys.readouterr()
    assert KEY not in one.everything + two.everything + streams.out + streams.err
    assert (
        one.sheet["key_fingerprint"]
        == (
            hmac.new(
                KEY.encode(), b"dlightrag-export-key-fingerprint-v1", hashlib.sha256
            ).hexdigest()[:8]
        )
    )

    # Another key writes the same dataset under other pseudonyms, and says it is another key.
    scratch.configure(key="another-key-for-the-pseudonyms-" + "z" * 16)
    assert await scratch.dataset(third) == 0
    other = Written(third)
    assert other.sheet["key_fingerprint"] != one.sheet["key_fingerprint"]
    assert {r["run"]["run_id"] for r in other.records}.isdisjoint(
        {r["run"]["run_id"] for r in records}
    )
    assert len(other.records) == len(records)


async def test_no_stored_id_survives_into_a_dataset_while_every_link_still_resolves(
    scratch: Scratch,
) -> None:
    """Every id the product stored for an owner, a Session, a Run, an Entry or a Child is replaced
    wherever it stands, in a field, in a tool's arguments or in the words of a result, and replaced
    alike each time, so a parent entry, a Child and a fork point still point at what they named,
    and the context a Run ran on can be rebuilt from the steps and the carried entries alone. An
    id the product did not issue is left as the agent had it. Context a pruned Run wrote is
    carried by the Runs that ran on it, and is nobody's record."""
    world = await build_world(scratch)
    directory = scratch.root.parent / "dataset"
    assert await scratch.dataset(directory) == 0
    written = Written(directory)
    records = {record["run"]["run_id"]: record for record in written.records}
    assert pseudonym(world.r0) not in records, "the pruned Run is nobody's record"
    second, third = records[pseudonym(world.r1)], records[pseudonym(world.r2)]
    entries = scratch.entries

    # No id the product stored for anyone is anywhere; the page the agent was handed still is.
    stored = {str(session) for session in (world.s1, world.s2, world.s3)}
    stored |= {world.child, *scratch.markers}
    for row in await scratch.pool.fetch(
        "SELECT entry_id::text, parent_entry_id::text, session_id::text"
        " FROM dlightrag_agent_session_entries"
    ):
        stored |= {value for value in row.values() if value}
    stored |= {
        row[0]
        for row in await scratch.pool.fetch(
            "SELECT fork_point_entry_id FROM dlightrag_answer_run_routing"
        )
        if row[0]
    }
    # The ids minted for an operation are the database's too: a tool call's intent and attempt, a
    # compaction's projection, the intent a Child names, and the lane of a fork.
    for query in (
        "SELECT payload_json->>'intent_id', payload_json->>'attempt_id',"
        " payload_json->>'projection_id' FROM dlightrag_agent_session_entries",
        "SELECT parent_intent_id::text FROM dlightrag_answer_child_sessions",
        "SELECT agent_lane_id, source_lane_id FROM dlightrag_answer_run_routing",
    ):
        for row in await scratch.pool.fetch(query):
            stored |= {value for value in row.values() if value and value != "main"}
    assert not [raw for raw in stored if raw in written.everything]
    assert '"alice"' not in written.everything
    assert str(world.foreign) in written.everything
    # An owner id is a digest, and it is found inside text as well as where a value is exactly it:
    # a Skill that failed to publish wrote the path of the owner's directory into its result.
    assert DAVE not in written.everything
    dave = records[pseudonym(world.r4)]
    assert dave["run"]["owner_id"] == dave["run"]["submitted_by"] == pseudonym(DAVE)
    (failed,) = (step for step in dave["steps"] if step["type"] == "tool_result")
    assert failed["payload"]["content"][0]["text"] == skill_error(pseudonym(DAVE))
    # A digest that names no owner is ordinary text and stays as written.
    assert (
        failed["payload"]["content"][1]["text"]
        == f"checksum {hashlib.sha256(b'a file').hexdigest()}"
    )

    # The same id is the same pseudonym wherever it stands.
    assert third["run"]["run_id"] == pseudonym(world.r2)
    assert third["run"]["owner_id"] == third["run"]["submitted_by"] == pseudonym("alice")
    assert third["session"]["id"] == pseudonym(world.s1)
    assert third["input"]["accepted"] == {
        "query": "fast-follow-up",
        "agent_session_id": pseudonym(world.s1),
        "parent_run_id": pseudonym(world.r1),
        "agent_lane_id": pseudonym(world.lane),
        "source_lane_id": "main",
    }
    assert third["routing"]["agent_lane_id"] == pseudonym(world.lane)
    assert third["routing"]["source_lane_id"] == "main"
    assert second["input"]["accepted"]["parent_run_id"] == pseudonym(world.r0)
    assert third["steps"][0]["payload"]["acceptance_id"] == pseudonym(world.r2)
    compaction = third["steps"][1]["payload"]
    assert compaction["covered_through_entry_id"] == pseudonym(entries[world.r0][0])
    assert compaction["first_retained_entry_id"] == pseudonym(entries[world.r1][-1])
    assert compaction["projection_id"] == pseudonym(world.projection)
    assert second["steps"][2]["payload"]["intent_id"] == pseudonym(world.intent)
    spawn, observed = second["steps"][1]["payload"], second["steps"][2]["payload"]
    assert spawn["tool_calls"][0]["arguments"] == {
        "child_session_id": pseudonym(world.child),
        "page_id": str(world.foreign),
    }
    assert observed["content"][0]["text"] == (
        f"Child session {pseudonym(world.child)} finished, page {world.foreign}"
    )
    (child,) = second["children"]
    assert child["child_session_id"] == pseudonym(world.child)
    assert child["parent_session_id"] == pseudonym(world.s1)
    assert child["parent_intent_id"] == pseudonym(world.intent)

    # The context a Run was given is rebuilt from the files, root first, and is not listed.
    assert set(third["session"]) == {"id", "head_entry_id", "carried"}
    assert rebuilt_chain(written.records, second) == [pseudonym(e) for e in entries[world.r0]]
    assert rebuilt_chain(written.records, third) == [
        pseudonym(entry) for entry in (*entries[world.r0], *entries[world.r1])
    ]
    assert third["session"]["head_entry_id"] == pseudonym(entries[world.r2][-1])
    assert third["routing"]["fork_point_entry_id"] == third["session"]["head_entry_id"]
    for record in (second, third):
        assert [entry["entry_id"] for entry in record["session"]["carried"]] == [
            pseudonym(entry) for entry in entries[world.r0]
        ]
        assert [entry["type"] for entry in record["session"]["carried"]] == [
            "user_message",
            "assistant_message",
        ]
    # The pruned Run's words are in the files as the context two Runs saw, and in no Run's steps.
    runs = (world.r0, world.r1, world.r2, world.r3, world.r4)
    assert scratch.words_in(written.everything) == set(runs)
    assert not any("pruned-parent-words" in json.dumps(r["steps"]) for r in written.records)

    # No parent id dangles: every Run's chain rebuilds to the root from steps and carried alone,
    # and so does the parent of every step and every carried entry.
    for record in written.records:
        rebuilt_chain(written.records, record)
        reach = entries_a_run_can_reach(written.records, record)
        for step in (*record["steps"], *record["session"]["carried"]):
            assert step["parent_entry_id"] is None or step["parent_entry_id"] in reach


async def test_a_record_holds_the_episode_and_leaves_out_what_the_design_excludes(
    scratch: Scratch,
) -> None:
    """The provider's replay state, the Run's stream of events, its error message, the text it
    streamed, the whole of its stored result and the text of its evidence snapshots are in no
    file. What is left is the episode: the steps with their payload format, the sources the answer
    cites with the passages cited, and the digests that tell the evidence snapshots apart."""
    world = await build_world(scratch)
    directory = scratch.root.parent / "dataset"
    assert await scratch.dataset(directory) == 0
    written = Written(directory)
    records = {record["run"]["run_id"]: record for record in written.records}

    in_the_records = written.text("train.jsonl") + written.text("eval.jsonl")
    for left_out in (
        "provider-state-marker",
        "provider_state",
        "error-message-marker",
        "evidence-snapshot-marker",
        "result-answer-marker",
        "result-contexts-marker",
        "result-trace-marker",
        "event-text-marker",
        "tool_events",
    ):
        assert left_out not in in_the_records, left_out
    assert "the-cited-passage" in in_the_records

    researched = records[pseudonym(world.r1)]
    assert set(researched) == {
        "schema_version",
        "split",
        "run",
        "input",
        "routing",
        "session",
        "steps",
        "children",
        "evidence",
        "session_notes",
        "result",
        "labels",
    }
    assert researched["result"] == {"sources": SOURCES}
    assert "error_message" not in researched["run"]
    assert records[pseudonym(world.r3)]["run"]["error_kind"] == "run_execution_failed"
    assert records[pseudonym(world.r3)]["result"] is None
    # A step says which payload format it is in, and a Run's evidence is told apart by digests.
    assert [step["schema_version"] for step in researched["steps"]] == [2, 2, 2, 2]
    assert [step["type"] for step in researched["steps"]] == [
        "user_message",
        "assistant_message",
        "tool_result",
        "assistant_message",
    ]
    (snapshot,) = researched["evidence"]
    assert set(snapshot) == {
        "ordinal",
        "content_digest",
        "locator",
        "locator_digest",
        "created_at",
    }
    # The Run is an episode: what it did, what it saw, how it ended, and what its owner said.
    assert researched["run"]["status"] == "succeeded"
    assert researched["steps"][2]["tool"]["duration_ms"] == 900
    assert set(researched["session"]) == {"id", "head_entry_id", "carried"}
    assert researched["session"]["head_entry_id"] == researched["steps"][-1]["entry_id"]


async def test_a_sessions_side_is_a_hash_of_its_id_and_never_moves(scratch: Scratch) -> None:
    """A Session, its Runs and its Children are on one side, which only its stored id decides: no
    secret and no list, so exporting more never moves a Session, and a larger share held out only
    takes Sessions from train."""
    since = await scratch.turn("alice", True)
    assert since is not None
    start = since + HOUR
    sessions = {
        "held": session_where(0.0, 0.1, "held"),
        "middle": session_where(0.1, 0.5, "middle"),
        "kept": session_where(0.5, 1.0, "kept"),
    }

    async def converse(session: uuid.UUID, name: str, number: int) -> None:
        """Two turns of a conversation, the first with a Child."""
        at = start + number * HOUR
        first = await scratch.accept("alice", at=at, session=session, text=f"{name}-first")
        await scratch.child("alice", session, first, at=at + SECOND, text=f"{name}-child")
        await scratch.accept(
            "alice",
            at=at + MINUTE,
            session=session,
            text=f"{name}-second",
            after=scratch.entries[first][-1],
        )

    async def made(name: str, *options: str) -> dict[str, str]:
        """The side of each Session's Runs, by the Session's pseudonym, once it is read back from
        the files; every Run of a Session must be on the side its record says."""
        directory = scratch.root.parent / name
        assert await scratch.dataset(directory, *options) == 0
        written = Written(directory)
        found: dict[str, set[str]] = {}
        for side, part in (("train", written.train), ("eval", written.held_out)):
            for record in part:
                assert record["split"] == side
                found.setdefault(record["session"]["id"], set()).add(side)
        assert all(len(places) == 1 for places in found.values())
        return {session: next(iter(places)) for session, places in found.items()}

    for number, (name, session) in enumerate(sessions.items()):
        await converse(session, name, number)
    one = await made("one")
    assert one == {
        pseudonym(sessions["held"]): "eval",
        pseudonym(sessions["middle"]): "train",
        pseudonym(sessions["kept"]): "train",
    }

    # Sessions added later leave the earlier ones where they were.
    later = {"held": session_where(0.0, 0.1, "later-held"), "kept": session_where(0.5, 1, "later")}
    for number, (name, session) in enumerate(later.items(), start=len(sessions)):
        await converse(session, f"later-{name}", number)
    two = await made("two")
    assert {session: side for session, side in two.items() if session in one} == one
    assert two[pseudonym(later["held"])] == "eval"
    assert two[pseudonym(later["kept"])] == "train"

    # A larger share takes Sessions from train and gives none back, and a Session is where the
    # hash puts it at that share.
    three = await made("three", "--eval-fraction", "0.5")
    assert three[pseudonym(sessions["middle"])] == "eval"
    assert {session for session, side in two.items() if side == "eval"} <= {
        session for session, side in three.items() if side == "eval"
    }
    everyone = {pseudonym(session): session for session in (*sessions.values(), *later.values())}
    for session, side in three.items():
        assert side == side_of(everyone[session], 0.5)
    assert set((await made("none", "--eval-fraction", "0")).values()) == {"train"}
    assert set((await made("all", "--eval-fraction", "1")).values()) == {"eval"}
    assert (scratch.root.parent / "none" / "eval.jsonl").read_bytes() == b""
    assert (scratch.root.parent / "all" / "train.jsonl").read_bytes() == b""


async def test_a_dataset_takes_whole_sessions_of_the_owners_who_contribute_and_no_run_still_going(
    scratch: Scratch,
) -> None:
    """The Session is the unit of consent, so it is the unit of the dataset: it is in whole or not
    at all, by when it began, whatever the options say of when its Runs were made. A Session begun
    before the permission is out however it went on, an owner who never chose or took it back is
    absent, a Run still going is not an episode yet and holds secrets, and an owner who turns the
    switch off is gone from the next dataset."""
    alice, dave = await scratch.turn("alice", True), await scratch.turn("dave", True)
    await scratch.turn("carol", True)
    await scratch.turn("carol", False)
    assert alice is not None and dave is not None
    base = max(alice, dave) + HOUR
    early, whole, late, mine = (uuid.uuid4() for _ in range(4))
    await scratch.accept("alice", at=alice - HOUR, session=early, text="begun-before-the-grant")
    await scratch.accept("alice", at=base, session=early, text="continued-after-the-grant")
    w1 = await scratch.accept("alice", at=base + HOUR, session=whole, text="whole-1")
    w2 = await scratch.accept(
        "alice",
        at=base + 10 * HOUR,
        session=whole,
        text="whole-2",
        after=scratch.entries[w1][-1],
    )
    l1 = await scratch.accept("alice", at=base + 20 * HOUR, session=late, text="late-1")
    d1 = await scratch.accept("dave", at=base + HOUR, session=mine, text="dave-finished")
    going = await scratch.accept(
        "dave",
        at=base + 2 * HOUR,
        session=mine,
        text="dave-going",
        status="running",
        after=scratch.entries[d1][-1],
    )
    await scratch.accept("bob", at=base, session=uuid.uuid4(), text="never-chose")
    await scratch.accept("carol", at=base, session=uuid.uuid4(), text="withdrew")
    await scratch.accept(
        "alice", at=base, session=uuid.uuid4(), text="a-retrieval", kind="retrieval"
    )
    counter = itertools.count()

    async def made(*options: str) -> Written:
        directory = scratch.root.parent / f"dataset-{next(counter)}"
        assert await scratch.dataset(directory, *options) == 0
        return Written(directory)

    def runs(written: Written) -> set[str]:
        return {record["run"]["run_id"] for record in written.records}

    everything = await made()
    assert runs(everything) == {pseudonym(run) for run in (w1, w2, l1, d1)}
    assert scratch.words_in(everything.everything) == {w1, w2, l1, d1}
    assert f"secret-{going}" not in everything.everything, "a Run still going holds secrets"

    # The bounds are of when a Session began: `whole` began early and its second Run came late.
    after = await made("--since", (base + 5 * HOUR).isoformat())
    assert runs(after) == {pseudonym(l1)}
    before = await made("--until", (base + 5 * HOUR).isoformat())
    assert runs(before) == {pseudonym(run) for run in (w1, w2, d1)}
    assert runs(await made("--owner", "dave")) == {pseudonym(d1)}
    assert (await made("--owner", "dave")).sheet["selection"]["owner"] == pseudonym("dave")

    # Turning the switch off takes the owner out of the next dataset, and the last out of the
    # dataset altogether, which is then empty and says so.
    await scratch.turn("alice", False)
    assert runs(await made()) == {pseudonym(d1)}
    await scratch.turn("dave", False)
    empty = await made()
    assert empty.records == []
    assert empty.text("train.jsonl") == empty.text("eval.jsonl") == ""
    assert empty.sheet["counts"]["runs"] == empty.sheet["counts"]["sessions"] == 0
    assert empty.sheet["consent"]["contributing_owners"] == 0
    assert empty.sheet["consent"]["earliest_contribute_since"] is None
    assert empty.sheet["time_window"] == {"first_run_created_at": None, "last_run_created_at": None}
    assert empty.sheet["labels"]["unlabelled_share"] is None


async def test_a_dataset_is_refused_rather_than_made_wrongly(
    scratch: Scratch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No key, a key too short to protect anything, a directory that holds something, and an
    option that would cut a Session or leave the directory, each stop the command before it reads
    or writes anything. The database here is not even listening: reaching it would be a different
    failure."""
    scratch.configure(key=None, port=1)
    directory = scratch.root.parent / "dataset"

    async def refused(*options: str, at: Path = directory) -> str:
        assert await scratch.dataset(at, *options) == 2
        return capsys.readouterr().err

    assert export_traces.KEY_NAME in await refused()
    scratch.configure(key="k" * 31, port=1)
    assert "at least 32 characters" in await refused()
    scratch.configure(key="k" * 32, port=1)
    # With a key of exactly enough characters it gets as far as the database, which is not there.
    assert await scratch.dataset(directory) == 1
    assert "cannot read" in capsys.readouterr().err
    assert not directory.exists()

    occupied, a_file = scratch.root.parent / "occupied", scratch.root.parent / "a-file"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("mine", encoding="utf-8")
    a_file.write_text("mine", encoding="utf-8")
    for place in (occupied, a_file):
        assert "not an empty directory" in await refused(at=place)
    assert sorted(path.name for path in occupied.iterdir()) == ["keep.txt"]
    assert (occupied / "keep.txt").read_text(encoding="utf-8") == "mine"
    assert a_file.read_text(encoding="utf-8") == "mine"

    for option, given, because in (
        ("--limit", ("5",), "whole Sessions"),
        ("--run-id", ("some-run",), "whole Sessions"),
        ("--dedupe", (), "whole Sessions"),
        ("--out", ("elsewhere.jsonl",), "its own directory"),
    ):
        message = await refused(option, *given)
        assert f"{option} cannot be used with --dataset" in message
        assert because in message
    assert "between 0 and 1" in await refused("--eval-fraction", "1.5")
    assert not directory.exists()

    # And a share to hold out means nothing to the raw export.
    assert await asyncio.to_thread(_status, ["--eval-fraction", "0.2"]) == 2
    assert "--eval-fraction applies to --dataset only" in capsys.readouterr().err


def _status(argv: list[str]) -> int:
    """The exit status of the command for these arguments, as a shell sees it."""
    try:
        return export_traces.main(argv)
    except SystemExit as stopped:
        return int(stopped.code or 0)


async def test_a_dataset_says_when_each_tool_ran_and_how_long_each_child_took(
    scratch: Scratch,
) -> None:
    """A tool's time is the interval between the two events its call left, and none where those
    events do not say it exactly: a start with no end, an end with no start, a start made twice,
    an id two results claim, a Run whose events were trimmed. A Child took from its row's making to
    its last entry, or to its row's last change when it wrote none, and has not finished if it is
    still running. The model call's own timings are the step's, as the Runtime stored them."""
    since = await scratch.turn("alice", True)
    assert since is not None
    session, start = uuid.uuid4(), since + HOUR
    run = await scratch.accept("alice", at=start, session=session, text="timed", mode="research")
    at = start + 20 * SECOND
    model_call = {
        "model": {"role": "query", "model": "a-model", "served_model": "a-model-2026"},
        "params": {"temperature": 1.0, "max_tokens": 65536},
        "queued_ms": 7,
        "duration_ms": 8412,
        "prompt": {"system": "a" * 64, "tools": "b" * 64, "request": "c" * 64},
        "plan_digest": "d" * 64,
    }
    calls = [
        "call-ok",
        "call-down",
        "call-up",
        "call-open",
        "call-late",
        "call-twice",
        "call-shared",
        "call-none",
    ]
    await scratch.chat(
        "alice",
        session,
        run,
        at=at,
        text="",
        payload={
            "content": "",
            "stop_reason": "tool_use",
            "tool_calls": [
                {"id": call, "name": "bash", "arguments": {}, "argument_error": None}
                for call in calls
            ],
            "call": model_call,
        },
    )
    for number, call in enumerate(calls):
        await scratch.chat(
            "alice",
            session,
            run,
            at=at + (number + 1) * SECOND,
            text="",
            kind="tool_result",
            payload={"call_id": call, "tool_name": "bash", "content": []},
        )

    child_start = start + 5 * SECOND
    finished, _ = await scratch.child(
        "alice",
        session,
        run,
        at=child_start,
        text="finished",
        call="call-spawn",
        also=tuple(
            ("tool_result", {"call_id": call, "tool_name": "bash", "content": []}, child_start + s)
            for call, s in (("call-child", 10 * SECOND), ("call-shared", 25 * SECOND))
        ),
    )
    silent, _ = await scratch.child(
        "alice",
        session,
        run,
        at=start + 6 * SECOND,
        text="silent",
        call="call-silent",
        wrote=False,
        updated=start + 40 * SECOND,
    )
    going, _ = await scratch.child(
        "alice",
        session,
        run,
        at=start + 7 * SECOND,
        text="going",
        call="call-going",
        status="running",
        updated=start + 50 * SECOND,
    )

    t = start + 60 * SECOND
    await scratch.events(
        "alice",
        run,
        ("tool_start", "call-ok", t),
        ("tool_end", "call-ok", t + 250 * MILLISECOND),
        ("tool_start", "call-down", t),
        ("tool_end", "call-down", t + datetime.timedelta(microseconds=1499)),
        ("tool_start", "call-up", t),
        ("tool_end", "call-up", t + datetime.timedelta(microseconds=1500)),
        ("tool_start", "call-open", t + SECOND),
        ("tool_end", "call-late", t + 2 * SECOND),
        ("tool_start", "call-twice", t),
        ("tool_start", "call-twice", t + SECOND),
        ("tool_end", "call-twice", t + 2 * SECOND),
        ("tool_start", "call-shared", t),
        ("tool_end", "call-shared", t + SECOND),
        ("tool_start", "call-child", t + 3 * SECOND),
        ("tool_end", "call-child", t + 4 * SECOND),
    )
    # A Run whose events retention has trimmed keeps its steps and has lost the times of its tools.
    trimmed_session = uuid.uuid4()
    trimmed = await scratch.accept("alice", at=start, session=trimmed_session, text="trimmed")
    await scratch.chat(
        "alice",
        trimmed_session,
        trimmed,
        at=at,
        text="",
        kind="tool_result",
        payload={"call_id": "call-trimmed", "tool_name": "bash", "content": []},
    )
    await scratch.pool.execute(
        "UPDATE dlightrag_runs SET events_trimmed_at = NOW() WHERE run_id = $1",
        uuid.UUID(trimmed),
    )

    directory = scratch.root.parent / "dataset"
    assert await scratch.dataset(directory) == 0
    records = {r["run"]["run_id"]: r for r in Written(directory).records}
    record = records[pseudonym(run)]

    def when(moment: datetime.datetime | None) -> str | None:
        return None if moment is None else moment.isoformat()

    def timed(start_at: datetime.datetime | None, end_at: datetime.datetime | None, ms: int | None):
        return {"started_at": when(start_at), "ended_at": when(end_at), "duration_ms": ms}

    nothing = timed(None, None, None)
    tools: dict[str, list[dict[str, Any]]] = {}
    for step in (*record["steps"], *(s for child in record["children"] for s in child["steps"])):
        if step["type"] == "tool_result":
            tools.setdefault(step["payload"]["call_id"], []).append(step["tool"])
    assert tools == {
        "call-ok": [timed(t, t + 250 * MILLISECOND, 250)],
        "call-down": [timed(t, t + datetime.timedelta(microseconds=1499), 1)],
        "call-up": [timed(t, t + datetime.timedelta(microseconds=1500), 2)],
        "call-open": [timed(t + SECOND, None, None)],
        "call-late": [timed(None, t + 2 * SECOND, None)],
        "call-twice": [nothing],
        "call-shared": [nothing, nothing],
        "call-none": [nothing],
        "call-child": [timed(t + 3 * SECOND, t + 4 * SECOND, 1000)],
    }
    (trimmed_tool,) = (
        step["tool"]
        for step in records[pseudonym(trimmed)]["steps"]
        if step["type"] == "tool_result"
    )
    assert trimmed_tool == nothing
    assert records[pseudonym(trimmed)]["run"]["events_trimmed_at"] is not None

    # The model call is the step's own, untouched; the Run's duration is the one it always had.
    assert record["steps"][1]["payload"]["call"] == model_call
    assert record["run"]["duration_seconds"] == 2.0
    assert record["run"]["started_at"] == start.isoformat()

    # A Child ends at its last entry, or at its row's last change if it wrote none.
    by_id = {child["child_session_id"]: child for child in record["children"]}
    done, quiet, running = (by_id[pseudonym(c)] for c in (finished, silent, going))
    assert (done["started_at"], done["finished_at"], done["duration_seconds"]) == (
        child_start.isoformat(),
        (child_start + 25 * SECOND).isoformat(),
        25.0,
    )
    assert (quiet["started_at"], quiet["finished_at"], quiet["duration_seconds"]) == (
        (start + 6 * SECOND).isoformat(),
        (start + 40 * SECOND).isoformat(),
        34.0,
    )
    assert (running["started_at"], running["finished_at"], running["duration_seconds"]) == (
        (start + 7 * SECOND).isoformat(),
        None,
        None,
    )
