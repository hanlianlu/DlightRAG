# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner-scoped export and erasure for a data-subject request (ADR 0049).

One operator command with the two operations a hosted deployment owes a person:
give them everything we hold, and remove it. Both are owner-scoped, both leave a
receipt, and both take their coverage from the schema rather than from a list —
a table added tomorrow is covered the day it appears. What the schema cannot
know is written down here: the two tables that name their owner `principal_id`,
the columns whose values are sealed credentials and stay out of an export, and
the workspace-scoped corpus tables an owner erase must not touch directly because
they belong to the customer, not to the person.

A Corpus Workspace the requester created is the one corpus an erase may cover, and
even then it does not delete it: it delivers a Workspace Delete Run through the
same Corpus Mutation channel every corpus write uses (`owner_data_corpora.py`), so
a concurrent ingest cannot write back what a delete removed. The delivery is
asynchronous, so the receipt states it — the Run ids, and the promise is checked
against their terminal states.

Erasure is not the same operation as deleting a conversation. A conversation
delete cascades its Runs; an owner erase also reaches Profile Memory, sealed
connection rows, Agent accounts, Skills, workspace roots and telemetry, which is
what "delete everything about me" means.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import asyncpg
import httpx
from dlightrag_observability.otlp import USER_ID

from dlightrag.adapters.postgres.core._pool import pg_pool
from dlightrag.adapters.postgres.owner_data_corpora import (
    CorpusDeletionDeliverer,
    corpus_deletion_deliverer,
)
from dlightrag.adapters.postgres.owner_data_sweep import SweptPackage, sweep_expired_exports
from dlightrag.application.access.principal import owner_id_from_principal
from dlightrag.application.config.loading import get_config, load_config, set_config
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.engine.agent.skills import owner_skill_root
from dlightrag.engine.answer.agentspace import owner_shard, resolve_agentspace_root, run_root

logger = logging.getLogger(__name__)

#: Tables that name the owner something other than `owner_id`; discovery cannot see these.
_OWNER_COLUMN_BY_TABLE: dict[str, str] = {
    "web_conversations": "principal_id",
    "web_conversation_turns": "principal_id",
    "web_conversation_commands": "principal_id",
}

#: Column-name fragments that mean "this value is a credential, not the person's data".
#: An export withholds them and says so, because handing a working credential to
#: whoever holds the package is worse than a stated omission. Matching by name is
#: deliberately lopsided: a false positive costs the subject a counter
#: (`connection_grants.secret_version` is a number, and stays out), while a false
#: negative hands out a working token. `endpoint_headers` is named because a
#: Connection states the headers every request to its endpoint carries.
_WITHHELD_COLUMN_HINTS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "credential",
    "envelope",
    "cipher",
    "encrypted",
    "authorization",
    "endpoint_headers",
)
#: A key's *identifier* is not a key: `key_id` names which sealing key wrapped a
#: value, so it stays in an export and the value it wrapped does not.

#: Workspace-scoped planes: the customer's corpus, not the person's. An owner erase
#: reports them and leaves them; deleting a Corpus Workspace is the customer's call
#: — except for one the requester created, which the erase delivers for deletion
#: rather than performing itself (see `owner_data_corpora.py`).
_CORPUS_TABLES: tuple[str, ...] = (
    "dlightrag_workspace_meta",
    "dlightrag_doc_metadata",
    "dlightrag_metadata_field_stats",
    "dlightrag_promotion_jobs",
)

#: A delete pass repeats until nothing changes, so foreign keys order themselves.
_ERASE_PASS_LIMIT = 8

#: The erasure promise a DPA states, and the ceiling a backup policy may not exceed:
#: a copy kept for R days still holds the data at request plus R, so the horizon and
#: the promise are the same number (docs/trust/retention.md).
_ERASE_PROMISE_DAYS = 30

#: How long a handover package may sit on disk. An export is a copy of a person's
#: life in cleartext: the clock that bounds the delivery bounds the copy, and
#: An operator's timer runs `sweep-expired`, which reads this and nothing else.
_EXPORT_TTL_DAYS = 7

#: The receipt table names the owner, so discovery finds it — and erasing it would
#: destroy the evidence of the erasure itself. It is kept: it holds an id and
#: counts, no content, and it is what answers "was this request served?".
_RECEIPTS_TABLE = "dlightrag_owner_data_requests"


class TelemetryEraser(Protocol):
    """Remove one owner's telemetry, returning the rows removed."""

    async def __call__(self, *, owner_id: str) -> int: ...


class OwnerEraseRefused(RuntimeError):
    """The owner still has work in flight, so erasing now would race it."""


class OwnerEraseUnauthorized(RuntimeError):
    """A B2B erase names no customer approval, and the customer is the controller."""


@dataclass(frozen=True)
class CorpusWorkspace:
    """A Corpus Workspace the requester created, and what an erase would take."""

    workspace: str
    documents: int


@dataclass(frozen=True)
class OwnerPlane:
    """One table that holds owner-scoped rows, and the column that names the owner."""

    table: str
    column: str


@dataclass(frozen=True)
class ExportOutcome:
    """What an export wrote, and what it deliberately withheld."""

    owner_id: str
    request_id: str
    path: Path
    rows: dict[str, int]
    withheld_columns: dict[str, tuple[str, ...]]
    left_in_place: tuple[str, ...]
    blob_count: int
    expires_at: datetime


@dataclass(frozen=True)
class EraseOutcome:
    """What an erase removed, and what it left for the customer to decide.

    `corpora_deletion_runs` is the asynchronous part: the Corpus Workspace
    deletions this erase delivered, as Run ids whose terminal states are the
    promise's evidence. `corpora_already_gone` are the ones a concurrent
    deletion had already taken, which needed no Run.
    """

    owner_id: str
    request_id: str
    deleted: dict[str, int]
    workspace_roots: int
    telemetry_rows: int
    left_in_place: tuple[str, ...]
    corpora_deletion_runs: tuple[str, ...] = ()
    corpora_already_gone: tuple[str, ...] = ()


#: The ledger's operations, in one tuple, which the table's check reads. A sweep is
#: maintenance rather than a request from a person, and it is still a ledger row — the
#: container log rotates, and a DPA audit reads durable evidence of what happened to a
#: package.
_RECEIPT_OPERATIONS: tuple[str, ...] = ("export", "erase", "sweep")
_RECEIPT_OPERATION_VALUES = ", ".join(f"'{operation}'" for operation in _RECEIPT_OPERATIONS)

_RECEIPTS_DDL = f"""
CREATE TABLE IF NOT EXISTS dlightrag_owner_data_requests (
    request_id   UUID        PRIMARY KEY,
    owner_id     TEXT        NOT NULL,
    operation    TEXT        NOT NULL,
    executor     TEXT        NOT NULL,
    line         TEXT,
    region       TEXT,
    outcome      TEXT        NOT NULL,
    counts       JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
    left_in_place JSONB      NOT NULL DEFAULT '[]'::jsonb,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT dlightrag_owner_data_requests_operation_check
        CHECK (operation IN ({_RECEIPT_OPERATION_VALUES})),
    CONSTRAINT dlightrag_owner_data_requests_outcome_check
        CHECK (outcome IN ('completed', 'refused', 'failed'))
)
"""

_RECEIPT_INSERT = """
INSERT INTO dlightrag_owner_data_requests
    (request_id, owner_id, operation, executor, line, region, outcome, counts, left_in_place,
     requested_at, finished_at)
VALUES ($1::uuid, $2, $3, $4, $5, $6, $7, $8::jsonb, $9::jsonb, $10, NOW())
"""


async def owner_planes(conn: Any) -> tuple[OwnerPlane, ...]:
    """Every table holding owner-scoped rows, discovered rather than listed.

    Discovery is what keeps this honest as the schema grows: a table with an
    `owner_id` column is covered without anyone remembering to add it.
    """
    rows = await conn.fetch(
        """
        SELECT table_name FROM information_schema.columns
        WHERE table_schema = 'public' AND column_name = 'owner_id'
        ORDER BY table_name
        """
    )
    planes = [
        OwnerPlane(table=str(row["table_name"]), column="owner_id")
        for row in rows
        if str(row["table_name"]) != _RECEIPTS_TABLE
    ]
    named = {plane.table for plane in planes}
    for table, column in _OWNER_COLUMN_BY_TABLE.items():
        if table in named:
            continue
        exists = await conn.fetchval(
            """
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2
            """,
            table,
            column,
        )
        if exists:
            planes.append(OwnerPlane(table=table, column=column))
    return tuple(planes)


async def _columns(conn: Any, table: str) -> tuple[str, ...]:
    rows = await conn.fetch(
        """
        SELECT column_name, data_type FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position
        """,
        table,
    )
    return tuple(str(row["column_name"]) for row in rows)


def _withheld(columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        column
        for column in columns
        if any(hint in column.lower() for hint in _WITHHELD_COLUMN_HINTS)
    )


async def _ensure_receipts(conn: Any) -> None:
    """Create the ledger, able to record every operation."""
    await conn.execute(_RECEIPTS_DDL)


async def _insert_receipt(
    conn: Any,
    *,
    request_id: uuid.UUID,
    owner_id: str,
    operation: str,
    executor: str,
    line: str | None,
    region: str | None,
    outcome: str,
    counts: dict[str, Any],
    left_in_place: Iterable[str],
    requested_at: datetime | None = None,
) -> None:
    """Append one row to a ledger this connection has already made writable."""
    await conn.execute(
        _RECEIPT_INSERT,
        request_id,
        owner_id,
        operation,
        executor,
        line,
        region,
        outcome,
        json.dumps(counts),
        json.dumps(list(left_in_place)),
        requested_at or datetime.now(UTC),
    )


async def _write_receipt(
    conn: Any,
    *,
    request_id: uuid.UUID,
    owner_id: str,
    operation: str,
    executor: str,
    line: str | None,
    region: str | None,
    outcome: str,
    counts: dict[str, Any],
    left_in_place: Iterable[str],
    requested_at: datetime | None = None,
) -> None:
    await _ensure_receipts(conn)
    await _insert_receipt(
        conn,
        request_id=request_id,
        owner_id=owner_id,
        operation=operation,
        executor=executor,
        line=line,
        region=region,
        outcome=outcome,
        counts=counts,
        left_in_place=left_in_place,
        requested_at=requested_at,
    )


async def record_sweep_receipts(
    conn: Any,
    *,
    executor: str,
    swept: Sequence[SweptPackage],
) -> list[str]:
    """Record one ledger row per package the sweep judged, and return their ids.

    A sweep is maintenance rather than a request from a person, and it is still an
    act on a person's data: the row says which package went, when its own manifest
    said it had to, and whether the removal actually happened. A dry run records
    nothing — promising a removal that was not performed is worse than silence.
    """
    await _ensure_receipts(conn)
    request_ids: list[str] = []
    for package in swept:
        request_id = uuid.uuid4()
        await _insert_receipt(
            conn,
            request_id=request_id,
            owner_id=package.owner_id,
            operation="sweep",
            executor=executor,
            line=None,
            region=None,
            outcome="completed" if package.removed else "failed",
            counts={
                "package_request_id": package.request_id,
                "expired_at": package.expires_at.isoformat(),
                "removed": package.removed,
            },
            # A package that outlived its expiry is exactly what was left in place.
            left_in_place=(str(package.path),) if not package.removed else (),
        )
        request_ids.append(str(request_id))
    return request_ids


async def export_owner(
    conn: Any, *, owner_id: str, out_dir: Path, owner_skills_root: Path | None = None
) -> ExportOutcome:
    """Write everything one owner holds, minus the credentials, and receipt it.

    The owner's Skills live on disk rather than in a table, so the package carries
    their files as they are; the key ring and the sealed columns it opens are the
    one class an export states and refuses.
    """
    request_id = uuid.uuid4()
    requested_at = datetime.now(UTC)
    expires_at = requested_at + timedelta(days=_EXPORT_TTL_DAYS)
    target = out_dir / str(request_id)
    (target / "blobs").mkdir(parents=True, exist_ok=True)
    planes = await owner_planes(conn)
    counts: dict[str, int] = {}
    withheld_by_table: dict[str, tuple[str, ...]] = {}
    for plane in planes:
        columns = await _columns(conn, plane.table)
        withheld = _withheld(columns)
        writable = [column for column in columns if column not in withheld]
        if withheld:
            withheld_by_table[plane.table] = withheld
        if not writable:
            continue
        selected = ", ".join(f'"{column}"' for column in writable)
        statement = f'SELECT {selected} FROM "{plane.table}" WHERE "{plane.column}" = $1'  # noqa: S608
        rows = await conn.fetch(statement, owner_id)
        counts[plane.table] = len(rows)
        if not rows:
            continue
        with (target / f"{plane.table}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), default=str, ensure_ascii=False) + "\n")
    blob_count = await _export_blobs(conn, owner_id=owner_id, target=target)
    skill_files = _export_owner_skills(owner_skills_root, owner_id=owner_id, target=target)
    manifest = {
        "owner_id": owner_id,
        "request_id": str(request_id),
        "operation": "export",
        "exported_at": datetime.now(UTC).isoformat(),
        "tables": counts,
        "withheld_columns": {table: list(cols) for table, cols in withheld_by_table.items()},
        "left_in_place": list(_CORPUS_TABLES),
        "blobs": blob_count,
        "expires_at": expires_at.isoformat(),
        "skill_files": skill_files,
        "note": (
            "Not in this package, and deliberately: the sealed credential columns "
            "named in withheld_columns; the key ring that unseals them, which never "
            "leaves the deployment; and the telemetry store, whose retention is "
            "stated in docs/trust/retention.md. What a credential buys the subject "
            "is rotation or deletion, not disclosure."
        ),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    await _write_receipt(
        conn,
        request_id=request_id,
        owner_id=owner_id,
        operation="export",
        executor="operator",
        line=None,
        region=None,
        outcome="completed",
        counts=counts | {"blobs": blob_count, "expires_at": expires_at.isoformat()},
        left_in_place=_CORPUS_TABLES,
        requested_at=requested_at,
    )
    return ExportOutcome(
        owner_id=owner_id,
        request_id=str(request_id),
        path=target,
        rows=counts,
        withheld_columns=withheld_by_table,
        left_in_place=_CORPUS_TABLES,
        blob_count=blob_count,
        expires_at=expires_at,
    )


def _export_owner_skills(root: Path | None, *, owner_id: str, target: Path) -> int:
    """Copy the owner's own Skills shard into the package, path for path."""
    if root is None:
        return 0
    shard = owner_skill_root(root, owner_id)
    if not shard.is_dir():
        return 0
    written = 0
    for path in sorted(shard.rglob("*")):
        if not path.is_file():
            continue
        destination = target / "skills" / path.relative_to(shard)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        written += 1
    return written


async def _export_blobs(conn: Any, *, owner_id: str, target: Path) -> int:
    """Reassemble the owner's content-addressed bytes, one file per digest."""
    digests = await conn.fetch(
        "SELECT digest FROM dlightrag_blobs WHERE owner_id = $1 ORDER BY digest", owner_id
    )
    written = 0
    for row in digests:
        digest = str(row["digest"])
        chunks = await conn.fetch(
            """
            SELECT content FROM dlightrag_blob_chunks
            WHERE owner_id = $1 AND digest = $2 ORDER BY chunk_index
            """,
            owner_id,
            digest,
        )
        if not chunks:
            continue
        (target / "blobs" / digest).write_bytes(b"".join(bytes(row["content"]) for row in chunks))
        written += 1
    return written


def require_customer_approval(*, line: str | None, customer_approval: str | None) -> str | None:
    """The reference an erase needs on the B2B line, or the refusal that says why.

    On B2C the requester is the data subject, so covering the Corpus Workspaces
    they created is their own call. On B2B the customer controls those Workspaces
    — they may be shared with people who never asked for anything — so deleting
    one needs the customer's approval, cited well enough for an auditor to find
    it. The check is here rather than in the flag parser because every surface
    that erases has to pass it.
    """
    reference = (customer_approval or "").strip()
    if line == "b2b" and not reference:
        raise OwnerEraseUnauthorized(
            "a b2b erase covers the Corpus Workspaces the customer's people created, "
            "so it needs the customer's approval: pass --customer-approval <ticket or "
            "email reference>"
        )
    return reference or None


async def erase_owner(
    conn: Any,
    *,
    owner_id: str,
    telemetry: TelemetryEraser | None,
    workspace_root: Path | None,
    owner_skills_root: Path | None,
    line: str | None = None,
    region: str | None = None,
    executor: str = "operator",
    corpora: CorpusDeletionDeliverer | None = None,
    keep_corpora: bool = False,
    customer_approval: str | None = None,
) -> EraseOutcome:
    """Remove everything one owner holds, then say what was left for the customer.

    The order is deliberate: rows first so no reader can resolve the owner, and
    the filesystem after, which leaves an interrupted erase as orphaned trees
    the reclaimer already knows how to sweep rather than as rows pointing at
    content that is gone.

    The Corpus Workspaces the requester created are covered by default — on B2C
    the creator is the data subject, and "delete everything about me" includes
    the knowledge base they built. `keep_corpora` keeps them instead, for a
    deployment that judges one a shared asset. Either way the erase delivers
    rather than deletes: each covered Workspace becomes a Workspace Delete Run on
    the Corpus Mutation lane (`corpora`), whose id goes in the receipt.
    """
    approval = require_customer_approval(line=line, customer_approval=customer_approval)
    request_id = uuid.uuid4()
    requested_at = datetime.now(UTC)
    busy = await conn.fetchval(
        """
        SELECT count(*) FROM dlightrag_runs
        WHERE owner_id = $1 AND status IN ('queued', 'running')
        """,
        owner_id,
    )
    if busy:
        await _write_receipt(
            conn,
            request_id=request_id,
            owner_id=owner_id,
            operation="erase",
            executor=executor,
            line=line,
            region=region,
            outcome="refused",
            counts={"in_flight_runs": int(busy)},
            left_in_place=(),
        )
        raise OwnerEraseRefused(
            f"owner {owner_id} has {int(busy)} Run(s) queued or running; cancel them "
            "and erase again, or the Run writes back what this removed"
        )

    deleted: dict[str, int] = {}
    workspace_roots = 0
    telemetry_rows = 0
    left_in_place = list(_CORPUS_TABLES)
    created: tuple[CorpusWorkspace, ...] = ()
    covered: tuple[str, ...] = ()
    deletion_runs: list[str] = []
    already_gone: list[str] = []
    try:
        created = await created_corpus_workspaces(conn, owner_id=owner_id)
        if corpora is not None and not keep_corpora:
            covered = tuple(item.workspace for item in created)
        elif created:
            # The receipt names what stayed, so "left in place" is a fact and not
            # an omission — the same reason a dry run names them.
            left_in_place.append(
                "corpus workspaces this requester created: "
                + ", ".join(item.workspace for item in created)
            )
        run_ids = [
            str(row["run_id"])
            for row in await conn.fetch(
                "SELECT run_id FROM dlightrag_runs WHERE owner_id = $1", owner_id
            )
        ]
        deleted = await _delete_owner_rows(conn, owner_id=owner_id)
        workspace_roots, orphans = _remove_workspace_roots(
            workspace_root, owner_id=owner_id, run_ids=run_ids
        )
        skills_removed = _remove_owner_skills(owner_skills_root, owner_id=owner_id)
        if orphans:
            # An orphan tree carries no Run row, so it cannot be attributed to this
            # owner: a shard is a hash prefix two owners can share. It is named
            # rather than deleted, and the reclaimer sweeps it on its own clock.
            left_in_place.append(f"{orphans} orphan tree(s) under the owner's shard")
        if telemetry is not None:
            telemetry_rows = await telemetry(owner_id=owner_id)
        # Last, so the delivery cannot be undone by the erase that asked for it:
        # a Workspace Delete Run is the deployment's act, submitted by whoever ran
        # the erase, and its own rows must survive this request.
        if corpora is not None:
            for workspace in covered:
                run_id = await corpora(workspace=workspace, submitted_by=executor)
                if run_id is None:
                    already_gone.append(workspace)
                else:
                    deletion_runs.append(run_id)
    except Exception as failure:
        # A request that fails is still a request: the ledger records what happened,
        # and what it recorded as removed is what a second run has to finish.
        await _write_failed_receipt(
            conn,
            request_id=request_id,
            owner_id=owner_id,
            executor=executor,
            line=line,
            region=region,
            deleted=deleted,
            left_in_place=left_in_place,
            corpora_deletion_runs=deletion_runs,
            corpora_already_gone=already_gone,
            error=failure,
        )
        raise
    await _write_receipt(
        conn,
        request_id=request_id,
        owner_id=owner_id,
        operation="erase",
        executor=executor,
        line=line,
        region=region,
        outcome="completed",
        counts=deleted
        | {
            "workspace_roots": workspace_roots,
            "owner_skill_trees": skills_removed,
            "telemetry_rows": telemetry_rows,
            # "skipped" is not "nothing to erase": the ledger says which it was.
            "telemetry": "skipped" if telemetry is None else "erased",
            "orphan_trees_left": orphans,
            "erase_due_at": (requested_at + timedelta(days=_ERASE_PROMISE_DAYS)).isoformat(),
            # The corpus deletions are delivered, not done: these Run ids are what
            # the 30-day promise is checked against once they reach a terminal state.
            "corpora_deletion_runs": deletion_runs,
            "corpora_already_gone": already_gone,
            "customer_approval": approval,
        },
        left_in_place=left_in_place,
        requested_at=requested_at,
    )
    return EraseOutcome(
        owner_id=owner_id,
        request_id=str(request_id),
        deleted=deleted,
        workspace_roots=workspace_roots,
        telemetry_rows=telemetry_rows,
        left_in_place=tuple(left_in_place),
        corpora_deletion_runs=tuple(deletion_runs),
        corpora_already_gone=tuple(already_gone),
    )


async def _write_failed_receipt(
    conn: Any,
    *,
    request_id: uuid.UUID,
    owner_id: str,
    executor: str,
    line: str | None,
    region: str | None,
    deleted: dict[str, int],
    left_in_place: Sequence[str],
    error: BaseException,
    corpora_deletion_runs: Sequence[str] = (),
    corpora_already_gone: Sequence[str] = (),
) -> None:
    """Record a failure without hiding it: the raise is what the operator acts on."""
    try:
        await _write_receipt(
            conn,
            request_id=request_id,
            owner_id=owner_id,
            operation="erase",
            executor=executor,
            line=line,
            region=region,
            outcome="failed",
            counts=deleted
            | {
                "error": f"{type(error).__name__}: {error}",
                # A delivery that happened before the failure is still a delivery.
                "corpora_deletion_runs": list(corpora_deletion_runs),
                "corpora_already_gone": list(corpora_already_gone),
            },
            left_in_place=tuple(left_in_place),
        )
    except Exception:
        logger.warning("Failed to record a failed erase for %s", owner_id, exc_info=True)


async def _delete_owner_rows(conn: Any, *, owner_id: str) -> dict[str, int]:
    """Delete every owner row, repeating passes so foreign keys order themselves."""
    planes = await owner_planes(conn)
    deleted: dict[str, int] = {}
    for _ in range(_ERASE_PASS_LIMIT):
        removed_this_pass = 0
        skipped = False
        for plane in planes:
            statement = f'DELETE FROM "{plane.table}" WHERE "{plane.column}" = $1'  # noqa: S608
            try:
                result = await conn.execute(statement, owner_id)
            except asyncpg.ForeignKeyViolationError:
                # A child table still references this row; the next pass has it gone,
                # so a pass that only skipped is not a pass that finished.
                skipped = True
                continue
            count = _affected(result)
            deleted[plane.table] = deleted.get(plane.table, 0) + count
            removed_this_pass += count
        if removed_this_pass == 0 and not skipped:
            return deleted
    raise RuntimeError(
        "an erase kept deleting after "
        f"{_ERASE_PASS_LIMIT} passes; a row is being recreated or a constraint blocks removal"
    )


def _affected(result: Any) -> int:
    """`DELETE n` as an int; anything else means no rows moved."""
    text = str(result or "")
    _, _, count = text.partition(" ")
    return int(count) if count.isdigit() else 0


async def created_corpus_workspaces(conn: Any, *, owner_id: str) -> tuple[CorpusWorkspace, ...]:
    """The Corpus Workspaces this owner created — the only corpus an erase may cover.

    Corpus documents belong to the customer, not to the person, so creation is the
    one relation that lets an erase reach a Workspace: membership would let an
    erase delete a colleague's knowledge base, and membership is not enumerable
    from here anyway. The document count comes with each row because a dry run has
    to say how much would go; nothing else reads it.
    """
    exists = await conn.fetchval(
        """
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'dlightrag_workspace_meta'
          AND column_name = 'created_by'
        """
    )
    if not exists:
        return ()
    rows = await conn.fetch(
        """
        SELECT workspace, ingested_docs_total FROM dlightrag_workspace_meta
        WHERE created_by = $1 ORDER BY workspace
        """,
        owner_id,
    )
    return tuple(
        CorpusWorkspace(
            workspace=str(row["workspace"]),
            documents=int(row["ingested_docs_total"] or 0),
        )
        for row in rows
    )


def _remove_workspace_roots(
    root: Path | None, *, owner_id: str, run_ids: Sequence[str]
) -> tuple[int, int]:
    """Remove the owner's Agent Workspace trees, and count the trees that stayed.

    A tree counts as removed only when it is gone afterwards: a sweep that quietly
    failed must not end up in a receipt that says the filesystem is clean.
    """
    if root is None:
        return 0, 0
    removed = 0
    for run_id in run_ids:
        path = run_root(root, owner_id, run_id)
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            logger.warning("Failed to remove Agent Workspace %s", path, exc_info=True)
        if not path.exists():
            removed += 1
    shard = root / owner_shard(owner_id)
    orphans = 0
    if shard.exists():
        known = set(run_ids)
        orphans = sum(1 for entry in shard.iterdir() if entry.is_dir() and entry.name not in known)
        if not orphans:
            try:
                shard.rmdir()
            except OSError:
                logger.warning("Failed to remove an empty Workspace shard %s", shard)
    return removed, orphans


def _remove_owner_skills(root: Path | None, *, owner_id: str) -> int:
    """Remove the owner's Skills shard, and report whether it is gone."""
    if root is None:
        return 0
    path = owner_skill_root(root, owner_id)
    if not path.exists():
        return 0
    try:
        shutil.rmtree(path)
    except OSError:
        logger.warning("Failed to remove owner Skills at %s", path, exc_info=True)
    return 0 if path.exists() else 1


def clickhouse_telemetry_eraser(
    *, url: str, user: str, password: str, timeout: float = 30.0
) -> TelemetryEraser:
    """A ClickHouse eraser for the deployment's own telemetry store.

    The predicate is the attribute the runtime writes — `user.id`, holding the
    owner id itself (`application/access/principal.py`) — and the id travels as
    a bound parameter, never as interpolated SQL. Traces carry it today; the log
    table is included because a log record that gains the attribute must be
    covered the day it does. Replay rows (`hyperdx_sessions`) carry no owner
    attribute at all: they are masked at the seam and age out under the
    deployment's telemetry TTL, which is why erasure cannot reach them.
    """
    tables = (
        ("default.otel_traces", "SpanAttributes"),
        ("default.otel_logs", "LogAttributes"),
    )

    async def erase(*, owner_id: str) -> int:
        removed = 0
        async with httpx.AsyncClient(timeout=timeout) as client:
            for table, attributes in tables:
                predicate = f"{attributes}['{USER_ID}'] = {{owner:String}}"
                counted = await client.post(
                    url,
                    params={
                        "query": f"SELECT count() FROM {table} WHERE {predicate}",  # noqa: S608 - constant table, bound owner
                        "param_owner": owner_id,
                    },
                    auth=(user, password),
                )
                counted.raise_for_status()
                removed += int(counted.text.strip() or 0)
                response = await client.post(
                    url,
                    params={
                        "query": (
                            f"ALTER TABLE {table} DELETE WHERE {predicate} "
                            "SETTINGS mutations_sync = 1"
                        ),
                        "param_owner": owner_id,
                    },
                    auth=(user, password),
                )
                response.raise_for_status()
        return removed

    return erase


async def _on_domain_pool[T](
    config: DlightragConfig, operation: Callable[[Any], Awaitable[T]]
) -> T:
    """Bind the process's domain pool to this config, run one operation, and close it.

    Every operation here is outcome-sensitive — a receipt, a deletion, a delivered
    Run — so it runs once (`run_once`) rather than being replayed on a transient
    error whose transaction may already have committed.
    """
    pg_pool.bind(config)
    try:
        return await pg_pool.run_once(operation)
    finally:
        await pg_pool.close()


async def _run_export(args: argparse.Namespace, config: DlightragConfig) -> None:
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome = await _on_domain_pool(
        config, lambda conn: export_owner(conn, owner_id=args.owner, out_dir=out_dir)
    )
    total = sum(outcome.rows.values())
    print(f"Exported owner {outcome.owner_id} to {outcome.path}: {total} row(s).")
    for table, count in sorted(outcome.rows.items()):
        if count:
            print(f"  {table}: {count}")
    print(f"  blobs: {outcome.blob_count}")
    for table, columns in sorted(outcome.withheld_columns.items()):
        print(f"  withheld from {table}: {', '.join(columns)}")
    print(f"Receipt: {outcome.request_id}")


async def _run_erase(args: argparse.Namespace, config: DlightragConfig) -> None:
    approval = require_customer_approval(line=args.line, customer_approval=args.customer_approval)
    if args.dry_run:
        await _report_erase_plan(args, config, approval=approval)
        return
    root = resolve_agentspace_root(
        execution_environment=config.answer.agent.execution_environment,
        workspace_root=config.answer.agent.workspace_root,
    )
    skills_root_value = config.answer.agent.owner_skills_root
    skills_root = Path(skills_root_value).expanduser().resolve() if skills_root_value else None
    telemetry: TelemetryEraser | None = None
    if not args.no_telemetry:
        settings = config.observability.telemetry_erasure
        if settings.url is None or settings.user is None or settings.password is None:
            raise SystemExit(
                "Telemetry erasure is not configured, so an erase would leave the owner's "
                "spans behind. Configure observability.telemetry_erasure with "
                "a delete-capable ClickHouse user, or pass --no-telemetry for a deployment "
                "that keeps no telemetry."
            )
        telemetry = clickhouse_telemetry_eraser(
            url=settings.url, user=settings.user, password=settings.password
        )
    pg_pool.bind(config)
    try:
        # A reader registers no Corpus Mutation executor and holds no corpus of its own,
        # so an erase there names the Workspaces it cannot reach instead of accepting
        # Runs nothing would ever run.
        corpora: CorpusDeletionDeliverer | None = (
            None
            if config.is_reader
            else corpus_deletion_deliverer(pool=await pg_pool.get(), config=config)
        )
        outcome = await pg_pool.run_once(
            lambda conn: erase_owner(
                conn,
                owner_id=args.owner,
                telemetry=telemetry,
                workspace_root=root,
                owner_skills_root=skills_root,
                line=args.line,
                region=args.region,
                executor=args.executor,
                corpora=corpora,
                keep_corpora=args.keep_corpora,
                customer_approval=approval,
            )
        )
    finally:
        await pg_pool.close()
    removed = sum(outcome.deleted.values())
    print(
        f"Erased owner {outcome.owner_id}: {removed} row(s), {outcome.workspace_roots} workspace root(s)."
    )
    for table, count in sorted(outcome.deleted.items()):
        if count:
            print(f"  {table}: {count}")
    print(f"  telemetry rows: {outcome.telemetry_rows}")
    print(f"  left in place (the customer's corpus): {', '.join(outcome.left_in_place)}")
    for run_id in outcome.corpora_deletion_runs:
        # The Run is the deletion; the receipt is the promise that one was asked for.
        print(f"  corpus deletion delivered as Run {run_id} — its terminal state is the promise")
    if outcome.corpora_deletion_runs:
        print(
            "  the deletions are delivered, not done: check those Runs reach a terminal "
            "state before the receipt's erase_due_at"
        )
    for workspace in outcome.corpora_already_gone:
        print(f"  corpus workspace {workspace} was already gone")
    print(f"Receipt: {outcome.request_id}")


async def _report_erase_plan(
    args: argparse.Namespace, config: DlightragConfig, *, approval: str | None
) -> None:
    """Say which Corpus Workspaces an erase would cover, and remove nothing."""
    created = await _on_domain_pool(
        config, lambda conn: created_corpus_workspaces(conn, owner_id=args.owner)
    )
    print(f"Dry run for owner {args.owner}: nothing was removed and no receipt was written.")
    if approval is not None:
        print(f"  customer approval on file: {approval}")
    if not created:
        print("  the requester created no Corpus Workspace, so there is none to cover.")
        return
    if args.keep_corpora:
        print(f"  --keep-corpora: these {len(created)} Corpus Workspace(s) would stay:")
    else:
        print(
            f"  an erase would deliver one Workspace Delete Run for each of these "
            f"{len(created)} Corpus Workspace(s) the requester created:"
        )
    for item in created:
        print(f"    {item.workspace}: {item.documents} document(s)")


async def _run_sweep_ledger(
    swept: Sequence[SweptPackage], *, executor: str, config: DlightragConfig
) -> list[str]:
    """Write the sweep's ledger rows through the domain pool, and return their ids."""
    return await _on_domain_pool(
        config, lambda conn: record_sweep_receipts(conn, executor=executor, swept=swept)
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `dlightrag-owner-data`."""
    parser = argparse.ArgumentParser(
        description=(
            "Export or erase everything one owner holds, with a receipt (ADR 0049). "
            "Erase refuses while the owner has Runs in flight."
        ),
        suggest_on_error=True,
    )
    parser.add_argument("--env-file", help="Path to .env configuration file")
    parser.add_argument(
        "--owner",
        help="The owner id as the runtime projects it (sha256 for jwt deployments).",
    )
    parser.add_argument("--principal", help="A jwt subject, to derive --owner from.")
    parser.add_argument("--issuer", help="The subject's issuer, with --principal.")
    subparsers = parser.add_subparsers(dest="operation", required=True)

    export = subparsers.add_parser("export", help="Write the owner's data to a directory.")
    export.add_argument("--out", required=True, help="Directory to write the package into.")

    sweep = subparsers.add_parser(
        "sweep-expired",
        help="Remove handover packages past their expiry, naming every package it judged.",
    )
    sweep.add_argument("--out", required=True, help="Directory holding the packages.")
    sweep.add_argument(
        "--dry-run", action="store_true", help="List what would go, and remove nothing."
    )
    sweep.add_argument(
        "--executor",
        default="operator",
        help="Who ran it; recorded on the sweep receipt of every package.",
    )

    erase = subparsers.add_parser(
        "erase",
        help=(
            "Remove the owner's data, covering the Corpus Workspaces they created "
            "unless --keep-corpora."
        ),
    )
    erase.add_argument(
        "--line",
        required=True,
        choices=("b2c", "b2b"),
        help=(
            "Which service line the request came through; b2b needs --customer-approval, "
            "because the customer controls the Corpus Workspaces its people created."
        ),
    )
    erase.add_argument(
        "--customer-approval",
        help="The customer's reference (ticket or email id) authorizing a b2b erase.",
    )
    erase.add_argument("--region", help="Which deployment region served it.")
    erase.add_argument("--executor", default="operator", help="Who ran it, for the receipt.")
    erase.add_argument(
        "--keep-corpora",
        action="store_true",
        help="Keep the Corpus Workspaces the requester created, and name them instead.",
    )
    erase.add_argument(
        "--dry-run",
        action="store_true",
        help="List the Corpus Workspaces an erase would cover, and remove nothing.",
    )
    erase.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Leave telemetry to its TTL; only for a deployment that keeps none.",
    )
    return parser


def resolve_owner(args: argparse.Namespace) -> str:
    """The owner id from `--owner`, or derived from `--principal` and `--issuer`."""
    if bool(args.principal) != bool(args.issuer):
        raise SystemExit("--principal and --issuer are given together")
    if args.owner and args.principal:
        raise SystemExit("give --owner or --principal/--issuer, not both")
    if args.owner:
        return str(args.owner)
    if args.principal:
        return owner_id_from_principal(
            auth_mode="jwt", user_id=str(args.principal), issuer=str(args.issuer)
        )
    raise SystemExit("give --owner, or --principal with --issuer")


def main() -> None:
    """Entry point for `dlightrag-owner-data`."""
    args = build_parser().parse_args()
    if args.operation != "sweep-expired":  # a sweep is owner-agnostic
        args.owner = resolve_owner(args)
    config = load_config(args.env_file) if args.env_file else get_config()
    set_config(config)
    logging.basicConfig(
        level=getattr(logging, config.observability.log_level.upper(), logging.INFO)
    )
    if args.operation == "sweep-expired":
        out = Path(args.out).expanduser().resolve()
        if not out.is_dir():
            raise SystemExit(f"{out} is not a directory, so nothing was looked at")
        swept = sweep_expired_exports(out, dry_run=args.dry_run)
        verb = "would go" if args.dry_run else "removed"
        for package in swept:
            state = "removed" if package.removed else "left in place"
            print(
                f"  {package.request_id} (owner {package.owner_id}) "
                f"expired {package.expires_at.isoformat()}: {state if not args.dry_run else verb}"
            )
        print(f"{len(swept)} package(s) past expiry under {out}.")
        problems: list[str] = []
        if swept and not args.dry_run:
            # One row per package, written after the removals: the container log
            # rotates, and this is the evidence a DPA audit reads. A dry run writes
            # nothing — a removal that was not performed was not promised.
            try:
                receipt_ids = asyncio.run(
                    _run_sweep_ledger(swept, executor=args.executor, config=config)
                )
                print(f"  {len(receipt_ids)} sweep receipt(s) written to the ledger.")
            except Exception as failure:
                problems.append(
                    f"the sweep ledger could not be written ({type(failure).__name__}: {failure})"
                )
        failed = [item for item in swept if not item.removed and not args.dry_run]
        if failed:
            problems.append(
                f"{len(failed)} package(s) outlived their expiry and could not be removed"
            )
        if problems:
            raise SystemExit("; ".join(problems))
        return

    if args.operation == "export":
        asyncio.run(_run_export(args, config))
        return
    try:
        asyncio.run(_run_erase(args, config))
    except OwnerEraseUnauthorized as refusal:
        # The rule is the operation's; the exit status an operator reads is the CLI's.
        raise SystemExit(str(refusal)) from None


__all__ = [
    "CorpusDeletionDeliverer",
    "CorpusWorkspace",
    "EraseOutcome",
    "ExportOutcome",
    "OwnerEraseRefused",
    "OwnerEraseUnauthorized",
    "OwnerPlane",
    "TelemetryEraser",
    "build_parser",
    "clickhouse_telemetry_eraser",
    "created_corpus_workspaces",
    "erase_owner",
    "export_owner",
    "main",
    "owner_planes",
    "record_sweep_receipts",
    "require_customer_approval",
    "resolve_owner",
]
