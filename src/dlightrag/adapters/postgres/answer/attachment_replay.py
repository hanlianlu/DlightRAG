# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Host-settled Entry occurrence references, retained without historical capabilities."""

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from dlightrag.engine.agent.session.entries import SessionEntry, ToolResultMessageEntry
from dlightrag.engine.agent.tool_content import ToolResourceAttachmentPart, encode_tool_content
from dlightrag.engine.answer.attachment_replay import (
    AttachmentOccurrence,
    AttachmentReplaySelection,
)
from dlightrag.engine.runtime.coordinator import LeaseLostError
from dlightrag.engine.runtime.records import RunFetchedResource
from dlightrag.engine.runtime.settlements import EffectHostUpdate

_INSERT_OCCURRENCE = """
INSERT INTO dlightrag_answer_resources (
    owner_id, run_id, resource_id, kind, safe_name, media_type, capabilities,
    ordinal, blob_digest, locator_digest, source_locator, session_id, intent_id
)
VALUES ($1, $2, $3, 'fetched_blob', $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12)
ON CONFLICT (owner_id, run_id, resource_id) DO NOTHING
"""

_SELECT_OCCURRENCE = """
SELECT run_id, resource_id, safe_name, media_type, capabilities, ordinal,
       blob_digest, locator_digest, source_locator, session_id, intent_id
FROM dlightrag_answer_resources
WHERE owner_id = $1 AND resource_id = $2
  AND kind = 'fetched_blob'
  AND capabilities->>'resource_kind' = 'attachment_occurrence'
  AND (NOT $4::boolean OR run_id = $3)
ORDER BY (run_id = $3) DESC, created_at, run_id
FOR SHARE
"""


async def _lock_consuming_run(
    conn: Any, owner_id: str, run_id: uuid.UUID, worker_id: str, fencing_epoch: int
) -> None:
    held = await conn.fetchval(
        "SELECT 1 FROM dlightrag_runs WHERE owner_id=$1 AND run_id=$2"
        " AND lease_owner=$3 AND fencing_epoch=$4 AND status='running'"
        " AND lease_expires_at > NOW() FOR UPDATE",
        owner_id,
        run_id,
        worker_id,
        fencing_epoch,
    )
    if held is None:
        raise LeaseLostError


async def _ancestry_entries(
    conn: Any, owner_id: str, session_id: uuid.UUID, head_entry_id: uuid.UUID
) -> dict[str, Any]:
    lineage = await conn.fetch(
        "WITH RECURSIVE ancestry AS ("
        " SELECT entry_id, parent_entry_id, entry_type, payload_json"
        " FROM dlightrag_agent_session_entries WHERE owner_id=$1 AND session_id=$2 AND entry_id=$3"
        " UNION ALL SELECT entry.entry_id, entry.parent_entry_id, entry.entry_type, entry.payload_json"
        " FROM dlightrag_agent_session_entries entry JOIN ancestry ON entry.entry_id=ancestry.parent_entry_id"
        " WHERE entry.owner_id=$1 AND entry.session_id=$2)"
        " SELECT entry_id, entry_type, payload_json FROM ancestry",
        owner_id,
        session_id,
        head_entry_id,
    )
    return {str(row["entry_id"]): row for row in lineage}


async def _occurrence_rows(
    conn: Any,
    owner_id: str,
    reference_id: str,
    run_id: uuid.UUID,
    *,
    current_run_only: bool,
) -> Sequence[Any]:
    # Retention can adopt authorized ancestry across Runs. Child hydration must
    # only load already-adopted rows; keep that restriction explicit at each call.
    return await conn.fetch(_SELECT_OCCURRENCE, owner_id, reference_id, run_id, current_run_only)


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _attachment_bytes(part: ToolResourceAttachmentPart) -> bytes:
    return json.dumps(
        encode_tool_content((part,))[0], sort_keys=True, separators=(",", ":")
    ).encode()


async def write_attachment_occurrences(
    conn: Any,
    *,
    owner_id: str,
    run_id: uuid.UUID,
    entries: Sequence[SessionEntry],
    update: EffectHostUpdate | None,
) -> None:
    """Bind each returned part to THIS atomic Host update, not a prior dedup intent."""
    for entry in entries:
        if not isinstance(entry, ToolResultMessageEntry):
            continue
        for index, part in enumerate(entry.result.parts):
            if not isinstance(part, ToolResourceAttachmentPart):
                continue
            writes = () if update is None else update.fetched
            fetched = next(
                (item for item in writes if item.resource.resource_id == part.resource_id), None
            )
            if (
                fetched is None
                or entry.intent_id is None
                or fetched.resource.session_id != entry.session_id.value
                or fetched.resource.intent_id != entry.intent_id.value
                or fetched.resource.blob_digest != part.content_digest
                or fetched.complete_blob.digest != part.content_digest
                or fetched.complete_blob.total_bytes != part.size_bytes
                or fetched.resource.media_type != part.media_type
                or fetched.resource.safe_name != part.safe_name
                or fetched.resource.capabilities.get("visual_source")
                != (asdict(part.source) if part.source is not None else None)
            ):
                raise ValueError("attachment occurrence does not match its Host settlement")
            occurrence = AttachmentOccurrence(entry.entry_id.value, index, part)
            locator = _attachment_bytes(part)
            capabilities = {
                "resource_kind": "attachment_occurrence",
                "origin_run_id": str(run_id),
                "entry_id": entry.entry_id.value,
                "part_index": index,
            }
            await conn.execute(
                _INSERT_OCCURRENCE,
                owner_id,
                run_id,
                occurrence.reference_id,
                part.safe_name,
                part.media_type,
                json.dumps(capabilities),
                index,
                part.content_digest,
                hashlib.sha256(locator).hexdigest(),
                locator,
                uuid.UUID(entry.session_id.value),
                uuid.UUID(entry.intent_id.value),
            )
            rows = await _occurrence_rows(
                conn, owner_id, occurrence.reference_id, run_id, current_run_only=False
            )
            stored = next((row for row in rows if row["run_id"] == run_id), None)
            if (
                stored is None
                or _json(stored["capabilities"]) != capabilities
                or bytes(stored["source_locator"]) != locator
            ):
                raise ValueError("attachment occurrence identity conflict")


async def retain_attachment_occurrences(
    conn: Any,
    *,
    owner_id: str,
    run_id: uuid.UUID,
    worker_id: str,
    fencing_epoch: int,
    selection: AttachmentReplaySelection,
) -> tuple[RunFetchedResource, ...]:
    """Authorize selected ancestry and retain every requested reference in one transaction.

    Lock the consuming Run before Session/registers and source reference rows. A
    source deletion either wins (explicit missing-reference failure), or waits
    until the consuming reference commits. No original-Run FK is introduced.
    """
    async with conn.transaction():
        await _lock_consuming_run(conn, owner_id, run_id, worker_id, fencing_epoch)
        routing = await conn.fetchrow(
            "SELECT agent_session_id, agent_lane_id, source_lane_id"
            " FROM dlightrag_answer_run_routing WHERE owner_id=$1 AND run_id=$2",
            owner_id,
            run_id,
        )
        if routing is None or str(routing["agent_session_id"]) != selection.session_id:
            raise ValueError("attachment replay Session is not selected by this Run")
        session_id = uuid.UUID(selection.session_id)
        await conn.fetchval(
            "SELECT 1 FROM dlightrag_agent_sessions WHERE owner_id=$1 AND session_id=$2 FOR SHARE",
            owner_id,
            session_id,
        )
        lanes = await conn.fetch(
            "SELECT register_key, payload_json FROM dlightrag_agent_session_registers"
            " WHERE owner_id=$1 AND session_id=$2 AND register_kind='lane_head' FOR SHARE",
            owner_id,
            session_id,
        )
        heads = {str(row["register_key"]): _json(row["payload_json"])["entry_id"] for row in lanes}
        selected_lane = next(
            (
                lane
                for lane in (routing["agent_lane_id"], routing["source_lane_id"], "main")
                if lane in heads
            ),
            None,
        )
        if (
            selected_lane != selection.lane_id
            or heads.get(selection.lane_id) != selection.head_entry_id
        ):
            raise ValueError("attachment replay selected Lane Head changed")
        entries = await _ancestry_entries(
            conn, owner_id, session_id, uuid.UUID(selection.head_entry_id)
        )
        results: list[RunFetchedResource] = []
        seen: set[str] = set()
        for occurrence in sorted(selection.occurrences, key=lambda item: item.reference_id):
            if occurrence.reference_id in seen:
                raise ValueError("duplicate attachment occurrence request")
            seen.add(occurrence.reference_id)
            entry = entries.get(occurrence.entry_id)
            part = occurrence.attachment
            rows = await _occurrence_rows(
                conn, owner_id, occurrence.reference_id, run_id, current_run_only=False
            )
            reference = _validate_occurrence(occurrence, session_id, entry, rows)
            locator = reference.source_locator
            locator_digest = hashlib.sha256(locator).hexdigest()
            row = rows[0]
            # The existing owner-scoped Blob FK protects the retained bytes after
            # this insert, including when cleanup deletes the original Run.
            await conn.execute(
                _INSERT_OCCURRENCE,
                owner_id,
                run_id,
                occurrence.reference_id,
                part.safe_name,
                part.media_type,
                json.dumps(_json(row["capabilities"])),
                occurrence.part_index,
                part.content_digest,
                locator_digest,
                locator,
                session_id,
                row["intent_id"],
            )
            results.append(reference)
        return tuple(results)


def _validate_occurrence(
    occurrence: AttachmentOccurrence,
    session_id: uuid.UUID,
    entry: Any,
    rows: Sequence[Any],
) -> RunFetchedResource:
    """One closed Entry/part/source/Blob binding for adoption and pinned hydration."""
    part = occurrence.attachment
    payload = {} if entry is None else _json(entry["payload_json"])
    parts = payload.get("content", [])
    if (
        entry is None
        or entry["entry_type"] != "tool_result"
        or occurrence.part_index < 0
        or occurrence.part_index >= len(parts)
        or parts[occurrence.part_index] != encode_tool_content((part,))[0]
    ):
        raise ValueError("attachment occurrence is not on the selected Entry ancestry")
    if not rows:
        raise ValueError("authorized attachment occurrence reference is missing")
    locator = _attachment_bytes(part)
    origin: str | None = None
    for row in rows:
        capabilities = _json(row["capabilities"])
        candidate_origin = capabilities.get("origin_run_id")
        if (
            not candidate_origin
            or (origin is not None and origin != candidate_origin)
            or capabilities
            != {
                "resource_kind": "attachment_occurrence",
                "origin_run_id": candidate_origin,
                "entry_id": occurrence.entry_id,
                "part_index": occurrence.part_index,
            }
            or row["session_id"] != session_id
            or str(row["intent_id"]) != payload.get("intent_id")
            or row["blob_digest"] != part.content_digest
            or row["safe_name"] != part.safe_name
            or row["media_type"] != part.media_type
            or row["ordinal"] != occurrence.part_index
            or row["locator_digest"] != hashlib.sha256(locator).hexdigest()
            or bytes(row["source_locator"]) != locator
        ):
            raise ValueError("retained attachment occurrence binding mismatch")
        uuid.UUID(candidate_origin)
        origin = candidate_origin
    return RunFetchedResource(
        resource_id=part.resource_id,
        ordinal=occurrence.part_index,
        digest=part.content_digest,
        filename=part.safe_name,
        mime_type=part.media_type,
        source_locator=locator,
        capabilities=_json(rows[0]["capabilities"]),
    )


async def load_child_attachment_occurrences(
    conn: Any,
    *,
    owner_id: str,
    run_id: uuid.UUID,
    worker_id: str,
    fencing_epoch: int,
    child_session_id: uuid.UUID,
    context_snapshot: dict[str, Any],
) -> tuple[RunFetchedResource, ...]:
    """Read only THIS Run's already-adopted refs in the exact accepted Child pin.

    The parent projection may have compacted these entries away. This read does
    not adopt historical rows, search other Runs, or issue Resource capabilities.
    """
    async with conn.transaction():
        await _lock_consuming_run(conn, owner_id, run_id, worker_id, fencing_epoch)
        child = await conn.fetchrow(
            "SELECT parent_session_id, context_snapshot_json FROM dlightrag_answer_child_sessions"
            " WHERE owner_id=$1 AND run_id=$2 AND child_session_id=$3 FOR SHARE",
            owner_id,
            run_id,
            child_session_id,
        )
        if (
            child is None
            or _json(child["context_snapshot_json"]) != context_snapshot
            or str(child["parent_session_id"]) != context_snapshot.get("parent_session_id")
        ):
            raise ValueError("Child attachment context is not the accepted pin")
        session_id = child["parent_session_id"]
        entries = await _ancestry_entries(
            conn, owner_id, session_id, uuid.UUID(context_snapshot["parent_entry_id"])
        )
        results: list[RunFetchedResource] = []
        seen: set[str] = set()
        for raw in context_snapshot["attachment_occurrences"]:
            occurrence = AttachmentOccurrence.from_payload(raw)
            if occurrence.reference_id in seen:
                raise ValueError("duplicate Child attachment occurrence pin")
            seen.add(occurrence.reference_id)
            rows = await _occurrence_rows(
                conn, owner_id, occurrence.reference_id, run_id, current_run_only=True
            )
            results.append(
                _validate_occurrence(occurrence, session_id, entries.get(occurrence.entry_id), rows)
            )
        return tuple(results)
