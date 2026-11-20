# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner export and erasure: coverage, withholding, refusal and the receipt."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest

from dlightrag.adapters.postgres.owner_data import (
    OwnerEraseRefused,
    OwnerEraseUnauthorized,
    build_parser,
    clickhouse_telemetry_eraser,
    erase_owner,
    export_owner,
    owner_planes,
    resolve_owner,
)
from dlightrag.application.access.principal import owner_id_from_principal

_OWNER = "a" * 64

_PLANES: dict[str, tuple[str, ...]] = {
    "dlightrag_runs": ("run_id", "owner_id", "status"),
    "dlightrag_memory_records": ("memory_id", "owner_id", "body"),
    "dlightrag_agent_accounts": ("account_id", "owner_id", "site", "encrypted_envelope"),
    "dlightrag_connection_heads": ("connection_id", "owner_id", "label", "endpoint_headers"),
    "web_conversations": ("conversation_id", "principal_id", "title"),
    "web_conversation_turns": ("turn_id", "principal_id", "conversation_id"),
    "web_conversation_commands": ("command_id", "principal_id", "conversation_id"),
    "dlightrag_owner_data_requests": ("request_id", "owner_id", "operation"),
    "dlightrag_blobs": ("owner_id", "digest", "byte_size"),
    "dlightrag_blob_chunks": ("owner_id", "digest", "chunk_index", "content"),
}

_ROWS: dict[str, list[dict[str, Any]]] = {
    "dlightrag_runs": [{"run_id": "run-1", "owner_id": _OWNER, "status": "succeeded"}],
    "dlightrag_memory_records": [
        {"memory_id": "m-1", "owner_id": _OWNER, "body": "The owner prefers Chinese."}
    ],
    "dlightrag_agent_accounts": [
        {
            "account_id": "acct-1",
            "owner_id": _OWNER,
            "site": "example.com",
            "encrypted_envelope": b"sealed",
        }
    ],
    "dlightrag_connection_heads": [
        {
            "connection_id": "conn-1",
            "owner_id": _OWNER,
            "label": "Notion",
            "endpoint_headers": {"Authorization": "Bearer x"},
        }
    ],
    "web_conversations": [{"conversation_id": "c-1", "principal_id": _OWNER, "title": "hi"}],
    "web_conversation_commands": [
        {"command_id": "cmd-1", "principal_id": _OWNER, "conversation_id": "c-1"}
    ],
    "dlightrag_blobs": [{"owner_id": _OWNER, "digest": "d" * 64, "byte_size": 3}],
    "dlightrag_blob_chunks": [
        {"owner_id": _OWNER, "digest": "d" * 64, "chunk_index": 0, "content": b"abc"}
    ],
}


class FakeConnection:
    """Answers the queries the two operations make, and records what it was asked.

    Deletes decay: a table that held rows reports them removed once, then reports
    nothing left, which is what makes the erase loop terminate the way the real one does.
    """

    def __init__(
        self,
        *,
        in_flight: int = 0,
        rows: dict[str, int] | None = None,
        fk_once: str | None = None,
        workspaces: tuple[str, ...] = (),
        documents: int = 0,
    ) -> None:
        self.in_flight = in_flight
        self.workspaces = workspaces
        self.documents = documents
        self.remaining = dict(rows or {})
        self.fk_once = fk_once
        self.statements: list[str] = []
        self.receipts: list[tuple[Any, ...]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        if "information_schema.columns" in query and "column_name = 'owner_id'" in query:
            return [
                {"table_name": table} for table, columns in _PLANES.items() if "owner_id" in columns
            ]
        if "information_schema.columns" in query and "table_name = $1" in query:
            return [{"column_name": column} for column in _PLANES.get(str(args[0]), ())]
        if "FROM dlightrag_blobs" in query:
            return list(_ROWS.get("dlightrag_blobs", ()))
        if "FROM dlightrag_blob_chunks" in query:
            return list(_ROWS.get("dlightrag_blob_chunks", ()))
        if "SELECT run_id FROM dlightrag_runs" in query:
            return [{"run_id": "run-1"}]
        if "FROM dlightrag_workspace_meta" in query:
            return [
                {"workspace": name, "ingested_docs_total": self.documents}
                for name in self.workspaces
            ]
        for table in _PLANES:
            if f'FROM "{table}"' in query:
                selected = re.findall(r'"([a-z_]+)"', query.split(" FROM ")[0])
                return [
                    {column: row[column] for column in selected if column in row}
                    for row in _ROWS.get(table, ())
                ]
        return []

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.statements.append(query)
        if "FROM dlightrag_runs" in query:
            return self.in_flight
        if "information_schema.columns" in query:
            if "table_name = 'dlightrag_workspace_meta'" in query:
                return 1  # the corpus table exists and records its creator
            table, column = str(args[0]), str(args[1])
            return 1 if column in _PLANES.get(table, ()) else None
        return None

    async def execute(self, query: str, *args: Any) -> str:
        self.statements.append(query)
        if query.strip().startswith("DELETE FROM"):
            table = query.split('"')[1]
            if self.fk_once == table:
                self.fk_once = None
                raise asyncpg.ForeignKeyViolationError("child rows still reference this owner")
            left = self.remaining.get(table, 0)
            if left <= 0:
                return "DELETE 0"
            self.remaining[table] = left - 1
            return "DELETE 1"
        if query.strip().startswith("INSERT INTO dlightrag_owner_data_requests"):
            self.receipts.append(args)
        return "INSERT 0 1"


async def _no_telemetry(*, owner_id: str) -> int:
    return 0


def _parser_args(*argv: str) -> Any:
    return build_parser().parse_args(list(argv))


async def test_discovery_covers_owner_columns_and_the_principal_tables() -> None:
    """What the schema can name is found; what it cannot is written down, not forgotten."""
    planes = await owner_planes(FakeConnection())
    tables = {plane.table for plane in planes}
    assert "dlightrag_agent_accounts" in tables
    assert {"web_conversations", "web_conversation_turns", "web_conversation_commands"} <= tables
    # The record of an erasure is not itself erased.
    assert "dlightrag_owner_data_requests" not in tables


async def test_export_withholds_credentials_and_says_so(tmp_path: Path) -> None:
    conn = FakeConnection()
    outcome = await export_owner(conn, owner_id=_OWNER, out_dir=tmp_path)
    manifest = json.loads((outcome.path / "manifest.json").read_text())
    assert manifest["withheld_columns"]["dlightrag_agent_accounts"] == ["encrypted_envelope"]
    assert manifest["withheld_columns"]["dlightrag_connection_heads"] == ["endpoint_headers"]
    assert "sealed" not in (outcome.path / "dlightrag_agent_accounts.jsonl").read_text()
    assert "Bearer x" not in (outcome.path / "dlightrag_connection_heads.jsonl").read_text()
    assert (
        "The owner prefers Chinese."
        in (outcome.path / "dlightrag_memory_records.jsonl").read_text()
    )
    assert manifest["left_in_place"] == list(outcome.left_in_place)
    # The request ledger is neither exported as content nor erased with it.
    assert "dlightrag_owner_data_requests" not in outcome.rows


async def test_export_includes_persisted_command_events(tmp_path: Path) -> None:
    """A command is the owner's transcript data; the export carries it like a turn."""
    outcome = await export_owner(FakeConnection(), owner_id=_OWNER, out_dir=tmp_path)

    assert outcome.rows["web_conversation_commands"] == 1
    exported = (outcome.path / "web_conversation_commands.jsonl").read_text()
    assert '"command_id": "cmd-1"' in exported


def test_an_export_states_when_the_package_must_go(tmp_path: Path) -> None:
    outcome = asyncio.run(
        export_owner(FakeConnection(), owner_id=_OWNER, out_dir=tmp_path / "packages")
    )
    manifest = json.loads((outcome.path / "manifest.json").read_text())
    assert manifest["expires_at"] == outcome.expires_at.isoformat()
    assert outcome.expires_at > datetime.now(UTC) + timedelta(days=6)


async def test_export_reassembles_blob_bytes_by_digest(tmp_path: Path) -> None:
    outcome = await export_owner(FakeConnection(), owner_id=_OWNER, out_dir=tmp_path)
    assert (outcome.path / "blobs" / ("d" * 64)).read_bytes() == b"abc"
    assert outcome.blob_count == 1


async def test_erase_refuses_while_the_owner_has_runs_in_flight(tmp_path: Path) -> None:
    conn = FakeConnection(in_flight=2)
    with pytest.raises(OwnerEraseRefused):
        await erase_owner(
            conn,
            owner_id=_OWNER,
            telemetry=_no_telemetry,
            workspace_root=tmp_path,
            owner_skills_root=tmp_path / "skills",
        )
    assert any(
        "INSERT INTO dlightrag_owner_data_requests" in statement for statement in conn.statements
    )
    assert conn.receipts[0][6] == "refused"
    assert conn.receipts[0][7] == json.dumps({"in_flight_runs": 2})


async def test_erase_repeats_until_a_pass_changes_nothing(tmp_path: Path) -> None:
    """A foreign key is an ordering problem: the next pass finds the parent gone."""
    conn = FakeConnection(
        rows={table: 1 for table in _PLANES if table != "dlightrag_owner_data_requests"}
    )
    erased: list[str] = []

    async def telemetry(*, owner_id: str) -> int:
        erased.append(owner_id)
        return 9

    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=telemetry,
        workspace_root=tmp_path,
        owner_skills_root=tmp_path / "skills",
        line="b2c",
        region="eu",
    )
    deletes = [s for s in conn.statements if s.startswith("DELETE FROM")]
    planes = len(_PLANES) - 1  # the request ledger is not erased
    assert len(deletes) > planes  # the last pass deleted nothing
    assert outcome.telemetry_rows == 9
    assert erased == [_OWNER]
    assert outcome.left_in_place  # the customer's corpus stays
    receipt = conn.receipts[-1]
    assert (receipt[2], receipt[4], receipt[5], receipt[6]) == ("erase", "b2c", "eu", "completed")
    assert json.loads(receipt[8]) == list(outcome.left_in_place)
    counts = json.loads(receipt[7])
    assert counts["telemetry"] == "erased" and counts["telemetry_rows"] == 9
    assert counts["owner_skill_trees"] == 0
    # The one number the DPA states: the deadline the whole promise fits inside.
    assert counts["erase_due_at"] > receipt[9].isoformat()


async def test_a_failed_erase_leaves_a_receipt_that_says_so(tmp_path: Path) -> None:
    """A request that fails is still a request: the ledger must not stay silent."""
    conn = FakeConnection(rows={"dlightrag_runs": 1})

    async def broken_telemetry(*, owner_id: str) -> int:
        raise RuntimeError("clickhouse refused the mutation")

    with pytest.raises(RuntimeError):
        await erase_owner(
            conn,
            owner_id=_OWNER,
            telemetry=broken_telemetry,
            workspace_root=tmp_path,
            owner_skills_root=tmp_path / "skills",
        )
    receipt = conn.receipts[-1]
    assert receipt[6] == "failed"
    counts = json.loads(receipt[7])
    assert counts["dlightrag_runs"] == 1  # what did happen is recorded
    assert "clickhouse refused the mutation" in counts["error"]


async def test_an_erase_names_the_corpus_workspaces_the_requester_created() -> None:
    """A customer's corpus is the customer's decision, so the receipt says which ones."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts", "shared-kb"))
    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
    )
    named = [entry for entry in outcome.left_in_place if "created" in entry]
    assert named and "acme-contracts" in named[0] and "shared-kb" in named[0]


class FakeCorpusDeletions:
    """The Corpus Mutation port, narrowed to what an erase says to it."""

    def __init__(self, *, already_gone: tuple[str, ...] = ()) -> None:
        self.already_gone = already_gone
        self.delivered: list[str] = []
        self.submitters: list[str] = []

    async def __call__(self, *, workspace: str, submitted_by: str) -> str | None:
        self.submitters.append(submitted_by)
        if workspace in self.already_gone:
            return None
        self.delivered.append(workspace)
        return f"run-{workspace}"


async def test_the_corpus_workspaces_the_requester_created_are_covered_by_default() -> None:
    """On B2C the creator is the data subject: erasing them covers what they built."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts", "shared-kb"))
    deletions = FakeCorpusDeletions()
    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
        corpora=deletions,
        line="b2c",
    )
    assert deletions.delivered == ["acme-contracts", "shared-kb"]
    assert outcome.corpora_deletion_runs == ("run-acme-contracts", "run-shared-kb")
    counts = json.loads(conn.receipts[-1][7])
    assert counts["corpora_deletion_runs"] == ["run-acme-contracts", "run-shared-kb"]
    assert counts["customer_approval"] is None
    # A Workspace being deleted is not also "left in place": two claims that disagree
    # are worse than none.
    assert not [entry for entry in outcome.left_in_place if "created" in entry]
    # Creation, not membership, is what an erase may act on: the created_by listing is
    # the only corpus listing it makes, and exactly that listing's rows are delivered.
    listings = [s for s in conn.statements if "FROM dlightrag_workspace_meta" in s]
    assert listings and all("created_by = $1" in statement for statement in listings)
    assert deletions.delivered == ["acme-contracts", "shared-kb"]


async def test_keep_corpora_names_them_and_delivers_nothing() -> None:
    """A Workspace judged a shared asset stays, and the receipt says so."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts",))
    deletions = FakeCorpusDeletions()
    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
        corpora=deletions,
        keep_corpora=True,
    )
    assert deletions.delivered == [] and outcome.corpora_deletion_runs == ()
    assert any("acme-contracts" in entry for entry in outcome.left_in_place)
    assert json.loads(conn.receipts[-1][7])["corpora_deletion_runs"] == []


async def test_a_workspace_delete_run_is_submitted_by_the_executor_not_the_subject() -> None:
    """The Run is the deployment's act: submitting it as the person would leave the
    person's id in a table the erase does not clean, and name them in a Run that
    outlives the request."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts",))
    deletions = FakeCorpusDeletions()
    await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
        corpora=deletions,
        executor="dsr-bot",
    )
    assert deletions.submitters == ["dsr-bot"]


async def test_a_b2b_erase_without_a_customer_approval_is_refused_before_anything() -> None:
    """The customer is the controller of the corpus, so their yes comes first."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts",))
    deletions = FakeCorpusDeletions()
    with pytest.raises(OwnerEraseUnauthorized):
        await erase_owner(
            conn,
            owner_id=_OWNER,
            telemetry=_no_telemetry,
            workspace_root=None,
            owner_skills_root=None,
            corpora=deletions,
            line="b2b",
        )
    assert conn.statements == [] and deletions.delivered == []


async def test_a_b2b_erase_records_the_customer_approval_it_ran_on() -> None:
    """An audit has to be able to answer "who approved this"."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts",))
    deletions = FakeCorpusDeletions()
    await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
        corpora=deletions,
        line="b2b",
        customer_approval="  ACME-4711  ",
    )
    counts = json.loads(conn.receipts[-1][7])
    assert conn.receipts[-1][4] == "b2b" and counts["customer_approval"] == "ACME-4711"
    assert deletions.delivered == ["acme-contracts"]


async def test_a_workspace_a_concurrent_deletion_already_took_is_not_a_delivery() -> None:
    """The Workspace row is gone, so the deletion happened; there is no Run to name,
    and a second one that keeps the erase from finishing the rest would be a bug."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts", "shared-kb"))
    deletions = FakeCorpusDeletions(already_gone=("acme-contracts",))
    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
        corpora=deletions,
    )
    assert outcome.corpora_already_gone == ("acme-contracts",)
    assert outcome.corpora_deletion_runs == ("run-shared-kb",)
    counts = json.loads(conn.receipts[-1][7])
    assert counts["corpora_already_gone"] == ["acme-contracts"]


async def test_a_reader_names_the_corpus_it_cannot_reach() -> None:
    """No Corpus Mutation port means no delivery, and the receipt still says which
    Workspaces the requester created rather than staying silent about them."""
    conn = FakeConnection(rows={"dlightrag_runs": 1}, workspaces=("acme-contracts",))
    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=None,
        owner_skills_root=None,
        corpora=None,
    )
    assert outcome.corpora_deletion_runs == ()
    assert any("acme-contracts" in entry for entry in outcome.left_in_place)


def test_erase_requires_a_line_and_a_b2b_line_requires_the_customer_reference() -> None:
    from dlightrag.adapters.postgres.owner_data import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--owner", _OWNER, "erase"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--owner", _OWNER, "erase", "--line", "smoke"])
    args = build_parser().parse_args(
        ["--owner", _OWNER, "erase", "--line", "b2b", "--customer-approval", "ACME-1"]
    )
    assert args.line == "b2b" and args.customer_approval == "ACME-1"
    assert args.keep_corpora is False and args.dry_run is False


def test_a_b2b_dry_run_still_needs_the_customer_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dry run rehearses the command that will run, so it answers to the same rule
    — and a refusal must not reach the database."""
    import sys
    from types import SimpleNamespace

    from dlightrag.adapters.postgres import owner_data

    monkeypatch.setattr(
        owner_data,
        "get_config",
        lambda: SimpleNamespace(observability=SimpleNamespace(log_level="info")),
    )
    monkeypatch.setattr(owner_data, "set_config", lambda config: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dlightrag-owner-data", "--owner", _OWNER, "erase", "--line", "b2b", "--dry-run"],
    )
    with pytest.raises(SystemExit) as refusal:
        owner_data.main()
    assert refusal.value.code not in (0, None)
    assert "--customer-approval" in str(refusal.value.code)


async def test_a_skipped_telemetry_erase_says_skipped_not_zero(tmp_path: Path) -> None:
    """`0 rows` and `never looked` are different facts, and the ledger keeps them apart."""
    conn = FakeConnection(rows={"dlightrag_runs": 1})
    await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=None,
        workspace_root=tmp_path,
        owner_skills_root=tmp_path / "skills",
    )
    counts = json.loads(conn.receipts[-1][7])
    assert counts["telemetry"] == "skipped" and counts["telemetry_rows"] == 0


async def test_export_carries_the_owners_own_skills(tmp_path: Path) -> None:
    from dlightrag.engine.agent.skills import owner_skill_root

    skills = tmp_path / "skills"
    skill = owner_skill_root(skills, _OWNER) / "mine" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: mine\ndescription: mine\n---\n\nBody.\n")

    outcome = await export_owner(
        FakeConnection(), owner_id=_OWNER, out_dir=tmp_path / "out", owner_skills_root=skills
    )
    assert (outcome.path / "skills" / "mine" / "SKILL.md").is_file()
    manifest = json.loads((outcome.path / "manifest.json").read_text())
    assert manifest["skill_files"] == 1 and "key ring" in manifest["note"]


async def test_erase_tolerates_a_foreign_key_and_finishes(tmp_path: Path) -> None:
    conn = FakeConnection(rows={"dlightrag_runs": 1}, fk_once="dlightrag_runs")
    outcome = await erase_owner(
        conn,
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=tmp_path,
        owner_skills_root=tmp_path / "skills",
    )
    assert outcome.deleted["dlightrag_runs"] == 1


async def test_erase_removes_the_workspace_trees_and_the_owner_skills(tmp_path: Path) -> None:
    from dlightrag.engine.agent.skills import owner_skill_root
    from dlightrag.engine.answer.agentspace import run_root

    root = tmp_path / "workspaces"
    tree = run_root(root, _OWNER, "run-1")
    tree.mkdir(parents=True)
    (tree / "notes").mkdir()
    skills = tmp_path / "skills"
    owner_skill_root(skills, _OWNER).mkdir(parents=True)

    outcome = await erase_owner(
        FakeConnection(),
        owner_id=_OWNER,
        telemetry=_no_telemetry,
        workspace_root=root,
        owner_skills_root=skills,
    )
    assert outcome.workspace_roots == 1
    assert not tree.exists()
    # The shard the tree lived in goes too, or an erase leaves a directory per owner.
    assert not tree.parent.exists()
    assert not owner_skill_root(skills, _OWNER).exists()


def test_resolve_owner_derives_the_id_the_runtime_projects() -> None:
    args = _parser_args(
        "--principal",
        "alice@acme.test",
        "--issuer",
        "https://idp.test",
        "export",
        "--out",
        "/tmp/x",
    )
    assert resolve_owner(args) == owner_id_from_principal(
        auth_mode="jwt", user_id="alice@acme.test", issuer="https://idp.test"
    )
    with pytest.raises(SystemExit):
        resolve_owner(_parser_args("--principal", "alice@acme.test", "export", "--out", "/tmp/x"))
    with pytest.raises(SystemExit):
        resolve_owner(_parser_args("export", "--out", "/tmp/x"))


async def test_the_clickhouse_eraser_counts_then_deletes_with_a_bound_parameter() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "SELECT count()" in request.url.params["query"]:
            return httpx.Response(200, text="3\n")
        return httpx.Response(200, text="")

    transport = httpx.MockTransport(handler)
    eraser = clickhouse_telemetry_eraser(url="http://clickhouse:8123/", user="u", password="p")
    with httpx.Client(transport=transport):
        pass
    import httpx as _httpx

    original = _httpx.AsyncClient

    class _Patched(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kwargs: Any) -> None:
            kwargs["transport"] = transport
            super().__init__(**kwargs)

    _httpx.AsyncClient = _Patched  # type: ignore[misc]
    try:
        removed = await eraser(owner_id=_OWNER)
    finally:
        _httpx.AsyncClient = original
    assert removed == 6  # two tables, three rows each
    assert all(request.url.params["param_owner"] == _OWNER for request in seen)
    assert not any(_OWNER in request.url.params["query"] for request in seen)
