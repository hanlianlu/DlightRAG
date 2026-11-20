# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Web Inspector's Run overview: the wire shape the service's whole-Run read is drawn as.

The read itself — the rollups, the findings cap, the content gate, and the usage record — is fixed
by the service and the overview module's own tests. These tests fix what this route owes a reader:
the one payload a panel reads, the cache validator it states, and the opaque misses.
"""

import datetime
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.application.answer_runs.overview import (
    RunAgentRollup,
    RunFinding,
    RunOverview,
    RunTotals,
)
from dlightrag.application.config.sections import DlightragConfig
from tests.support.application_double import application_double
from tests.unit.web.answer_run_fixtures import RUN_ID, linked_turn

_T0 = datetime.datetime(2026, 8, 12, 12, 0, 0, tzinfo=datetime.UTC)

_KIND_ZERO = {
    "thinking": 0,
    "tool": 0,
    "say": 0,
    "answer": 0,
    "instruction": 0,
    "delivery": 0,
}
_MARK_ZERO = {"failed": 0, "degraded": 0, "repeated": 0, "redirect": 0}


def _at(seconds: float) -> str:
    return (_T0 + datetime.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _totals(
    *,
    steps: int = 0,
    kinds: dict[str, int] | None = None,
    decisions: int = 0,
    marks: dict[str, int] | None = None,
    tool_time_ms: int = 0,
) -> RunTotals:
    return RunTotals(
        steps=steps,
        kinds={**_KIND_ZERO, **(kinds or {})},
        decisions=decisions,
        marks={**_MARK_ZERO, **(marks or {})},
        tool_time_ms=tool_time_ms,
    )


def _agent(
    *,
    agent: str = "main",
    label: str = "main",
    order: int = 0,
    steps: int = 0,
    tool_calls: int = 0,
    marks: dict[str, int] | None = None,
    tool_time_ms: int = 0,
    first_at: str | None = None,
    last_at: str | None = None,
) -> RunAgentRollup:
    return RunAgentRollup(
        agent=agent,
        label=label,
        order=order,
        steps=steps,
        tool_calls=tool_calls,
        marks={**_MARK_ZERO, **(marks or {})},
        tool_time_ms=tool_time_ms,
        first_at=first_at,
        last_at=last_at,
    )


def _finding(
    *,
    agent: str = "main",
    sequence: int = 42,
    kind: str = "tool",
    at: str | None = None,
    name: str | None = "search_web",
    subject: str | None = "the filing",
    outcome: str | None = "failed",
    marks: tuple[str, ...] = ("failed", "repeated"),
    excerpt: str = "the first line",
) -> RunFinding:
    return RunFinding(
        agent=agent,
        session_id="00000000-0000-7000-8000-000000000001",
        sequence=sequence,
        kind=kind,
        at=_at(0) if at is None else at,
        name=name,
        subject=subject,
        outcome=outcome,
        marks=marks,
        excerpt=excerpt,
    )


def _overview(
    *,
    totals: RunTotals | None = None,
    agents: tuple[RunAgentRollup, ...] | None = None,
    findings: tuple[RunFinding, ...] | None = None,
    findings_total: int | None = None,
    usage: dict[str, Any] | None = None,
    version: str = "version-1",
) -> RunOverview:
    selected = (_finding(),) if findings is None else findings
    return RunOverview(
        run_id=RUN_ID,
        totals=_totals() if totals is None else totals,
        agents=() if agents is None else agents,
        findings=selected,
        findings_total=len(selected) if findings_total is None else findings_total,
        usage=usage,
        version=version,
    )


@pytest.fixture
def application(test_config: DlightragConfig) -> Any:
    """The strict Application double the Web answer routes reach."""
    created = application_double(test_config)
    created.web_conversations.turn_for_run.return_value = linked_turn(conversation_id="conv-1")
    created.answers.run_overview.return_value = _overview()
    return created


@pytest.fixture
async def client(application: Any):
    app = create_app()
    app.state.application = application
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"dlightrag_workspace": "default"},
    ) as created:
        yield created


async def test_overview_states_the_whole_run_in_one_payload(
    client: AsyncClient, application: Any
) -> None:
    usage = {
        "usage_details": {"total_tokens": 15_000},
        "child_usage_details": {"total_tokens": 9_000},
        "inclusive_usage_details": {"total_tokens": 24_000},
    }
    application.answers.run_overview.return_value = _overview(
        totals=_totals(
            steps=300,
            kinds={
                "thinking": 70,
                "tool": 150,
                "say": 20,
                "answer": 1,
                "instruction": 2,
                "delivery": 3,
            },
            decisions=12,
            marks={"failed": 3, "degraded": 2, "repeated": 5, "redirect": 2},
            tool_time_ms=412_345,
        ),
        agents=(
            _agent(
                steps=120,
                tool_calls=40,
                marks={"failed": 1, "degraded": 0, "repeated": 2, "redirect": 1},
                tool_time_ms=12_345,
                first_at=_at(0),
                last_at=_at(9),
            ),
            _agent(agent="child-a", label="Look up filings", order=1),
        ),
        findings=(
            _finding(),
            _finding(
                agent="child-a",
                sequence=2,
                kind="delivery",
                name=None,
                subject=None,
                outcome=None,
                marks=(),
                excerpt="no Skill loaded",
            ),
        ),
        findings_total=40,
        usage=usage,
        version="version-7",
    )

    response = await client.get(f"/web/api/answer/{RUN_ID}/overview")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-cache"
    assert response.headers["etag"]
    assert response.json() == {
        "run_id": RUN_ID,
        "totals": {
            "steps": 300,
            "kinds": {
                "thinking": 70,
                "tool": 150,
                "say": 20,
                "answer": 1,
                "instruction": 2,
                "delivery": 3,
            },
            "decisions": 12,
            "marks": {"failed": 3, "degraded": 2, "repeated": 5, "redirect": 2},
            "tool_time_ms": 412_345,
        },
        "agents": [
            {
                "agent": "main",
                "label": "main",
                "order": 0,
                "steps": 120,
                "tool_calls": 40,
                "marks": {"failed": 1, "degraded": 0, "repeated": 2, "redirect": 1},
                "tool_time_ms": 12_345,
                "first_at": _at(0),
                "last_at": _at(9),
            },
            {
                "agent": "child-a",
                "label": "Look up filings",
                "order": 1,
                "steps": 0,
                "tool_calls": 0,
                "marks": {"failed": 0, "degraded": 0, "repeated": 0, "redirect": 0},
                "tool_time_ms": 0,
                "first_at": None,
                "last_at": None,
            },
        ],
        "findings": [
            {
                "agent": "main",
                "sequence": 42,
                "kind": "tool",
                "at": _at(0),
                "name": "search_web",
                "subject": "the filing",
                "outcome": "failed",
                "marks": ["failed", "repeated"],
                "excerpt": "the first line",
            },
            {
                "agent": "child-a",
                "sequence": 2,
                "kind": "delivery",
                "at": _at(0),
                "name": None,
                "subject": None,
                "outcome": None,
                "marks": [],
                "excerpt": "no Skill loaded",
            },
        ],
        "findings_total": 40,
        "usage": usage,
    }
    read_call = application.answers.run_overview.await_args.kwargs
    assert read_call["run_id"] == RUN_ID


async def test_overview_states_an_absent_usage_as_null(
    client: AsyncClient, application: Any
) -> None:
    application.answers.run_overview.return_value = _overview(usage=None)

    response = await client.get(f"/web/api/answer/{RUN_ID}/overview")

    body = response.json()
    # Unknown is not free: the route states null, never zeroes or an empty record.
    assert body["usage"] is None


async def test_overview_keeps_capped_findings_beside_the_uncapped_totals(
    client: AsyncClient, application: Any
) -> None:
    """The response carries the read's whole-Run numbers; it never rebuilds them from the rows."""
    application.answers.run_overview.return_value = _overview(
        totals=_totals(steps=300, decisions=12),
        agents=(_agent(steps=120),),
        findings=(_finding(),),
        findings_total=137,
    )

    body = (await client.get(f"/web/api/answer/{RUN_ID}/overview")).json()

    assert body["findings_total"] == 137
    assert len(body["findings"]) == 1
    assert body["totals"]["steps"] == 300
    assert body["totals"]["decisions"] == 12
    assert body["agents"][0]["steps"] == 120


async def test_overview_answers_304_for_the_version_the_client_holds(
    client: AsyncClient, application: Any
) -> None:
    first = await client.get(f"/web/api/answer/{RUN_ID}/overview")
    cached = await client.get(
        f"/web/api/answer/{RUN_ID}/overview",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert cached.status_code == 304
    assert cached.content == b""
    assert cached.headers["etag"] == first.headers["etag"]
    assert cached.headers["cache-control"] == "private, no-cache"

    # A Run whose index moved is a different overview, so the held validator is not answered.
    application.answers.run_overview.return_value = _overview(version="version-2")
    moved = await client.get(
        f"/web/api/answer/{RUN_ID}/overview",
        headers={"If-None-Match": first.headers["etag"]},
    )

    assert moved.status_code == 200
    assert moved.headers["etag"] != first.headers["etag"]


async def test_overview_unknown_run_is_404_before_any_read(
    client: AsyncClient, application: Any
) -> None:
    application.web_conversations.turn_for_run.return_value = None

    response = await client.get(f"/web/api/answer/{RUN_ID}/overview")

    assert response.status_code == 404
    application.answers.run_overview.assert_not_awaited()


async def test_overview_a_run_the_read_cannot_state_is_404(
    client: AsyncClient, application: Any
) -> None:
    application.answers.run_overview.return_value = None

    response = await client.get(f"/web/api/answer/{RUN_ID}/overview")

    assert response.status_code == 404
