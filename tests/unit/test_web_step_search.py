# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Web Inspector's Run-level step search: the wire shape one page of matches is drawn as.

The read itself — the whole-Run transcript search, the thinking exclusion, the page, and the
content gate — is fixed by the service and the timeline module's own tests. These tests fix what
this route owes a reader: the rows one page holds, the cursor it states, the marks the whole Run
gives its rows, and the refusals of a cursor it did not state.
"""

import datetime
import json
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.application.answer_runs.timeline import (
    RunStepSearch,
    RunTimelinePosition,
    RunTimelineStep,
)
from dlightrag.application.config.sections import DlightragConfig
from tests.support.application_double import application_double
from tests.unit.web.answer_run_fixtures import RUN_ID, linked_turn

_T0 = datetime.datetime(2026, 8, 12, 12, 0, 0, tzinfo=datetime.UTC)

_SEARCH_ROW_KEYS = {
    "agent",
    "agent_label",
    "at",
    "kind",
    "sequence",
    "name",
    "subject",
    "outcome",
    "duration_ms",
    "excerpt",
    "marks",
}


def _at(seconds: float) -> str:
    return (_T0 + datetime.timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _step(
    *,
    second: float,
    kind: str = "tool",
    agent: str = "main",
    agent_label: str = "main",
    sequence: int = 1,
    name: str | None = "read",
    subject: str | None = None,
    outcome: str | None = "succeeded",
    duration_ms: int | None = None,
    content: str = "",
) -> RunTimelineStep:
    """One matching step of the Run, as the search read states it once the page is chosen."""
    tool = kind == "tool"
    return RunTimelineStep(
        agent=agent,
        agent_label=agent_label,
        agent_order=0 if agent == "main" else 1,
        session_id="00000000-0000-7000-8000-000000000001" if agent == "main" else agent,
        sequence=sequence,
        instant=_T0 + datetime.timedelta(seconds=second),
        at=_at(second),
        kind=kind,
        name=name if tool else None,
        subject=subject if tool else None,
        outcome=outcome if tool else None,
        duration_ms=duration_ms if tool else None,
        content=content,
    )


def _read(
    page: list[RunTimelineStep],
    *,
    has_older: bool = False,
    matched: int = 0,
    marks: dict[tuple[str, int], tuple[str, ...]] | None = None,
    terms: tuple[str, ...] = ("filings",),
    match: str = "all",
) -> RunStepSearch:
    return RunStepSearch(
        page=tuple(page),
        has_older=has_older,
        matched=matched,
        marks={} if marks is None else marks,
        terms=terms,
        match=match,
    )


@pytest.fixture
def application(test_config: DlightragConfig) -> Any:
    """The strict Application double the Web answer routes reach."""
    created = application_double(test_config)
    created.web_conversations.turn_for_run.return_value = linked_turn(conversation_id="conv-1")
    created.answers.search_run_steps.return_value = _read([])
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


async def test_step_search_states_the_matching_page_and_its_cursor(
    client: AsyncClient, application: Any
) -> None:
    steps = [
        _step(
            second=0,
            sequence=1,
            subject="a.md",
            duration_ms=1234,
            content="stdout:\nsecond line",
        ),
        _step(
            second=1,
            sequence=2,
            kind="answer",
            agent="child-a",
            agent_label="Read the filing",
            content="Clause 9.2 caps it.",
        ),
    ]
    marks = {("main", 1): ("degraded",)}
    application.answers.search_run_steps.return_value = _read(
        steps, has_older=True, matched=7, marks=marks
    )

    response = await client.get(
        f"/web/api/answer/{RUN_ID}/steps/search", params={"q": "filings", "limit": 2}
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert set(body) == {"run_id", "steps", "next_before", "matched", "terms", "match"}
    assert body["run_id"] == RUN_ID
    assert body["matched"] == 7
    assert body["terms"] == ["filings"]
    assert body["match"] == "all"
    assert body["steps"] == [
        {
            "agent": "main",
            "agent_label": "main",
            "at": _at(0),
            "kind": "tool",
            "sequence": 1,
            "name": "read",
            "subject": "a.md",
            "outcome": "succeeded",
            "duration_ms": 1234,
            "excerpt": "stdout:",
            "marks": ["degraded"],
        },
        {
            "agent": "child-a",
            "agent_label": "Read the filing",
            "at": _at(1),
            "kind": "answer",
            "sequence": 2,
            "name": None,
            "subject": None,
            "outcome": None,
            "duration_ms": None,
            "excerpt": "Clause 9.2 caps it.",
            "marks": [],
        },
    ]
    # The retired timeline wire's fields are gone from both the page and its rows.
    for step in body["steps"]:
        assert set(step) == _SEARCH_ROW_KEYS
    assert body["next_before"] is not None
    assert json.loads(body["next_before"]) == [steps[0].instant.isoformat(), "main", 1, "tool"]
    read_call = application.answers.search_run_steps.await_args.kwargs
    assert read_call["run_id"] == RUN_ID
    assert read_call["query"] == "filings"
    assert read_call["limit"] == 2
    assert read_call["before"] is None


async def test_step_search_pages_by_the_cursor_it_states(
    client: AsyncClient, application: Any
) -> None:
    steps = [_step(second=second, sequence=second + 1) for second in range(3)]
    application.answers.search_run_steps.return_value = _read(steps, has_older=True, matched=3)
    first = (await client.get(f"/web/api/answer/{RUN_ID}/steps/search", params={"q": "a"})).json()
    cursor = first["next_before"]
    assert cursor is not None
    assert json.loads(cursor) == [steps[0].instant.isoformat(), "main", 1, "tool"]

    application.answers.search_run_steps.return_value = _read(steps[:1], matched=3)
    second = await client.get(
        f"/web/api/answer/{RUN_ID}/steps/search", params={"q": "a", "limit": 2, "before": cursor}
    )

    assert second.status_code == 200
    assert second.json()["next_before"] is None
    boundary = application.answers.search_run_steps.await_args.kwargs["before"]
    assert boundary == RunTimelinePosition(
        instant=steps[0].instant,
        agent=steps[0].agent,
        sequence=steps[0].sequence,
        kind=steps[0].kind,
    )


async def test_step_search_rejects_a_cursor_it_did_not_state(
    client: AsyncClient, application: Any
) -> None:
    response = await client.get(
        f"/web/api/answer/{RUN_ID}/steps/search",
        params={"q": "filings", "before": "not-a-cursor"},
    )

    assert response.status_code == 422
    application.answers.search_run_steps.assert_not_awaited()


async def test_step_search_unknown_run_is_404_before_cursor_or_read(
    client: AsyncClient, application: Any
) -> None:
    application.web_conversations.turn_for_run.return_value = None

    response = await client.get(
        f"/web/api/answer/{RUN_ID}/steps/search",
        params={"q": "filings", "before": "not-a-cursor"},
    )

    assert response.status_code == 404
    application.answers.search_run_steps.assert_not_awaited()


async def test_step_search_a_run_the_read_cannot_state_is_404(
    client: AsyncClient, application: Any
) -> None:
    application.answers.search_run_steps.return_value = None

    response = await client.get(f"/web/api/answer/{RUN_ID}/steps/search", params={"q": "filings"})

    assert response.status_code == 404


@pytest.mark.parametrize(
    "params",
    [
        pytest.param({}, id="no-query"),
        pytest.param({"q": ""}, id="empty-query"),
        pytest.param({"q": "x" * 201}, id="query-too-long"),
        pytest.param({"q": "filings", "limit": 0}, id="limit-below-one"),
        pytest.param({"q": "filings", "limit": 201}, id="limit-above-the-bound"),
    ],
)
async def test_step_search_refuses_a_page_it_cannot_state(
    client: AsyncClient, application: Any, params: dict[str, Any]
) -> None:
    response = await client.get(f"/web/api/answer/{RUN_ID}/steps/search", params=params)

    assert response.status_code == 422
    application.answers.search_run_steps.assert_not_awaited()
