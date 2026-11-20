# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Artifact surfaces publish the derived attestation, and never gate on it.

The two transports reach one derivation and one record class, so these tests assert the record
that comes back over REST is the record the Web presentation carries -- and that an unattested
Artifact is served, bytes and presentation included, with the reason travelling beside it.
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.rest.auth import get_current_user
from dlightrag.adapters.http.server import create_app
from dlightrag.application.access.principal import UserContext
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.runs import RunEvent, RunToolCall
from dlightrag.engine.agent.tools.contracts import TOOL_SUBJECT_MAX_CHARS
from tests.support.application_double import application_double
from tests.unit.web.answer_run_fixtures import RUN_ID, answer_run, linked_turn, stored_result

_ANON = UserContext(user_id="anonymous", auth_mode="none")
_NOW = datetime.datetime(2026, 8, 13, tzinfo=datetime.UTC)
pytestmark = pytest.mark.usefixtures("sealed_report_packaged")

_SKILL = "sealed-report"
_VERIFIER = "scripts/case_validate.py"
_RESOURCE = "artifact-report"
_FILENAME = "case.json"
_CONTENT = b"# Report\n"


def _event_log(*events: tuple[str, dict[str, Any]]) -> AsyncIterator[RunEvent]:
    """One Run's durable event replay, as the Artifact surfaces read it back."""

    async def _iterate() -> AsyncIterator[RunEvent]:
        for index, (kind, payload) in enumerate(events):
            yield RunEvent(sequence=index + 1, event_type=kind, payload=payload, created_at=_NOW)

    return _iterate()


def _load_events() -> list[tuple[str, dict[str, Any]]]:
    """The records of a Run that loaded the Skill and never ran its declared verifier."""
    return [
        ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
        ("tool_progress", {"tool_name": "load_skill", "call_id": "c1", "object_label": _SKILL}),
        ("tool_end", {"tool_name": "load_skill", "call_id": "c1", "outcome": "succeeded"}),
    ]


def _verifier_events() -> list[tuple[str, dict[str, Any]]]:
    return [
        *_load_events(),
        ("tool_start", {"tool_name": "bash", "call_id": "c2"}),
        (
            "tool_progress",
            {
                "tool_name": "bash",
                "call_id": "c2",
                "object_label": f"python tmp/{_SKILL}/{_VERIFIER} {_FILENAME}",
            },
        ),
        ("tool_end", {"tool_name": "bash", "call_id": "c2", "outcome": "succeeded"}),
    ]


def _settled_run(*, trimmed: bool = False) -> Any:
    result = stored_result(answer="[Report](artifact:artifact-report)")
    result["artifacts"] = [
        {
            "resource_id": _RESOURCE,
            "media_type": "text/markdown",
            "label": "Report",
            "filename": _FILENAME,
            "byte_size": 12,
            "digest": "a" * 64,
            "presentation": "markdown",
            "status": "available",
        }
    ]
    return answer_run(
        status="succeeded",
        result=result,
        events_trimmed_at=_NOW if trimmed else None,
    )


@pytest.fixture
def _app(test_config: DlightragConfig) -> Iterator[FastAPI]:
    application = create_app()
    application.dependency_overrides[get_current_user] = lambda: _ANON
    yield application
    application.dependency_overrides.clear()


def _replays(
    application: Any,
    *events: tuple[str, dict[str, Any]],
    arguments: tuple[RunToolCall, ...] = (),
) -> None:
    """Answer every subscription with a fresh replay of the same events.

    A real subscription is a new iterator per call; a bare ``return_value`` is one generator,
    which the first transport to read the Run would exhaust for the second. ``arguments`` is the
    Run's transcript read, empty unless the test records one.
    """
    application.runs.subscribe.side_effect = lambda **_: _event_log(*events)
    application.runs.load_tool_call_arguments.return_value = arguments


async def _stream(blob: bytes) -> AsyncIterator[bytes]:
    yield blob


def _install(_app: FastAPI, config: DlightragConfig) -> Any:
    created = application_double(config)
    created.answers.list_artifacts.return_value = ()
    created.answers.read_artifact.return_value = _CONTENT
    created.answers.artifact_size.return_value = len(_CONTENT)
    created.answers.open_artifact.side_effect = lambda **_: _stream(_CONTENT)
    created.runs.get.return_value = _settled_run()
    created.web_conversations.turn_for_run.return_value = linked_turn(_settled_run())
    _replays(created)
    _app.state.application = created
    return created


@pytest.fixture
def application(_app: FastAPI, test_config: DlightragConfig) -> Any:
    return _install(_app, test_config)


@pytest.fixture
async def client(_app: FastAPI) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=_app)
    async with AsyncClient(
        transport=transport, base_url="http://test", cookies={"dlightrag_workspace": "default"}
    ) as api:
        yield api


async def test_rest_artifact_list_carries_the_derived_record(
    client: AsyncClient, application: Any
) -> None:
    _replays(application, *_verifier_events())

    response = await client.get(f"/answer/{RUN_ID}/artifacts")

    assert response.status_code == 200
    artifact = response.json()["artifacts"][0]
    assert artifact["status"] == "available"
    # A Run whose loads carry no read fact shows the reads its Skill declares as missing.
    assert artifact["attestation"]["required_reads_missing"]
    assert artifact["attestation"] | {"required_reads_missing": None} == {
        "status": "attested",
        "skill": _SKILL,
        "verifier": _VERIFIER,
        "reason": None,
        "basis": "named_in_a_recorded_command",
        "required_reads_missing": None,
    }
    assert response.json()["artifact_outcome"]["status"] == "complete"


async def test_a_cut_command_is_attested_from_the_transcripts_full_arguments(
    client: AsyncClient, application: Any
) -> None:
    """The bytes route reads the same records: the full command is what attests the Artifact."""
    command = (
        f"cd /srv/sealed-report/run-workspaces/0123456789abcdef && python {_VERIFIER} case.json"
    )
    subject = command[: TOOL_SUBJECT_MAX_CHARS - 1] + "…"
    assert _VERIFIER not in subject
    _replays(
        application,
        *_load_events(),
        ("tool_start", {"tool_name": "bash", "call_id": "c2"}),
        ("tool_progress", {"tool_name": "bash", "call_id": "c2", "object_label": subject}),
        ("tool_end", {"tool_name": "bash", "call_id": "c2", "outcome": "succeeded"}),
        arguments=(RunToolCall(call_id="c2", tool_name="bash", arguments={"command": command}),),
    )

    response = await client.get(f"/answer/{RUN_ID}/artifacts")

    artifact = response.json()["artifacts"][0]
    assert artifact["status"] == "available"
    assert artifact["attestation"]["status"] == "attested"
    assert application.runs.load_tool_call_arguments.await_count == 1


async def test_rest_artifact_list_marks_a_run_that_never_loaded_the_skill(
    client: AsyncClient, application: Any
) -> None:
    _replays(
        application,
        ("tool_start", {"tool_name": "bash", "call_id": "c1"}),
        (
            "tool_progress",
            {"tool_name": "bash", "call_id": "c1", "object_label": "python build.py"},
        ),
    )

    response = await client.get(f"/answer/{RUN_ID}/artifacts")

    assert response.json()["artifacts"][0]["attestation"] == {
        "status": "unattested",
        "skill": None,
        "verifier": None,
        "reason": "no_skill_loaded",
        "basis": None,
        "required_reads_missing": None,
    }
    assert response.json()["artifacts"][0]["status"] == "available"


async def test_an_unattested_artifact_is_served_with_its_reason_beside_it(
    client: AsyncClient, application: Any
) -> None:
    """A verifier that never ran is a finding, not a delivery refusal: the file and record ship."""
    _replays(application, *_load_events())

    listed = await client.get(f"/answer/{RUN_ID}/artifacts")
    streamed = await client.get(f"/answer/{RUN_ID}/artifacts/{_RESOURCE}?download=1")

    artifact = listed.json()["artifacts"][0]
    assert artifact["status"] == "available"
    assert artifact["issue"] is None
    assert artifact["download_url"]
    assert artifact["attestation"]["reason"] == "skill_loaded_without_verifier"
    assert listed.json()["artifact_outcome"]["status"] == "complete"
    assert streamed.status_code == 200
    assert streamed.content == _CONTENT


async def test_a_trimmed_run_is_unknown_rather_than_accused(
    client: AsyncClient, application: Any
) -> None:
    trimmed = _settled_run(trimmed=True)
    application.runs.get.return_value = trimmed
    application.web_conversations.turn_for_run.return_value = linked_turn(trimmed)

    response = await client.get(f"/answer/{RUN_ID}/artifacts")

    assert response.json()["artifacts"][0]["attestation"]["status"] == "unknown"
    assert response.json()["artifacts"][0]["attestation"]["reason"] == "records_pruned"


async def test_both_transports_publish_one_record_for_the_same_artifact(
    client: AsyncClient, application: Any
) -> None:
    _replays(application, *_verifier_events())

    listed = await client.get(f"/answer/{RUN_ID}/artifacts")
    presented = await client.get(f"/web/api/answer/{RUN_ID}/artifacts/{_RESOURCE}/presentation")

    assert presented.status_code == 200
    rest_record = listed.json()["artifacts"][0]["attestation"]
    web_record = presented.json()["artifacts"][0]["attestation"]
    assert rest_record == web_record
    assert rest_record["status"] == "attested"
    assert rest_record["basis"] == "named_in_a_recorded_command"


async def test_both_transports_publish_the_reads_a_run_skipped_beside_its_artifact(
    client: AsyncClient, application: Any
) -> None:
    """``interactive-html`` declares no verifier and a required read: the record exists for the read."""
    hub = {
        "tool_name": "load_skill",
        "call_id": "c1",
        "object_label": "interactive-html",
        "skill_path": "SKILL.md",
        "skill_window_start": 0,
        "skill_window_end": 70,
        "skill_document_chars": 70,
    }
    _replays(
        application,
        ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
        ("tool_progress", hub),
        ("tool_end", {"tool_name": "load_skill", "call_id": "c1", "outcome": "succeeded"}),
    )

    listed = await client.get(f"/answer/{RUN_ID}/artifacts")
    presented = await client.get(f"/web/api/answer/{RUN_ID}/artifacts/{_RESOURCE}/presentation")

    assert presented.status_code == 200
    rest_record = listed.json()["artifacts"][0]["attestation"]
    web_record = presented.json()["artifacts"][0]["attestation"]
    assert rest_record == web_record
    assert rest_record["reason"] == "skill_has_no_verifier_contract"
    (missing,) = rest_record["required_reads_missing"]
    assert missing["skill"] == "interactive-html"
    assert missing["paths"][0] == "references/fragment.md"
    assert missing["when"].startswith("before writing or restructuring")


async def test_an_unattested_artifact_is_served_through_both_presentation_routes(
    client: AsyncClient, application: Any
) -> None:
    """The document and the bytes are served; the reason rides the record, never a 409."""
    _replays(application, *_load_events())

    web = await client.get(f"/web/api/answer/{RUN_ID}/artifacts/{_RESOURCE}/presentation")
    rest = await client.get(f"/answer/{RUN_ID}/artifacts/{_RESOURCE}/presentation")

    for response in (web, rest):
        assert response.status_code == 200
        assert response.json()["artifacts"][0]["attestation"]["reason"] == (
            "skill_loaded_without_verifier"
        )


async def test_bytes_are_served_for_a_run_that_loaded_no_skill(
    client: AsyncClient, application: Any
) -> None:
    """A Run that no Skill stood behind is served like any other; the record is the only addition."""
    response = await client.get(f"/answer/{RUN_ID}/artifacts/{_RESOURCE}")

    assert response.status_code == 200
    assert response.content == _CONTENT
    assert application.runs.subscribe.call_count == 0
