# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The history surface carries the Attestation record the live Answer published.

The record was derived for the Artifact list, the presentation and the byte routes, and the
conversation-history listing -- what a browser loads when a conversation is reopened -- arrived
without it, so a reader who came back saw a card that said nothing about whether any Skill stood
behind the file. These pin the two seams that close that: the projection stamps the record its
caller derived, and the route derives for exactly the turns that published something.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from dlightrag.adapters.http.browser import conversations as projection
from dlightrag.adapters.http.browser.routes import conversations as route
from dlightrag.application.answer.provenance import (
    ArtifactAttestation,
    MissingRead,
    stamp_when_publishing,
)


def _attested() -> ArtifactAttestation:
    return ArtifactAttestation(
        status="attested", skill="sealed-report", verifier="scripts/validate_model.py"
    )


def _turn(
    run_id: str, *, status: str = "succeeded", result: Any = None, trimmed: bool = False
) -> Any:
    return SimpleNamespace(
        turn_id=f"turn-{run_id}",
        run=SimpleNamespace(
            run_id=run_id,
            status=status,
            result=result,
            events_trimmed_at="2026-10-15T00:00:00Z" if trimmed else None,
        ),
    )


def _page(*turns: Any) -> Any:
    return SimpleNamespace(
        conversation=SimpleNamespace(), turns=turns, next_cursor=None, fetched_rows=len(turns)
    )


def test_the_projection_stamps_the_record_its_caller_derived(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A run absent from the records map renders as before; a run present in it is stamped."""
    seen: list[dict[str, Any]] = []

    def capture(turn: Any, **kwargs: Any) -> Any:
        seen.append({"run_id": turn.run.run_id, **kwargs})
        return SimpleNamespace(turn_id=turn.turn_id)

    monkeypatch.setattr(projection, "project_conversation_turn", capture)
    monkeypatch.setattr(projection, "_head_summary", lambda _: SimpleNamespace())
    # The transport model is built from the turns this test stands in for; recording the call is the
    # assertion, so the model itself is not the subject here.
    built: list[Any] = []
    monkeypatch.setattr(
        projection, "ConversationHistory", lambda **kwargs: built.append(kwargs) or kwargs
    )

    projection.project_conversation_history(
        _page(_turn("run-1"), _turn("run-2")),
        attestations={"run-1": _attested()},
    )
    assert [turn.turn_id for turn in built[0]["turns"]] == ["turn-run-1", "turn-run-2"]

    assert [entry["run_id"] for entry in seen] == ["run-1", "run-2"]
    assert seen[0]["attestation"] == _attested()
    assert seen[1]["attestation"] is None


@pytest.mark.asyncio
async def test_the_route_derives_only_for_runs_that_published_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One derivation per Artifact-publishing Run: a page of chatter costs nothing."""
    derived: list[str] = []

    async def fake_attestation(**kwargs: Any) -> ArtifactAttestation:
        derived.append(kwargs["run_id"])
        assert kwargs["records_pruned"] is True  # the caller's own trim knowledge travels
        return _attested()

    monkeypatch.setattr(route, "run_attestation", fake_attestation)
    monkeypatch.setattr(route, "owner_id_from_user", lambda _: "owner-1")
    monkeypatch.setattr(
        route,
        "get_application",
        lambda _: SimpleNamespace(config=SimpleNamespace(), runs=SimpleNamespace()),
    )
    request = SimpleNamespace(state=SimpleNamespace(user_context=None))
    page = _page(
        _turn("published", result={"artifacts": [{"resource_id": "r1"}]}, trimmed=True),
        _turn("chatter", result={"answer": "no artifacts"}),
        _turn("queued", status="running", result=None),
        _turn("nothing-published", result={}),
    )

    # Stand-ins for the request and the page: this test is about which Runs are derived for.
    records = await route._turn_attestations(cast(Any, request), cast(Any, page))

    assert derived == ["published"]
    assert records == {"published": _attested()}


def test_a_reopened_conversation_shows_the_reads_the_live_answer_showed() -> None:
    """The history projection stamps through the one seam, so a skipped read survives a reload.

    A Skill that declares no verifier publishes no record, except that this Run left a declared
    read unread: the reader who comes back is owed the same line the live Answer drew.
    """
    projected = {"artifacts": [{"resource_id": "r1", "status": "available"}]}
    skipped = ArtifactAttestation(
        status="unattested",
        skill="interactive-html",
        reason="skill_has_no_verifier_contract",
        required_reads_missing=(
            MissingRead(
                skill="interactive-html",
                paths=("references/example-brief.html",),
                when="before writing the fragment",
            ),
        ),
    )

    marked = stamp_when_publishing(projected, attestation=skipped)
    quiet = stamp_when_publishing(
        projected, attestation=replace(skipped, required_reads_missing=())
    )

    assert marked["artifacts"][0]["attestation"]["required_reads_missing"] == [
        {
            "skill": "interactive-html",
            "paths": ["references/example-brief.html"],
            "when": "before writing the fragment",
        }
    ]
    assert quiet["artifacts"][0]["attestation"] is None
