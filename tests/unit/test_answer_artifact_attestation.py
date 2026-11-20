# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Runtime-derived Artifact attestation: the vocabulary, the derivation, and the one shape.

Each test here fails if the mechanism it names is removed: the reasons are asserted one by one
against synthetic stored records, the two transport models are asserted to read one record, and
the refusal is asserted to name the Run, the Artifact and the vocabulary's own reason.
"""

from __future__ import annotations

import datetime
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from dlightrag.adapters.http.browser.presentation import build_answer_presentation
from dlightrag.adapters.http.rest.models import AnswerArtifactResponse
from dlightrag.application.answer.provenance import (
    ATTESTATION_REASON_TEXT,
    ATTESTATION_REASONS,
    ArtifactAttestation,
    ArtifactAttestationResponse,
    MissingRead,
    RecordedToolCall,
    SkillContract,
    attest_projected_artifacts,
    derive_run_attestation,
    stamp_projected_artifacts,
    tool_calls_from_events,
)
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.runs import RunEvent, RunToolCall
from dlightrag.engine.agent.skill_reads import (
    RequiredRead,
    SkillRead,
    skill_reads_from_events,
)
from dlightrag.engine.agent.tools.contracts import TOOL_SUBJECT_MAX_CHARS

_NOW = datetime.datetime(2026, 8, 13, tzinfo=datetime.UTC)
_RUN_ID = "0199a0a0-0000-7000-8000-000000000001"
pytestmark = pytest.mark.usefixtures("sealed_report_packaged")

_SKILL = "sealed-report"
_VERIFIER = "scripts/case_validate.py"
_CONTRACTS: dict[str, SkillContract] = {_SKILL: SkillContract(_VERIFIER)}
_ARTIFACT: dict[str, Any] = {
    "resource_id": "artifact-report",
    "media_type": "text/markdown",
    "label": "Report",
    "filename": "case.json",
    "byte_size": 12,
    "digest": "a" * 64,
    "presentation": "markdown",
    "status": "available",
    "uri": f"dlightrag://answer/{_RUN_ID}/artifacts/artifact-report",
    "issue": None,
    "data_url": f"/answer/{_RUN_ID}/artifacts/artifact-report",
    "download_url": f"/answer/{_RUN_ID}/artifacts/artifact-report?download=1",
    "presentation_url": f"/answer/{_RUN_ID}/artifacts/artifact-report/presentation",
}


def _load(
    name: str = _SKILL, *, subject: str | None = None, outcome: str | None = "succeeded"
) -> RecordedToolCall:
    return RecordedToolCall(
        tool_name="load_skill",
        call_id="call-load",
        subject=name if subject is None else subject,
        outcome=outcome,
    )


def _cut(command: str) -> str:
    """A command as the runtime publishes it: one bounded line, cut where the bound falls."""
    return command[: TOOL_SUBJECT_MAX_CHARS - 1] + "…"


def _bash(command: str = f"python tmp/{_SKILL}/{_VERIFIER} case.json") -> RecordedToolCall:
    return RecordedToolCall(
        tool_name="bash", call_id="call-bash", subject=command, outcome="succeeded"
    )


def _verdict(
    *calls: RecordedToolCall,
    contracts: dict[str, SkillContract] | None = None,
    records_complete: bool = True,
) -> ArtifactAttestation:
    return derive_run_attestation(
        contracts=_CONTRACTS if contracts is None else contracts,
        calls=calls,
        records_complete=records_complete,
    ).verdict


def _projected(
    *, parts: bool = False, artifacts: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    values = [dict(_ARTIFACT)] if artifacts is None else [dict(item) for item in artifacts]
    projected: dict[str, Any] = {
        "artifacts": values,
        "artifact_outcome": {"status": "complete", "issues": []},
    }
    if parts:
        projected["parts"] = [
            {"type": "markdown", "text": ""},
            {"type": "artifact", "artifact": dict(values[0])},
        ]
    return projected


def test_attested_only_when_a_skill_with_a_contract_was_loaded_and_its_verifier_ran() -> None:
    derived = derive_run_attestation(contracts=_CONTRACTS, calls=[_load(), _bash()])

    assert derived.skills_loaded == (_SKILL,)
    assert derived.verifier_runs[0].ran is True
    assert derived.verifier_runs[0].evidence_tool == "bash"
    assert derived.verdict.status == "attested"
    assert derived.verdict.skill == _SKILL
    assert derived.verdict.verifier == _VERIFIER
    assert derived.verdict.reason is None


def test_unattested_no_skill_loaded() -> None:
    """The measured defect: a Run that ran tools and never loaded the Skill at all."""
    verdict = _verdict(
        RecordedToolCall(
            tool_name="bash", call_id="c", subject="python build.py", outcome="succeeded"
        )
    )

    assert verdict.status == "unattested"
    assert verdict.reason == "no_skill_loaded"
    assert verdict.skill is None


def test_unattested_skill_loaded_without_verifier() -> None:
    """The other measured defect: the Skill was loaded and its verifier never ran."""
    verdict = _verdict(_load())

    assert verdict.status == "unattested"
    assert verdict.reason == "skill_loaded_without_verifier"
    assert verdict.skill == _SKILL
    assert verdict.verifier == _VERIFIER


def test_unattested_skill_has_no_verifier_contract() -> None:
    verdict = _verdict(_load(), contracts={_SKILL: SkillContract()})

    assert verdict.status == "unattested"
    assert verdict.reason == "skill_has_no_verifier_contract"
    assert verdict.skill == _SKILL
    assert verdict.verifier is None


def test_unknown_when_the_records_are_not_whole() -> None:
    """A trimmed log settles nothing, so it produces an unknown rather than an accusation."""
    verdict = _verdict(_load(), records_complete=False)

    assert verdict.status == "unknown"
    assert verdict.reason == "records_pruned"


def test_unknown_when_a_skill_load_is_recorded_without_its_name() -> None:
    verdict = _verdict(_load(subject=""))

    assert verdict.status == "unknown"
    assert verdict.reason == "skill_name_unavailable"
    assert verdict.skill is None


def test_unknown_when_a_loaded_skill_is_no_longer_in_the_catalog() -> None:
    verdict = _verdict(_load(), contracts={})

    assert verdict.status == "unknown"
    assert verdict.reason == "contract_unavailable"
    assert verdict.skill == _SKILL


def test_a_positive_finding_survives_a_record_that_is_not_whole() -> None:
    """Whatever was pruned, a Skill with a contract was loaded and its verifier ran."""
    verdict = _verdict(_load(), _bash(), records_complete=False)

    assert verdict.status == "attested"


def test_a_cut_record_is_unknown_rather_than_an_accusation() -> None:
    """A Tool's subject is one bounded line: a verifier named after the bound is not "not run"."""
    hidden = _cut(f"cd /somebody/else/very/long/workspace/path && python {_VERIFIER}")
    verdict = _verdict(_load(), _bash(hidden))

    assert verdict.status == "unknown"
    assert verdict.reason == "arguments_truncated"
    assert verdict.skill == _SKILL


def test_a_cut_record_that_still_names_the_verifier_is_a_positive_finding() -> None:
    """The common shape: the command opens with the script, so the bound cuts the document."""
    visible = _cut(f"python tmp/{_SKILL}/{_VERIFIER} case.json --write-audit")

    verdict = _verdict(_load(), _bash(visible))

    assert verdict.status == "attested"


def test_a_failed_load_is_not_a_load() -> None:
    verdict = _verdict(_load(outcome="failed"), _bash())

    assert verdict.status == "unattested"
    assert verdict.reason == "no_skill_loaded"


def test_an_unnameable_load_prevents_a_complaint_that_would_guess() -> None:
    """A named Skill's missing verifier is not a finding while another load cannot be named."""
    verdict = _verdict(_load(), _load(subject=""), contracts={_SKILL: SkillContract(_VERIFIER)})

    assert verdict.status == "unknown"
    assert verdict.reason == "skill_name_unavailable"


def test_a_verifier_is_recognised_by_its_declared_path_and_by_its_file_name() -> None:
    """A Run that copies the script into its workspace runs the same document under its name."""
    by_name = _verdict(_load(), _bash("python case_validate.py case.json"))
    neighbouring_file = _verdict(_load(), _bash("python case_validate.py.bak case.json"))

    assert by_name.status == "attested"
    assert neighbouring_file.status == "unattested"


def test_an_artifact_attestation_must_name_its_reason_unless_it_is_attested() -> None:
    with pytest.raises(ValueError, match="must name its reason"):
        ArtifactAttestation(status="unattested")
    with pytest.raises(ValueError, match="carries no reason"):
        ArtifactAttestation(status="attested", reason="no_skill_loaded")
    with pytest.raises(ValueError, match="unknown attestation reason"):
        ArtifactAttestation(status="unattested", reason="made_up")  # type: ignore[arg-type]


def test_the_vocabulary_is_closed_and_every_reason_is_reachable() -> None:
    """One set of reasons, all of them produced by some record, none of them invented ad hoc."""
    produced = {
        _verdict().reason,
        _verdict(_load()).reason,
        _verdict(_load(), contracts={_SKILL: SkillContract()}).reason,
        _verdict(_load(), contracts={}).reason,
        _verdict(_load(), records_complete=False).reason,
        _verdict(_load(subject="")).reason,
        _verdict(
            _load(), _bash(_cut(f"cd /somebody/else/very/long/workspace && python {_VERIFIER}"))
        ).reason,
        _verdict(
            _load(),
            RecordedToolCall(
                tool_name="bash",
                call_id="call-failed",
                subject=f"python3 {_VERIFIER} out.json",
                outcome="failed",
            ),
        ).reason,
    }

    assert produced == set(ATTESTATION_REASONS)
    assert set(ATTESTATION_REASON_TEXT) == set(ATTESTATION_REASONS)
    assert all(ATTESTATION_REASON_TEXT[reason] for reason in ATTESTATION_REASONS)  # type: ignore[index]
    # The sentences are pinned to their own tokens: swapping two of them used to pass every test,
    # and a refusal would then explain one reason with another reason's words.
    assert "loaded no Agent Skill" in ATTESTATION_REASON_TEXT["no_skill_loaded"]
    assert "no command recorded" in ATTESTATION_REASON_TEXT["skill_loaded_without_verifier"]
    # and it must not assert what reading today's declaration cannot show
    assert "republished" in ATTESTATION_REASON_TEXT["skill_loaded_without_verifier"]
    assert "declares no verifier" in ATTESTATION_REASON_TEXT["skill_has_no_verifier_contract"]
    assert "reported failures" in ATTESTATION_REASON_TEXT["verifier_reported_failures"]


def test_tool_events_merge_into_one_record_per_call_in_call_order() -> None:
    """A call is one call: its start, its own subject update, and its settlement are one line."""
    calls = tool_calls_from_events(
        [
            ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
            ("tool_progress", {"tool_name": "load_skill", "call_id": "c1", "object_label": _SKILL}),
            ("tool_end", {"tool_name": "load_skill", "call_id": "c1", "outcome": "succeeded"}),
            ("tool_start", {"tool_name": "bash", "call_id": "c2"}),
            ("token", {"text": "not a tool event"}),
            ("tool_start", {"tool_name": "bash"}),
        ]
    )

    assert calls == (
        RecordedToolCall(tool_name="load_skill", call_id="c1", subject=_SKILL, outcome="succeeded"),
        RecordedToolCall(tool_name="bash", call_id="c2"),
    )


# ---------------------------------------------------------------------------
# The one shape both transports read
# ---------------------------------------------------------------------------


def test_the_projected_artifact_carries_one_shape_for_both_transports() -> None:
    stamped = stamp_projected_artifacts(_projected(), verdict=_verdict(_load(), _bash()))
    value = stamped["artifacts"][0]

    rest = AnswerArtifactResponse.model_validate(value)
    web = build_answer_presentation(
        answer="",
        sources=[],
        evidence_images=[],
        artifacts=[value],
        artifact_outcome=stamped["artifact_outcome"],
    ).artifacts[0]

    assert rest.attestation is not None and web.attestation is not None
    assert rest.attestation.model_dump() == web.attestation.model_dump()
    assert rest.attestation.model_dump() == {
        "status": "attested",
        "skill": _SKILL,
        "verifier": _VERIFIER,
        "reason": None,
        "basis": "named_in_a_recorded_command",
        "required_reads_missing": None,
    }


def test_a_verifier_that_reported_failures_attests_nothing_and_says_so() -> None:
    """The failure the Skill exists to catch must not read as a pass, or as a missing verifier.

    A verifier invoked and exiting non-zero is a Run that checked its document and found it wanting.
    The record has to say that, rather than reporting the strongest possible finding as ``attested``
    or the weakest as "did not run".
    """
    load = RecordedToolCall(tool_name="load_skill", call_id="c1", subject=_SKILL)
    failed_run = (
        load,
        RecordedToolCall(
            tool_name="bash",
            call_id="c2",
            subject=f"python3 {_VERIFIER} out.json",
            outcome="failed",
        ),
    )
    passed_run = failed_run + (
        RecordedToolCall(
            tool_name="bash",
            call_id="c3",
            subject=f"python3 {_VERIFIER} out.json",
            outcome="succeeded",
        ),
    )

    failing = _verdict(*failed_run)
    retried = _verdict(*passed_run)

    assert failing.status == "unattested"
    assert failing.reason == "verifier_reported_failures"
    assert ATTESTATION_REASON_TEXT["verifier_reported_failures"].startswith(
        "the Skill's verifier ran"
    )
    assert retried.status == "attested"


def test_reading_the_verifier_is_not_running_it() -> None:
    """Only a command line can attest: a read, a write or a listing of the script cannot.

    This is the difference between the record saying "this Run ran the verifier" and "this Run
    mentioned the file", and before the evidence was limited to execution tools a ``read`` of
    ``scripts/case_validate.py`` produced ``attested`` with no caveat.
    """
    loaded = (RecordedToolCall(tool_name="load_skill", call_id="c1", subject=_SKILL),)
    read_only = loaded + (
        RecordedToolCall(tool_name="read", call_id="c2", subject=f"case/{_VERIFIER}"),
        RecordedToolCall(tool_name="view", call_id="c3", subject=_VERIFIER),
        RecordedToolCall(tool_name="write", call_id="c4", subject=f"copy/{_VERIFIER}"),
    )
    ran = loaded + (
        RecordedToolCall(tool_name="bash", call_id="c5", subject=f"python3 {_VERIFIER} out.json"),
    )

    read_verdict = derive_run_attestation(
        contracts={_SKILL: SkillContract(_VERIFIER)}, calls=read_only, records_complete=True
    ).verdict
    ran_verdict = derive_run_attestation(
        contracts={_SKILL: SkillContract(_VERIFIER)}, calls=ran, records_complete=True
    ).verdict

    assert read_verdict.status == "unattested"
    assert read_verdict.reason == "skill_loaded_without_verifier"
    assert ran_verdict.status == "attested"
    assert ran_verdict.payload()["basis"] == "named_in_a_recorded_command"


def test_mark_publishes_an_unattested_artifact_with_the_record_beside_it() -> None:
    """The default changes nothing about publication: the Artifact ships, labelled."""
    stamped = stamp_projected_artifacts(_projected(), verdict=_verdict())
    value = stamped["artifacts"][0]

    assert value["status"] == "available"
    assert value["issue"] is None
    assert value["data_url"] == _ARTIFACT["data_url"]
    assert value["attestation"] == {
        "status": "unattested",
        "skill": None,
        "verifier": None,
        "reason": "no_skill_loaded",
        "basis": None,
        "required_reads_missing": None,
    }
    assert stamped["artifact_outcome"] == {"status": "complete", "issues": []}


def test_stamping_rewrites_an_inline_markdown_placement_as_well() -> None:
    stamped = stamp_projected_artifacts(_projected(parts=True), verdict=_verdict(_load(), _bash()))

    placement = stamped["parts"][1]["artifact"]
    assert placement["attestation"]["status"] == "attested"
    assert placement is stamped["artifacts"][0]


# ---------------------------------------------------------------------------
# A Skill that declares no verifier is not a finding about the file
# ---------------------------------------------------------------------------

#: A real builtin Skill whose front matter declares no ``attestation:`` verifier, so its Runs
#: reach the "the Skill declares no verifier" branch through the real catalog, not a fixture.
_NO_CONTRACT_SKILL = "interactive-html"


def _loads_no_contract_skill() -> tuple[RecordedToolCall, ...]:
    return (RecordedToolCall(tool_name="load_skill", call_id="c1", subject=_NO_CONTRACT_SKILL),)


def test_a_skill_without_a_verifier_contract_publishes_no_record() -> None:
    """The Artifact is published with no attestation at all, so no surface can draw an accusation.

    ``interactive-html`` ships no verifier; a Run that loads it produced a file no gate promised to
    check. Reporting that as "not attested" told every reader of every such Answer that something
    was missing, when the Skill simply never made the claim.
    """
    verdict = _verdict(*_loads_no_contract_skill(), contracts={_NO_CONTRACT_SKILL: SkillContract()})
    assert verdict.no_verifier_declared()

    stamped = stamp_projected_artifacts(_projected(parts=True), verdict=verdict)

    assert stamped["artifacts"][0]["attestation"] is None
    assert stamped["artifacts"][0]["status"] == "available"
    # The inline placement carries the same absent record, so the two surfaces cannot disagree.
    assert stamped["parts"][1]["artifact"]["attestation"] is None


def test_a_skill_that_declares_no_verifier_makes_no_claim_to_publish() -> None:
    """The label stays off and the file is served: both follow from the Skill never making a claim.

    A Skill that ships no verifier promised no gate over what it produces, so there is nothing to
    publish beside the Artifact and nothing for a delivery rule to hold it to.
    """
    verdict = _verdict(*_loads_no_contract_skill(), contracts={_NO_CONTRACT_SKILL: SkillContract()})

    stamped = stamp_projected_artifacts(_projected(), verdict=verdict)
    assert stamped["artifacts"][0]["attestation"] is None
    assert stamped["artifacts"][0]["status"] == "available"


def test_a_skill_declaring_a_verifier_that_never_ran_is_still_reported() -> None:
    """The suppression is narrow: a real gate that did not run stays a finding."""
    verdict = _verdict(_load(), _bash("python unrelated.py"))

    assert not verdict.no_verifier_declared()
    stamped = stamp_projected_artifacts(_projected(), verdict=verdict)
    assert stamped["artifacts"][0]["attestation"] is not None
    assert stamped["artifacts"][0]["attestation"]["reason"] == "skill_loaded_without_verifier"


def test_a_run_that_loaded_nothing_still_carries_the_record() -> None:
    """``no_skill_loaded`` stays published: an Artifact published with no Skill behind it is the
    silent case this module exists to remove, so it is not suppressed."""
    verdict = _verdict(_bash())

    assert not verdict.no_verifier_declared()


async def test_a_no_verifier_run_is_drawn_silently(test_config: DlightragConfig) -> None:
    """End to end through the real catalog: ``interactive-html`` declares nothing, so the label
    stays off and the Artifact is served with no record beside it."""
    calls = tuple(
        event
        for call_id, path in (
            ("c1", "SKILL.md"),
            ("c2", "references/example-brief.html"),
            ("c3", "references/fragment.md"),
        )
        for event in (
            ("tool_start", {"tool_name": "load_skill", "call_id": call_id}),
            (
                "tool_progress",
                {
                    "tool_name": "load_skill",
                    "call_id": call_id,
                    "object_label": _NO_CONTRACT_SKILL,
                    "skill_path": path,
                    "skill_window_start": 0,
                    "skill_window_end": 10,
                    "skill_document_chars": 10,
                },
            ),
            ("tool_end", {"tool_name": "load_skill", "call_id": call_id, "outcome": "succeeded"}),
        )
    )

    stamped = await attest_projected_artifacts(
        _projected(),
        runs=_Reader(*calls),
        config=test_config,
        owner_id="owner-1",
        run_id=_RUN_ID,
    )
    assert stamped["artifacts"][0]["status"] == "available"
    assert stamped["artifacts"][0]["attestation"] is None


# ---------------------------------------------------------------------------
# The reader a transport calls
# ---------------------------------------------------------------------------


class _Reader:
    """A durable event replay and transcript with no store behind them: the transport's seam."""

    def __init__(
        self,
        *events: tuple[str, dict[str, Any]],
        arguments: tuple[RunToolCall, ...] = (),
    ) -> None:
        self._events = events
        self._arguments = arguments
        self.reads = 0
        self.argument_reads = 0

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RunEvent]:
        del owner_id, run_id, after_sequence
        self.reads += 1
        events = [
            RunEvent(sequence=index + 1, event_type=kind, payload=payload, created_at=_NOW)
            for index, (kind, payload) in enumerate(self._events)
        ]

        async def _iterate() -> AsyncIterator[RunEvent]:
            for event in events:
                yield event

        return _iterate()

    async def load_tool_call_arguments(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunToolCall, ...]:
        del owner_id, run_id
        self.argument_reads += 1
        return self._arguments


async def test_an_unloaded_skill_run_stays_unattested_over_the_transport_seam(
    test_config: DlightragConfig,
) -> None:
    """The whole path: the Run's own events, the catalog's declarations, and the verdict."""
    reader = _Reader(
        ("tool_start", {"tool_name": "bash", "call_id": "c1"}),
        (
            "tool_progress",
            {"tool_name": "bash", "call_id": "c1", "object_label": "python build.py"},
        ),
        ("tool_end", {"tool_name": "bash", "call_id": "c1", "outcome": "succeeded"}),
    )

    stamped = await attest_projected_artifacts(
        _projected(),
        runs=reader,
        config=test_config,
        owner_id="owner-1",
        run_id=_RUN_ID,
    )

    assert reader.reads == 1
    assert stamped["artifacts"][0]["attestation"]["status"] == "unattested"
    assert stamped["artifacts"][0]["attestation"]["reason"] == "no_skill_loaded"


async def test_a_loaded_skill_whose_verifier_ran_is_attested_over_the_transport_seam(
    test_config: DlightragConfig,
) -> None:
    reader = _Reader(
        ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
        ("tool_progress", {"tool_name": "load_skill", "call_id": "c1", "object_label": _SKILL}),
        ("tool_end", {"tool_name": "load_skill", "call_id": "c1", "outcome": "succeeded"}),
        ("tool_start", {"tool_name": "bash", "call_id": "c2"}),
        (
            "tool_progress",
            {
                "tool_name": "bash",
                "call_id": "c2",
                "object_label": f"python tmp/{_SKILL}/{_VERIFIER} case.json",
            },
        ),
    )

    stamped = await attest_projected_artifacts(
        _projected(), runs=reader, config=test_config, owner_id="owner-1", run_id=_RUN_ID
    )

    attestation = stamped["artifacts"][0]["attestation"]
    # No load carries a read fact, so the reads the Skill declares show as missing.
    assert {row["skill"] for row in attestation["required_reads_missing"]} == {_SKILL}
    assert attestation | {"required_reads_missing": None} == {
        "status": "attested",
        "skill": _SKILL,
        "verifier": _VERIFIER,
        "reason": None,
        "basis": "named_in_a_recorded_command",
        "required_reads_missing": None,
    }


async def test_an_artifact_free_projection_never_reads_the_run(
    test_config: DlightragConfig,
) -> None:
    """Nothing to judge, nothing to read: an answer with no Artifacts costs no event replay."""
    reader = _Reader()

    stamped = await attest_projected_artifacts(
        {"artifacts": [], "artifact_outcome": {"status": "complete", "issues": []}},
        runs=reader,
        config=test_config,
        owner_id="owner-1",
        run_id=_RUN_ID,
    )

    assert stamped["artifacts"] == []
    assert reader.reads == 0


# ---------------------------------------------------------------------------
# The transcript supplies the command the display bound cut
# ---------------------------------------------------------------------------


def _skill_load_events() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
        ("tool_progress", {"tool_name": "load_skill", "call_id": "c1", "object_label": _SKILL}),
        ("tool_end", {"tool_name": "load_skill", "call_id": "c1", "outcome": "succeeded"}),
    ]


def _cut_verifier_events() -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """A bash call whose bounded subject stops before the verifier it named."""
    command = (
        f"cd /srv/sealed-report/run-workspaces/0123456789abcdef && python {_VERIFIER} case.json"
    )
    cut = _cut(command)
    assert _VERIFIER not in cut
    return [
        *_skill_load_events(),
        ("tool_start", {"tool_name": "bash", "call_id": "c2"}),
        ("tool_progress", {"tool_name": "bash", "call_id": "c2", "object_label": cut}),
        ("tool_end", {"tool_name": "bash", "call_id": "c2", "outcome": "succeeded"}),
    ], command


async def test_the_full_command_attests_where_the_bounded_subject_could_not(
    test_config: DlightragConfig,
) -> None:
    """The event subject is cut before the verifier; the transcript's command carries it."""
    events, command = _cut_verifier_events()
    argument = RunToolCall(call_id="c2", tool_name="bash", arguments={"command": command})

    event_only = _Reader(*events)
    unknown = await attest_projected_artifacts(
        _projected(), runs=event_only, config=test_config, owner_id="owner-1", run_id=_RUN_ID
    )
    with_arguments = _Reader(*events, arguments=(argument,))
    attested = await attest_projected_artifacts(
        _projected(), runs=with_arguments, config=test_config, owner_id="owner-1", run_id=_RUN_ID
    )

    assert unknown["artifacts"][0]["attestation"]["reason"] == "arguments_truncated"
    assert attested["artifacts"][0]["attestation"]["status"] == "attested"
    assert attested["artifacts"][0]["attestation"]["verifier"] == _VERIFIER
    assert with_arguments.argument_reads == 1


async def test_a_transcript_call_the_event_log_never_ran_is_not_evidence(
    test_config: DlightragConfig,
) -> None:
    """Arguments alone are not evidence: only a call the event log shows executed is merged."""
    events, command = _cut_verifier_events()
    reader = _Reader(
        *events,
        arguments=(RunToolCall(call_id="c9", tool_name="bash", arguments={"command": command}),),
    )

    stamped = await attest_projected_artifacts(
        _projected(), runs=reader, config=test_config, owner_id="owner-1", run_id=_RUN_ID
    )

    assert stamped["artifacts"][0]["attestation"]["reason"] == "arguments_truncated"


async def test_a_transcript_entry_for_another_tool_does_not_rewrite_the_call(
    test_config: DlightragConfig,
) -> None:
    """A call id names one call: an entry whose Tool disagrees is not the same call."""
    events, command = _cut_verifier_events()
    reader = _Reader(
        *events,
        arguments=(RunToolCall(call_id="c2", tool_name="read", arguments={"command": command}),),
    )

    stamped = await attest_projected_artifacts(
        _projected(), runs=reader, config=test_config, owner_id="owner-1", run_id=_RUN_ID
    )

    assert stamped["artifacts"][0]["attestation"]["reason"] == "arguments_truncated"


# ---------------------------------------------------------------------------
# Required reads: evidence beside the verdict, never inside it
# ---------------------------------------------------------------------------

_READER_SKILL = "interactive-html"
_EXAMPLES = ("references/example-brief.html", "references/example-dashboard.html")
_FRAGMENT = "references/fragment.md"
_REQUIRE_EXAMPLE = RequiredRead(paths=_EXAMPLES, when="before writing the fragment")
_REQUIRE_FRAGMENT = RequiredRead(paths=(_FRAGMENT,), when="before the build")
_READS: dict[str, SkillContract] = {
    _READER_SKILL: SkillContract(required_reads=(_REQUIRE_EXAMPLE, _REQUIRE_FRAGMENT))
}


def _read_call(
    path: str,
    start: int | None = 0,
    end: int | None = 100,
    chars: int | None = 100,
    *,
    skill: str = _READER_SKILL,
    call_id: str = "call-read",
    outcome: str | None = "succeeded",
) -> RecordedToolCall:
    """One ``load_skill`` call carrying the read fact; ``start=None`` is a ``write_to`` copy."""
    return RecordedToolCall(
        tool_name="load_skill",
        call_id=call_id,
        subject=skill,
        outcome=outcome,
        read=SkillRead(skill=skill, path=path, start=start, end=end, document_chars=chars),
    )


def _hub(skill: str = _READER_SKILL, call_id: str = "call-hub") -> RecordedToolCall:
    """The load of the Skill's own SKILL.md, read whole."""
    return _read_call("SKILL.md", 0, 50, 50, skill=skill, call_id=call_id)


def _missing(*calls: RecordedToolCall, **kwargs: Any) -> tuple[MissingRead, ...] | None:
    return _verdict(
        *calls, contracts=kwargs.pop("contracts", _READS), **kwargs
    ).required_reads_missing


def test_a_read_is_carried_from_the_progress_event_onto_the_recorded_call() -> None:
    calls = tool_calls_from_events(
        [
            ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
            (
                "tool_progress",
                {
                    "tool_name": "load_skill",
                    "call_id": "c1",
                    "object_label": _READER_SKILL,
                    "skill_path": _FRAGMENT,
                    "skill_window_start": 0,
                    "skill_window_end": 90,
                    "skill_document_chars": 90,
                },
            ),
            ("tool_end", {"tool_name": "load_skill", "call_id": "c1", "outcome": "succeeded"}),
            ("tool_start", {"tool_name": "bash", "call_id": "c2"}),
            (
                "tool_progress",
                {"tool_name": "bash", "call_id": "c2", "skill_path": "not-a-load.md"},
            ),
        ]
    )

    assert calls[0].read == SkillRead(
        skill=_READER_SKILL, path=_FRAGMENT, start=0, end=90, document_chars=90
    )
    # Only a load_skill call has a read: another tool that carried the key is not reading a Skill.
    assert calls[1].read is None


def test_a_read_the_runtime_cut_is_carried_without_its_window() -> None:
    """The cut is on the settlement, not on the progress event the window came from."""
    events = [
        ("tool_start", {"tool_name": "load_skill", "call_id": "c1"}),
        (
            "tool_progress",
            {
                "tool_name": "load_skill",
                "call_id": "c1",
                "object_label": _READER_SKILL,
                "skill_path": _FRAGMENT,
                "skill_window_start": 0,
                "skill_window_end": 90,
                "skill_document_chars": 90,
            },
        ),
        (
            "tool_end",
            {
                "tool_name": "load_skill",
                "call_id": "c1",
                "outcome": "succeeded",
                "observed_truncated": True,
            },
        ),
    ]

    (call,) = tool_calls_from_events(events)

    assert call.read == SkillRead(skill=_READER_SKILL, path=_FRAGMENT)
    # The two merges of the same events apply one rule, so they cannot say different things.
    (shared,) = skill_reads_from_events(events)
    assert shared.read == call.read


def test_a_document_whose_only_read_was_cut_is_listed_as_missing() -> None:
    """A post-fact record, so the Run is settled; but a cut window is not a read."""
    cut = RecordedToolCall(
        tool_name="load_skill",
        call_id="cut",
        subject=_READER_SKILL,
        outcome="succeeded",
        read=SkillRead(skill=_READER_SKILL, path=_FRAGMENT),
    )
    example = _read_call(_EXAMPLES[0], call_id="a")

    missing = _missing(_hub(), example, cut)

    assert missing == (
        MissingRead(skill=_READER_SKILL, paths=(_FRAGMENT,), when="before the build"),
    )
    # The same document read whole by another call is read.
    assert _missing(_hub(), example, cut, _read_call(_FRAGMENT, call_id="whole")) == ()


def test_every_declared_read_read_to_the_end_leaves_nothing_missing() -> None:
    missing = _missing(
        _hub(),
        _read_call(_EXAMPLES[1], call_id="a"),
        _read_call(_FRAGMENT, call_id="b"),
    )

    assert missing == ()


def test_an_any_of_entry_is_satisfied_by_one_of_its_paths() -> None:
    assert _missing(_hub(), _read_call(_EXAMPLES[0], call_id="a"), _read_call(_FRAGMENT)) == ()


def test_an_unread_entry_is_named_with_its_paths_and_its_when() -> None:
    missing = _missing(_hub(), _read_call(_EXAMPLES[0], call_id="a"))

    assert missing == (
        MissingRead(skill=_READER_SKILL, paths=(_FRAGMENT,), when="before the build"),
    )
    nothing_read = _missing(_hub())
    assert nothing_read == (
        MissingRead(
            skill=_READER_SKILL,
            paths=_EXAMPLES,
            when="before writing the fragment",
        ),
        MissingRead(skill=_READER_SKILL, paths=(_FRAGMENT,), when="before the build"),
    )


def test_windows_across_several_calls_add_up_to_a_read_and_the_first_one_alone_does_not() -> None:
    first = _read_call(_FRAGMENT, 0, 40, 100, call_id="w1")
    rest = _read_call(_FRAGMENT, 40, 100, 100, call_id="w2")
    example = _read_call(_EXAMPLES[0], call_id="a")

    assert _missing(_hub(), example, first) == (
        MissingRead(skill=_READER_SKILL, paths=(_FRAGMENT,), when="before the build"),
    )
    assert _missing(_hub(), example, first, rest) == ()


def test_a_failed_load_is_silent_without_unsettling_the_rest() -> None:
    """A refused call hands over nothing and records no fact on purpose."""
    refused = RecordedToolCall(
        tool_name="load_skill", call_id="bad", subject=_READER_SKILL, outcome="failed"
    )

    missing = _missing(_hub(), refused, _read_call(_EXAMPLES[0], call_id="a"))

    assert missing == (
        MissingRead(skill=_READER_SKILL, paths=(_FRAGMENT,), when="before the build"),
    )


#: Each way the records cannot say, as the calls and keywords a derivation is given.
_UNSETTLED: dict[str, Any] = {
    "the log is incomplete": lambda: _missing(
        _hub(), _read_call(_FRAGMENT), records_complete=False
    ),
    "a load cannot be named": lambda: _missing(_hub(), _read_call(_FRAGMENT), _load(subject="")),
    "a loaded Skill is gone from the catalog": lambda: _missing(_hub(), contracts={}),
    "no loaded Skill declares a read": lambda: _missing(
        _hub(), contracts={_READER_SKILL: SkillContract()}
    ),
    "nothing was loaded": lambda: _missing(_bash()),
}


@pytest.mark.parametrize("settle", _UNSETTLED.values(), ids=_UNSETTLED.keys())
def test_the_reads_cannot_be_settled_when_the_records_cannot_say(settle: Any) -> None:
    """``None`` is not ``()``: an unsettled record must never read as "nothing was missed"."""
    assert settle() is None


def test_missing_reads_change_neither_the_status_nor_the_reason() -> None:
    """Evidence beside the verdict, never inside it: the verdict is what it was without them."""
    calls = (_hub(),)
    contracts = {_READER_SKILL: SkillContract(required_reads=(_REQUIRE_FRAGMENT,))}

    with_reads = _verdict(*calls, contracts=contracts)
    without = _verdict(*calls, contracts={_READER_SKILL: SkillContract()})

    assert with_reads.required_reads_missing != without.required_reads_missing
    assert (
        (with_reads.status, with_reads.reason, with_reads.skill)
        == (
            without.status,
            without.reason,
            without.skill,
        )
        == ("unattested", "skill_has_no_verifier_contract", _READER_SKILL)
    )

    # Nor does a verdict that is attested stop being so because a read was skipped.
    both = {
        _SKILL: SkillContract(_VERIFIER, (_REQUIRE_FRAGMENT,)),
    }
    attested = _verdict(_read_call("SKILL.md", 0, 5, 5, skill=_SKILL), _bash(), contracts=both)
    assert attested.status == "attested" and attested.reason is None
    assert attested.required_reads_missing == (
        MissingRead(skill=_SKILL, paths=(_FRAGMENT,), when="before the build"),
    )


def test_the_payload_carries_none_an_empty_list_or_the_entries() -> None:
    unsettled = ArtifactAttestation(status="unattested", reason="no_skill_loaded")
    complete = replace(unsettled, required_reads_missing=())
    missing = replace(
        unsettled,
        required_reads_missing=(MissingRead(skill=_READER_SKILL, paths=_EXAMPLES, when="before"),),
    )

    assert unsettled.payload()["required_reads_missing"] is None
    assert complete.payload()["required_reads_missing"] == []
    assert missing.payload()["required_reads_missing"] == [
        {"skill": _READER_SKILL, "paths": list(_EXAMPLES), "when": "before"}
    ]
    # The wire model accepts what the payload says and refuses a key it does not know.
    parsed = ArtifactAttestationResponse.model_validate(missing.payload())
    assert parsed.required_reads_missing is not None
    assert parsed.required_reads_missing[0].paths == list(_EXAMPLES)
    assert (
        ArtifactAttestationResponse.model_validate(unsettled.payload()).required_reads_missing
        is None
    )


def _no_verifier_verdict(*, missing: tuple[MissingRead, ...] | None) -> ArtifactAttestation:
    return ArtifactAttestation(
        status="unattested",
        skill=_READER_SKILL,
        reason="skill_has_no_verifier_contract",
        required_reads_missing=missing,
    )


_ONE_MISSING = (MissingRead(skill=_READER_SKILL, paths=_EXAMPLES, when="before writing"),)


def test_a_skill_with_no_verifier_publishes_a_record_only_when_a_read_is_missing() -> None:
    def stamped(missing: tuple[MissingRead, ...] | None) -> dict[str, Any]:
        return stamp_projected_artifacts(
            _projected(parts=True),
            verdict=_no_verifier_verdict(missing=missing),
        )

    for silent in (None, ()):
        assert stamped(silent)["artifacts"][0]["attestation"] is None
        assert stamped(silent)["parts"][1]["artifact"]["attestation"] is None

    value = stamped(_ONE_MISSING)
    record = value["artifacts"][0]["attestation"]
    assert record is not None
    assert record["reason"] == "skill_has_no_verifier_contract"
    assert record["required_reads_missing"] == [
        {"skill": _READER_SKILL, "paths": list(_EXAMPLES), "when": "before writing"}
    ]
    # The inline placement carries the same record, so the two surfaces cannot disagree.
    assert value["parts"][1]["artifact"]["attestation"] == record
    assert value["artifacts"][0]["status"] == "available"


async def test_the_whole_path_flags_a_report_built_without_reading_its_reference(
    test_config: DlightragConfig,
) -> None:
    """Real catalog, real events: ``interactive-html`` declares a read and no verifier."""

    def load(call_id: str, path: str, start: int, end: int, chars: int) -> list[Any]:
        base = {"tool_name": "load_skill", "call_id": call_id}
        return [
            ("tool_start", dict(base)),
            (
                "tool_progress",
                {
                    **base,
                    "object_label": _READER_SKILL,
                    "skill_path": path,
                    "skill_window_start": start,
                    "skill_window_end": end,
                    "skill_document_chars": chars,
                },
            ),
            ("tool_end", {**base, "outcome": "succeeded"}),
        ]

    hub = load("c1", "SKILL.md", 0, 70, 70)
    reference = load("c2", "references/fragment.md", 0, 500, 500)

    skipped = await attest_projected_artifacts(
        _projected(),
        runs=_Reader(*hub),
        config=test_config,
        owner_id="owner-1",
        run_id=_RUN_ID,
    )
    read = await attest_projected_artifacts(
        _projected(),
        runs=_Reader(*hub, *reference),
        config=test_config,
        owner_id="owner-1",
        run_id=_RUN_ID,
    )
    required = await attest_projected_artifacts(
        _projected(),
        runs=_Reader(*hub),
        config=test_config,
        owner_id="owner-1",
        run_id=_RUN_ID,
    )

    (entry,) = skipped["artifacts"][0]["attestation"]["required_reads_missing"]
    assert entry["skill"] == _READER_SKILL
    assert entry["paths"] == ["references/fragment.md"]
    assert entry["when"] == "before writing or restructuring the fragment"
    assert skipped["artifacts"][0]["status"] == "available"
    # Read to the end: nothing to report, and a Skill with no verifier publishes no record.
    assert read["artifacts"][0]["attestation"] is None
    # `require` enforces a verifier claim; this Skill made none, so the Artifact is served with
    # the record that names the skipped read -- the missing read is evidence, not a gate.
    assert required["artifacts"][0]["status"] == "available"
    assert required["artifacts"][0]["attestation"]["required_reads_missing"] == [entry]
