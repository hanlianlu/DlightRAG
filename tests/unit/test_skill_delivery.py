# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The attempt's delivery ledger: reads, verifier evidence, and the preview an attach carries."""

import pytest

from dlightrag.engine.agent.skill_delivery import (
    DeliveryLedger,
    PageProvenance,
    executes_verifier,
)
from dlightrag.engine.agent.skill_reads import RequiredRead, SkillRead
from dlightrag.engine.agent.skills import SealedArtifactDeclaration

_LABEL = "Unsealed research memo / 未封存研究备忘"
_SEALED_DECLARATION = SealedArtifactDeclaration(
    name="case.json", gate="scripts/verify_sealed_case.py", unsealed_label=_LABEL
)

_SKILL = "sealed-report"
_VERIFIER = "scripts/validate_model.py"
_GROUNDING = RequiredRead(
    paths=("references/grounding-protocol.md",),
    when="before the first number — identity, as-of, provenance tags, currency, halt conditions",
)


def _read(
    path: str,
    *,
    start: int = 0,
    end: int = 10,
    chars: int = 10,
    skill: str = _SKILL,
) -> SkillRead:
    return SkillRead(skill=skill, path=path, start=start, end=end, document_chars=chars)


def _ledger(
    *,
    verifier: str | None = _VERIFIER,
    required_reads: tuple[RequiredRead, ...] = (_GROUNDING,),
) -> DeliveryLedger:
    ledger = DeliveryLedger()
    ledger.record_load(
        skill=_SKILL,
        verifier=verifier,
        required_reads=required_reads,
        read=_read("SKILL.md"),
    )
    return ledger


def test_a_declared_read_the_attempt_did_not_make_is_reported() -> None:
    ledger = _ledger()

    preview = ledger.preview()

    assert preview is not None
    assert preview.skills == (_SKILL,)
    assert [(row.skill, row.paths) for row in preview.missing_reads] == [
        (_SKILL, ("references/grounding-protocol.md",))
    ]
    text = preview.text()
    assert "Delivery record preview" in text
    assert (
        f"not read: references/grounding-protocol.md (declared by {_SKILL}: before the first number"
        in text
    )


def test_a_whole_document_read_covers_the_declaration() -> None:
    ledger = _ledger()
    ledger.record_load(
        skill=_SKILL,
        verifier=_VERIFIER,
        required_reads=(_GROUNDING,),
        read=_read("references/grounding-protocol.md", start=0, end=10, chars=10),
    )

    preview = ledger.preview()

    assert preview is not None
    assert preview.missing_reads == ()


def test_a_command_naming_the_verifier_is_evidence_that_it_ran() -> None:
    ledger = _ledger()
    ledger.record_load(
        skill=_SKILL,
        verifier=_VERIFIER,
        required_reads=(_GROUNDING,),
        read=_read("references/grounding-protocol.md", start=0, end=10, chars=10),
    )
    ledger.record_execution(
        tool_name="bash",
        command="python scripts/validate_model.py case.draft.json --write-audit "
        "--page artifacts/report.html",
        failed=False,
    )

    preview = ledger.preview()

    assert preview is not None
    assert preview.verifiers[0].named is True
    # Everything the record would report is covered: the preview says nothing.
    assert preview.text() == ""


def test_a_failed_command_naming_the_verifier_is_not_attestation() -> None:
    ledger = _ledger()
    ledger.record_execution(
        tool_name="bash",
        command="python scripts/validate_model.py case.draft.json",
        failed=True,
    )

    preview = ledger.preview()

    assert preview is not None
    assert preview.verifiers[0].failed is True
    text = preview.text()
    assert f"a command naming {_SKILL}'s verifier {_VERIFIER} ran and failed" in text


def test_a_skill_with_no_verifier_and_no_missing_read_has_no_preview_text() -> None:
    ledger = _ledger(verifier=None, required_reads=())
    ledger.record_load(
        skill="interactive-html",
        verifier=None,
        required_reads=(),
        read=_read("SKILL.md", skill="interactive-html"),
    )

    preview = ledger.preview()

    assert preview is not None
    assert preview.verifiers == ()
    assert preview.missing_reads == ()
    assert preview.text() == ""


def test_no_loaded_skill_means_no_preview() -> None:
    assert DeliveryLedger().preview() is None
    assert DeliveryLedger().preview_text() == ""


def test_an_attachment_with_no_loaded_skill_states_the_reader_record() -> None:
    """The reader will meet ``unattested``; the attach and the checkpoint say so while it can be fixed."""
    ledger = DeliveryLedger()

    assert ledger.no_skill_delivery() == ()

    ledger.record_attachment("report.html")
    lines = ledger.no_skill_delivery()

    assert len(lines) == 1
    assert "unattested" in lines[0]
    assert "report.html" in lines[0]
    # A second attach of the same file is the same finding, said once.
    ledger.record_attachment("report.html")
    assert ledger.no_skill_delivery() == lines

    # Once a Skill is loaded, the preview speaks for the delivery and this line is gone.
    ledger.record_load(
        skill=_SKILL,
        verifier=_VERIFIER,
        required_reads=(_GROUNDING,),
        read=_read("SKILL.md"),
    )
    assert ledger.no_skill_delivery() == ()


def test_an_html_page_without_the_toolkit_provenance_is_reported() -> None:
    ledger = _ledger(verifier=None, required_reads=())

    text = ledger.preview_text(page=PageProvenance(file="report.html", toolkit_built=False))

    assert "carries no report-source block" in text
    assert "report.html" in text


def test_an_unreadable_page_is_no_finding() -> None:
    ledger = _ledger(verifier=None, required_reads=())

    assert ledger.preview_text(page=None) == ""


def test_the_preview_carries_the_record_never_a_refusal() -> None:
    text = _ledger().preview_text()

    assert "nothing here is enforced" in text
    assert "withholds nothing" in text


@pytest.mark.parametrize(
    ("command", "runs"),
    [
        ("python3 -u -X utf8 -W ignore scripts/case_validate.py case.json", True),
        ("cd scripts && ./case_validate.py case.json", True),
        ("PYTHONUTF8=1 python case_validate.py case.json", True),
        ("grep -n python scripts/case_validate.py", False),
        ("echo python case_validate.py", False),
        ("cd scripts && grep -n 'def check_' case_validate.py", False),
        ("python scripts/other.py", False),
        ("cp scripts/case_validate.py .", False),
        ("ls artifacts && python3 scripts/case_validate.py x", True),
        ("cp -r kit/scripts . && python3 scripts/case_validate.py x", True),
        ("cd scripts\npython3.14 case_validate.py x", True),
        ("find . -name case_validate.py", False),
        ("chmod +x scripts/case_validate.py", False),
        ("python3 -c \"open('scripts/case_validate.py')\"", False),
        ("python3 scripts/case_validate.py x | tail -20", False),
        ("python3 scripts/case_validate.py x; echo done", False),
        ("python3 scripts/case_validate.py x || true", False),
        ("python3 scripts/case_validate.py x && echo sealed", True),
        ("python3 scripts/case_validate.py.bak x", False),
    ],
)
def test_a_command_runs_the_verifier_by_what_it_starts_with(command: str, runs: bool) -> None:
    assert executes_verifier(command, "scripts/case_validate.py") is runs


def test_a_verifier_passes_by_its_exit_code_and_not_by_what_it_printed() -> None:
    ledger = _ledger()
    command = "python scripts/validate_model.py case.draft.json --write-audit"

    ledger.record_execution(tool_name="bash", command=command, failed=True)
    ledger.record_execution(tool_name="bash", command=command, failed=False)
    preview = ledger.preview()
    assert preview is not None and preview.verifiers[0].passed is True

    unrun = _ledger()
    unrun.record_execution(tool_name="bash", command=f"grep -n check_ {_VERIFIER}", failed=False)
    reading = unrun.preview()
    assert reading is not None and reading.verifiers[0].named is False
    assert "no command recorded so far" in reading.text()


def test_the_answer_label_follows_the_gate_attestation_not_the_reads_or_the_verifier() -> None:
    """A loaded Skill that declares a sealed artifact labels the answer until its gate attests one."""
    ledger = DeliveryLedger()
    ledger.record_load(
        skill="interactive-html",
        verifier=None,
        required_reads=(),
        read=SkillRead(skill="interactive-html", path="SKILL.md"),
    )
    other = ledger.preview()
    assert other is not None and other.unsealed_labels == ()  # no declaration, no label

    ledger.record_load(
        skill=_SKILL,
        verifier=_VERIFIER,
        required_reads=(),
        read=_read("SKILL.md"),
        sealed_artifact=_SEALED_DECLARATION,
    )
    loaded = ledger.preview()
    assert loaded is not None
    assert loaded.sealed_skills == (_SKILL,)
    assert loaded.unsealed_labels == (_LABEL,)

    ledger.record_execution(tool_name="bash", command=f"python {_VERIFIER} c.json", failed=False)
    verified = ledger.preview()
    assert verified is not None
    assert verified.unsealed_labels == (_LABEL,)  # a clean verifier run is not a seal

    ledger.record_sealed_attestation(_SKILL)
    sealed = ledger.preview()
    assert sealed is not None
    assert sealed.unsealed_labels == ()
    assert sealed.sealed_attested == (_SKILL,)


def test_a_gate_refusal_is_a_finding_and_a_later_attestation_supersedes_it() -> None:
    ledger = DeliveryLedger()
    ledger.record_sealed_refusal(
        _SKILL, "case.json", ("the seal does not cover the current bytes",)
    )
    assert (
        ledger.preview() is None
    )  # no Skill was loaded: the preview is not this record's business

    ledger.record_load(
        skill=_SKILL,
        verifier=_VERIFIER,
        required_reads=(),
        read=_read("SKILL.md"),
        sealed_artifact=_SEALED_DECLARATION,
    )
    ledger.record_execution(tool_name="bash", command=f"python {_VERIFIER} c.json", failed=False)
    refused = ledger.preview()
    assert refused is not None
    assert refused.sealed_refused == (
        ("case.json", _SKILL, "the seal does not cover the current bytes"),
    )
    assert refused.unsealed_labels == (_LABEL,)
    assert "gate refused case.json" in refused.text()

    ledger.record_sealed_attestation(_SKILL)
    repaired = ledger.preview()
    assert repaired is not None
    assert repaired.sealed_refused == ()
    assert repaired.lines() == ()
