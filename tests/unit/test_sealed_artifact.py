# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The sealed-artifact gate: candidates go to the Skill's gate, and its three states are honoured."""

from __future__ import annotations

import zipfile
from pathlib import Path
from types import SimpleNamespace

from dlightrag.engine.agent.sealed_artifact import (
    CHANGED,
    UNREADABLE,
    SealedArtifact,
    attest_bytes,
    attest_refused,
    contracts,
)

_SEALED_JSON = '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}'


def _gate_script(directory: Path, claim: str, sealed: bool) -> Path:
    """A minimal Skill gate: prints the three-state report, exits per contract."""
    directory.mkdir(parents=True, exist_ok=True)
    gate = directory / "gate.py"
    exit_code = 0 if sealed or claim == "none" else 1
    gate.write_text(
        "import json\n"
        "print(json.dumps({\n"
        f"    'claim': {claim!r},\n"
        f"    'sealed': {sealed!r},\n"
        "    'reasons': ['the seal is stale'] if not "
        f"{sealed!r} and {claim!r} == 'unsealed' else [],\n"
        "}))\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return gate


def _contract(
    directory: Path, *, name: str = "case.json", claim: str = "sealed", sealed: bool = True
) -> SealedArtifact:
    return SealedArtifact(
        skill="sealed-report",
        name=name,
        gate=_gate_script(directory, claim, sealed),
    )


def _verdict(data: bytes, artifact: str, contract: SealedArtifact, *, size_limit: int = 1024):
    return attest_bytes(data, artifact, (contract,), size_limit=size_limit)


def test_contracts_are_builtin_declarations_with_a_filesystem_gate(tmp_path: Path) -> None:
    builtin = tmp_path / "builtin"
    (builtin / "scripts").mkdir(parents=True)
    (builtin / "scripts" / "gate.py").write_text("", encoding="utf-8")
    declaration = SimpleNamespace(name="case.json", gate="scripts/gate.py")
    metadata = [
        SimpleNamespace(
            name="sealed-report",
            source="builtin",
            root=builtin,
            sealed_artifact=declaration,
        ),
        SimpleNamespace(
            name="owner-skill", source="owner", root=builtin, sealed_artifact=declaration
        ),
        SimpleNamespace(name="plain", source="builtin", root=builtin, sealed_artifact=None),
    ]

    found = contracts(metadata)

    assert [contract.skill for contract in found] == ["sealed-report"]
    assert found[0].name == "case.json"
    assert found[0].gate == builtin / "scripts" / "gate.py"


def test_a_sealed_report_attests_the_candidate(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill", claim="sealed", sealed=True)

    verdict = _verdict(_SEALED_JSON.encode(), "case.json", contract)

    assert verdict is not None
    assert verdict.claimed is True and verdict.sealed is True


def test_an_unsealed_report_withholds_the_candidate(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill", claim="unsealed", sealed=False)

    verdict = _verdict(_SEALED_JSON.encode(), "case.json", contract)

    assert verdict is not None
    assert verdict.claimed is True and verdict.sealed is False
    assert verdict.reasons == ("the seal is stale",)
    assert "refused case.json" in verdict.refusal()


def test_a_none_report_attaches_the_name_as_an_ordinary_document(tmp_path: Path) -> None:
    """A draft under the reserved name is ordinary; the gate, not the harness, says so."""
    contract = _contract(tmp_path / "skill", claim="none", sealed=False)

    assert _verdict(b'{"case_version": "2.0"}', "case.json", contract) is None


def test_a_file_under_no_declared_name_never_reaches_a_gate(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill")

    assert _verdict(b"Grounded analysis.", "analysis.md", contract) is None


def test_a_zip_candidate_is_its_member(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill", claim="sealed", sealed=True)
    archive = tmp_path / "deliverable.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("case.json", _SEALED_JSON)
        zf.writestr("notes.txt", "notes")

    verdict = _verdict(archive.read_bytes(), archive.name, contract)

    assert verdict is not None and verdict.sealed is True


def test_a_zip_with_two_candidates_is_refused_before_reading_either(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill")
    archive = tmp_path / "deliverable.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("case.json", _SEALED_JSON)
        zf.writestr("copies/case.json", _SEALED_JSON)

    verdict = _verdict(archive.read_bytes(), archive.name, contract)

    assert verdict is not None and verdict.sealed is False
    assert "2 files named" in verdict.refusal()


def test_an_oversized_zip_candidate_is_refused(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill")
    archive = tmp_path / "deliverable.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("case.json", "x" * 4096)

    verdict = _verdict(archive.read_bytes(), archive.name, contract, size_limit=64)

    assert verdict is not None and verdict.sealed is False
    assert "larger than one Artifact may carry" in verdict.refusal()


def test_a_zip_with_no_candidate_name_is_ordinary(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill")
    archive = tmp_path / "deliverable.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("notes.txt", "notes")

    assert _verdict(archive.read_bytes(), archive.name, contract) is None


def test_a_malformed_gate_report_is_a_refusal_not_a_pass(tmp_path: Path) -> None:
    directory = tmp_path / "skill"
    directory.mkdir()
    gate = directory / "gate.py"
    gate.write_text('print(\'{"sealed": "yes", "reasons": "nope"}\')\n', encoding="utf-8")
    contract = SealedArtifact(skill="sealed-report", name="case.json", gate=gate)

    verdict = _verdict(_SEALED_JSON.encode(), "case.json", contract)

    assert verdict is not None and verdict.sealed is False
    assert "malformed" in verdict.refusal()


def test_a_gate_that_prints_no_report_is_a_refusal(tmp_path: Path) -> None:
    directory = tmp_path / "skill"
    directory.mkdir()
    gate = directory / "gate.py"
    gate.write_text("raise SystemExit(1)\n", encoding="utf-8")
    contract = SealedArtifact(skill="sealed-report", name="case.json", gate=gate)

    verdict = _verdict(_SEALED_JSON.encode(), "case.json", contract)

    assert verdict is not None and verdict.sealed is False
    assert verdict.reasons


def test_a_name_matching_file_the_gate_cannot_judge_is_refused_not_passed(tmp_path: Path) -> None:
    contract = _contract(tmp_path / "skill")

    for reason in (UNREADABLE, CHANGED):
        refused = attest_refused("artifacts/case.json", (contract,), reason)
        assert refused is not None and refused.claimed is True and refused.sealed is False
        assert refused.reasons == (reason,)
        assert attest_refused("artifacts/other.json", (contract,), reason) is None
    refusal = refused.refusal()
    assert "Regenerate it as the sealed-report Skill instructs" in refusal
    assert "blocking 0" not in refusal


def test_the_declared_name_matches_without_case(tmp_path: Path) -> None:
    """A name that differs only in case is the same claim; the baseline was case-insensitive."""
    contract = _contract(tmp_path / "skill", name="case.json", claim="unsealed", sealed=False)

    verdict = _verdict(_SEALED_JSON.encode(), "Case.json", contract)

    assert verdict is not None and verdict.claimed is True and verdict.sealed is False
