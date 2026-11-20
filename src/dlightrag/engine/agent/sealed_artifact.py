# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The one Artifact a Skill lets claim *sealed*, and the gate that attests it from its bytes.

ADR 0066 makes the harness the owner of a sealed case's construction: the Agent must not be able
to hand the reader a file that says "sealed" when the Skill's own gate never verified it, and must
be able to hand over an *already sealed* file again on a later turn. Both are content questions,
not filename or ledger questions:

- **The declared name sends a candidate to the gate; the gate says whether it claims.** A Skill
  declares one plain artifact name and the read-only gate it runs. A file under that name — or a
  member of that name inside a zip — is always a *candidate*. The gate reports the three states the
  harness needs: ``none`` (an ordinary file under the name, e.g. a draft — attach it as a
  document), ``unsealed`` (it claims sealed and does not hold — withhold it), ``sealed`` (attach
  it). Malformed reports, crashes, timeouts and oversized or unreadable candidates are refusals,
  never passes: the name is reserved, so the gate must say "ordinary" explicitly.
- **Attestation is the Skill's gate run on the bytes.** The gate reads one path and never writes;
  it is run on a temporary copy of bytes the Artifact check already read in the same workspace hold,
  so it judges the file the check accepted, not a later read of its path. The harness never
  re-implements a Skill's seal format: a second implementation of a seal is a second seal.
- **The current attempt's ledger is preview, not gate.** A continuation or a later turn re-attaches
  an Artifact whose sealing command ran in another attempt; the bytes are the same and the gate is
  the same, so the result is the same. A successful gate attestation *is* recorded in the ledger, so
  the reader-facing label can follow the fact instead of an older attempt's missing read.

Only a built-in Skill may declare a sealed artifact: its gate is product code the harness runs in
the serving process, not an owner's or an operator's installed script.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

#: The gate is a Skill's own check of a small JSON artifact, not a build. The bound is operational
#: (a gate that hangs must not hold an attach), not a correctness threshold.
_GATE_TIMEOUT_SECONDS = 120


@dataclass(frozen=True, slots=True)
class SealedArtifact:
    """One built-in Skill's declaration: the name that may claim sealed, and the gate that judges it.

    ``name`` is a plain file name, not a path: an Artifact under it (or a zip member under it) is a
    candidate for the gate. ``gate`` is the absolute path of the Skill's read-only gate script,
    already checked to be a file inside the Skill directory at discovery. The harness runs it; it
    does not read the Skill's documents to decide what sealed means.
    """

    skill: str
    name: str
    gate: Path


@dataclass(frozen=True, slots=True)
class SealedVerdict:
    """What one candidate Artifact turned out to be: a claim or an ordinary file, and the gate's word."""

    artifact: str
    skill: str
    name: str
    claimed: bool
    sealed: bool
    reasons: tuple[str, ...] = ()

    def refusal(self) -> str:
        """The ToolResult text a refused attach carries: the reasons, and what to do next."""
        reasons = "; ".join(self.reasons) or "the gate reported no reason"
        return (
            f"refused {self.artifact}: it claims the sealed {self.name!r} of the {self.skill} "
            f"Skill, but that Skill's gate did not attest it: {reasons}. Regenerate it as the "
            f"{self.skill} Skill instructs and attach it again. The answer may still be delivered "
            "unsealed; only the sealed artifact is withheld."
        )


def contracts(metadata: Iterable[object]) -> tuple[SealedArtifact, ...]:
    """The sealed-artifact declarations among discovered built-in Skills.

    Discovery has already refused a declaration that names a gate the Skill does not carry, a
    non-built-in declaring one, and two built-ins declaring the same name; this function resolves
    the declaration to the gate file the harness can run. A built-in root that is not a filesystem
    path (an importlib zip rather than the packaged install) cannot be executed in place: that is a
    loud deployment error, not a silent gate-off.
    """
    found: list[SealedArtifact] = []
    for skill in metadata:
        declaration = getattr(skill, "sealed_artifact", None)
        source = getattr(skill, "source", "")
        root = getattr(skill, "root", None)
        name = getattr(declaration, "name", None)
        gate = getattr(declaration, "gate", None)
        if not name or not gate or source != "builtin":
            continue
        if not isinstance(root, Path):
            raise ValueError(
                f"built-in Skill {getattr(skill, 'name', '?')!r} declares a sealed artifact but its "
                "root is not a filesystem path; the harness cannot run its gate in place"
            )
        found.append(
            SealedArtifact(
                skill=str(getattr(skill, "name", "?")),
                name=name,
                gate=root / gate,
            )
        )
    return tuple(found)


@dataclass(frozen=True, slots=True)
class _Claim:
    """One candidate under a declaration: the bytes to gate, or why they cannot be gated."""

    contract: SealedArtifact
    payloads: tuple[bytes, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _GateReport:
    claimed: bool
    sealed: bool
    reasons: tuple[str, ...] = ()


_MALFORMED = "the gate's report is malformed: it states claim/sealed and a list of reasons"


def _zip_candidate(
    data: bytes,
    contract: SealedArtifact,
    *,
    size_limit: int,
) -> _Claim | None:
    """The single zip member under the declared name, or a refusal for an unbounded/ambiguous one.

    A memory-bomb archive cannot stage one: only the members carrying the declared name are
    decompressed, each is bounded by the same limit the Artifact check applies to one file, and more
    than one such member is refused before any of them is read (a sealed artifact is one file).
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile, OSError:
        return None
    with archive:
        members = [
            member
            for member in archive.infolist()
            if PurePosixPath(member.filename).name.casefold() == contract.name.casefold()
        ]
        if not members:
            return None
        if len(members) > 1:
            return _Claim(
                contract=contract,
                payloads=(),
                reasons=(
                    f"the archive carries {len(members)} files named {contract.name!r}: a sealed "
                    "artifact is one file, not a set to choose from",
                ),
            )
        member = members[0]
        if member.file_size > size_limit:
            return _Claim(
                contract=contract,
                payloads=(),
                reasons=(f"the archive's {contract.name!r} is larger than one Artifact may carry",),
            )
        try:
            payload = archive.read(member)
        except OSError, RuntimeError, zipfile.BadZipFile:
            return _Claim(
                contract=contract,
                payloads=(),
                reasons=(f"the archive's {contract.name!r} cannot be read: the zip is corrupt",),
            )
    return _Claim(contract=contract, payloads=(payload,))


def _candidates(
    artifact_name: str,
    data: bytes,
    contracts: Sequence[SealedArtifact],
    *,
    size_limit: int,
) -> list[_Claim]:
    """Every declaration this Artifact is a candidate for; the gate, not a marker, decides claim.

    A file matching a declared name is always a candidate, including a draft or a data file under
    that name: only the Skill's gate may call it ordinary. A file matching no name is a zip candidate
    for every declaration, since a sealed artifact may be shipped inside an archive.
    """
    direct = [
        contract for contract in contracts if contract.name.casefold() == artifact_name.casefold()
    ]
    candidates = [_Claim(contract=contract, payloads=(data,)) for contract in direct]
    for contract in contracts:
        if direct and contract.name.casefold() == artifact_name.casefold():
            continue
        claim = _zip_candidate(data, contract, size_limit=size_limit)
        if claim is not None:
            candidates.append(claim)
    return candidates


def _gate_result(stdout: str) -> _GateReport:
    """The gate's three-state report: what it claims, whether it is sealed, and why (ADR 0074)."""
    candidate = ""
    for line in reversed(stdout.splitlines()):
        if line.strip():
            candidate = line.strip()
            break
    try:
        report = json.loads(candidate)
    except ValueError:
        return _GateReport(claimed=True, sealed=False, reasons=("the gate printed no report",))
    if not isinstance(report, dict):
        return _GateReport(claimed=True, sealed=False, reasons=(_MALFORMED,))
    sealed = report.get("sealed")
    claim = report.get("claim")
    raw_reasons = report.get("reasons", [])
    well_formed_reasons = raw_reasons is None or (
        isinstance(raw_reasons, list) and all(isinstance(item, str) for item in raw_reasons)
    )
    if not isinstance(sealed, bool) or not well_formed_reasons:
        return _GateReport(claimed=True, sealed=False, reasons=(_MALFORMED,))
    reasons = tuple(item for item in (raw_reasons or []) if item.strip())
    if sealed is False and claim in (False, None, "none"):
        return _GateReport(claimed=False, sealed=False, reasons=())
    if sealed is False and (claim is True or claim in ("sealed", "unsealed")):
        return _GateReport(claimed=True, sealed=False, reasons=reasons)
    if sealed is True:
        return _GateReport(claimed=True, sealed=True, reasons=())
    return _GateReport(claimed=True, sealed=False, reasons=(_MALFORMED,))


def _run_gate(contract: SealedArtifact, payload: bytes) -> _GateReport:
    """Run one Skill gate over one candidate payload; every failure is a refusal, never a pass."""
    with tempfile.TemporaryDirectory(prefix="dlight-sealed-") as directory:
        target = Path(directory) / contract.name
        target.write_bytes(payload)
        try:
            # A fixed program from the built-in Skill's own root, executed by path with no shell:
            # the declaration is checked at discovery, and the payload is a file in a private
            # temporary directory, never a command string.
            completed = subprocess.run(  # noqa: S603
                [sys.executable, str(contract.gate), str(target)],
                cwd=directory,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": directory,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONIOENCODING": "utf-8",
                },
                capture_output=True,
                text=True,
                timeout=_GATE_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _GateReport(
                claimed=True,
                sealed=False,
                reasons=("the gate did not finish within the operational bound",),
            )
        except OSError as exc:
            return _GateReport(
                claimed=True,
                sealed=False,
                reasons=(f"the gate could not be run: {exc.strerror or exc}",),
            )
    report = _gate_result(completed.stdout)
    if report.claimed and completed.returncode != 0 and not report.reasons:
        tail = completed.stderr.strip().splitlines()[-1:] or ["no output"]
        report = _GateReport(
            claimed=True,
            sealed=False,
            reasons=(f"the gate exited {completed.returncode}: {tail[0]}",),
        )
    elif not report.claimed and completed.returncode != 0:
        # An ordinary file is not a refusal; a gate that says "ordinary" and then fails is broken.
        return _GateReport(claimed=True, sealed=False, reasons=(_MALFORMED,))
    return report


CHANGED = "the bytes changed after the Artifact check; attach it again"
UNREADABLE = "the artifact could not be re-read for its sealed gate; attach it again"


def attest_refused(
    artifact_name: str,
    contracts: Sequence[SealedArtifact],
    reason: str,
) -> SealedVerdict | None:
    """The refusal (``CHANGED``, ``UNREADABLE``) for a name-matching candidate the gate cannot judge.

    The reserved name must not become a way around the gate because of a host fault.
    """
    name = PurePosixPath(artifact_name).name
    contract = next(
        (contract for contract in contracts if contract.name.casefold() == name.casefold()), None
    )
    if contract is None:
        return None
    return SealedVerdict(
        artifact=artifact_name,
        skill=contract.skill,
        name=contract.name,
        claimed=True,
        sealed=False,
        reasons=(reason,),
    )


def attest_bytes(
    data: bytes,
    artifact_name: str,
    contracts: Sequence[SealedArtifact],
    *,
    size_limit: int,
) -> SealedVerdict | None:
    """The verdict for already-read candidate bytes, or ``None`` when the gate calls them ordinary.

    ``None`` means every declaration that matched called the file ordinary (or none matched): the
    caller attaches it as a document, not as a sealed claim. A non-``None`` verdict is the claim the
    caller must honour.
    """
    if not contracts:
        return None
    name = PurePosixPath(artifact_name).name
    verdicts: list[SealedVerdict] = []
    for claim in _candidates(name, data, contracts, size_limit=size_limit):
        contract = claim.contract
        if claim.reasons:
            verdicts.append(
                SealedVerdict(
                    artifact=artifact_name,
                    skill=contract.skill,
                    name=contract.name,
                    claimed=True,
                    sealed=False,
                    reasons=claim.reasons,
                )
            )
            continue
        report = _run_gate(contract, claim.payloads[0])
        verdicts.append(
            SealedVerdict(
                artifact=artifact_name,
                skill=contract.skill,
                name=contract.name,
                claimed=report.claimed,
                sealed=report.sealed,
                reasons=report.reasons,
            )
        )
    claims = [verdict for verdict in verdicts if verdict.claimed]
    if not claims:
        return None
    refused = next((verdict for verdict in claims if not verdict.sealed), None)
    if refused is not None:
        return refused
    return claims[0]


__all__ = [
    "CHANGED",
    "UNREADABLE",
    "SealedArtifact",
    "SealedVerdict",
    "attest_bytes",
    "attest_refused",
    "contracts",
]
