# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Derive, from a Run's own stored records, whether a Skill attested what it published.

A packaged Agent Skill can ship a verifier and still be bypassed: a Run may never load the Skill
at all, or may load it and never run the verifier, and the Artifact is published as though the
Skill's gates had applied. Nothing in the Answer says either way, so this module reads the Run's
durable records back and *derives* the answer. Nothing here trusts a field a model wrote on its
own, and nothing here makes a Skill load: the record is a consequence of what the Run did, not a
claim about what it should have done.

Three records are read, and no one of them is trusted alone:

* the Run's durable tool events -- ``tool_start`` names a call, ``tool_progress`` carries the
  Tool's own subject (the Skill a ``load_skill`` call named, the command line a ``bash`` call
  ran) and, for a ``load_skill`` call, the *read fact*: the document it answered and the window of
  characters it returned (``engine/agent/skill_reads.py``), and ``tool_end`` carries its outcome.
  The Tool publishes its subject and its fact from its *validated* input and its own result, which
  is why they are evidence about a call rather than a description of one.
* the Session transcript's Assistant entries, which hold the full arguments the model asked each
  call to run with. That text is the model's own, so it becomes evidence only where the event log
  independently shows the same call was admitted: the two records are joined by ``call_id``, and an
  entry with no matching event -- the transcript also holds calls the runtime refused -- is never
  read. A subject is one bounded line (``TOOL_SUBJECT_MAX_CHARS``), so a command naming the
  verifier after that bound is invisible in the event log; the transcript holds the whole command.
* the Skill catalog's own declarations, read through the same factory a Run's catalog is
  discovered through, so a Skill that declares ``attestation: scripts/case_validate.py`` is
  known to have a contract and one that declares nothing is known not to; the documents a Skill
  declares in ``required_reads`` come from the same discovery.

**This is evidence, not proof.** A command line that runs the verifier (``executes_verifier``) is not
proof that the verifier produced *this* Artifact: the run may have run it against another document,
or run it and ignored the result. A command that only names the file -- ``cat``, ``grep``, ``cp``,
``chmod`` -- is no run at all, and neither is one that pipes the verifier's output onward, because
the exit code is then the pipeline's last command's. A Run may also load a Skill and ignore every
instruction in it. What the derivation removes is the *silent* case -- an Artifact published with
no Skill and no verifier behind it, presented as though the gates applied -- not every false case.
The same holds for the documents a Skill says a Run must read (``required_reads``, ADR 0057): a
recorded window shows the text was put in front of the model, not that the model attended to it.
It removes the case where a Skill says "read this" and the Run never opened it.
Where the records cannot settle the question (an event log trimmed by retention, a Skill load
whose name was never recorded, a Skill deleted since the Run ran) the verdict is ``unknown`` with
the reason, because an honest unknown is not an accusation.

Two limits of the record are worth naming, because both are reported rather than papered over.
When a Run's transcript cannot be read -- an entry trimmed by retention, a transcript the store
cannot name -- a call falls back to the bounded subject its events published, and a cut subject
that could have named the verifier reads as ``arguments_truncated`` rather than as a missing
verifier. And the verdict is one per Run, not one per Artifact: the records say a verifier ran in
this Run, never which document it was handed.

**What this module deliberately does not do.** It never opens an Artifact, never parses a Skill's
own document format, and never re-runs a Skill's verifier. A Skill's in-loop gate already recomputes
a delivered document's audit against its verifier's own output; repeating that work here would put a
second implementation of a Skill's rules inside the product, which is the wrong layer and the copy
that drifts. The verifier's own counts -- ``blocking`` and ``warnings`` -- are not carried for the
same reason: the durable record holds a call's bounded subject and not its output text, so a count
here could only be recomputed or guessed, and a guessed count is precisely what this module exists
to prevent.

**Required reads sit beside the verdict and never inside it.** ``required_reads_missing`` lists the
documents a loaded Skill declares and the Run did not read to the end, and it changes neither the
status nor the reason: an Artifact built without a reading its Skill asked for is not thereby
unattested, and a missing read never changes delivery. ``None`` says the records
cannot settle it -- they are incomplete, a load cannot be named, a loaded Skill has left the
catalog, or no loaded Skill declares a read -- and an empty tuple says
every declared read was read to the end; the two are never the same answer.

The derivation is deliberately not stored with the Answer: it is a property of the Run's records
read at the moment a reader asks, so it cannot be edited by the Run that is being judged, and a
re-derivation after more records exist says more rather than contradicting itself.

The verdict is published beside every Artifact, always: the record is what the reader is owed, and
it names what the Run did and did not do. Nothing in the product turns that record into a delivery
gate. A configuration once could -- ``artifact_attestation: require`` withheld the bytes of an
Artifact whose verifier claim it could not confirm -- and that was withdrawn (ADR 0058, final
amendment): a deployment setting must not reach back into Answers already served, a Run must not
finish with its files silently unavailable to the reader, and errors belong to the Skill's own
in-loop gate, which the Agent sees and can correct, not to the reader after the fact. A Run is
never failed for its provenance, and the file it wrote is always handed over with its record.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.runs import RunEvent, RunToolCall
from dlightrag.application.skills import skills_bundle_factory
from dlightrag.engine.agent.skill_delivery import EXECUTION_TOOLS, executes_verifier
from dlightrag.engine.agent.skill_reads import (
    LOAD_SKILL_TOOL,
    RequiredRead,
    SkillRead,
    documents_read_to_end,
    fact_values,
    missing_required_reads,
    skill_read,
)
from dlightrag.engine.agent.tools.contracts import TOOL_SUBJECT_MAX_CHARS
from dlightrag.engine.answer.client_contracts import ClientContractModel

#: How many of a Run's durable events one derivation reads before it gives up on the record.
#: The bound exists because a long Run's log is dominated by token events this module does not
#: want; reaching it is reported as pruned records rather than scanned past, so an unusually
#: large Run reads as ``unknown`` instead of as an accusation.
DEFAULT_EVENT_SCAN_LIMIT = 5_000

#: The tools whose subject is a command line rather than a file are declared in
#: ``engine/agent/skill_delivery.py``, imported above: the in-loop preview and this durable
#: derivation must not disagree about which calls could have run a verifier.

#: What an attested record rests on, in one token: the strongest thing these records can establish.
#: A command line that runs the verifier is not proof the verifier produced the Artifact -- the Run
#: may have run it against another document -- and a reader that
#: sees only ``attested`` would take a stronger claim than the evidence supports.
ATTESTATION_BASIS_NAMED_IN_A_COMMAND = "named_in_a_recorded_command"

#: The durable event types a Tool call writes. Every other event is scanned past.
TOOL_EVENT_TYPES = frozenset({"tool_start", "tool_progress", "tool_end"})

type AttestationStatus = Literal["attested", "unattested", "unknown"]

#: The closed vocabulary every reason comes from. One tuple, one mapping below, and one Literal:
#: a refusal message, an API client and a test all read the same set, and a new reason cannot be
#: introduced by a caller inventing a string.
ATTESTATION_REASONS: tuple[str, ...] = (
    "no_skill_loaded",
    "skill_loaded_without_verifier",
    "skill_has_no_verifier_contract",
    "contract_unavailable",
    "records_pruned",
    "skill_name_unavailable",
    "arguments_truncated",
    "verifier_reported_failures",
)

type AttestationReason = Literal[
    "no_skill_loaded",
    "skill_loaded_without_verifier",
    "skill_has_no_verifier_contract",
    "contract_unavailable",
    "records_pruned",
    "skill_name_unavailable",
    "arguments_truncated",
    "verifier_reported_failures",
]

#: What each reason says in a reader's words. The token is carried verbatim beside this text, so
#: a client that maps the vocabulary itself never has to parse the sentence.
ATTESTATION_REASON_TEXT: Mapping[AttestationReason, str] = {
    "no_skill_loaded": "the Run loaded no Agent Skill at all",
    "skill_loaded_without_verifier": (
        "no command recorded in this Run ran the verifier the Skill declares today; if the Skill "
        "was republished with a different verifier since the Run, that cannot be told from here"
    ),
    "skill_has_no_verifier_contract": (
        "the Skill that was loaded declares no verifier for what it produces"
    ),
    "contract_unavailable": (
        "a Skill the Run loaded is no longer in the catalog, so its declaration cannot be read"
    ),
    "verifier_reported_failures": (
        "the Skill's verifier ran in this Run and reported failures, so the document it checked "
        "did not pass the Skill's own gates"
    ),
    "records_pruned": (
        "the records read here are not the whole Run -- trimmed by retention, or past this reader's "
        "bound -- so no Skill's verifier can be ruled in or out"
    ),
    "skill_name_unavailable": (
        "a Skill load is recorded without the Skill's name, so which Skill it was cannot be read"
    ),
    "arguments_truncated": (
        "part of the Run's record is missing, so whether the Skill's verifier ran cannot be "
        "confirmed"
    ),
}


class MissingReadResponse(ClientContractModel):
    """One declared read the Run did not make, as the wire carries it.

    ``paths`` are the Skill-relative files that would have satisfied the entry -- one for a
    ``path`` declaration, several for ``any_of`` -- and ``when`` is the Skill author's own words
    for the step or condition, so a reader sees what the Skill asked for and when.
    """

    skill: str
    paths: list[str]
    when: str


class ArtifactAttestationResponse(ClientContractModel):
    """The serialized attestation record, shared by the REST and Web Artifact models.

    One class rather than two field-identical ones: the REST payload and the Web presentation
    answer the same question about the same Artifact, and a client that reads both must not have
    to reconcile two shapes.
    """

    status: AttestationStatus
    skill: str | None = None
    verifier: str | None = None
    reason: AttestationReason | None = None
    basis: str | None = None
    """What an attested record rests on, so the wire never claims more than the evidence.

    ``attested`` means a command recorded in this Run named the Skill's verifier. It does not mean
    the verifier produced this Artifact, and a client that shows the status alone would read it as
    though the Skill's gates had passed over the delivered file. The token names the observation;
    the docstring's honesty travels with the record instead of staying in the module.
    """
    required_reads_missing: list[MissingReadResponse] | None = None
    """The documents a loaded Skill says a Run must read that this Run did not read to the end.

    Evidence beside the verdict, never part of it. ``None`` means the records cannot settle it
    (see ``ArtifactAttestation.required_reads_missing``), an empty list means every declared read
    was made, and an entry names what was not.
    """


@dataclass(frozen=True, slots=True)
class RecordedToolCall:
    """One Tool call as the Run's own durable records describe it.

    ``subject`` is what the Tool said it was acting on -- a ``load_skill`` call's Skill name, a
    ``bash`` call's command line -- and is ``None`` when the record carries no subject, which is
    a fact about the record and never a reason to guess one. When the Session transcript holds the
    call's full arguments, ``subject`` is that entire command instead of the bounded line the event
    log published, and ``subject_cut`` is ``False``; a transcript the store cannot read leaves the
    bounded line and ``subject_cut`` says whether it is the runtime's own truncation.

    ``read`` is the fact a ``load_skill`` call recorded about the document it answered, and is
    ``None`` for every other call and for a ``load_skill`` call whose record carries none: a call
    that failed, or one recorded before the fact existed.
    """

    tool_name: str
    call_id: str = ""
    subject: str | None = None
    outcome: str | None = None
    read: SkillRead | None = None
    subject_cut: bool | None = None
    """Whether ``subject`` is the runtime's bounded line rather than the whole value.

    ``None`` asks this constructor to read it off the value, which is how a record built straight
    from an event log gets it. A caller that filled ``subject`` from the durable full arguments
    sets ``False``: no bound was applied to that value.
    """

    def __post_init__(self) -> None:
        if self.subject_cut is None:
            object.__setattr__(self, "subject_cut", _is_cut(self.subject))


@dataclass(frozen=True, slots=True)
class SkillContract:
    """What one Skill declares about its own work, as the catalog serves it today.

    Both are claims the Skill makes about itself: the verifier it says checks what it produces,
    and the documents it says a Run must read. A Skill that declares neither is a Skill that
    makes no claim, which is different from a Skill the catalog no longer holds.
    """

    verifier: str | None = None
    required_reads: tuple[RequiredRead, ...] = ()


@dataclass(frozen=True, slots=True)
class MissingRead:
    """One declared read the Run did not make: the Skill, the files that would satisfy it, and when."""

    skill: str
    paths: tuple[str, ...]
    when: str


@dataclass(frozen=True, slots=True)
class VerifierRun:
    """Whether one loaded Skill's declared verifier ran, and the call that says so."""

    skill: str
    verifier: str
    ran: bool
    evidence_tool: str | None = None
    failed: bool = False


@dataclass(frozen=True, slots=True)
class ArtifactAttestation:
    """What the records settle about one Artifact, in the closed vocabulary.

    ``attested`` is the only status that carries no reason, because it is the only one that makes
    no complaint: every other status must name why, or a reader is left to guess at an accusation
    the records did not make.
    """

    status: AttestationStatus
    skill: str | None = None
    verifier: str | None = None
    reason: AttestationReason | None = None
    required_reads_missing: tuple[MissingRead, ...] | None = None
    """The documents a loaded Skill declares and the Run did not read to the end.

    Evidence beside the verdict, never a gate: it changes neither ``status`` nor ``reason``, and it
    changes no Artifact's delivery. ``None`` is "the records cannot
    settle it" -- incomplete, an unnameable load, a loaded Skill gone from the catalog, or no loaded
    Skill declaring a read -- and is a different answer from
    ``()``, which says every declared read was read to the end.
    """

    def __post_init__(self) -> None:
        if self.status == "attested":
            if self.reason is not None:
                raise ValueError("an attested Artifact carries no reason")
            return
        if self.reason is None:
            raise ValueError(f"a {self.status} Artifact must name its reason")
        if self.reason not in ATTESTATION_REASONS:
            raise ValueError(f"unknown attestation reason: {self.reason!r}")

    def payload(self) -> dict[str, Any]:
        """The one serialized shape, shared by the REST and Web transport models."""
        return {
            "status": self.status,
            "skill": self.skill,
            "verifier": self.verifier,
            "reason": self.reason,
            "basis": ATTESTATION_BASIS_NAMED_IN_A_COMMAND if self.status == "attested" else None,
            "required_reads_missing": (
                None
                if self.required_reads_missing is None
                else [
                    {"skill": missing.skill, "paths": list(missing.paths), "when": missing.when}
                    for missing in self.required_reads_missing
                ]
            ),
        }

    def no_verifier_declared(self) -> bool:
        """Whether this verdict means the loaded Skill declared no verifier for what it produces.

        A Skill that declares no verifier is not a Skill that skipped one: it publishes no gate to
        apply, so there is nothing an Artifact could have failed. The verdict still says
        ``unattested`` -- that is the honest description of a Run no gate stood behind -- but it is a
        fact about the Skill rather than about the file, and a reader who never asked whether a
        photograph was verified should not be told it was not. So a projection publishes no record
        at all for it, and draws nothing -- unless the Run also left a read its Skill declared
        unread (``required_reads_missing``): that is a statement about this Run, not about the
        file's verifier, and the record is published so the reader can see it.
        """
        return self.status == "unattested" and self.reason == "skill_has_no_verifier_contract"

    def publishes_record(self) -> bool:
        """Whether a projection publishes a record beside an Artifact for this verdict.

        A verdict that says nothing about a verifier publishes none, as before. It is published
        when it makes a verifier claim, and also when it carries a declared read the Run did not
        make, because that is the one thing a Skill with no verifier can still be found short of.
        """
        return not self.no_verifier_declared() or bool(self.required_reads_missing)


@dataclass(frozen=True, slots=True)
class RunAttestation:
    """One Run's derived provenance: what it loaded, what ran, and the verdict for its Artifacts."""

    skills_loaded: tuple[str, ...]
    verifier_runs: tuple[VerifierRun, ...]
    verdict: ArtifactAttestation
    records_complete: bool = True


class RunEventReader(Protocol):
    """The owner-scoped durable event replay a transport already reaches (``RunService``)."""

    def subscribe(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> AsyncIterator[RunEvent]: ...


class RunRecordReader(RunEventReader, Protocol):
    """The two durable records an attestation verdict reads: the Run's events and its transcript.

    The event log settles what the Run executed; the transcript carries the arguments each call
    was made with. A reader with only the former can see that a command ran but not what it said,
    which is exactly the gap that reported a real verifier as unknown.
    """

    async def load_tool_call_arguments(
        self, *, owner_id: str, run_id: str
    ) -> tuple[RunToolCall, ...]: ...


def tool_calls_from_events(
    events: Iterable[tuple[str, Mapping[str, Any]]],
) -> tuple[RecordedToolCall, ...]:
    """Merge one Run's durable Tool events into one record per call, in call order.

    A call missing an event keeps what the events it has say: a ``tool_start`` with no subject is
    a call whose arguments are not in the record, which ``derive_run_attestation`` reads as an
    unknown rather than as an empty command line.

    The same pass keeps a ``load_skill`` call's read fact -- the keys its progress event carries --
    so the one scan over the events answers both what ran and what was read. The first event to
    carry a key supplies it, which is the rule ``skill_reads_from_events`` applies to the same
    records for the callers that have no use for the rest of the tool calls.
    """
    order: dict[str, int] = {}
    names: dict[str, str] = {}
    subjects: dict[str, str] = {}
    outcomes: dict[str, str] = {}
    facts: dict[str, dict[str, Any]] = {}
    for event_type, payload in events:
        call_id = str(payload.get("call_id") or "")
        if not call_id:
            continue
        if call_id not in order:
            order[call_id] = len(order)
        name = str(payload.get("tool_name") or "")
        if name:
            names.setdefault(call_id, name)
        subject = str(payload.get("object_label") or "")
        if subject:
            subjects.setdefault(call_id, subject)
        if event_type == "tool_end":
            outcome = str(payload.get("outcome") or "")
            if outcome:
                outcomes[call_id] = outcome
        for key, value in fact_values(payload).items():
            facts.setdefault(call_id, {}).setdefault(key, value)
    return tuple(
        RecordedToolCall(
            tool_name=names.get(call_id, ""),
            call_id=call_id,
            subject=subjects.get(call_id),
            outcome=outcomes.get(call_id),
            read=(
                skill_read(subjects.get(call_id, ""), facts.get(call_id, {}))
                if names.get(call_id) == LOAD_SKILL_TOOL
                else None
            ),
        )
        for call_id in sorted(order, key=lambda value: order[value])
    )


def derive_run_attestation(
    *,
    contracts: Mapping[str, SkillContract],
    calls: Sequence[RecordedToolCall],
    records_complete: bool = True,
) -> RunAttestation:
    """Derive one Run's provenance from its records and the catalog's declarations.

    ``contracts`` maps every Skill name this Run could have loaded to what that Skill declares:
    the verifier, or ``None`` for a Skill that declares none, and the documents it says must be
    read. A name absent from the mapping is a Skill the catalog no longer serves, which is
    reported as an unknown rather than assumed to have no contract -- the difference between "it
    declares nothing" and "its declaration cannot be read" is the difference between a finding
    and a guess.

    ``records_complete`` is the caller's statement that the Run's event log is whole. A trimmed
    log settles nothing, so it can only ever produce ``unknown`` -- unless a Skill with a
    declaration was seen both loaded and verified in what survives, which is a positive finding
    about the Run regardless of what was dropped.
    """
    loaded: list[str] = []
    unnameable = 0
    for call in calls:
        if call.tool_name != LOAD_SKILL_TOOL:
            continue
        # A refused load hands over nothing, so it is not a load; every other recorded load is
        # one, including a load whose outcome the runtime left uncertain.
        if call.outcome == "failed":
            continue
        name = (call.subject or "").strip()
        # A load subject is bounded, so a long Skill name arrives cut. Treating the cut name as the
        # Skill's name made the catalog lookup miss and reported "no longer in the catalog" about a
        # Skill that is right there; the records cannot name it, which is what that token says.
        if not name or _is_cut(name):
            unnameable += 1
        elif name not in loaded:
            loaded.append(name)

    verifier_runs: list[VerifierRun] = []
    for skill in loaded:
        verifier = contracts[skill].verifier if skill in contracts else None
        if not verifier:
            continue
        naming = [
            call
            for call in calls
            if call.tool_name in EXECUTION_TOOLS and executes_verifier(call.subject, verifier)
        ]
        # A verifier that exited non-zero ran and reported failures. Counting that as evidence that
        # the Skill checked the document would report the failure the Skill exists to catch as a
        # pass, so it attests nothing -- but it is not "the verifier did not run" either, and the
        # vocabulary says which of the two happened.
        succeeded = next((call for call in naming if call.outcome != "failed"), None)
        verifier_runs.append(
            VerifierRun(
                skill=skill,
                verifier=verifier,
                ran=succeeded is not None,
                evidence_tool=succeeded.tool_name if succeeded else None,
                failed=bool(naming) and succeeded is None,
            )
        )

    verdict = _verdict(
        loaded=loaded,
        unnameable=unnameable,
        verifier_runs=verifier_runs,
        contracts=contracts,
        records_complete=records_complete,
        # Only a bounded command line can be hiding the verifier; a cut file path cannot, and a
        # full argument the transcript supplied is never cut.
        arguments_cut=any(call.subject_cut for call in calls if call.tool_name in EXECUTION_TOOLS),
    )
    verdict = replace(
        verdict,
        required_reads_missing=_required_reads_missing(
            loaded=loaded,
            unnameable=unnameable,
            contracts=contracts,
            calls=calls,
            records_complete=records_complete,
        ),
    )
    return RunAttestation(
        skills_loaded=tuple(loaded),
        verifier_runs=tuple(verifier_runs),
        verdict=verdict,
        records_complete=records_complete,
    )


def _verdict(
    *,
    loaded: Sequence[str],
    unnameable: int,
    verifier_runs: Sequence[VerifierRun],
    contracts: Mapping[str, SkillContract],
    records_complete: bool,
    arguments_cut: bool,
) -> ArtifactAttestation:
    """One Artifact's verdict, in the order the evidence settles it.

    A positive finding outranks every gap: whatever was pruned, a Skill with a declaration was
    loaded and its verifier ran. After that, the first thing the records cannot answer decides,
    and only a whole record that answers everything reaches a complaint.
    """
    attested = next((run for run in verifier_runs if run.ran), None)
    if attested is not None:
        return ArtifactAttestation(
            status="attested", skill=attested.skill, verifier=attested.verifier
        )
    failed = next((run for run in verifier_runs if run.failed), None)
    if failed is not None:
        return ArtifactAttestation(
            status="unattested",
            skill=failed.skill,
            verifier=failed.verifier,
            reason="verifier_reported_failures",
        )
    if unnameable:
        # Something was loaded that cannot be named, so no complaint about what loaded would be
        # safe: the unnameable load may be the one that carried a verifier.
        return ArtifactAttestation(status="unknown", reason="skill_name_unavailable")
    if not records_complete:
        return ArtifactAttestation(status="unknown", reason="records_pruned")
    if not loaded:
        return ArtifactAttestation(status="unattested", reason="no_skill_loaded")
    unreadable = next((name for name in loaded if name not in contracts), None)
    if unreadable is not None:
        return ArtifactAttestation(
            status="unknown", skill=unreadable, reason="contract_unavailable"
        )
    declared = next(iter(verifier_runs), None)
    if declared is not None:
        if arguments_cut:
            # A Tool's subject is published as one bounded line, and the verifier may be named
            # after that bound. Saying it did not run would be an accusation the record cannot
            # support, so the cut is reported as what it is.
            return ArtifactAttestation(
                status="unknown",
                skill=declared.skill,
                verifier=declared.verifier,
                reason="arguments_truncated",
            )
        return ArtifactAttestation(
            status="unattested",
            skill=declared.skill,
            verifier=declared.verifier,
            reason="skill_loaded_without_verifier",
        )
    return ArtifactAttestation(
        status="unattested", skill=loaded[0], reason="skill_has_no_verifier_contract"
    )


def _required_reads_missing(
    *,
    loaded: Sequence[str],
    unnameable: int,
    contracts: Mapping[str, SkillContract],
    calls: Sequence[RecordedToolCall],
    records_complete: bool,
) -> tuple[MissingRead, ...] | None:
    """The reads the loaded Skills declare and the Run did not make, or ``None`` if unsettled.

    ``None`` whenever the records cannot answer, each case being one where an empty answer would
    be a claim and a non-empty one an accusation:

    * the log is not the whole Run, so the read may be in the part that is gone;
    * a load cannot be named, so the Skill whose reads it made is unknown;
    * a loaded Skill has left the catalog, so what it declares cannot be read;
    * no loaded Skill declares a read, so there is nothing to have missed. That is a statement
      about the Skills and not about the Run, and an empty list would say the opposite.

    Otherwise the entries of the loaded Skills that were not satisfied by a document read to its
    end, possibly none.
    """
    if unnameable or not records_complete:
        return None
    if any(skill not in contracts for skill in loaded):
        return None
    loads = [call for call in calls if call.tool_name == LOAD_SKILL_TOOL]
    declared = {skill: contracts[skill].required_reads for skill in loaded}
    if not any(declared.values()):
        return None
    read = documents_read_to_end(call.read for call in loads if call.read is not None)
    return tuple(
        MissingRead(skill=skill, paths=entry.paths, when=entry.when)
        for skill, entry in missing_required_reads(declared, loaded, read)
    )


def _is_cut(subject: str | None) -> bool:
    """Whether one recorded subject is the runtime's own truncation rather than the whole value.

    A subject longer than the published bound comes back as its first characters and an ellipsis,
    so a reader can tell a value that ends in an ellipsis from one that was cut. Anything shorter
    than the bound is what the Tool reported, whatever characters it ends with.
    """
    return bool(subject) and len(subject) >= TOOL_SUBJECT_MAX_CHARS and subject.endswith("…")


async def read_run_tool_calls(
    runs: RunRecordReader,
    *,
    owner_id: str,
    run_id: str,
    scan_limit: int = DEFAULT_EVENT_SCAN_LIMIT,
) -> tuple[tuple[RecordedToolCall, ...], bool]:
    """Read one settled Run's Tool calls back from its durable events and transcript.

    The event log says which calls the runtime took and how each ended; the transcript supplies the
    full arguments of each of those calls. Only the executable tools gain anything from the join --
    a ``load_skill`` subject is a Skill name, not an argument -- and a transcript entry with no
    matching call in the event log is never used, because the transcript also records calls the
    runtime refused.

    Returns the calls and whether the scan saw the Run's whole event log. A Run whose log was
    trimmed, or one longer than ``scan_limit`` events, comes back incomplete; the caller turns
    that into an unknown rather than into a complaint.
    """
    scanned = 0
    complete = True
    seen: list[tuple[str, Mapping[str, Any]]] = []
    stream = runs.subscribe(owner_id=owner_id, run_id=run_id, after_sequence=0)
    try:
        async for event in stream:
            scanned += 1
            if scanned > scan_limit:
                complete = False
                break
            if event.event_type in TOOL_EVENT_TYPES:
                seen.append((event.event_type, event.payload))
    finally:
        # Closed explicitly: a bounded read stops the subscription instead of leaving it to the
        # garbage collector, which for a live Run would keep a broker waiter armed.
        close = getattr(stream, "aclose", None)
        if close is not None:
            await close()
    recorded = await runs.load_tool_call_arguments(owner_id=owner_id, run_id=run_id)
    return _with_full_arguments(tool_calls_from_events(seen), recorded), complete


def _with_full_arguments(
    calls: Sequence[RecordedToolCall], recorded: Sequence[RunToolCall]
) -> tuple[RecordedToolCall, ...]:
    """Put each executed call's full arguments where its bounded subject stood, by ``call_id``.

    A transcript entry is used only when the event log proves that call ran and the two records
    name the same Tool; anything else keeps the bounded subject, whose ``subject_cut`` already says
    what the record can no longer settle. A call with no usable ``command`` keeps its subject for
    the same reason: there is nothing to read in its place.
    """
    by_id: dict[str, RunToolCall] = {}
    for argument in recorded:
        by_id.setdefault(argument.call_id, argument)
    merged: list[RecordedToolCall] = []
    for call in calls:
        argument = by_id.get(call.call_id)
        command = argument.arguments.get("command") if argument is not None else None
        if (
            argument is None
            or argument.tool_name != call.tool_name
            or call.tool_name not in EXECUTION_TOOLS
            or not isinstance(command, str)
            or not command
        ):
            merged.append(call)
            continue
        merged.append(replace(call, subject=command, subject_cut=False))
    return tuple(merged)


def skill_contracts(config: DlightragConfig, *, owner_id: str) -> dict[str, SkillContract]:
    """Every Skill this owner's Runs could load, as name to what it declares today.

    Read through the same factory a Run's own catalog is discovered through, and in one discovery
    for both declarations, so the verifier a Skill carries now and the reads it declares now are
    the ones its name is judged by. A Skill republished or deleted since the Run ran is therefore
    judged by what it says today -- and when the name is gone entirely, the derivation reports
    ``contract_unavailable`` rather than assuming no contract.
    """
    catalog = skills_bundle_factory(config)(owner_id).catalog()
    if catalog is None:
        return {}
    return {
        skill.name: SkillContract(verifier=skill.attestation, required_reads=skill.required_reads)
        for skill in catalog.metadata
    }


async def run_verdict(
    *,
    runs: RunRecordReader,
    config: DlightragConfig,
    owner_id: str,
    run_id: str,
    records_pruned: bool,
) -> ArtifactAttestation:
    """Read one settled Run's records and derive the verdict its Artifacts share.

    Public because a transport that builds its own projection -- conversation history does, from the
    same stored result the live Answer came from -- derives here and stamps with
    ``stamp_projected_artifacts``, so the label a reader sees on reopening a conversation comes from
    the one implementation that decides what an Artifact's record is.
    """
    calls, complete = await read_run_tool_calls(runs, owner_id=owner_id, run_id=run_id)
    return derive_run_attestation(
        contracts=skill_contracts(config, owner_id=owner_id),
        calls=calls,
        records_complete=complete and not records_pruned,
    ).verdict


async def attest_projected_artifacts(
    projected: Mapping[str, Any],
    *,
    runs: RunRecordReader,
    config: DlightragConfig,
    owner_id: str,
    run_id: str,
    records_pruned: bool = False,
) -> dict[str, Any]:
    """The one projection entry point a transport calls, with its attestations on it.

    ``records_pruned`` is the caller's own knowledge that the Run's event log was trimmed
    (``events_trimmed_at``), which the derivation cannot see for itself.
    """
    if not projected.get("artifacts"):
        return dict(projected)
    verdict = await run_verdict(
        runs=runs,
        config=config,
        owner_id=owner_id,
        run_id=run_id,
        records_pruned=records_pruned,
    )
    return stamp_projected_artifacts(projected, verdict=verdict)


async def run_attestation(
    *,
    runs: RunRecordReader,
    config: DlightragConfig,
    owner_id: str,
    run_id: str,
    records_pruned: bool = False,
) -> ArtifactAttestation:
    """The record one Run's Artifact projections stamp, derived from that Run's records.

    The live Answer and the conversation a reader reopens both derive through this one entry point,
    so a reader who watches an Answer finish and a reader who reloads it later are told one thing.
    The projection that consumes the record is synchronous, which is why the derivation happens
    here and travels as an argument. Nothing is read until a caller has Artifacts to stamp.
    """
    return await run_verdict(
        runs=runs,
        config=config,
        owner_id=owner_id,
        run_id=run_id,
        records_pruned=records_pruned,
    )


def stamp_when_publishing(
    projected: Mapping[str, Any],
    *,
    attestation: ArtifactAttestation | None,
) -> dict[str, Any]:
    """Stamp one projection when it published Artifacts, and leave it untouched otherwise.

    Both browser projections are synchronous and both carry the record as an argument, so the same
    three lines -- is there anything to label, is there a record to put beside it, put it there --
    stood in each of them. It lives here once because the rule is one rule: a projection with no
    Artifacts is never judged, and one whose caller resolved no record is returned exactly as its
    own read produced it.
    """
    if attestation is None or not projected.get("artifacts"):
        return dict(projected)
    return stamp_projected_artifacts(projected, verdict=attestation)


def stamp_projected_artifacts(
    projected: Mapping[str, Any],
    *,
    verdict: ArtifactAttestation,
) -> dict[str, Any]:
    """Put the attestation record on every Artifact one projected Answer lists.

    The one shape both transports read: an Artifact carries ``attestation`` wherever it appears --
    in the Artifact list, in an inline Markdown placement, in the Web presentation built from the
    same list -- so a client never has to reconcile two answers to one question.

    An Artifact of a Run whose loaded Skill declares no verifier carries no record at all, because
    there is no claim to publish beside it (``ArtifactAttestation.no_verifier_declared``) -- unless
    the Run left unread a document its Skill declares (``required_reads_missing``), which is a
    finding about the Run that the reader is owed whether or not a verifier stood behind the file.
    Nothing here changes an Artifact's status or bytes: the record is told, never enforced, and a
    Run that skipped a gate is served with that finding beside its files.
    """
    artifacts = [
        dict(item) for item in projected.get("artifacts") or () if isinstance(item, Mapping)
    ]
    publishes_record = verdict.publishes_record()
    values: list[dict[str, Any]] = []
    by_resource: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        value: dict[str, Any] = {
            **item,
            "attestation": verdict.payload() if publishes_record else None,
        }
        values.append(value)
        by_resource[str(value.get("resource_id") or "")] = value
    result: dict[str, Any] = {**projected, "artifacts": values}
    parts = projected.get("parts")
    if isinstance(parts, Sequence) and not isinstance(parts, str):
        result["parts"] = [_restamp_part(part, by_resource=by_resource) for part in parts]
    return result


def _restamp_part(part: Any, *, by_resource: Mapping[str, Mapping[str, Any]]) -> Any:
    """One Answer part, with an inline Artifact placement carrying the stamped copy."""
    if not isinstance(part, Mapping):
        return part
    artifact = part.get("artifact")
    if not isinstance(artifact, Mapping):
        return part
    replacement = by_resource.get(str(artifact.get("resource_id") or ""))
    return part if replacement is None else {**part, "artifact": replacement}


__all__ = [
    "ATTESTATION_REASONS",
    "ATTESTATION_REASON_TEXT",
    "ArtifactAttestation",
    "ArtifactAttestationResponse",
    "AttestationReason",
    "AttestationStatus",
    "DEFAULT_EVENT_SCAN_LIMIT",
    "LOAD_SKILL_TOOL",
    "MissingRead",
    "MissingReadResponse",
    "RecordedToolCall",
    "RunAttestation",
    "RunEventReader",
    "RunRecordReader",
    "SkillContract",
    "TOOL_EVENT_TYPES",
    "VerifierRun",
    "attest_projected_artifacts",
    "derive_run_attestation",
    "read_run_tool_calls",
    "run_attestation",
    "skill_contracts",
    "stamp_projected_artifacts",
    "stamp_when_publishing",
    "tool_calls_from_events",
]
