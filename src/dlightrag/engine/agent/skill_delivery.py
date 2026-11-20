# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The delivery record a Run can see before it finishes, from what the Run itself recorded.

A Skill declares what a Run must read and which verifier checks what it delivers (ADR 0057,
ADR 0058). Until now that record was derived only when a reader asked, after the Run could no
longer act on it. This module keeps a small ledger of the current attempt — the Skill loads and
their read facts, and the command lines the Run executed — so the Tool that attaches an Artifact
can tell the Agent what the reader's record is about to say, while the Agent can still correct it.

**Evidence, not a verdict.** The preview states what the attempt recorded: the declared reads it
has not covered, whether a command executed a Skill's declared verifier (reading the file is not
execution), and whether an HTML page
carries the toolkit's provenance (ADR 0038). It deliberately does not re-derive the reader-facing
status and reason: that derivation stays the one in `application/answer/provenance.py`, judged from
the Run's durable records when the Run settles. A preview that restated the verdict would be a
second implementation of it, and the two would drift.

**It imports nothing of the product but `skill_reads.py`**: the read rule and the verifier-execution
rule are one rule each, imported by the durable derivation (which this module may not reach upward to) and by
the in-loop preview. The rule for whether a document was read is `documents_read_to_end`; a
`write_to` copy has a path and no window, so it is present in the ledger and never satisfies a
declared read.

**What the ledger cannot see.** It is the current attempt's memory, not the Run's durable record. A
Run that was reclaimed and continued starts a new ledger and gets back the Skill loads its earlier
attempts made (`SkillsBundle.restore_loads`) and the sealed artifacts the gate attested then (the
`sealed_skill` fact of a successful `attach_artifact`), both from the Run's durable events, so the
label follows the Run and not the attempt. A command that ran before the continuation left no
durable event this ledger can read. That is why the preview says "so far in this attempt", and why
the durable derivation remains the authority.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from dlightrag.engine.agent.skill_reads import (
    RequiredRead,
    SkillRead,
    documents_read_to_end,
    missing_required_reads,
)

if TYPE_CHECKING:
    from dlightrag.engine.agent.skills import SealedArtifactDeclaration

#: The tools whose recorded subject is a command line rather than a file. Only these can run a
#: verifier: a ``read`` or ``view`` of the script names it without executing it. The set is declared
#: once here and imported by the durable derivation, because a Tool's name is the only thing a
#: record carries about what it does.
EXECUTION_TOOLS = frozenset({"bash", "shell", "sh", "run", "exec"})

#: Command separators. A verifier runs when the *segment* naming it starts it, and a segment that a
#: pipe feeds is not judged by its own exit code: the command's status is the pipeline's last one.
_SEPARATORS = frozenset({"&&", "||", ";", "|", "|&"})
#: After the verifier's segment, only ``&&`` keeps its exit code the command's own: a pipe, ``;``,
#: a newline or ``||`` hands the exit code to whatever runs next.
_KEEPS_EXIT = frozenset({"&&"})
_END = "\0"
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=")
#: What may stand between a segment's start and a script it runs: the interpreters ``python``,
#: ``python3`` and ``python3.N``, and the launchers ``uv``, ``env``, ``bash`` and ``sh``.
_INTERPRETER = re.compile(r"python(?:3(?:\.\d+)?)?|uv|env|bash|sh")


def executes_verifier(subject: str | None, verifier: str) -> bool:
    """Whether a command runs the verifier rather than only naming its file.

    The command is cut at ``&&``, ``||``, ``;``, newlines and ``|``, and only a segment that names
    the verifier's file counts. It runs the verifier when, after any ``VAR=value`` prefixes, its first
    word is the verifier itself (``./case_validate.py``) or an interpreter (``python3``, ``uv run``,
    ``env``, ``bash``, ``sh``) that takes the verifier as a script argument, never inside a ``-c``
    string. The verifier is judged by the command's exit code, so its segment counts only when that
    exit code is the verifier's: it ends the command or is followed by ``&&``, never by ``|``, ``;``
    or ``||`` (``case_validate.py x | tail``, ``case_validate.py x || true``). ``grep x case_validate.py``,
    ``cp scripts/case_validate.py .`` and
    ``find . -name case_validate.py`` name the file and run nothing.
    """
    name = PurePosixPath(verifier).name
    if not subject or not name:
        return False
    lexer = shlex.shlex(
        re.sub(r"\\\n", " ", subject).replace("\n", ";"), posix=True, punctuation_chars=True
    )
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    segment: list[str] = []
    for token in (*tokens, _END):
        if token not in _SEPARATORS and token != _END:
            segment.append(token)
            continue
        if (token in _KEEPS_EXIT or token == _END) and _starts_verifier(segment, name):
            return True
        segment = []
    return False


def _starts_verifier(segment: list[str], name: str) -> bool:
    """Whether one command segment starts the verifier file ``name``, itself or by an interpreter."""
    words = iter(word for word in segment if not _ASSIGNMENT.match(word))
    first = next(words, None)
    if first is None:
        return False
    if PurePosixPath(first).name == name:
        return True
    if _INTERPRETER.fullmatch(PurePosixPath(first).name) is None:
        return False
    for word in words:
        if word.startswith("-c") and not word.startswith("--"):
            return False  # everything after -c is a code string, not a script to run
        if PurePosixPath(word).name == name:
            return True
    return False


@dataclass(frozen=True, slots=True)
class MissingRead:
    """One declared read this attempt has not covered: the Skill, the paths, and when it was due."""

    skill: str
    paths: tuple[str, ...]
    when: str


@dataclass(frozen=True, slots=True)
class VerifierEvidence:
    """Whether a recorded command in this attempt executed a Skill's declared verifier, and its result.

    ``named`` is true only when a command that executed it did not fail: the verifier passes by its
    exit code, the one signal every verifier states the same way. A command that executed it and
    failed is not attestation, which the reader's record reports as a failure; the preview says the
    same thing in one line rather than staying silent about a run that tried.
    """

    skill: str
    verifier: str
    named: bool
    failed: bool = False

    @property
    def passed(self) -> bool:
        return self.named


@dataclass(frozen=True, slots=True)
class PageProvenance:
    """Whether an HTML Artifact carries the toolkit's ``report-source`` block (ADR 0038, ADR 0051)."""

    file: str
    toolkit_built: bool


@dataclass(frozen=True, slots=True)
class DeliveryFindings:
    """What the pre-delivery checkpoint tells the model, and what the answer must open with.

    ``lines`` is the human-readable evidence. ``unsealed_labels`` are the declaring Skill's own
    labels for every loaded sealed-artifact Skill the gate has not attested this attempt: the answer
    opens with them whatever the lines say, because the label follows the record, not the prose.
    """

    lines: tuple[str, ...] = ()
    unsealed_labels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DeliveryPreview:
    """What this attempt recorded: the uncovered reads, the verifier evidence and the page."""

    skills: tuple[str, ...]
    missing_reads: tuple[MissingRead, ...]
    verifiers: tuple[VerifierEvidence, ...]
    page: PageProvenance | None = None
    sealed_skills: tuple[str, ...] = ()
    sealed_attested: tuple[str, ...] = ()
    sealed_refused: tuple[tuple[str, str, str], ...] = ()
    unsealed_labels: tuple[str, ...] = ()

    def lines(self) -> tuple[str, ...]:
        """What a reader's record would report that this attempt can state: attach and warning alike."""
        lines = [
            f"not read: {' or '.join(missing.paths)} (declared by {missing.skill}: {missing.when})"
            for missing in self.missing_reads
        ]
        for verifier in self.verifiers:
            if verifier.passed:
                continue
            if verifier.failed:
                lines.append(
                    f"a command naming {verifier.skill}'s verifier {verifier.verifier} ran and "
                    "failed; that is not attestation"
                )
            else:
                lines.append(
                    f"no command recorded so far named {verifier.skill}'s verifier "
                    f"{verifier.verifier}; until it runs, the deliverable has no verified output. "
                    "Produce what it checks, then run it as the command's last step (no pipe, `;` "
                    "or `||` after it): its exit code is what counts"
                )
        for artifact, skill, reason in self.sealed_refused:
            lines.append(
                f"the {skill} Skill's gate refused {artifact}: {reason}; fix the sealed artifact "
                "and attach it again, or deliver the answer unsealed"
            )
        if self.page is not None and not self.page.toolkit_built:
            lines.append(
                f"this HTML page carries no report-source block, so it was not built by "
                f"html-report: {self.page.file}"
            )
        return tuple(lines)

    def text(self) -> str:
        """The neutral block an attach result carries, or ``""`` when the record is clean so far."""
        lines = self.lines()
        if not lines:
            return ""
        return "\n".join(
            (
                "Delivery record preview (recorded so far in this attempt; nothing here is enforced):",
                *lines,
                "The reader's record is derived from the Run's durable events when the Run "
                "settles; this preview withholds nothing and changes no file.",
            )
        )


@dataclass(slots=True)
class DeliveryLedger:
    """What the current attempt recorded about its Skills, kept for the attach-time preview.

    One ledger belongs to one Run. A Child Session is composed with its parent Run's ledger, so a
    Skill a Child loads or a command it runs is recorded here too: a sealed-artifact Skill loaded only
    inside a Child labels the parent's answer.
    """

    _declared: dict[
        str, tuple[str | None, SealedArtifactDeclaration | None, tuple[RequiredRead, ...]]
    ] = field(default_factory=dict)
    _reads: list[SkillRead] = field(default_factory=list)
    _executions: list[tuple[str | None, bool]] = field(default_factory=list)
    _sealed_attested: set[str] = field(default_factory=set)
    _sealed_refused: dict[str, tuple[str, str]] = field(default_factory=dict)
    _attached: list[str] = field(default_factory=list)

    def record_load(
        self,
        *,
        skill: str,
        verifier: str | None,
        required_reads: Sequence[RequiredRead],
        read: SkillRead | None,
        sealed_artifact: SealedArtifactDeclaration | None = None,
    ) -> None:
        """One successful Skill load, with the declaration the catalog served and its read fact.

        ``sealed_artifact`` is the served declaration itself; the ledger keeps it for the label an
        answer without an attested artifact opens with. Its gate is the attach Tool's business, not
        this record's. ``read`` is ``None`` for a load whose durable record carries no read fact: the
        Skill was loaded, and no document is credited.
        """
        self._declared.setdefault(skill, (verifier, sealed_artifact, tuple(required_reads)))
        if read is not None:
            self._reads.append(read)

    def record_execution(self, *, tool_name: str, command: str | None, failed: bool) -> None:
        """One executed Tool call, when the Tool is one whose subject can name a verifier."""
        if tool_name in EXECUTION_TOOLS:
            self._executions.append((command, failed))

    def record_sealed_attestation(self, skill: str) -> None:
        """Record that the Skill's gate attested a sealed artifact this attempt attached.

        The gate ran on the artifact's bytes, which is stronger evidence than the attempt's own load
        and command history: a later turn re-attaching an already sealed file must not have the
        answer labelled unsealed because the sealing command ran in another attempt. A successful
        re-attach after an earlier refusal is exactly that repair: it supersedes the refusal for
        this Skill, or ``complete`` would be refused forever after one bad attempt.
        """
        self._sealed_attested.add(skill)
        for artifact, (refused_skill, _reason) in tuple(self._sealed_refused.items()):
            if refused_skill == skill:
                del self._sealed_refused[artifact]

    def record_sealed_refusal(self, skill: str, artifact: str, reasons: Sequence[str] = ()) -> None:
        """Record that the Skill's gate refused a sealed claim this attempt tried to attach.

        The refusal is evidence the answer's headline must follow: a sealed claim was attempted
        and denied, so an unqualified "sealed" answer would be false even if every other delivery
        finding is clean by the time the final answer is proposed.
        """
        self._sealed_refused[artifact] = (
            skill,
            "; ".join(reason for reason in reasons if reason) or "the gate did not attest it",
        )

    def record_attachment(self, artifact: str) -> None:
        """Record that this attempt attached an Artifact.

        Kept apart from the Skill facts: an Artifact published by a Run that loaded no Skill still
        has a reader record (``unattested``, ``no_skill_loaded``), and the in-loop preview must be
        able to say so before the Answer ends.
        """
        self._attached.append(artifact)

    @property
    def has_attachment(self) -> bool:
        """Whether this attempt attached an Artifact: without one there is no delivery to warn about."""
        return bool(self._attached)

    def no_skill_delivery(self) -> tuple[str, ...]:
        """The line for an Artifact whose Run loaded no Skill, empty when that is not the case.

        ``preview()`` returns ``None`` when nothing was loaded, which used to read as "nothing to
        tell": the attachment was accepted silently and the reader met an ``unattested`` record after
        the Run had already settled. The model can still correct that while the Run is live -- load
        the Skill, run its verifier, attach again -- so the record is said here, at the attach and
        before delivery, not only in the reader's derivation.
        """
        if self._declared or not self._attached:
            return ()
        artifacts = ", ".join(dict.fromkeys(self._attached))
        return (
            f"this Run loaded no Agent Skill, so the reader's record will publish {artifacts} as "
            "unattested; if a Skill governs the deliverable, load it and run its verifier before the "
            "Answer ends, then attach the file again",
        )

    def preview(self, *, page: PageProvenance | None = None) -> DeliveryPreview | None:
        """The record this attempt can state so far, or ``None`` when no Skill was loaded.

        The page finding is only carried when a Skill was loaded: without one the page is not a
        Skill delivery, and a hand-written HTML file may be exactly what the user asked for.
        """
        if not self._declared:
            return None
        loaded = tuple(self._declared)
        declared = {skill: reads for skill, (_, _, reads) in self._declared.items()}
        sealed = {
            skill: declaration
            for skill, (_, declaration, _) in self._declared.items()
            if declaration is not None
        }
        covered = documents_read_to_end(self._reads)
        missing = tuple(
            MissingRead(skill=skill, paths=entry.paths, when=entry.when)
            for skill, entry in missing_required_reads(declared, loaded, covered)
        )
        verifiers: list[VerifierEvidence] = []
        for skill, (verifier, _, _) in self._declared.items():
            if not verifier:
                continue
            ran = [
                failed
                for command, failed in self._executions
                if executes_verifier(command, verifier)
            ]
            named = any(not failed for failed in ran)
            verifiers.append(
                VerifierEvidence(
                    skill=skill, verifier=verifier, named=named, failed=bool(ran) and not named
                )
            )
        return DeliveryPreview(
            skills=loaded,
            missing_reads=missing,
            verifiers=tuple(verifiers),
            page=page,
            sealed_skills=tuple(sealed),
            sealed_attested=tuple(sorted(self._sealed_attested)),
            sealed_refused=tuple(
                (artifact, *claim) for artifact, claim in sorted(self._sealed_refused.items())
            ),
            unsealed_labels=tuple(
                declaration.unsealed_label
                for skill, declaration in sealed.items()
                if skill not in self._sealed_attested
            ),
        )

    def preview_text(self, *, page: PageProvenance | None = None) -> str:
        """The block to append to an attach result: the preview's text prefixed, or ``""``."""
        preview = self.preview(page=page)
        text = "" if preview is None else preview.text()
        return "" if not text else f"\n\n{text}"


__all__ = [
    "EXECUTION_TOOLS",
    "DeliveryFindings",
    "DeliveryLedger",
    "DeliveryPreview",
    "MissingRead",
    "PageProvenance",
    "VerifierEvidence",
    "executes_verifier",
]
