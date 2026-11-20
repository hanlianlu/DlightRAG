# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Model-facing attachment of completed Agent Workspace artifacts."""

import asyncio
import hashlib
from collections.abc import Callable, Collection, Sequence
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
    AgentspaceAccess,
    PathAccess,
)
from dlightrag.engine.agent.environment.errors import PathRejected
from dlightrag.engine.agent.environment.execution import ExecutionEnvironment
from dlightrag.engine.agent.sealed_artifact import (
    CHANGED,
    UNREADABLE,
    SealedArtifact,
    SealedVerdict,
    attest_bytes,
    attest_refused,
)
from dlightrag.engine.agent.skill_delivery import DeliveryLedger, PageProvenance
from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.agent.skill_reads import FACT_SEALED_SKILL
from dlightrag.engine.agent.tools.contracts import (
    TOOL_PROGRESS_FACT_MAX_CHARS,
    AdmittedBytesReader,
    AgentspaceInventoryFacts,
    AgentspacePathFact,
    AgentTool,
    DeclaredSource,
    ToolDeclaration,
    ToolEffects,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.agent.tools.files import (
    agentspace_integrity_refusal,
    replace_agentspace_file,
)
from dlightrag.engine.answer.publication import (
    ArtifactValidationError,
    PublicationLimits,
    artifact_closure,
    artifact_link,
    artifact_read_call,
    artifact_resource_id,
    prepare_artifact_attachment,
    run_artifact_check,
)
from dlightrag.engine.answer.resources.models import PUBLISHED_ARTIFACT_HANDLE_PREFIX

NOT_AN_ARTIFACT = (
    "resource_id takes an earlier turn's Artifact handle ({prefix}…); attach a workspace "
    "file by path."
)
NO_EARLIER = "This Run cannot read earlier Artifacts; attach a workspace file by path."
NO_FILENAME = "{handle} has no filename to restore it under; materialize it, then attach by path."
OCCUPIED = (
    "artifacts/{name} already holds a different file. Attach that one by path, or move it and "
    "attach {handle} again."
)


class RestoredSources(NamedTuple):
    """What reopening a published product did with the files it was built from (ADR 0052)."""

    restored: tuple[str, ...] = ()
    """Brought back into artifacts/, and re-declared so the next publication carries them."""
    kept: tuple[str, ...] = ()
    """Already in the workspace, which is this Run's own copy and is left alone."""
    unreadable: tuple[str, ...] = ()
    """Retention no longer holds them; the product opens, and the short kit is reported."""


class AttachArtifactArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    path: str | None = Field(
        default=None,
        min_length=1,
        max_length=500,
        description="Path relative to the artifacts/ directory.",
    )
    resource_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="An earlier turn's Artifact handle (artifact-…), to publish that version "
        "again instead of a workspace file.",
    )
    label: str | None = Field(
        default=None,
        max_length=200,
        description="Optional user-facing label; defaults to the safe filename.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="The artifacts/ files this deliverable was built from: the script that "
        "generates it and the inputs it reads. They are published with it, are not deliverables "
        "of their own, and come back when a later turn reopens the published version. Name them "
        "for anything a program produced — a workbook, a Word file, a report — because those "
        "cannot link to what made them, and the Run that made them does not outlive itself. How "
        "many fit is the publication budget's to say, not this field's.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        if (self.path is None) == (self.resource_id is None):
            raise ValueError("attach_artifact takes exactly one of path or resource_id")
        if self.resource_id is not None and self.sources:
            raise ValueError("sources describe files a Run attaches by path")
        return self


def attach_artifact_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="attach_artifact",
        description="Attach one completed artifacts/ file as a root user deliverable, after its "
        "final write or edit, or an earlier turn's Artifact by resource_id, which is restored "
        "to artifacts/ under its filename. Files that an attached Markdown or HTML root links "
        "to are included automatically, so attach only the root. Name the files the deliverable "
        "was built from in sources: a product a program produced cannot link to its own script "
        "and inputs, and the Run that wrote them is gone by the next turn. Place the returned "
        "Artifact link where it helps in the Answer; the Host adds an attached root at the end "
        "if the Answer omits it. The Answer links only roots attached in this Run and the files "
        "they link, so an earlier turn's Artifact it links is attached again. Restoring an "
        "Artifact by resource_id also restores the sources it was published with, and says "
        "which it kept because the workspace already held one and which it could not read back.",
        input_model=AttachArtifactArgs,
        replay_policy="replayable",
        contract_version=2,
    )


def attach_artifact_tool(
    artifacts_root: Path,
    *,
    scheduler: AccessScheduler,
    limits: PublicationLimits,
    environment: ExecutionEnvironment | None = None,
    admitted_bytes_reader: AdmittedBytesReader | None = None,
    protected_digests: Collection[str] = frozenset(),
    protected_index: Callable[[], ProtectedTextIndex] | None = None,
    delivery_ledger: DeliveryLedger | None = None,
    sealed_artifacts: Sequence[SealedArtifact] = (),
) -> AgentTool:
    """Build the optional parent-Research publication-intent tool.

    An earlier turn's Artifact is published again rather than linked across Runs: the new row
    pins the same content-addressed Blob, so the copy costs no bytes and the Answer outlives
    the Run that first published it (ADR 0013, ADR 0023).

    ``protected_digests`` withholds a Skill document's own bytes; ``protected_index`` withholds
    a text deliverable that reproduces the document's own words, and is asked for only when a
    file is actually validated (ADR 0046, ADR 0061).

    ``delivery_ledger`` is the current attempt's record, if the Run composed one. When it has
    something the reader's record would report, the success text also carries that preview: the
    declared reads not yet covered, whether a command named a declared verifier, and whether an
    HTML page carries the toolkit's provenance (ADR 0058). The preview changes no file and never
    fails the attachment.

    ``sealed_artifacts`` are the built-in declarations that let one Artifact name claim sealed
    (ADR 0066, ADR 0074). An Artifact under a declared name — directly, or as a member of the zip
    it is — is put to that Skill's gate on its bytes; its refusal withholds this one file and says
    what to do. A file under no declared name is an ordinary document, not a claim.
    """

    async def restore_sources(
        sources: tuple[DeclaredSource, ...],
        runtime: ToolRuntime,
        upserts: list[AgentspacePathFact],
    ) -> RestoredSources:
        """Bring back what the earlier product was built from, under artifacts/.

        A path the workspace already holds is left alone: the Run that asked to reopen a
        product may have written its own copy of a source already, and that copy is its work.
        A source retention has dropped is skipped so the product still opens, and it is
        reported rather than counted, because a kit that came back short is not the kit that
        was published.
        """
        if environment is None or admitted_bytes_reader is None:
            return RestoredSources(unreadable=tuple(source.path for source in sources))
        artifacts_relative = artifacts_root.resolve().relative_to(environment.root.resolve())
        restored: list[str] = []
        kept: list[str] = []
        unreadable: list[str] = []
        for source in sources:
            try:
                target = environment.resolve((artifacts_relative / source.path).as_posix())
            except PathRejected, ValueError:
                unreadable.append(source.path)
                continue
            if environment.stat_kind(target) == "file":
                kept.append(source.path)
                continue
            admitted = await admitted_bytes_reader(source.resource_id, runtime)
            if isinstance(admitted, ToolResult):
                unreadable.append(source.path)
                continue
            if hashlib.sha256(admitted.content).hexdigest() in protected_digests:
                unreadable.append(source.path)
                continue
            inventory = await replace_agentspace_file(
                environment, scheduler, target, admitted.content
            )
            if isinstance(inventory, ToolResult):
                unreadable.append(source.path)
                continue
            upserts.extend(inventory.upserts)
            restored.append(source.path)
        return RestoredSources(tuple(restored), tuple(kept), tuple(unreadable))

    async def restore(
        handle: str, runtime: ToolRuntime
    ) -> tuple[str, ToolEffects, RestoredSources] | ToolResult:
        """Copy an earlier Artifact's bytes back to artifacts/, and the files built it."""
        if environment is None or admitted_bytes_reader is None:
            return ToolResult.text(NO_EARLIER, is_error=True)
        if not handle.startswith(PUBLISHED_ARTIFACT_HANDLE_PREFIX):
            return ToolResult.text(
                NOT_AN_ARTIFACT.format(prefix=PUBLISHED_ARTIFACT_HANDLE_PREFIX), is_error=True
            )
        if blocked := agentspace_integrity_refusal(environment):
            return blocked
        admitted = await admitted_bytes_reader(handle, runtime)
        if isinstance(admitted, ToolResult):
            return admitted
        if hashlib.sha256(admitted.content).hexdigest() in protected_digests:
            # Refused before the copy, so a refused attach does not leave the document in the
            # workspace it was refused from.
            return ToolResult.text(
                "protected_content: That earlier Artifact is a Skill document, which is its "
                "author's to publish and not this Run's.",
                is_error=True,
            )
        name = PurePosixPath(admitted.filename or "").name
        if not name:
            return ToolResult.text(NO_FILENAME.format(handle=handle), is_error=True)
        try:
            # The environment's root is the resolved workspace; compare like with like, or a
            # workspace reached through a symlink would refuse every restore.
            relative = artifacts_root.resolve().relative_to(environment.root.resolve()) / name
            target = environment.resolve(relative.as_posix())
        except (PathRejected, ValueError) as exc:
            return ToolResult.text(str(exc), is_error=True)
        async with scheduler.hold(PathAccess(path=str(target), kind="read")):
            present = environment.stat_kind(target) == "file"
            if present and environment.read_bytes(target) != admitted.content:
                return ToolResult.text(OCCUPIED.format(name=name, handle=handle), is_error=True)
        upserts: list[AgentspacePathFact] = []
        if not present:
            inventory = await replace_agentspace_file(
                environment, scheduler, target, admitted.content
            )
            if isinstance(inventory, ToolResult):
                return inventory
            upserts.extend(inventory.upserts)
        declared = await restore_sources(admitted.sources, runtime, upserts)
        effects = ToolEffects(
            workspace_inventory=AgentspaceInventoryFacts(upserts=tuple(upserts))
            if upserts
            else None
        )
        return name, effects, declared

    async def execute(args: BaseModel, runtime: ToolRuntime) -> ToolResult:
        attachment_args = cast(AttachArtifactArgs, args)
        await runtime.emit_update(
            ToolResult.text("", subject=attachment_args.path or attachment_args.resource_id)
        )
        path, effects, restored = attachment_args.path, ToolEffects(), ""

        def refuse(verdict: SealedVerdict) -> ToolResult:
            """A sealed claim the gate did not attest: recorded for the label, refused to the model."""
            if delivery_ledger is not None:
                delivery_ledger.record_sealed_refusal(
                    verdict.skill, verdict.artifact, verdict.reasons
                )
            return ToolResult.text(verdict.refusal(), is_error=True, effects=effects)

        sources = tuple(attachment_args.sources)
        if attachment_args.resource_id is not None:
            outcome = await restore(attachment_args.resource_id, runtime)
            if isinstance(outcome, ToolResult):
                return outcome
            path, effects, back = outcome
            # What came back is re-declared, so the next turn publishes the same kit again; a
            # source left in place is the workspace's own copy, and one that could not be read
            # back is named rather than counted, because a short kit is not the kit that shipped.
            sources = (*back.restored, *back.kept)
            restored = f"restored {attachment_args.resource_id} to artifacts/{path}"
            if back.restored:
                restored += f", with {', '.join(back.restored)}"
            if back.kept:
                restored += f"; kept the workspace's own {', '.join(back.kept)}"
            if back.unreadable:
                restored += f"; could not read back {', '.join(back.unreadable)}"
            restored += "; "
        async with scheduler.hold(AgentspaceAccess()):
            try:
                # A built-in Skill's declared sealed artifact is machine data: the exact-byte
                # digest gate still protects it, but the engine/verifier prose it carries is not a
                # reader-facing copy of the Skill document. Every other file is text-scanned.
                machine_names = frozenset(contract.name.casefold() for contract in sealed_artifacts)
                # The check scans the Agent Workspace and decodes the file.
                attachment = await run_artifact_check(
                    prepare_artifact_attachment,
                    artifacts_root,
                    path=cast(str, path),
                    label=attachment_args.label or "",
                    sources=sources,
                    limits=limits,
                    protected_digests=protected_digests,
                    protected_index=protected_index,
                    protected_text_exempt_names=machine_names,
                )
            except ArtifactValidationError as exc:
                return ToolResult.text(
                    f"{exc.kind}: {exc.description}",
                    is_error=True,
                    effects=effects,
                )
            sealed_candidates: list[tuple[str, bytes]] = []
            unjudged = None
            if sealed_artifacts:
                # Inside the same hold: the gate must judge the bytes the check accepted, not a
                # later read of the root a parallel tool could have replaced. A read that fails
                # after the check already read the file is a host fault; under a reserved name it
                # is a refusal, not a way around the gate.
                names = {contract.name.casefold() for contract in sealed_artifacts}
                closure = artifact_closure(
                    artifacts_root,
                    (attachment.relative_path, *attachment.sources),
                    limits=limits,
                )
                for relative in closure:
                    suffix = PurePosixPath(relative).suffix.lower()
                    if PurePosixPath(relative).name.casefold() not in names and suffix != ".zip":
                        continue
                    try:
                        data = (artifacts_root / relative).read_bytes()
                    except OSError:
                        unjudged = unjudged or attest_refused(
                            relative, sealed_artifacts, UNREADABLE
                        )
                        continue
                    if (
                        relative == attachment.relative_path
                        and hashlib.sha256(data).hexdigest() != attachment.content_digest
                    ):
                        unjudged = unjudged or attest_refused(relative, sealed_artifacts, CHANGED)
                        continue
                    sealed_candidates.append((relative, data))
        if unjudged is not None:
            return refuse(unjudged)
        refused = None
        attested: set[str] = set()
        for relative, data in sealed_candidates:
            # The gate is a small read-only Skill script over a temporary copy of the bytes; it
            # answers for the bytes, so a later turn re-attaching an already sealed Artifact gets
            # the same answer. The whole publication closure is gated, not only the root.
            verdict = await asyncio.to_thread(
                attest_bytes,
                data,
                relative,
                sealed_artifacts,
                size_limit=limits.max_file_bytes,
            )
            if verdict is None:
                continue
            if not verdict.sealed:
                refused = verdict
                break
            attested.add(verdict.skill)
        if refused is not None:
            return refuse(refused)
        sealed = ",".join(sorted(attested))
        if sealed and len(sealed) <= TOOL_PROGRESS_FACT_MAX_CHARS:
            # Only a live update reaches the durable ``tool_progress`` event (the settlement carries
            # none of the result's facts), and that event is the one record a resumed attempt's
            # ledger can read the gate's conclusion back from.
            await runtime.emit_update(ToolResult.text("", details={FACT_SEALED_SKILL: sealed}))
        if delivery_ledger is not None:
            for skill in attested:
                # A gate attestation is stronger evidence than this attempt's own load history, so
                # the reader's label can follow the fact instead of an older attempt's missing read.
                delivery_ledger.record_sealed_attestation(skill)
        link = artifact_link(attachment)
        resource_id = artifact_resource_id(attachment.relative_path)
        read_call = artifact_read_call(attachment.relative_path)
        built_from = (
            f" It ships with the {len(attachment.sources)} file(s) it was built from — "
            f"{', '.join(attachment.sources)} — so a later turn that reopens it gets them back "
            "and can run them again."
            if attachment.sources
            else ""
        )
        record = ""
        if delivery_ledger is not None:
            delivery_ledger.record_attachment(attachment.relative_path)
            record = delivery_ledger.preview_text(
                page=_page_provenance(artifacts_root, attachment.relative_path)
            )
            # A delivery no Skill stands behind still has a reader record. Say it while the Run can
            # correct it, instead of leaving the attachment accepted and the reader met by an
            # unattested record after the Run settles.
            unattested = delivery_ledger.no_skill_delivery()
            if unattested:
                record = f"{record}\n\n{' '.join(unattested)}"
        return ToolResult.text(
            f"{restored}attached {attachment.relative_path} ({attachment.size_bytes} bytes); "
            f"place it with {link}; a later turn continues from this published version "
            f"with {read_call}.{built_from}{record}",
            details={
                "artifact_attachment": {
                    "relative_path": attachment.relative_path,
                    "label": attachment.label,
                    "content_digest": attachment.content_digest,
                    "size_bytes": attachment.size_bytes,
                    "presentation": attachment.presentation,
                    "resource_id": resource_id,
                    "sources": list(attachment.sources),
                }
            },
            effects=effects,
        )

    return attach_artifact_declaration().bind(execute)


def _page_provenance(artifacts_root: Path, relative_path: str) -> PageProvenance | None:
    """Whether a delivered HTML page carries the toolkit's provenance, read from its bytes.

    ``report-source`` is the block html-report writes into every document it builds (ADR 0038,
    ADR 0051). An unreadable file is no finding: the report is already validated by the time this
    runs, and a preview that failed an attachment would be the gate this one is not.
    """
    if not relative_path.lower().endswith((".html", ".htm")):
        return None
    try:
        data = (artifacts_root / relative_path).read_bytes()
    except OSError:
        return None
    return PageProvenance(
        file=relative_path,
        toolkit_built=b'id="report-source"' in data,
    )


__all__ = ["AttachArtifactArgs", "attach_artifact_tool"]
