# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structured root Artifact attachment through the parent Research tool."""

import hashlib
import json
import threading
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.environment.access import (
    AccessScheduler,
)
from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
from dlightrag.engine.agent.sealed_artifact import SealedArtifact
from dlightrag.engine.agent.skill_delivery import DeliveryLedger
from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.agent.skill_reads import RequiredRead, SkillRead
from dlightrag.engine.agent.skills import SealedArtifactDeclaration
from dlightrag.engine.agent.tools.contracts import (
    AdmittedBytes,
    DeclaredSource,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.publication import PublicationLimits, artifact_resource_id
from dlightrag.engine.answer.tools.artifacts import (
    AttachArtifactArgs,
    attach_artifact_declaration,
    attach_artifact_tool,
)
from tests.support.tool_helpers import recording_tool_runtime, tool_runtime

_PROTECTED_DOCUMENT = (
    "Every claim in a draft is checked against the passage it cites before the answer leaves "
    "the Run. When the passage does not support the claim, the sentence is removed instead of "
    "softened, and the gap is named in the same paragraph. The check runs again on the final "
    "draft in the same order, so a reader can follow what changed between two versions "
    "without reading the whole ledger. Nothing is filled in from memory: an absence is stated, "
    "not guessed, because a confident sentence that no evidence carries is the one failure this "
    "method exists to prevent."
)


def _protected_index() -> ProtectedTextIndex:
    return ProtectedTextIndex.build(
        [("method", "references/method.md", _PROTECTED_DOCUMENT.encode("utf-8"))]
    )


def _ledger_with_a_missing_read() -> DeliveryLedger:
    """One attempt that loaded the sealed Skill and read none of its declared references."""
    ledger = DeliveryLedger()
    ledger.record_load(
        skill="sealed-report",
        verifier="scripts/validate_model.py",
        required_reads=(
            RequiredRead(
                paths=("references/grounding-protocol.md",),
                when="before the first number",
            ),
        ),
        read=SkillRead(skill="sealed-report", path="SKILL.md"),
    )
    return ledger


def _ledger_for_a_skill_without_a_verifier() -> DeliveryLedger:
    ledger = DeliveryLedger()
    ledger.record_load(
        skill="interactive-html",
        verifier=None,
        required_reads=(),
        read=SkillRead(skill="interactive-html", path="SKILL.md"),
    )
    return ledger


@pytest.mark.asyncio
async def test_attach_artifact_validates_and_returns_a_structured_receipt(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    )

    result = await tool.execute(
        AttachArtifactArgs(path="analysis.md", label="Open analysis"),
        tool_runtime(),
    )

    assert tool.replay_policy == "replayable"
    # The tool says how attaching works; when to attach is the system prompt's judgment.
    assert "included automatically" in tool.description
    assert result.is_error is False
    assert "[Open analysis](artifact:artifact-431b1900963e6cd2f4a1)" in result.text_content
    assert result.details is not None
    # The receipt names the address a later turn reads the published version with:
    # it is the same deterministic id publication binds, so it is known now.
    assert result.details["artifact_attachment"] == {
        "relative_path": "analysis.md",
        "label": "Open analysis",
        "content_digest": hashlib.sha256(b"Grounded analysis.").hexdigest(),
        "size_bytes": len(b"Grounded analysis."),
        "presentation": "markdown",
        "resource_id": "artifact-431b1900963e6cd2f4a1",
        "sources": [],
    }


@pytest.mark.asyncio
async def test_attaching_without_a_loaded_skill_says_the_reader_record_is_unattested(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text('<html lang="zh">report</html>', encoding="utf-8")
    ledger = DeliveryLedger()
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        delivery_ledger=ledger,
    )

    result = await tool.execute(AttachArtifactArgs(path="report.html"), tool_runtime())

    assert result.is_error is False, result.text_content
    assert "reader's record will publish report.html as unattested" in result.text_content
    assert ledger.no_skill_delivery() != ()


@pytest.mark.asyncio
async def test_attach_artifact_validates_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dlightrag.engine.answer.tools import artifacts

    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    prepare = artifacts.prepare_artifact_attachment
    threads: list[threading.Thread] = []

    def probe(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread())
        return prepare(*args, **kwargs)

    monkeypatch.setattr(artifacts, "prepare_artifact_attachment", probe)
    tool = attach_artifact_tool(root, scheduler=AccessScheduler(), limits=PublicationLimits())

    result = await tool.execute(AttachArtifactArgs(path="analysis.md"), tool_runtime())

    assert result.is_error is False
    assert threads and threads[0] is not threading.current_thread()
    assert threads[0].name.startswith("artifact-check")


@pytest.mark.asyncio
async def test_attach_artifact_rejects_missing_and_unsafe_paths(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    )

    missing = await tool.execute(AttachArtifactArgs(path="missing.md"), tool_runtime())
    unsafe = await tool.execute(AttachArtifactArgs(path="../secret.md"), tool_runtime())

    assert missing.is_error is True
    assert missing.text_content.startswith("missing_file:")
    assert unsafe.is_error is True
    assert unsafe.text_content.startswith("invalid_reference:")


@pytest.mark.asyncio
async def test_attach_artifact_reports_its_path_as_the_live_subject(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    updates: list[ToolResult] = []

    await attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
    ).execute(
        AttachArtifactArgs(path="analysis.md"),
        recording_tool_runtime(updates, tool_name="attach_artifact"),
    )

    assert [update.subject for update in updates] == ["analysis.md"]


def test_the_description_states_that_an_earlier_turns_artifact_is_attached_again() -> None:
    # Publication refuses an Answer that links an Artifact outside this Run's attachment
    # closure, and the correction pass is the first place a model would otherwise learn it.
    description = attach_artifact_declaration().description

    assert "attached in this Run" in description
    assert "earlier turn's Artifact" in description
    assert "attached again" in description


EARLIER = "artifact-5f0c9e1d2a3b4c5d6e7f"


def _restoring_tool(tmp_path: Path, content: bytes, calls: list[str] | None = None):
    """The tool as composition builds it, with a reader that holds one earlier Artifact."""

    async def reader(resource_id: str, runtime: ToolRuntime) -> AdmittedBytes | ToolResult:
        if calls is not None:
            calls.append(resource_id)
        return AdmittedBytes("res-adopted", "text/markdown", content, "analysis.md")

    root = tmp_path / "artifacts"
    root.mkdir(exist_ok=True)
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
        admitted_bytes_reader=reader,
    )
    return tool, root


@pytest.mark.asyncio
async def test_an_earlier_artifact_is_restored_and_attached_in_one_call(tmp_path: Path) -> None:
    tool, root = _restoring_tool(tmp_path, b"Earlier analysis.")

    result = await tool.execute(AttachArtifactArgs(resource_id=EARLIER), tool_runtime())

    assert result.is_error is False, result.text_content
    assert (root / "analysis.md").read_bytes() == b"Earlier analysis."
    assert result.text_content.startswith(f"restored {EARLIER} to artifacts/analysis.md; attached")
    # Restored under its own filename, it keeps the address it was published at.
    assert result.details is not None
    assert result.details["artifact_attachment"]["resource_id"] == artifact_resource_id(
        "analysis.md"
    )
    assert result.effects.workspace_inventory is not None


@pytest.mark.asyncio
async def test_an_identical_file_already_in_place_is_attached_without_a_write(
    tmp_path: Path,
) -> None:
    tool, root = _restoring_tool(tmp_path, b"Earlier analysis.")
    (root / "analysis.md").write_bytes(b"Earlier analysis.")

    result = await tool.execute(AttachArtifactArgs(resource_id=EARLIER), tool_runtime())

    assert result.is_error is False
    assert result.effects.workspace_inventory is None


@pytest.mark.asyncio
async def test_a_different_file_in_the_way_is_never_overwritten(tmp_path: Path) -> None:
    tool, root = _restoring_tool(tmp_path, b"Earlier analysis.")
    (root / "analysis.md").write_bytes(b"This turn's new draft.")

    result = await tool.execute(AttachArtifactArgs(resource_id=EARLIER), tool_runtime())

    assert result.is_error is True
    assert "already holds a different file" in result.text_content
    assert (root / "analysis.md").read_bytes() == b"This turn's new draft."


@pytest.mark.asyncio
async def test_resource_id_takes_only_an_artifact_handle(tmp_path: Path) -> None:
    calls: list[str] = []
    tool, _root = _restoring_tool(tmp_path, b"x", calls)

    result = await tool.execute(AttachArtifactArgs(resource_id="res-upload-1"), tool_runtime())

    assert result.is_error is True
    assert "earlier turn's Artifact handle (artifact-" in result.text_content
    assert calls == []


@pytest.mark.asyncio
async def test_a_run_that_cannot_read_earlier_artifacts_says_so(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    tool = attach_artifact_tool(root, scheduler=AccessScheduler(), limits=PublicationLimits())

    result = await tool.execute(AttachArtifactArgs(resource_id=EARLIER), tool_runtime())

    assert result.is_error is True
    assert "cannot read earlier Artifacts" in result.text_content


def test_a_call_names_exactly_one_of_path_or_resource_id() -> None:
    with pytest.raises(ValidationError):
        AttachArtifactArgs()
    with pytest.raises(ValidationError):
        AttachArtifactArgs(path="analysis.md", resource_id=EARLIER)
    # What produced a file is declared where the file is attached, never on a restore: the
    # earlier publication already carries its own sources.
    with pytest.raises(ValidationError):
        AttachArtifactArgs(resource_id=EARLIER, sources=["build.py"])


def _docx_bytes() -> bytes:
    """The smallest container publication accepts as the Office document it claims to be."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        archive.writestr("word/document.xml", "<w:document/>")
    return buffer.getvalue()


def _source_tool(tmp_path: Path, *, sources: list[str]):
    """The tool as composition builds it, over a workspace holding a deliverable and its files."""
    root = tmp_path / "artifacts"
    root.mkdir(exist_ok=True)
    (root / "report.docx").write_bytes(_docx_bytes())
    (root / "build.py").write_text("print('build')\n", encoding="utf-8")
    (root / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
    )
    return tool, root


@pytest.mark.asyncio
async def test_a_deliverable_declares_the_files_it_was_built_from(tmp_path: Path) -> None:
    tool, _root = _source_tool(tmp_path, sources=[])

    result = await tool.execute(
        AttachArtifactArgs(path="report.docx", sources=["build.py", "data.csv"]), tool_runtime()
    )

    assert result.is_error is False, result.text_content
    assert result.details is not None
    assert result.details["artifact_attachment"]["sources"] == ["build.py", "data.csv"]
    assert "the 2 file(s) it was built from" in result.text_content
    assert "build.py, data.csv" in result.text_content


@pytest.mark.asyncio
async def test_a_declared_source_the_workspace_does_not_hold_is_refused(tmp_path: Path) -> None:
    tool, _root = _source_tool(tmp_path, sources=[])

    result = await tool.execute(
        AttachArtifactArgs(path="report.docx", sources=["build.py", "nowhere.py"]), tool_runtime()
    )

    assert result.is_error is True
    assert result.text_content.startswith("missing_file:")
    assert "nowhere.py" in result.text_content


@pytest.mark.asyncio
async def test_a_deliverable_cannot_be_its_own_source(tmp_path: Path) -> None:
    tool, _root = _source_tool(tmp_path, sources=[])

    result = await tool.execute(
        AttachArtifactArgs(path="report.docx", sources=["report.docx"]), tool_runtime()
    )

    assert result.is_error is True
    assert "cannot be its own source" in result.text_content


@pytest.mark.asyncio
async def test_reopening_a_published_deliverable_brings_back_what_built_it(tmp_path: Path) -> None:
    """The loop the feature exists for: restore the product, and its sources come with it."""

    async def reader(resource_id: str, runtime: ToolRuntime) -> AdmittedBytes | ToolResult:
        if resource_id == "res-build":
            return AdmittedBytes("res-build", "text/x-python", b"print('build')\n", "build.py")
        if resource_id == "res-data":
            return AdmittedBytes("res-data", "text/csv", b"a,b\n1,2\n", "data.csv")
        return AdmittedBytes(
            "res-adopted",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            _docx_bytes(),
            "report.docx",
            (DeclaredSource("build.py", "res-build"), DeclaredSource("data.csv", "res-data")),
        )

    root = tmp_path / "artifacts"
    root.mkdir()
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
        admitted_bytes_reader=reader,
    )

    # A file this Run already wrote for one source is its own work and is left alone.
    (root / "data.csv").write_text("a,b\n9,9\n", encoding="utf-8")

    result = await tool.execute(AttachArtifactArgs(resource_id=EARLIER), tool_runtime())

    assert result.is_error is False, result.text_content
    assert (root / "report.docx").read_bytes() == _docx_bytes()
    assert (root / "build.py").read_text(encoding="utf-8") == "print('build')\n"
    assert (root / "data.csv").read_text(encoding="utf-8") == "a,b\n9,9\n"
    # Both are declared again, so the next turn's publication keeps the kit intact.
    assert result.details is not None
    assert result.details["artifact_attachment"]["sources"] == ["build.py", "data.csv"]
    # The receipt names what came back, what the workspace already had, and what it could not
    # read: a kit that returned short is not the kit that was published.
    assert "with build.py" in result.text_content
    assert "kept the workspace's own data.csv" in result.text_content


@pytest.mark.asyncio
async def test_a_workspace_reached_through_a_symlink_restores_like_any_other(
    tmp_path: Path,
) -> None:
    # The environment resolves its root and the artifacts directory may not be: the tool
    # must compare the two as the same place.
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    (linked / "artifacts").mkdir()

    async def reader(resource_id: str, runtime: ToolRuntime) -> AdmittedBytes | ToolResult:
        return AdmittedBytes("res-adopted", "text/csv", b"a,b\n1,2\n", "ledger.csv")

    tool = attach_artifact_tool(
        linked / "artifacts",
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(linked),
        admitted_bytes_reader=reader,
    )

    result = await tool.execute(AttachArtifactArgs(resource_id=EARLIER), tool_runtime())

    assert result.is_error is False, result.text_content
    assert (real / "artifacts" / "ledger.csv").read_bytes() == b"a,b\n1,2\n"


# --- the delivery record preview ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_successful_attach_previews_the_record_the_reader_will_see(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
        delivery_ledger=_ledger_with_a_missing_read(),
    )

    result = await tool.execute(AttachArtifactArgs(path="analysis.md"), tool_runtime())

    assert result.is_error is False
    assert "(artifact:artifact-" in result.text_content
    assert "Delivery record preview" in result.text_content
    assert "not read: references/grounding-protocol.md" in result.text_content
    assert "no command recorded so far named sealed-report's verifier" in result.text_content
    assert "withholds nothing and changes no file" in result.text_content


@pytest.mark.asyncio
async def test_an_html_page_without_the_toolkit_provenance_is_named_in_the_preview(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text(
        "<html><body><h1>Hand written</h1></body></html>", encoding="utf-8"
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
        delivery_ledger=_ledger_for_a_skill_without_a_verifier(),
    )

    result = await tool.execute(AttachArtifactArgs(path="report.html"), tool_runtime())

    assert result.is_error is False, result.text_content
    assert "carries no report-source block" in result.text_content
    assert "report.html" in result.text_content


@pytest.mark.asyncio
async def test_a_toolkit_built_page_and_a_covered_record_add_no_preview_line(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text(
        '<html><body><script type="application/json" id="report-source">{}</script>'
        "<p>Built.</p></body></html>",
        encoding="utf-8",
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
        delivery_ledger=_ledger_for_a_skill_without_a_verifier(),
    )

    result = await tool.execute(AttachArtifactArgs(path="report.html"), tool_runtime())

    assert result.is_error is False, result.text_content
    assert "Delivery record preview" not in result.text_content


@pytest.mark.asyncio
async def test_an_attach_without_a_ledger_says_nothing_new(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        environment=LocalExecutionEnvironment(tmp_path),
    )

    result = await tool.execute(AttachArtifactArgs(path="analysis.md"), tool_runtime())

    assert result.is_error is False
    assert "Delivery record preview" not in result.text_content


_SEALED_DECLARATION = SealedArtifactDeclaration(
    name="case.json", gate="scripts/verify_sealed_case.py", unsealed_label="Unsealed memo"
)


def _gate(tmp_path: Path, *, claim: str, sealed: bool) -> Path:
    """A minimal sealed-artifact gate: prints the three-state report a Skill's gate prints, and exits."""
    gate = tmp_path / "gate.py"
    reasons = (
        "[]" if sealed or claim == "none" else "['the case seal does not cover the current bytes']"
    )
    gate.write_text(
        "import json\n"
        f"print(json.dumps({{'claim': {claim!r}, 'sealed': {sealed!r}, 'reasons': {reasons}}}))\n"
        f"raise SystemExit({0 if sealed or claim == 'none' else 1})\n",
        encoding="utf-8",
    )
    return gate


def _case_contract(
    tmp_path: Path, *, claim: str = "sealed", sealed: bool = True, name: str = "case.json"
) -> SealedArtifact:
    return SealedArtifact(
        skill="sealed-report",
        name=name,
        gate=_gate(tmp_path, claim=claim, sealed=sealed),
    )


@pytest.mark.asyncio
async def test_a_claimed_sealed_artifact_is_refused_when_its_skill_gate_declines(
    tmp_path: Path,
) -> None:
    """The gate runs on the bytes: an unsealed `case.json` is withheld with the gate's reason."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text(
        '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}', encoding="utf-8"
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path, claim="unsealed", sealed=False),),
    )

    result = await tool.execute(AttachArtifactArgs(path="case.json", label="Case"), tool_runtime())

    assert result.is_error is True
    assert "refused case.json" in result.text_content
    assert "does not cover the current bytes" in result.text_content
    assert "withheld" in result.text_content
    assert not (tmp_path / "published").exists()


@pytest.mark.asyncio
async def test_a_sealed_artifact_attaches_from_its_bytes_not_the_current_attempt(
    tmp_path: Path,
) -> None:
    """A later turn re-attaches a case whose sealing command ran in another attempt: the bytes rule.

    The delivery ledger here is the *unsealed* one (a missing read, no verifier run), which is what
    a continuation sees. The Artifact is sealed, and the Skill's gate says so, so it attaches.
    """
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text(
        '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}', encoding="utf-8"
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        delivery_ledger=_ledger_with_a_missing_read(),
        sealed_artifacts=(_case_contract(tmp_path),),
    )

    result = await tool.execute(AttachArtifactArgs(path="case.json", label="Case"), tool_runtime())

    assert result.is_error is False, result.text_content
    assert "attached case.json" in result.text_content


@pytest.mark.asyncio
async def test_an_ordinary_document_under_no_declared_name_is_not_gated(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Grounded analysis.", encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path, claim="unsealed", sealed=False),),
    )

    result = await tool.execute(AttachArtifactArgs(path="analysis.md"), tool_runtime())

    assert result.is_error is False
    assert "attached analysis.md" in result.text_content


@pytest.mark.asyncio
async def test_a_same_named_file_without_the_marker_attaches_as_an_ordinary_document(
    tmp_path: Path,
) -> None:
    """A draft or data file named `case.json` is not a sealed claim: no gate, no refusal."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text('{"case_version": "2.0"}', encoding="utf-8")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path, claim="none", sealed=False),),
    )

    result = await tool.execute(AttachArtifactArgs(path="case.json"), tool_runtime())

    assert result.is_error is False, result.text_content
    assert "attached case.json" in result.text_content


@pytest.mark.asyncio
async def test_a_gate_attested_attach_lifts_the_unsealed_label(tmp_path: Path) -> None:
    """A later turn attaching the sealed file has proof; the old missing read does not label it."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text(
        '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}', encoding="utf-8"
    )
    read = RequiredRead(paths=("references/gate-audit.md",), when="before delivery")
    ledger = DeliveryLedger()
    ledger.record_load(
        skill="sealed-report",
        verifier="scripts/validate_model.py",
        required_reads=(read,),
        read=SkillRead(skill="sealed-report", path="SKILL.md"),
        sealed_artifact=_SEALED_DECLARATION,
    )
    preview = ledger.preview()
    assert preview is not None
    assert preview.unsealed_labels == ("Unsealed memo",)
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        delivery_ledger=ledger,
        sealed_artifacts=(_case_contract(tmp_path),),
    )

    result = await tool.execute(AttachArtifactArgs(path="case.json"), tool_runtime())

    assert result.is_error is False, result.text_content
    preview = ledger.preview()
    assert preview is not None
    assert preview.unsealed_labels == ()


@pytest.mark.asyncio
async def test_a_sealed_case_linked_from_an_ordinary_page_is_gated_too(tmp_path: Path) -> None:
    """A rejected sealed case cannot reach the reader as a dependency of a page that attaches."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "notes.md").write_text("[the case](artifact:case.json)", encoding="utf-8")
    (root / "case.json").write_text(
        '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}', encoding="utf-8"
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path, claim="unsealed", sealed=False),),
    )

    result = await tool.execute(AttachArtifactArgs(path="notes.md"), tool_runtime())

    assert result.is_error is True
    assert "refused case.json" in result.text_content


@pytest.mark.asyncio
async def test_a_sealed_case_declared_as_an_ordinary_pages_source_is_gated_too(
    tmp_path: Path,
) -> None:
    """A workbook cannot link its inputs, so declared sources ship with it; they are gated too."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "book.md").write_text("The workbook.", encoding="utf-8")
    (root / "case.json").write_text(
        '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}', encoding="utf-8"
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path, claim="unsealed", sealed=False),),
    )

    result = await tool.execute(
        AttachArtifactArgs(path="book.md", sources=["case.json"]), tool_runtime()
    )

    assert result.is_error is True
    assert "refused case.json" in result.text_content


@pytest.mark.asyncio
async def test_a_refused_sealed_attach_is_recorded_in_the_delivery_record(
    tmp_path: Path,
) -> None:
    """A refused sealed claim stays a delivery finding and leaves the answer labelled."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text(
        '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}', encoding="utf-8"
    )
    read = RequiredRead(paths=("references/gate-audit.md",), when="before delivery")
    ledger = DeliveryLedger()
    ledger.record_load(
        skill="sealed-report",
        verifier="scripts/validate_model.py",
        required_reads=(read,),
        read=SkillRead(skill="sealed-report", path="SKILL.md"),
        sealed_artifact=_SEALED_DECLARATION,
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        delivery_ledger=ledger,
        sealed_artifacts=(_case_contract(tmp_path, claim="unsealed", sealed=False),),
    )

    result = await tool.execute(AttachArtifactArgs(path="case.json"), tool_runtime())

    assert result.is_error is True
    preview = ledger.preview()
    assert preview is not None and preview.unsealed_labels == ("Unsealed memo",)
    assert [row[0] for row in preview.sealed_refused] == ["case.json"]
    assert "refused case.json" in preview.text()


@pytest.mark.asyncio
async def test_a_zip_that_carries_the_sealed_name_is_gated_by_its_member(tmp_path: Path) -> None:
    """A sealed case shipped inside a zip is the same bytes; the gate reads the member, not a name."""
    root = tmp_path / "artifacts"
    root.mkdir()
    archive = root / "deliverable.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "case.json", '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}'
        )
        zf.writestr("notes.txt", "notes")
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path, claim="unsealed", sealed=False),),
    )

    result = await tool.execute(AttachArtifactArgs(path="deliverable.zip"), tool_runtime())

    assert result.is_error is True
    assert "refused deliverable.zip" in result.text_content


@pytest.mark.asyncio
async def test_a_zip_with_two_copies_of_the_sealed_name_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    archive = root / "deliverable.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        claimed = '{"delivery": {"case_seal": {"algorithm": "jcs-sha256", "digest": "x"}}}'
        zf.writestr("case.json", claimed)
        zf.writestr("copies/case.json", claimed)
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        sealed_artifacts=(_case_contract(tmp_path),),
    )

    result = await tool.execute(AttachArtifactArgs(path="deliverable.zip"), tool_runtime())

    assert result.is_error is True
    assert "2 files named" in result.text_content


@pytest.mark.asyncio
async def test_a_sealed_case_is_machine_data_not_reader_prose(tmp_path: Path) -> None:
    """The declared sealed artifact legitimately carries engine/verifier prose: the exact-byte
    gate still protects the Skill's documents, but the text gate does not refuse the case."""
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text(
        json.dumps({"delivery": {"audit": {"detail": _PROTECTED_DOCUMENT}}}), encoding="utf-8"
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        protected_index=_protected_index,
        sealed_artifacts=(_case_contract(tmp_path),),
    )

    result = await tool.execute(AttachArtifactArgs(path="case.json", label="Case"), tool_runtime())

    assert result.is_error is False, result.text_content
    assert "attached case.json" in result.text_content


@pytest.mark.asyncio
async def test_a_page_that_reproduces_protected_prose_is_still_refused(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "report.html").write_text(
        f"<!doctype html><html><body><p>{_PROTECTED_DOCUMENT}</p></body></html>",
        encoding="utf-8",
    )
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        protected_index=_protected_index,
        sealed_artifacts=(_case_contract(tmp_path),),
    )

    result = await tool.execute(AttachArtifactArgs(path="report.html"), tool_runtime())

    assert result.is_error is True
    assert "protected_content" in result.text_content


@pytest.mark.asyncio
async def test_a_zip_carrying_the_sealed_case_is_exempt_for_that_member(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "case.json", json.dumps({"delivery": {"audit": {"detail": _PROTECTED_DOCUMENT}}})
        )
        archive.writestr("README.md", "Delivery archive.")
    (root / "delivery.zip").write_bytes(buffer.getvalue())
    tool = attach_artifact_tool(
        root,
        scheduler=AccessScheduler(),
        limits=PublicationLimits(),
        protected_index=_protected_index,
        sealed_artifacts=(_case_contract(tmp_path),),
    )

    result = await tool.execute(
        AttachArtifactArgs(path="delivery.zip", label="Delivery"), tool_runtime()
    )

    assert result.is_error is False, result.text_content
    assert "attached delivery.zip" in result.text_content
