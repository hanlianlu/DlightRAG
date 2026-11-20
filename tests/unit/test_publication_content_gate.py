# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Artifact publication refuses a text deliverable that reproduces a protected document.

The digest gate reads a file's bytes, so a copy of a Skill document under any name is already
refused; the content gate reads what a text file says, which is the road the digest cannot see:
the Run writing the document's method into a page, a note, or a report it produced (ADR 0061).
"""

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.answer.publication import (
    ArtifactValidationError,
    prepare_artifact_attachment,
    validate_publication,
)

DOCUMENT = (
    "Every claim in a draft is checked against the passage it cites before the answer leaves "
    "the Run. When the passage does not support the claim, the sentence is removed instead of "
    "softened, and the gap is named in the same paragraph. The check runs again on the final "
    "draft in the same order, so a reader can follow what changed between two versions "
    "without reading the whole ledger. Nothing is filled in from memory: an absence is stated, "
    "not guessed, because a confident sentence that no evidence carries is the one failure this "
    "method exists to prevent."
)
"""One protected document's own text, long past the publication gate's minimum run."""


def _index() -> ProtectedTextIndex:
    return ProtectedTextIndex.build([("method", "references/method.md", DOCUMENT.encode("utf-8"))])


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    root.mkdir()
    return root


@pytest.mark.parametrize("quoted", [DOCUMENT[:200], DOCUMENT[:300]])
def test_a_quotation_below_the_gate_is_published(tmp_path: Path, quoted: str) -> None:
    """A deliverable may quote a Skill's line, term, or table row; a reproduction is refused."""
    root = _root(tmp_path)
    (root / "notes.md").write_text(
        f"# Notes\n\nThe method says: {quoted}\n\nDone.", encoding="utf-8"
    )

    attachment = prepare_artifact_attachment(root, path="notes.md", protected_index=_index)

    assert attachment.relative_path == "notes.md"


def test_a_reproduced_document_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "copy.md").write_text(f"# Copied\n\n{DOCUMENT}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.md", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_run_at_the_gate_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "copy.md").write_text(f"# Copied\n\n{DOCUMENT[:400]}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.md", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_an_unrelated_text_deliverable_is_published(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "report.md").write_text("# Report\n\n" + "The numbers rose. " * 40, encoding="utf-8")

    attachment = prepare_artifact_attachment(root, path="report.md", protected_index=_index)

    assert attachment.relative_path == "report.md"


def test_terminal_publication_withholds_a_reproduced_dependency(tmp_path: Path) -> None:
    root = _root(tmp_path)
    (root / "copy.md").write_text(f"# Copied\n\n{DOCUMENT}\n", encoding="utf-8")
    (root / "note.md").write_text("See [copy](artifact:copy.md).\n", encoding="utf-8")
    attachment = prepare_artifact_attachment(root, path="note.md", protected_index=_index)

    plan = validate_publication(
        root,
        answer="Done. [Note](artifact:note.md)",
        attachments=(attachment,),
        protected_index=_index,
    )

    assert plan.outcome["status"] == "partial"
    assert [issue.kind for issue in plan.issues] == ["protected_content"]


def test_a_container_member_that_reproduces_a_document_is_refused(tmp_path: Path) -> None:
    """A document inside an archive is the same document, however the archive was named."""
    root = _root(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.md", f"# Notes\n\n{DOCUMENT}\n")
    (root / "kit.zip").write_bytes(buffer.getvalue())

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="kit.zip", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_renamed_text_deliverable_is_still_read(tmp_path: Path) -> None:
    """A name that claims binary does not hide the words a reader can read."""
    root = _root(tmp_path)
    (root / "copy.dat").write_text(f"# Copied\n\n{DOCUMENT}\n", encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.dat", protected_index=_index)

    assert raised.value.kind == "protected_content"


def _pdf(text: str) -> bytes:
    """A one-page PDF that shows ``text``, built here so the gate has a real document to read."""
    stream = f"BT /F1 10 Tf 40 750 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _docx(text: str) -> bytes:
    """A docx whose paragraph is split into one run per word, the way a writer saves it."""
    words = "".join(f"<w:r><w:t>{word} </w:t></w:r>" for word in text.split())
    document = (
        '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
        f'wordprocessingml/2006/main"><w:body><w:p>{words}</w:p></w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buffer.getvalue()


def _encoded(text: str) -> str:
    return "".join(
        f"&#x{ord(character):x};" if character.isalnum() else character for character in text
    )


@pytest.mark.parametrize(
    ("label", "filename", "content"),
    [
        ("markdown", "copy.md", DOCUMENT.encode("utf-8")),
        ("html-entities", "copy.html", _encoded(DOCUMENT).encode("utf-8")),
        ("utf-16-bom", "copy.txt", DOCUMENT.encode("utf-16")),
        ("utf-16-renamed", "copy.bin", DOCUMENT.encode("utf-16-le")),
        ("pdf", "copy.pdf", _pdf(DOCUMENT[:400])),
        ("docx-split-runs", "copy.docx", _docx(DOCUMENT[:400])),
    ],
)
def test_a_reproduction_is_refused_however_the_file_spells_it(
    tmp_path: Path, label: str, filename: str, content: bytes
) -> None:
    """A reader opens the format; so does the gate (ADR 0061).

    The digest gate only sees the document's own bytes. A Run that retypes the method into a
    page, a PDF, a Word file or a UTF-16 text is publishing something the digest cannot name,
    and each of those formats reaches a reader as words.
    """
    root = _root(tmp_path)
    (root / filename).write_bytes(content)

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path=filename, protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_pdf_of_unrelated_words_is_published(tmp_path: Path) -> None:
    """The gate reads a PDF's words; a page that states none of them is a delivery."""
    root = _root(tmp_path)
    (root / "report.pdf").write_bytes(_pdf("Revenue rose by four percent in the quarter."))

    attachment = prepare_artifact_attachment(root, path="report.pdf", protected_index=_index)

    assert attachment.relative_path == "report.pdf"


def test_a_named_character_reference_is_read_as_its_character(tmp_path: Path) -> None:
    """``&nbsp;`` and ``&mdash;`` are what a browser shows; the gate reads the same words."""
    encoded = DOCUMENT.replace(" ", "&nbsp;")
    root = _root(tmp_path)
    (root / "copy.md").write_text(encoded, encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.md", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_docx_that_splits_words_across_runs_is_read_as_its_sentence(tmp_path: Path) -> None:
    """A writer splits runs at formatting choices, including mid-word; the words are one."""
    words = "".join(
        f"<w:r><w:t>{DOCUMENT[offset : offset + 5]}</w:t></w:r>" for offset in range(0, 400, 5)
    )
    document = (
        '<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/'
        f'wordprocessingml/2006/main"><w:body><w:p>{words}</w:p></w:body></w:document>'
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", document)
    root = _root(tmp_path)
    (root / "copy.docx").write_bytes(buffer.getvalue())

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.docx", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_pdf_inside_a_zip_is_read_as_a_pdf(tmp_path: Path) -> None:
    """A container member is read as the media type its own name claims."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("report.pdf", _pdf(DOCUMENT[:400]))
    root = _root(tmp_path)
    (root / "bundle.zip").write_bytes(buffer.getvalue())

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="bundle.zip", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_gzipped_text_deliverable_is_read_after_it_is_opened(tmp_path: Path) -> None:
    """A wrapper is transparent: a reader opens a .gz and reads the words inside."""
    import gzip

    root = _root(tmp_path)
    (root / "copy.md.gz").write_bytes(gzip.compress(DOCUMENT.encode("utf-8")))

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.md.gz", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_an_encoded_run_straddling_the_read_window_is_still_refused(tmp_path: Path) -> None:
    """A raw-expanded run can be ten times its normalized length; the overlap covers it."""
    encoded = "".join(f"&#x{ord(character):x};" for character in DOCUMENT)
    # Push the run so that its raw start sits inside one window's tail and its end past it.
    root = _root(tmp_path)
    (root / "padded.md").write_text("x " * 32_000 + encoded, encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="padded.md", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_utf16_cjk_text_without_a_bom_is_read(tmp_path: Path) -> None:
    """CJK text in UTF-16 has few NUL bytes; the order is decided by what reads as text."""
    cjk = "这是一段受保护的方法文档，用来验证没有字节序标记的 UTF-16 文本也会被读取。" * 20
    index = ProtectedTextIndex.build([("method", "references/method.md", cjk.encode("utf-8"))])
    root = _root(tmp_path)
    (root / "copy.bin").write_bytes(cjk.encode("utf-16-le"))

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.bin", protected_index=lambda: index)

    assert raised.value.kind == "protected_content"


def test_a_reproduction_split_into_markdown_links_is_refused(tmp_path: Path) -> None:
    linked = "".join(f"[{DOCUMENT[offset : offset + 40]}](#)" for offset in range(0, 400, 40))
    root = _root(tmp_path)
    (root / "copy.md").write_text(linked, encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.md", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_inline_tags_in_an_xhtml_page_do_not_split_a_reproduction(tmp_path: Path) -> None:
    split = "".join(
        f"{DOCUMENT[offset : offset + 30]}<span></span>" for offset in range(0, len(DOCUMENT), 30)
    )
    root = _root(tmp_path)
    (root / "copy.xhtml").write_text(f"<p>{split}</p>", encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.xhtml", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_container_member_is_read_as_words_whatever_its_name_claims(tmp_path: Path) -> None:
    """A member's name is chosen by whoever built the archive; words are words under any name."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.png", DOCUMENT)
    root = _root(tmp_path)
    (root / "bundle.zip").write_bytes(buffer.getvalue())

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="bundle.zip", protected_index=_index)

    assert raised.value.kind == "protected_content"


def test_a_run_straddling_the_window_is_found_however_its_raw_length_inflates(
    tmp_path: Path,
) -> None:
    """Whitespace collapses and invisibles vanish, so a raw overlap cannot bound the window."""
    padded = DOCUMENT.replace(" ", " " + "\u200b" * 3_201)
    root = _root(tmp_path)
    (root / "copy.md").write_text("Lead. " + padded, encoding="utf-8")

    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(root, path="copy.md", protected_index=_index)

    assert raised.value.kind == "protected_content"


#: One protected document's prose, long enough that a machine artifact carrying it would trip
#: the text gate if it were scanned like a reader-facing page.
_MACHINE_DOCUMENT = DOCUMENT


def test_a_declared_machine_artifact_is_not_scanned_as_reader_prose(tmp_path: Path) -> None:
    """The built-in sealed ``case.json`` is machine data: its exact bytes stay digest-gated, and
    the engine/verifier prose it carries does not refuse its own publication. A page is scanned."""
    root = _root(tmp_path)
    (root / "case.json").write_text(
        '{"audit": {"detail": ' + __import__("json").dumps(_MACHINE_DOCUMENT) + "}}",
        encoding="utf-8",
    )
    attachment = prepare_artifact_attachment(
        root,
        path="case.json",
        protected_index=_index,
        protected_text_exempt_names=("case.json",),
    )
    assert attachment.relative_path == "case.json"

    (root / "report.html").write_text(
        f"<!doctype html><html><body><p>{_MACHINE_DOCUMENT}</p></body></html>",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(
            root,
            path="report.html",
            protected_index=_index,
            protected_text_exempt_names=("case.json",),
        )
    assert raised.value.kind == "protected_content"


def test_the_machine_exemption_keeps_the_exact_digest_gate(tmp_path: Path) -> None:
    root = _root(tmp_path)
    protected_bytes = _MACHINE_DOCUMENT.encode("utf-8")
    (root / "case.json").write_bytes(protected_bytes)
    protected_digests = frozenset({hashlib.sha256(protected_bytes).hexdigest()})
    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(
            root,
            path="case.json",
            protected_digests=protected_digests,
            protected_index=_index,
            protected_text_exempt_names=("case.json",),
        )
    assert raised.value.kind == "protected_content"


def test_only_the_declared_member_of_a_container_is_exempt(tmp_path: Path) -> None:
    """A zip carrying ``case.json`` is published; a copied reference passage elsewhere in it is not."""
    root = _root(tmp_path)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "case.json", '{"audit": ' + __import__("json").dumps(_MACHINE_DOCUMENT) + "}"
        )
        archive.writestr("report.html", f"<html><body><p>{_MACHINE_DOCUMENT}</p></body></html>")
    (root / "delivery.zip").write_bytes(buffer.getvalue())
    with pytest.raises(ArtifactValidationError) as raised:
        prepare_artifact_attachment(
            root,
            path="delivery.zip",
            protected_index=_index,
            protected_text_exempt_names=("case.json",),
        )
    assert raised.value.kind == "protected_content"

    clean = io.BytesIO()
    with zipfile.ZipFile(clean, "w") as archive:
        archive.writestr(
            "case.json", '{"audit": ' + __import__("json").dumps(_MACHINE_DOCUMENT) + "}"
        )
        archive.writestr("README.md", "Delivery archive.")
    (root / "delivery.zip").write_bytes(clean.getvalue())
    attachment = prepare_artifact_attachment(
        root,
        path="delivery.zip",
        protected_index=_index,
        protected_text_exempt_names=("case.json",),
    )
    assert attachment.relative_path == "delivery.zip"


def test_terminal_publication_accepts_the_machine_artifact_with_its_exemption(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    (root / "case.json").write_text(
        '{"audit": ' + __import__("json").dumps(_MACHINE_DOCUMENT) + "}", encoding="utf-8"
    )
    attachment = prepare_artifact_attachment(
        root,
        path="case.json",
        protected_index=_index,
        protected_text_exempt_names=("case.json",),
    )
    plan = validate_publication(
        root,
        answer="Done. [Case](artifact:case.json)",
        attachments=(attachment,),
        protected_index=_index,
        protected_text_exempt_names=("case.json",),
    )
    assert plan.outcome["status"] == "complete", plan.issues
