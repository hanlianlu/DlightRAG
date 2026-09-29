from pathlib import Path

import pytest
from lightrag.constants import PARSED_DIR_NAME

from dlightrag.engine.rag.corpus.ingestion.paths import (
    REMOTE_INGEST_DIR_NAME,
    REMOTE_SOURCES_DIR_NAME,
    UPLOADS_DIR_NAME,
    discard_parser_input,
    document_name,
    excluded_from_directory_scan,
    parser_input_path,
    place_parser_input,
    remote_ingest_batch_root,
    remote_parser_input_path,
    retained_remote_source_path,
    workspace_input_root,
)


@pytest.mark.parametrize(
    ("name", "is_dir"),
    [
        (".cache", True),
        (".hidden.pdf", False),
        (PARSED_DIR_NAME, True),
        (UPLOADS_DIR_NAME, True),
        (REMOTE_INGEST_DIR_NAME, True),
        (REMOTE_SOURCES_DIR_NAME, True),
    ],
)
def test_a_local_source_listing_skips_dot_entries_and_parser_folders(
    name: str, is_dir: bool
) -> None:
    assert excluded_from_directory_scan(name, is_dir=is_dir)


@pytest.mark.parametrize(("name", "is_dir"), [("report.pdf", False), ("nested", True)])
def test_a_local_source_listing_keeps_ordinary_entries(name: str, is_dir: bool) -> None:
    assert not excluded_from_directory_scan(name, is_dir=is_dir)


def test_a_documents_name_is_its_basename_without_a_parser_hint() -> None:
    """LightRAG derives a document's id from this name, so two files with it collide."""
    assert document_name("/stage/0/report.[mineru].pdf") == document_name("a/report.pdf")
    assert document_name("report.pdf") != document_name("report.docx")


def test_a_parser_input_is_placed_flat_under_its_basename(tmp_path: Path) -> None:
    """LightRAG looks a parser input up by basename only, never in a subfolder."""
    input_root = workspace_input_root(tmp_path / "corpus", "default")
    source = tmp_path / "stage" / "nested" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")

    target = place_parser_input(source, input_root)

    assert target == parser_input_path(input_root, source) == input_root / "report.pdf"
    assert target.read_bytes() == b"%PDF"
    assert source.read_bytes() == b"%PDF"
    assert sorted(path.name for path in input_root.iterdir()) == ["report.pdf"]


def test_placing_replaces_the_same_documents_earlier_input_and_keeps_one_in_place(
    tmp_path: Path,
) -> None:
    input_root = workspace_input_root(tmp_path / "corpus", "default")
    input_root.mkdir(parents=True)
    (input_root / "report.pdf").write_bytes(b"stale")
    source = tmp_path / "stage" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"fresh")

    placed = place_parser_input(source, input_root)

    assert placed.read_bytes() == b"fresh"
    assert place_parser_input(placed, input_root) == placed
    assert placed.read_bytes() == b"fresh"


def test_discarding_a_parser_input_keeps_its_sidecar(tmp_path: Path) -> None:
    parser_input = tmp_path / "report.pdf"
    parser_input.write_bytes(b"input")
    archived = tmp_path / PARSED_DIR_NAME / "report.pdf"
    sidecar = tmp_path / PARSED_DIR_NAME / "report.pdf.parsed"
    sidecar.mkdir(parents=True)
    archived.write_bytes(b"archived")

    discard_parser_input(parser_input)

    assert not parser_input.exists()
    assert not archived.exists()
    assert sidecar.is_dir()


def test_remote_parser_input_path_uses_ephemeral_hash_name(tmp_path: Path) -> None:
    root = remote_ingest_batch_root(
        input_root=tmp_path / "inputs" / "default",
        source_type="s3",
        batch_id="batch-1",
    )
    assert root == tmp_path / "inputs" / "default" / "__remote_ingest__" / "s3" / "batch-1"

    first = remote_parser_input_path(
        batch_root=root,
        source_uri="s3://bucket/team-a/report.pdf",
        key="team-a/report.pdf",
    )
    second = remote_parser_input_path(
        batch_root=root,
        source_uri="s3://bucket/team-b/report.pdf",
        key="team-b/report.pdf",
    )

    assert first.parent == root
    assert second.parent == root
    assert first.name.startswith("report__")
    assert first.suffix == ".pdf"
    assert first.name != second.name

    with pytest.raises(ValueError, match="remote object key is empty"):
        remote_parser_input_path(
            batch_root=root,
            source_uri="s3://bucket/",
            key="../",
        )


def test_retained_remote_source_path_uses_stable_workspace_location(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs" / "default"

    first = retained_remote_source_path(
        input_root=input_root,
        source_type="s3",
        source_uri="s3://bucket/team-a/report.pdf",
        key="team-a/report.pdf",
    )
    second = retained_remote_source_path(
        input_root=input_root,
        source_type="s3",
        source_uri="s3://bucket/team-a/report.pdf",
        key="team-a/report.pdf",
    )

    assert first == second
    assert first.parent == input_root / REMOTE_SOURCES_DIR_NAME / "s3"
    assert first.name.startswith("report__")
    assert first.suffix == ".pdf"
