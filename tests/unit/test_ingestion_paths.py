from pathlib import Path

import pytest
from lightrag.constants import PARSED_DIR_NAME

from dlightrag.core.ingestion.paths import (
    UPLOADS_DIR_NAME,
    iter_ingestable_files,
    remote_namespace_root,
    remote_object_path,
    stage_input_file,
    staged_input_path,
    workspace_input_root,
)


def test_iter_ingestable_files_skips_parser_upload_and_hidden_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "docs"
    (root / "nested").mkdir(parents=True)
    (root / PARSED_DIR_NAME / "report.pdf.parsed").mkdir(parents=True)
    (root / UPLOADS_DIR_NAME / "old-batch").mkdir(parents=True)
    (root / ".cache").mkdir(parents=True)

    keep = root / "nested" / "keep.pdf"
    keep.write_bytes(b"ok")
    (root / PARSED_DIR_NAME / "report.pdf.parsed" / "report.blocks.jsonl").write_text("{}\n")
    (root / UPLOADS_DIR_NAME / "old-batch" / "stale.pdf").write_bytes(b"stale")
    (root / ".cache" / "hidden.pdf").write_bytes(b"hidden")
    (root / ".hidden.pdf").write_bytes(b"hidden")

    assert iter_ingestable_files(root) == [keep]


def test_iter_ingestable_files_accepts_explicit_upload_batch(tmp_path: Path) -> None:
    batch = tmp_path / "docs" / UPLOADS_DIR_NAME / "batch"
    batch.mkdir(parents=True)
    uploaded = batch / "uploaded.pdf"
    uploaded.write_bytes(b"ok")

    assert iter_ingestable_files(batch) == [uploaded]


def test_stage_input_file_preserves_relative_path(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source = source_root / "nested" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF")
    input_root = workspace_input_root(tmp_path / "inputs", "default")

    target = stage_input_file(input_root=input_root, file_path=source, relative_to=source_root)

    assert target == input_root / "nested" / "report.pdf"
    assert target.read_bytes() == b"%PDF"


def test_staged_input_path_falls_back_to_basename_when_source_escapes_root(
    tmp_path: Path,
) -> None:
    input_root = workspace_input_root(tmp_path / "inputs", "default")
    source = tmp_path / "outside" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF")

    target = staged_input_path(
        input_root=input_root,
        file_path=source,
        relative_to=tmp_path / "source",
    )

    assert target == input_root / "report.pdf"


def test_remote_object_path_sanitizes_namespace_and_rejects_empty_key(tmp_path: Path) -> None:
    root = remote_namespace_root(
        working_dir=tmp_path,
        source_type="s3",
        namespace="bucket/name",
    )
    assert root == tmp_path / "sources" / "s3" / "bucket_name"

    path = remote_object_path(
        working_dir=tmp_path,
        source_type="s3",
        namespace="bucket/name",
        key="../folder/./report.pdf",
    )
    assert path == root / "folder" / "report.pdf"

    with pytest.raises(ValueError, match="remote object key is empty"):
        remote_object_path(
            working_dir=tmp_path,
            source_type="s3",
            namespace="bucket/name",
            key="../",
        )
