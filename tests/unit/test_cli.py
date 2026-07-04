# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for CLI argument validation."""

import importlib.util
from pathlib import Path
from typing import Any

import pytest

# Load scripts/cli.py as a module (it's a script, not a package)
_cli_path = Path(__file__).resolve().parents[2] / "scripts" / "cli.py"
_spec = importlib.util.spec_from_file_location("cli", _cli_path)
assert _spec is not None and _spec.loader is not None
_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cli)

build_parser = _cli.build_parser
_build_answer_payload = _cli._build_answer_payload
_build_ingest_kwargs = _cli._build_ingest_kwargs
_build_retrieve_payload = _cli._build_retrieve_payload
_validate_ingest_args = _cli._validate_ingest_args
cmd_ragas_eval = _cli.cmd_ragas_eval


def _image_block(url: str) -> dict:
    return {"type": "image_url", "image_url": {"url": url}}


def _parse_ingest(args: list[str]):
    """Parse CLI args for the ingest subcommand."""
    return build_parser().parse_args(["ingest", *args])


def _parse_query(args: list[str]):
    """Parse CLI args for the query subcommand."""
    return build_parser().parse_args(["query", *args])


def _parse_answer(args: list[str]):
    """Parse CLI args for the answer subcommand."""
    return build_parser().parse_args(["answer", *args])


def _parse_chat(args: list[str]):
    """Parse CLI args for the chat subcommand."""
    return build_parser().parse_args(["chat", *args])


# ---------------------------------------------------------------------------
# Test current REST payload options
# ---------------------------------------------------------------------------


def test_query_payload_supports_current_retrieval_options() -> None:
    args = _parse_query(
        [
            "find diagrams",
            "--top-k",
            "8",
            "--chunk-top-k",
            "5",
            "--workspaces",
            "finance",
            "legal",
            "--filter-doc-title",
            "Manual",
            "--filter-custom-json",
            '{"department":"finance"}',
            "--query-image",
            "data:image/png;base64,abc",
            "--session-id",
            "session-1",
            "--referenced-image-id",
            "img_1",
        ]
    )

    assert _build_retrieve_payload(args) == {
        "query": "find diagrams",
        "top_k": 8,
        "chunk_top_k": 5,
        "workspaces": ["finance", "legal"],
        "filters": {
            "doc_title": "Manual",
            "custom": {"department": "finance"},
        },
        "query_images": [_image_block("data:image/png;base64,abc")],
        "session_id": "session-1",
        "referenced_image_ids": ["img_1"],
    }


def test_answer_payload_supports_current_answer_options() -> None:
    args = _parse_answer(
        [
            "summarize",
            "--chunk-top-k",
            "9",
            "--answer-context-top-k",
            "4",
            "--filter-doc-author",
            "Ada",
            "--query-image",
            "https://example.test/chart.png",
        ]
    )

    assert _build_answer_payload(args, query=args.query) == {
        "query": "summarize",
        "stream": False,
        "chunk_top_k": 9,
        "answer_context_top_k": 4,
        "filters": {"doc_author": "Ada"},
        "query_images": [_image_block("https://example.test/chart.png")],
    }


def test_chat_payload_preserves_history_and_current_answer_options() -> None:
    args = _parse_chat(
        [
            "--answer-context-top-k",
            "3",
            "--session-id",
            "session-1",
            "--referenced-image-id",
            "img_1",
        ]
    )
    history = [{"role": "user", "content": "Previous"}]

    assert _build_answer_payload(args, query="Follow up", conversation_history=history) == {
        "query": "Follow up",
        "stream": False,
        "answer_context_top_k": 3,
        "conversation_history": history,
        "session_id": "session-1",
        "referenced_image_ids": ["img_1"],
    }


def test_ingest_kwargs_support_document_metadata_options() -> None:
    args = _parse_ingest(
        [
            "./docs/report.pdf",
            "--title",
            "Quarterly Report",
            "--author",
            "Ada",
            "--metadata-json",
            '{"department":"finance"}',
            "--metadata-policy",
            "reject_unknown",
        ]
    )

    assert _build_ingest_kwargs(args) == {
        "path": "./docs/report.pdf",
        "replace": False,
        "title": "Quarterly Report",
        "author": "Ada",
        "metadata": {"department": "finance"},
        "metadata_policy": "reject_unknown",
    }


def test_ingest_kwargs_support_s3_region_and_retention() -> None:
    args = _parse_ingest(
        [
            "--source",
            "s3",
            "--bucket",
            "bucket",
            "--key",
            "docs/report.pdf",
            "--region",
            "eu-north-1",
            "--retain-source-file",
        ]
    )

    assert _build_ingest_kwargs(args) == {
        "bucket": "bucket",
        "key": "docs/report.pdf",
        "region": "eu-north-1",
        "retain_source_file": True,
        "replace": False,
    }


def test_ingest_kwargs_support_url_source() -> None:
    args = _parse_ingest(
        [
            "--source",
            "url",
            "--url",
            "https://cdn.example.com/download?id=asset-1",
            "--filename",
            "asset.pdf",
            "--source-uri",
            "bynder://asset/asset-1",
        ]
    )

    assert _build_ingest_kwargs(args) == {
        "url": "https://cdn.example.com/download?id=asset-1",
        "filename": "asset.pdf",
        "source_uri": "bynder://asset/asset-1",
        "replace": False,
    }


def test_json_object_arg_rejects_non_object_json() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["query", "q", "--filter-custom-json", '["not", "object"]'])


def test_ragas_eval_cli_requires_dataset() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ragas_eval"])


def test_ragas_eval_cli_forwards_api_key_without_printing_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import subprocess

    class Result:
        returncode = 7

    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], *, check: bool) -> Result:
        captured["cmd"] = cmd
        captured["check"] = check
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = build_parser().parse_args(
        [
            "ragas_eval",
            "--api",
            "http://localhost:8100",
            "--dataset",
            "tests.json",
            "--api-key",
            "secret-token",
            "--output-dir",
            "out",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        cmd_ragas_eval(args)

    assert exc.value.code == 7
    assert captured["check"] is False
    assert captured["cmd"][-6:] == [
        "--dataset",
        "tests.json",
        "--api-key",
        "secret-token",
        "--output-dir",
        "out",
    ]
    assert "secret-token" not in capsys.readouterr().out


def test_ragas_eval_cli_leaves_api_auto_resolution_to_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    class Result:
        returncode = 0

    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], *, check: bool) -> Result:
        captured["cmd"] = cmd
        return Result()

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = build_parser().parse_args(["ragas_eval", "--dataset", "tests.json"])

    with pytest.raises(SystemExit) as exc:
        cmd_ragas_eval(args)

    assert exc.value.code == 0
    assert "--api" not in captured["cmd"]


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — local source
# ---------------------------------------------------------------------------


class TestValidateLocal:
    """Validation for local source (default)."""

    def test_valid_local(self) -> None:
        args = _parse_ingest(["./docs"])
        _validate_ingest_args(args)  # should not raise

    def test_valid_local_with_flags(self) -> None:
        args = _parse_ingest(["./docs", "--replace"])
        _validate_ingest_args(args)

    def test_local_requires_path(self) -> None:
        args = _parse_ingest([])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_local_rejects_container(self) -> None:
        args = _parse_ingest(["./docs", "--container", "c"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_local_rejects_bucket(self) -> None:
        args = _parse_ingest(["./docs", "--bucket", "b"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — azure_blob source
# ---------------------------------------------------------------------------


class TestValidateAzureBlob:
    """Validation for azure_blob source."""

    def test_valid_azure_container_only(self) -> None:
        args = _parse_ingest(["--source", "azure_blob", "--container", "c"])
        _validate_ingest_args(args)

    def test_valid_azure_with_prefix(self) -> None:
        args = _parse_ingest(["--source", "azure_blob", "--container", "c", "--prefix", "docs/"])
        _validate_ingest_args(args)

    def test_valid_azure_with_blob_path(self) -> None:
        args = _parse_ingest(["--source", "azure_blob", "--container", "c", "--blob-path", "f.pdf"])
        _validate_ingest_args(args)

    def test_azure_requires_container(self) -> None:
        args = _parse_ingest(["--source", "azure_blob"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_azure_rejects_positional_path(self) -> None:
        args = _parse_ingest(["./docs", "--source", "azure_blob", "--container", "c"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_azure_blob_path_and_prefix_mutually_exclusive(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "azure_blob",
                "--container",
                "c",
                "--blob-path",
                "f.pdf",
                "--prefix",
                "docs/",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_azure_rejects_bucket(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "azure_blob",
                "--container",
                "c",
                "--bucket",
                "b",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)


# ---------------------------------------------------------------------------
# TestValidateIngestArgs — s3 source
# ---------------------------------------------------------------------------


class TestValidateS3:
    """Validation for s3 source."""

    def test_valid_s3_with_key(self) -> None:
        args = _parse_ingest(["--source", "s3", "--bucket", "my-bucket", "--key", "doc.pdf"])
        _validate_ingest_args(args)

    def test_valid_s3_with_prefix(self) -> None:
        args = _parse_ingest(["--source", "s3", "--bucket", "my-bucket", "--prefix", "docs/"])
        _validate_ingest_args(args)

    def test_s3_requires_bucket(self) -> None:
        args = _parse_ingest(["--source", "s3"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_s3_rejects_positional_path(self) -> None:
        args = _parse_ingest(["./docs", "--source", "s3", "--bucket", "b"])
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_s3_key_and_prefix_mutually_exclusive(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "s3",
                "--bucket",
                "b",
                "--key",
                "doc.pdf",
                "--prefix",
                "docs/",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)

    def test_s3_rejects_container(self) -> None:
        args = _parse_ingest(
            [
                "--source",
                "s3",
                "--bucket",
                "b",
                "--key",
                "doc.pdf",
                "--container",
                "c",
            ]
        )
        with pytest.raises(SystemExit):
            _validate_ingest_args(args)
