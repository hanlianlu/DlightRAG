# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG's offline VDB rebuild wrapper."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import NoopTelemetry
from tests.config_helpers import mutate_config


def test_parser_defaults_to_check_only() -> None:
    from dlightrag.adapters.postgres.rebuild_vdb import build_parser

    args = build_parser().parse_args([])

    assert args.target == "check"
    assert args.yes is False
    assert args.restore_sidecar_alignment is True


def test_rebuild_targets_require_yes() -> None:
    from dlightrag.adapters.postgres.rebuild_vdb import build_parser, validate_args

    args = build_parser().parse_args(["--target", "chunks"])

    with pytest.raises(SystemExit):
        validate_args(args)


def test_pyproject_exposes_rebuild_console_script() -> None:
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["dlightrag-rebuild-vdb"] == (
        "dlightrag.adapters.postgres.rebuild_vdb:main"
    )


async def test_runner_uses_dlightrag_embedding_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    fake_embedding = object()
    fake_embedder = AsyncMock()
    config = _fake_config()
    calls: dict[str, Any] = {}

    async def fake_setup(self) -> bool:
        calls["workspace"] = self.workspace
        calls["storage_names"] = self.resolve_storage_names()
        calls["embedding_func"] = self.build_embedding_func()
        calls["global_config"] = self.build_global_config()
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = AsyncMock()
        return True

    def create_embedding_model(*_args: Any, **kwargs: Any) -> Any:
        calls["scheduler"] = kwargs["scheduler"]
        return fake_embedder

    monkeypatch.setattr(module, "create_embedding_model", create_embedding_model)
    monkeypatch.setattr(
        module,
        "build_lightrag_embedding",
        lambda _settings, _embedder: fake_embedding,
    )
    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(module.DlightRAGRebuildTool, "run_check", AsyncMock())

    exit_code = await module.run_rebuild(config=cast(Any, config), target="check", assume_yes=False)

    assert exit_code == 0
    assert calls["workspace"] == "research"
    assert calls["storage_names"] == {
        "graph": "PGTableGraphStorage",
        "vector": "PGVectorStorage",
        "kv": "PGKVStorage",
        "doc_status": "PGDocStatusStorage",
    }
    assert calls["embedding_func"] is fake_embedding
    assert isinstance(calls["scheduler"], ModelScheduler)
    assert calls["scheduler"].max_concurrency == config.models.max_concurrency
    assert calls["global_config"]["working_dir"] == str(Path("/tmp/dlightrag").resolve())
    assert calls["global_config"]["embedding_func"] is fake_embedding


async def test_chunks_rebuild_restores_sidecar_image_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    lightrag = SimpleNamespace(
        doc_status=AsyncMock(
            get_docs_by_status=AsyncMock(
                return_value={
                    "doc-1": SimpleNamespace(chunks_list=["chunk-a", "doc-1-mm-drawing-000"]),
                    "doc-2": SimpleNamespace(chunks_list=[]),
                }
            )
        )
    )
    stores = AsyncMock()

    async def _status_pages(*_args, **_kwargs):
        yield {"doc-1": object(), "doc-2": object()}

    stores.iter_doc_status_pages = _status_pages
    stores.get_full_doc_statuses.return_value = {
        "doc-1": SimpleNamespace(chunks_list=["chunk-a", "doc-1-mm-drawing-000"]),
        "doc-2": SimpleNamespace(chunks_list=[]),
    }
    stores.get_full_doc.side_effect = [
        {"sidecar_location": "file:///tmp/doc-1.parsed"},
        {"sidecar_location": "file:///tmp/doc-2.parsed"},
    ]
    embedder = object()
    settings = cast(
        Any,
        SimpleNamespace(
            embedding=SimpleNamespace(dim=8),
            parser_min_image_pixel=80,
            embedding_func_max_async=4,
            parser_rules="*:mineru-iteP",
        ),
    )
    document_embedder = MagicMock()
    build_document_embedder = MagicMock(return_value=document_embedder)
    monkeypatch.setattr(module, "build_document_embedder", build_document_embedder)
    calls: list[dict[str, object]] = []

    async def fake_overwrite(self, **kwargs) -> None:
        assert self._document_embedder is document_embedder
        calls.append(kwargs)

    monkeypatch.setattr(
        module.UnifiedIngestionEngine,
        "_overwrite_sidecar_image_vectors",
        fake_overwrite,
    )

    stats = await module.restore_sidecar_image_vectors(
        workspace_id=config.deployment.workspace,
        settings=settings,
        lightrag=lightrag,
        stores=stores,
        multimodal_embedder=embedder,
        telemetry=NoopTelemetry(),
    )

    assert stats == {"processed_docs": 1, "skipped_docs": 1}
    stores.get_full_doc_statuses.assert_awaited_once_with(["doc-1", "doc-2"])
    build_document_embedder.assert_called_once_with(settings, embedder, image_enabled=True)
    assert calls == [
        {
            "doc_id": "doc-1",
            "sidecar_location": "file:///tmp/doc-1.parsed",
            "chunk_ids": {"chunk-a", "doc-1-mm-drawing-000"},
        }
    ]


async def test_chunks_rebuild_relabels_bm25_languages_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    mutate_config(config, "corpus.retrieval.bm25_enabled", True)
    embedder = AsyncMock()
    monkeypatch.setattr(
        module,
        "create_embedding_model",
        lambda *_args, **_kwargs: embedder,
    )
    monkeypatch.setattr(module, "build_lightrag_embedding", lambda *_args: object())
    rebuild_mock = AsyncMock(return_value={"processed_chunks": 2, "updated_chunks": 2})
    monkeypatch.setattr(module, "run_rebuild_bm25", rebuild_mock)

    async def fake_setup(self) -> bool:
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = AsyncMock()
        return True

    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(
        module.DlightRAGRebuildTool,
        "run_rebuild_chunks",
        AsyncMock(return_value=[{"label": "chunks", "errors": []}]),
    )
    monkeypatch.setattr(module.DlightRAGRebuildTool, "report_rebuild", lambda self, stats: False)

    exit_code = await module.run_rebuild(
        config=cast(Any, config),
        target="chunks",
        assume_yes=True,
        restore_sidecar_alignment=False,
    )

    assert exit_code == 0
    rebuild_mock.assert_awaited_once()
    await_args = rebuild_mock.await_args
    assert await_args is not None
    assert await_args.args == ()
    assert await_args.kwargs == {
        "config": config,
        "assume_yes": True,
        "batch_size": module.DEFAULT_BATCH_SIZE,
    }


async def test_graph_rebuild_does_not_restore_sidecar_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    embedder = AsyncMock()
    monkeypatch.setattr(
        module,
        "create_embedding_model",
        lambda *_args, **_kwargs: embedder,
    )
    monkeypatch.setattr(module, "build_lightrag_embedding", lambda *_args: object())
    restore_mock = AsyncMock()
    rebuild_mock = AsyncMock()
    monkeypatch.setattr(module, "restore_sidecar_image_vectors", restore_mock)
    monkeypatch.setattr(module, "run_rebuild_bm25", rebuild_mock)

    async def fake_setup(self) -> bool:
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = AsyncMock()
        return True

    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(module.DlightRAGRebuildTool, "run_rebuild_entities_relations", AsyncMock())

    exit_code = await module.run_rebuild(config=cast(Any, config), target="graph", assume_yes=True)

    assert exit_code == 0
    restore_mock.assert_not_awaited()
    rebuild_mock.assert_not_awaited()


async def test_failed_chunks_rebuild_skips_sidecar_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    embedder = AsyncMock()
    monkeypatch.setattr(module, "create_embedding_model", lambda *_args, **_kwargs: embedder)
    monkeypatch.setattr(module, "build_lightrag_embedding", lambda *_args: object())
    resolve_mock = AsyncMock(return_value=True)
    restore_mock = AsyncMock()
    rebuild_mock = AsyncMock()
    monkeypatch.setattr(module, "resolve_direct_image_embedding_enabled", resolve_mock)
    monkeypatch.setattr(module, "restore_sidecar_image_vectors", restore_mock)
    monkeypatch.setattr(module, "run_rebuild_bm25", rebuild_mock)

    async def fake_setup(self) -> bool:
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = AsyncMock()
        return True

    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(
        module.DlightRAGRebuildTool,
        "run_rebuild_chunks",
        AsyncMock(return_value=[{"label": "chunks", "errors": [{"batch": 1}]}]),
    )
    monkeypatch.setattr(module.DlightRAGRebuildTool, "report_rebuild", lambda self, stats: True)

    exit_code = await module.run_rebuild(config=cast(Any, config), target="chunks", assume_yes=True)

    assert exit_code == 1
    resolve_mock.assert_awaited_once()
    restore_mock.assert_not_awaited()
    rebuild_mock.assert_not_awaited()


async def test_chunks_rebuild_delegates_bm25_before_embedder_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    mutate_config(config, "corpus.retrieval.bm25_enabled", True)
    mutate_config(config, "models.embedding.startup_probe", False)
    embedder = AsyncMock()
    embedder.aclose.side_effect = RuntimeError("embedder close failed")
    monkeypatch.setattr(module, "create_embedding_model", lambda *_args, **_kwargs: embedder)
    monkeypatch.setattr(module, "build_lightrag_embedding", lambda *_args: object())
    rebuild_bm25 = AsyncMock(return_value={"processed_chunks": 0, "updated_chunks": 0})
    monkeypatch.setattr(module, "run_rebuild_bm25", rebuild_bm25)
    monkeypatch.setattr(
        module,
        "resolve_direct_image_embedding_enabled",
        AsyncMock(return_value=False),
    )

    async def fake_setup(self) -> bool:
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = AsyncMock()
        return True

    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(
        module.DlightRAGRebuildTool,
        "run_rebuild_chunks",
        AsyncMock(return_value=[{"label": "chunks", "errors": []}]),
    )
    monkeypatch.setattr(module.DlightRAGRebuildTool, "report_rebuild", lambda self, stats: False)

    with pytest.raises(RuntimeError, match="embedder close failed"):
        await module.run_rebuild(
            config=cast(Any, config),
            target="chunks",
            assume_yes=True,
        )

    rebuild_bm25.assert_awaited_once_with(
        config=config,
        assume_yes=True,
        batch_size=module.DEFAULT_BATCH_SIZE,
    )


async def test_runner_addresses_the_workspace_by_its_canonical_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A display label addresses the id the service stored the workspace under."""
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    mutate_config(config, "deployment.workspace", "My Space")
    mutate_config(config, "corpus.retrieval.bm25_enabled", True)
    embedder = AsyncMock()
    monkeypatch.setattr(module, "create_embedding_model", lambda *_args, **_kwargs: embedder)
    monkeypatch.setattr(module, "build_lightrag_embedding", lambda *_args: object())
    rebuild_bm25 = AsyncMock(return_value={"processed_chunks": 0, "updated_chunks": 0})
    monkeypatch.setattr(module, "run_rebuild_bm25", rebuild_bm25)
    monkeypatch.setattr(
        module, "resolve_direct_image_embedding_enabled", AsyncMock(return_value=True)
    )
    restore = AsyncMock(return_value={"processed_docs": 0, "skipped_docs": 0})
    monkeypatch.setattr(module, "restore_sidecar_image_vectors", restore)
    setup_workspaces: list[str] = []

    async def fake_setup(self) -> bool:
        setup_workspaces.append(self.workspace)
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = AsyncMock()
        return True

    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(
        module.DlightRAGRebuildTool,
        "run_rebuild_chunks",
        AsyncMock(return_value=[{"label": "chunks", "errors": []}]),
    )
    monkeypatch.setattr(module.DlightRAGRebuildTool, "report_rebuild", lambda self, stats: False)

    exit_code = await module.run_rebuild(config=config, target="chunks", assume_yes=True)

    assert exit_code == 0
    assert setup_workspaces == ["my_space"]
    restore_args = restore.await_args
    assert restore_args is not None
    assert restore_args.kwargs["workspace_id"] == "my_space"
    bm25_args = rebuild_bm25.await_args
    assert bm25_args is not None
    assert bm25_args.kwargs["config"].deployment.workspace == "my_space"


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("chunks", ["probe", "chunks", "restore"]),
        ("all", ["probe", "graph", "chunks", "restore"]),
    ],
)
async def test_rebuild_settles_the_embedding_mode_before_rewriting_any_vector(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    expected: list[str],
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    events: list[str] = []
    embedder = _stub_rebuild(monkeypatch, module, events)

    exit_code = await module.run_rebuild(
        config=cast(Any, _fake_config()), target=cast(Any, target), assume_yes=True
    )

    assert exit_code == 0
    assert events == expected
    embedder.aclose.assert_awaited_once()


@pytest.mark.parametrize("target", ["chunks", "all"])
async def test_transient_embedding_probe_failure_rebuilds_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    config = _fake_config()
    mutate_config(config, "models.embedding.startup_probe", True)
    events: list[str] = []
    embedder = _stub_rebuild(
        monkeypatch,
        module,
        events,
        probe=module.resolve_direct_image_embedding_enabled,
    )
    embedder.supports_images = True
    embedder.probe_image_embedding.side_effect = ConnectionError("provider refused")
    storages: list[Any] = []
    setup = module.DlightRAGRebuildTool.setup_storages

    async def recording_setup(self) -> bool:
        ready = await setup(self)
        storages.extend(self.all_storages())
        return ready

    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", recording_setup)

    exit_code = await module.run_rebuild(
        config=cast(Any, config), target=cast(Any, target), assume_yes=True
    )

    assert exit_code == 1
    assert events == []
    embedder.probe_image_embedding.assert_awaited_once()
    error = capsys.readouterr().err
    assert "Nothing was rebuilt" in error
    assert "Run the same command again" in error
    assert "ConnectionError" not in error
    assert "provider refused" not in error
    assert storages and all(storage.finalize.await_count == 1 for storage in storages)
    embedder.aclose.assert_awaited_once()


async def test_definitive_embedding_probe_failure_rebuilds_nothing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    events: list[str] = []
    _stub_rebuild(
        monkeypatch,
        module,
        events,
        probe=AsyncMock(
            side_effect=ValueError(
                "embedding.input_modality='multimodal' requires working image-query and "
                "fused-document embeddings, but the startup probe failed"
            )
        ),
    )

    exit_code = await module.run_rebuild(
        config=cast(Any, _fake_config()), target="chunks", assume_yes=True
    )

    assert exit_code == 1
    assert events == []
    assert capsys.readouterr().err.startswith(
        "Nothing was rebuilt: embedding.input_modality='multimodal' requires"
    )


async def test_non_transient_probe_error_is_not_reported_as_an_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    events: list[str] = []
    _stub_rebuild(
        monkeypatch,
        module,
        events,
        probe=AsyncMock(side_effect=RuntimeError("embedder bug")),
    )

    with pytest.raises(RuntimeError, match="embedder bug"):
        await module.run_rebuild(config=cast(Any, _fake_config()), target="chunks", assume_yes=True)

    assert events == []


async def test_restoration_names_a_drifted_doc_status_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.adapters.postgres import rebuild_vdb as module

    events: list[str] = []
    _stub_rebuild(
        monkeypatch,
        module,
        events,
        doc_status=SimpleNamespace(get_full_docs_by_ids=AsyncMock(), finalize=AsyncMock()),
    )

    with pytest.raises(RuntimeError, match="'get_docs_by_statuses_page'"):
        await module.run_rebuild(config=cast(Any, _fake_config()), target="chunks", assume_yes=True)

    assert events == ["probe"]


def _stub_rebuild(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    events: list[str],
    *,
    doc_status: Any | None = None,
    probe: Any | None = None,
) -> AsyncMock:
    """Stub every rebuild step and record the order they run in."""
    embedder = AsyncMock()
    monkeypatch.setattr(module, "create_embedding_model", lambda *_args, **_kwargs: embedder)
    monkeypatch.setattr(module, "build_lightrag_embedding", lambda *_args: object())

    async def recording_probe(*_args: Any, **_kwargs: Any) -> bool:
        events.append("probe")
        return True

    async def restore(**_kwargs: Any) -> dict[str, int]:
        events.append("restore")
        return {"processed_docs": 1, "skipped_docs": 0}

    async def rebuild(label: str) -> list[dict[str, Any]]:
        events.append(label)
        return [{"label": label, "errors": []}]

    async def fake_setup(self) -> bool:
        self.graph = AsyncMock()
        self.entities_vdb = AsyncMock()
        self.relationships_vdb = AsyncMock()
        self.chunks_vdb = AsyncMock()
        self.text_chunks = AsyncMock()
        self.full_docs = AsyncMock()
        self.doc_status = doc_status if doc_status is not None else AsyncMock()
        return True

    monkeypatch.setattr(module, "resolve_direct_image_embedding_enabled", probe or recording_probe)
    monkeypatch.setattr(module, "restore_sidecar_image_vectors", restore)
    monkeypatch.setattr(module.DlightRAGRebuildTool, "setup_storages", fake_setup)
    monkeypatch.setattr(
        module.DlightRAGRebuildTool,
        "run_rebuild_entities_relations",
        lambda self: rebuild("graph"),
    )
    monkeypatch.setattr(
        module.DlightRAGRebuildTool, "run_rebuild_chunks", lambda self: rebuild("chunks")
    )
    monkeypatch.setattr(module.DlightRAGRebuildTool, "report_rebuild", lambda self, stats: False)
    return embedder


def _fake_config():
    from dlightrag.application.config import DlightragConfig

    return DlightragConfig.model_validate(
        {
            "deployment": {"workspace": "research", "working_dir": "/tmp/dlightrag"},
            "storage": {
                "lightrag": {
                    "vector_storage": "PGVectorStorage",
                    "graph_storage": "PGTableGraphStorage",
                    "kv_storage": "PGKVStorage",
                    "doc_status_storage": "PGDocStatusStorage",
                    "vector_db_kwargs": {},
                }
            },
            "models": {
                "max_concurrency": 3,
                "embedding": {
                    "provider": "openai_compatible",
                    "model": "text-embedding-3-small",
                    "api_key": "key",
                    "base_url": "https://api.example/v1",
                    "dim": 1536,
                    "startup_probe": False,
                    "batch_size": 7,
                },
            },
            "corpus": {
                "retrieval": {
                    "metadata_filter_exact_vector_threshold": 8192,
                    "bm25_enabled": False,
                },
                "sidecars": {"vlm": {"min_image_pixel": 64}},
            },
        }
    )
