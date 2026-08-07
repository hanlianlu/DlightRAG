# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for DlightRAG's offline VDB rebuild wrapper."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_parser_defaults_to_check_only() -> None:
    from dlightrag.tools.rebuild_vdb import build_parser

    args = build_parser().parse_args([])

    assert args.target == "check"
    assert args.yes is False
    assert args.restore_sidecar_alignment is True


def test_rebuild_targets_require_yes() -> None:
    from dlightrag.tools.rebuild_vdb import build_parser, validate_args

    args = build_parser().parse_args(["--target", "chunks"])

    with pytest.raises(SystemExit):
        validate_args(args)


def test_pyproject_exposes_rebuild_console_script() -> None:
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["dlightrag-rebuild-vdb"] == (
        "dlightrag.tools.rebuild_vdb:main"
    )


def test_rebuild_vdb_exports_are_resolvable() -> None:
    from dlightrag.tools import rebuild_vdb

    assert all(hasattr(rebuild_vdb, name) for name in rebuild_vdb.__all__)


async def test_runner_uses_dlightrag_embedding_and_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from dlightrag.tools import rebuild_vdb as module

    fake_embedding = object()
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

    monkeypatch.setattr(module, "get_embedding_func", lambda cfg, *, embedder=None: fake_embedding)
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
    assert calls["global_config"]["working_dir"] == "/tmp/dlightrag"
    assert calls["global_config"]["embedding_func"] is fake_embedding


async def test_chunks_rebuild_restores_sidecar_image_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.tools import rebuild_vdb as module

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
    stores.get_full_doc.side_effect = [
        {"sidecar_location": "file:///tmp/doc-1.parsed"},
        {"sidecar_location": "file:///tmp/doc-2.parsed"},
    ]
    embedder = object()
    document_embedder = MagicMock()
    build_document_embedder = MagicMock(return_value=document_embedder)
    monkeypatch.setattr(module.RAGService, "_build_document_embedder", build_document_embedder)
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
        config=cast(Any, config),
        lightrag=lightrag,
        stores=stores,
        multimodal_embedder=embedder,
    )

    assert stats == {"processed_docs": 1, "skipped_docs": 1}
    build_document_embedder.assert_called_once_with(config, embedder, image_enabled=True)
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
    from dlightrag.tools import rebuild_vdb as module

    config = _fake_config()
    config.bm25_enabled = True
    monkeypatch.setattr(module, "get_embedding_func", lambda cfg, *, embedder=None: object())
    rebuild_mock = AsyncMock(return_value={"processed_chunks": 2, "updated_chunks": 2})
    monkeypatch.setattr(
        module,
        "rebuild_postgres_bm25",
        rebuild_mock,
    )
    fake_pool = SimpleNamespace(bind=MagicMock(), close=AsyncMock())
    monkeypatch.setattr(module, "pg_pool", fake_pool)

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
    assert await_args.args == (config,)
    assert await_args.kwargs == {
        "pool": fake_pool,
        "batch_size": module.DEFAULT_BATCH_SIZE,
    }
    fake_pool.bind.assert_called_once_with(config)
    fake_pool.close.assert_awaited_once()


async def test_graph_rebuild_does_not_restore_sidecar_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.tools import rebuild_vdb as module

    config = _fake_config()
    monkeypatch.setattr(module, "get_embedding_func", lambda cfg, *, embedder=None: object())
    monkeypatch.setattr(module, "get_multimodal_embedder", lambda cfg: object())
    restore_mock = AsyncMock()
    rebuild_mock = AsyncMock()
    monkeypatch.setattr(module, "restore_sidecar_image_vectors", restore_mock)
    monkeypatch.setattr(module, "rebuild_postgres_bm25", rebuild_mock)

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
    from dlightrag.tools import rebuild_vdb as module

    config = _fake_config()
    embedder = AsyncMock()
    monkeypatch.setattr(module, "get_multimodal_embedder", lambda cfg: embedder)
    monkeypatch.setattr(module, "get_embedding_func", lambda cfg, *, embedder=None: object())
    resolve_mock = AsyncMock()
    restore_mock = AsyncMock()
    rebuild_mock = AsyncMock()
    monkeypatch.setattr(module.RAGService, "_resolve_direct_image_embedding_enabled", resolve_mock)
    monkeypatch.setattr(module, "restore_sidecar_image_vectors", restore_mock)
    monkeypatch.setattr(module, "rebuild_postgres_bm25", rebuild_mock)

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
    resolve_mock.assert_not_awaited()
    restore_mock.assert_not_awaited()
    rebuild_mock.assert_not_awaited()


async def test_chunks_rebuild_closes_domain_pool_when_embedder_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.tools import rebuild_vdb as module

    config = _fake_config()
    config.bm25_enabled = True
    config.embedding.startup_probe = False
    embedder = AsyncMock()
    embedder.aclose.side_effect = RuntimeError("embedder close failed")
    fake_pool = SimpleNamespace(bind=MagicMock(), close=AsyncMock())
    monkeypatch.setattr(module, "get_multimodal_embedder", lambda cfg: embedder)
    monkeypatch.setattr(module, "get_embedding_func", lambda cfg, *, embedder=None: object())
    monkeypatch.setattr(module, "pg_pool", fake_pool)
    monkeypatch.setattr(
        module,
        "rebuild_postgres_bm25",
        AsyncMock(return_value={"processed_chunks": 0, "updated_chunks": 0}),
    )
    monkeypatch.setattr(
        module.RAGService,
        "_resolve_direct_image_embedding_enabled",
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

    fake_pool.close.assert_awaited_once()


def _fake_config() -> SimpleNamespace:
    return SimpleNamespace(
        workspace="research",
        working_dir="/tmp/dlightrag",
        vector_storage="PGVectorStorage",
        graph_storage="PGTableGraphStorage",
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        embedding_batch_num=7,
        vector_db_kwargs={},
        metadata_filter_exact_vector_threshold=8192,
        parser_rules="*:mineru-iteP",
        parser_sidecars=SimpleNamespace(
            vlm=SimpleNamespace(min_image_pixel=64),
        ),
        embedding=SimpleNamespace(
            model="text-embedding-3-small",
            input_modality="auto",
        ),
        bm25_enabled=False,
        bm25_profiles=[],
        apply_lightrag_backend_env=lambda force=False: None,
        apply_lightrag_runtime_env=lambda force=False: None,
        apply_lightrag_sidecar_env=lambda force=False: None,
    )
