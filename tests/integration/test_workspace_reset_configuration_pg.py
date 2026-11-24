# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A workspace reset or delete leaves LightRAG's shared configuration container usable.

LightRAG keeps every workspace's embedding baselines, and the container's own identity,
in one table shared by the whole deployment. Dropping that storage with the workspace's others
erased them all and made every later start refuse. These run the real reset against real
LightRAG storages on a scratch database.
"""

import hashlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import pytest

from tests.support.pg import (
    PG_CONN_KWARGS,
    drop_database,
    drop_scratch_database,
    skip_without_postgres,
)

pytestmark = [
    pytest.mark.integration,
    # The storages create asyncpg pools bound to the loop they were born on.
    pytest.mark.asyncio(loop_scope="module"),
]

_DIM = 8
_EXTENSIONS = ("vector", "pg_textsearch", "pg_trgm")
_IDENTITY_KEY = "$meta/storage_identity"


def _kwargs(database: str) -> dict[str, Any]:
    return {**PG_CONN_KWARGS, "database": database}


def _embedding_func() -> Any:
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], *, context: str = "document") -> Any:
        import numpy as np

        values = []
        for text in texts:
            digest = hashlib.sha256(f"{context}:{text}".encode()).digest()
            values.append([((digest[i] / 255.0) * 2.0) - 1.0 for i in range(_DIM)])
        return np.array(values, dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=_DIM,
        max_token_size=512,
        func=embed,
        model_name="reset-cfg-fake",
        supports_asymmetric=True,
    )


@dataclass
class _Deployment:
    database: str
    working_dir: Path
    maintenance: Any
    opened: list[Any]

    def config(self, workspace: str) -> Any:
        from dlightrag.application.config.sections import DlightragConfig
        from dlightrag.engine.ai.settings import (
            EmbeddingSettings,
            ModelRoleSettings,
            ModelSettings,
        )

        return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
            deployment={"workspace": workspace, "working_dir": str(self.working_dir)},
            storage={
                "postgres": {
                    "host": str(PG_CONN_KWARGS["host"]),
                    "port": int(PG_CONN_KWARGS["port"]),
                    "user": str(PG_CONN_KWARGS["user"]),
                    "password": str(PG_CONN_KWARGS["password"]),
                    "database": self.database,
                    "pool_min_size": 1,
                    "pool_max_size": 3,
                }
            },
            models={
                "max_concurrency": 1,
                "chat": ModelRoleSettings(
                    default=ModelSettings(
                        provider="openai",
                        model="reset-cfg-fake-llm",
                        api_key="reset-cfg-fake-key",
                        timeout=30,
                    )
                ),
                "embedding": EmbeddingSettings(
                    provider="voyage",
                    model="reset-cfg-fake",
                    api_key="reset-cfg-fake-key",
                    dim=_DIM,
                    max_token_size=512,
                    max_concurrency=1,
                    batch_size=2,
                    startup_probe=False,
                ),
                "rerank": {"enabled": False},
            },
            corpus={
                "ingestion": {"chunk_token_size": 128, "pipeline": {"max_concurrency": 1}},
                "retrieval": {"bm25_enabled": False},
            },
        )

    async def start(self, workspace: str) -> Any:
        """A LightRAG for ``workspace`` with its storages initialized, as a start does."""
        from dlightrag.adapters.postgres.corpus.corpus import PGCorpusRuntimeBinder
        from dlightrag.application.settings.projections import rag_settings
        from dlightrag.engine.rag.workspace.ports import CorpusRuntimeModels

        config = self.config(workspace)
        lightrag = PGCorpusRuntimeBinder(config).create(
            models=CorpusRuntimeModels(
                default_llm_func=lambda _prompt, **_kwargs: "{}",
                embedding_func=_embedding_func(),
                role_llm_configs=None,
            ),
            settings=rag_settings(config),
        )
        self.opened.append(lightrag)
        await lightrag.initialize_storages()
        return lightrag

    async def configuration_ids(self) -> set[str]:
        conn = await asyncpg.connect(**_kwargs(self.database))
        try:
            return {row["id"] for row in await conn.fetch("SELECT id FROM lightrag_config")}
        finally:
            await conn.close()

    async def reset(self, workspace: str, lightrag: Any, tmp_path: Path) -> dict[str, Any]:
        from dlightrag.engine.rag.corpus.reset import areset

        return await areset(
            workspace_id=workspace,
            input_root=tmp_path / "corpus",
            lightrag=lightrag,
            metadata_index=None,
            maintenance=self.maintenance,
        )


def _baselines(workspace: str) -> set[str]:
    return {f"{workspace}/embedding/{target}" for target in ("entities", "relationships", "chunks")}


@pytest.fixture(scope="module")
async def deployment(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[_Deployment]:
    """One scratch database and one working directory shared by every workspace of the module."""
    await skip_without_postgres()
    from dlightrag.adapters.postgres.core._pool import pg_pool
    from dlightrag.adapters.postgres.corpus.corpus import build_pg_corpus_backend
    from dlightrag.application.config.loading import set_config

    database = f"dlightrag_reset_cfg_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_kwargs("postgres"))
    try:
        await drop_scratch_database(admin, database)
        await admin.execute(f'CREATE DATABASE "{database}"')
    finally:
        await admin.close()
    db = await asyncpg.connect(**_kwargs(database))
    try:
        for extension in _EXTENSIONS:
            await db.execute(f"CREATE EXTENSION IF NOT EXISTS {extension}")
    finally:
        await db.close()

    state = _Deployment(
        database=database,
        working_dir=tmp_path_factory.mktemp("lightrag_working_dir"),
        maintenance=None,
        opened=[],
    )
    config = state.config("reset_cfg_default")
    set_config(config)
    pg_pool.bind(config)
    try:
        backend = build_pg_corpus_backend(config)
        await backend.maintenance.initialize(validate_only=False)
        state.maintenance = backend.maintenance
        yield state
    finally:
        for lightrag in state.opened:
            try:
                await lightrag.finalize_storages()
            except Exception:  # noqa: BLE001 - teardown must still drop the database
                pass
        await pg_pool.close()
        await drop_database(database)


async def test_a_reset_workspace_starts_again_over_the_same_working_dir(
    deployment: _Deployment, tmp_path: Path
) -> None:
    lightrag = await deployment.start("reset_restart")
    assert _baselines("reset_restart") <= await deployment.configuration_ids()

    stats = await deployment.reset("reset_restart", lightrag, tmp_path)
    assert stats["errors"] == []
    await lightrag.finalize_storages()

    # The container's identity survived, so this start is not refused.
    restarted = await deployment.start("reset_restart")
    assert _IDENTITY_KEY in await deployment.configuration_ids()
    assert restarted.workspace == "reset_restart"


async def test_reset_and_delete_keep_every_other_workspaces_configuration(
    deployment: _Deployment, tmp_path: Path
) -> None:
    resets = await deployment.start("reset_gone")
    deletes = await deployment.start("reset_deleted")
    await deployment.start("reset_kept")
    kept = _baselines("reset_kept")
    assert kept | _baselines("reset_gone") | _baselines("reset_deleted") <= (
        await deployment.configuration_ids()
    )

    # A reset: the workspace's own records go, the others and the identity stay.
    assert (await deployment.reset("reset_gone", resets, tmp_path))["errors"] == []
    after_reset = await deployment.configuration_ids()
    assert not _baselines("reset_gone") & after_reset
    assert kept | _baselines("reset_deleted") | {_IDENTITY_KEY} <= after_reset

    # A delete resets the same way, and the corpus rows it then clears include the records.
    assert (await deployment.reset("reset_deleted", deletes, tmp_path))["errors"] == []
    after_delete = await deployment.configuration_ids()
    assert not _baselines("reset_deleted") & after_delete
    assert kept | {_IDENTITY_KEY} <= after_delete

    # The row sweep alone, without the workspace's LightRAG, reaches them too.
    await deployment.maintenance.clean_orphan_rows("reset_kept")
    after_sweep = await deployment.configuration_ids()
    assert not kept & after_sweep
    assert _IDENTITY_KEY in after_sweep
