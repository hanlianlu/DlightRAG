# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Helpers for the opt-in PostgreSQL 18 + LightRAG smoke tests."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    MetadataConfig,
    ModelConfig,
)

RUN_E2E_ENV = "DLIGHTRAG_RUN_E2E_PG18"
REQUIRED_EXTENSIONS = ("vector", "age", "pg_textsearch")
REQUIRED_PRELOAD_LIBRARIES = ("age", "pg_textsearch")


def e2e_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether PG18 E2E tests were explicitly enabled."""
    value = (env or os.environ).get(RUN_E2E_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def pg_conn_kwargs_from_env(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build asyncpg kwargs, preferring E2E-specific env over app env."""
    source = env or os.environ

    def get(name: str, default: str) -> str:
        return (
            source.get(f"DLIGHTRAG_E2E_POSTGRES_{name}")
            or source.get(f"DLIGHTRAG_POSTGRES_{name}")
            or default
        )

    return {
        "host": get("HOST", "localhost"),
        "port": int(get("PORT", "5432")),
        "user": get("USER", "dlightrag"),
        "password": get("PASSWORD", "dlightrag"),
        "database": get("DATABASE", "dlightrag"),
    }


def missing_preload_libraries(setting: str) -> list[str]:
    """Return required libraries missing from shared_preload_libraries."""
    loaded = {item.strip().strip('"').strip("'") for item in setting.split(",") if item.strip()}
    return [name for name in REQUIRED_PRELOAD_LIBRARIES if name not in loaded]


@dataclass(frozen=True)
class PgPrereqReport:
    server_version: str
    server_major: int
    installed_extensions: tuple[str, ...]
    shared_preload_libraries: str

    @property
    def missing_extensions(self) -> list[str]:
        installed = set(self.installed_extensions)
        return [name for name in REQUIRED_EXTENSIONS if name not in installed]

    @property
    def missing_preload_libraries(self) -> list[str]:
        return missing_preload_libraries(self.shared_preload_libraries)


async def fetch_pg_prereq_report(conn: Any) -> PgPrereqReport:
    """Read PG major version, installed extensions, and preload settings."""
    version_num = int(await conn.fetchval("SHOW server_version_num"))
    version = str(await conn.fetchval("SHOW server_version"))
    preload = str(await conn.fetchval("SHOW shared_preload_libraries"))
    rows = await conn.fetch(
        "SELECT extname FROM pg_extension WHERE extname = ANY($1::text[]) ORDER BY extname",
        list(REQUIRED_EXTENSIONS),
    )
    return PgPrereqReport(
        server_version=version,
        server_major=version_num // 10000,
        installed_extensions=tuple(row["extname"] for row in rows),
        shared_preload_libraries=preload,
    )


def make_workspace_name(prefix: str = "e2e_pg18") -> str:
    """Build a PostgreSQL-safe workspace identifier."""
    token = hashlib.sha1(os.urandom(16)).hexdigest()[:10]
    return f"{prefix}_{token}"


def make_e2e_config(
    *,
    working_dir: Path,
    workspace: str,
    conn_kwargs: Mapping[str, Any],
    runtime_role: str = "ingest",
) -> DlightragConfig:
    """Create a compact config for the local fake-model E2E smoke."""
    return DlightragConfig(  # type: ignore[call-arg]
        _env_file=None,
        runtime_role=runtime_role,
        postgres_host=str(conn_kwargs["host"]),
        postgres_port=int(conn_kwargs["port"]),
        postgres_user=str(conn_kwargs["user"]),
        postgres_password=str(conn_kwargs["password"]),
        postgres_database=str(conn_kwargs["database"]),
        postgres_replica_host=str(conn_kwargs["host"]),
        postgres_replica_port=int(conn_kwargs["port"]),
        postgres_replica_user=str(conn_kwargs["user"]),
        postgres_replica_password=str(conn_kwargs["password"]),
        postgres_replica_database=str(conn_kwargs["database"]),
        postgres_pool_min_size=1,
        postgres_pool_max_size=2,
        workspace=workspace,
        working_dir=str(working_dir),
        chunk_size=128,
        chunk_overlap=12,
        max_async=1,
        embedding_func_max_async=1,
        embedding_batch_num=2,
        bm25_enabled=True,
        bm25_top_k=5,
        rerank={"enabled": False},
        llm=LLMConfig(
            default=ModelConfig(
                provider="openai",
                model="e2e-fake-llm",
                api_key="e2e-fake-key",
                timeout=30,
            )
        ),
        embedding=EmbeddingConfig(
            provider="voyage",
            model="e2e-fake-multimodal",
            api_key="e2e-fake-key",
            dim=8,
            max_token_size=1024,
            asymmetric="auto",
            startup_probe=False,
        ),
        metadata=MetadataConfig(
            allow_ad_hoc_json=True,
            fields={
                "e2e_case": {
                    "type": "string",
                    "normalizer": "casefold_trim",
                    "filter_ops": ["exact"],
                    "indexed": True,
                }
            },
        ),
    )


def stable_vector(seed: str | bytes, *, dim: int = 8) -> list[float]:
    """Return a deterministic non-zero embedding vector."""
    raw = seed if isinstance(seed, bytes) else seed.encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    values = [((digest[i] / 255.0) * 2.0) - 1.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


def image_seed(image: Image.Image) -> bytes:
    """Build a deterministic seed for a PIL image."""
    normalized = image.convert("RGB")
    return b"|".join(
        [
            str(normalized.size).encode("ascii"),
            normalized.mode.encode("ascii"),
            normalized.tobytes(),
        ]
    )


class FakeMultimodalEmbedder:
    """Small deterministic multimodal embedder for local E2E runs."""

    supports_asymmetric = True

    def __init__(self, *, dim: int = 8) -> None:
        self.dim = dim
        self.model = "e2e-fake-multimodal"

    async def aclose(self) -> None:
        return None

    async def probe_image_embedding(self) -> None:
        return None

    async def embed_texts(
        self, texts: list[str], *, context: str = "document"
    ) -> list[list[float]]:
        return [stable_vector(f"{context}:{text}", dim=self.dim) for text in texts]

    async def embed_index_images(self, images: list[Image.Image]) -> list[list[float]]:
        return [stable_vector(image_seed(image), dim=self.dim) for image in images]

    async def embed_query_images(self, images: list[Image.Image]) -> list[list[float]]:
        return await self.embed_index_images(images)


def fake_embedding_func(*, dim: int = 8) -> Any:
    """Build a LightRAG EmbeddingFunc backed by deterministic local vectors."""
    from lightrag.utils import EmbeddingFunc

    async def embed(texts: list[str], *, context: str = "document") -> np.ndarray:
        return np.array([stable_vector(f"{context}:{text}", dim=dim) for text in texts])

    return EmbeddingFunc(
        embedding_dim=dim,
        max_token_size=1024,
        func=embed,
        model_name="e2e-fake-multimodal",
        supports_asymmetric=True,
    )


async def fake_lightrag_llm(prompt: str, **_: Any) -> str:
    """Return a valid empty LightRAG extraction payload."""
    if re.search(r"keyword", prompt, re.IGNORECASE):
        return '{"high_level_keywords": ["image"], "low_level_keywords": ["native image"]}'
    return "<|COMPLETE|>"


async def fake_vlm_func(*, prompt: str, image_path: str, **_: Any) -> str:
    """Return a compact visual semantic description."""
    del prompt
    return f"Test image at {image_path} containing a green square for pg18 smoke."


def install_fake_model_functions(monkeypatch: Any, *, dim: int = 8) -> FakeMultimodalEmbedder:
    """Patch RAGService model factories to avoid external network calls."""
    from dlightrag.core import service as service_module

    multimodal_embedder = FakeMultimodalEmbedder(dim=dim)
    monkeypatch.setattr(
        service_module,
        "get_chat_model_func_for_lightrag",
        lambda _config: fake_lightrag_llm,
    )
    monkeypatch.setattr(service_module, "get_vlm_model_func", lambda _config: fake_vlm_func)
    monkeypatch.setattr(service_module, "get_rerank_func", lambda _config: None)
    monkeypatch.setattr(service_module, "build_role_llm_configs", lambda _config: None)
    monkeypatch.setattr(
        service_module, "get_embedding_func", lambda _config: fake_embedding_func(dim=dim)
    )
    monkeypatch.setattr(
        service_module,
        "get_multimodal_embedder",
        lambda _config: multimodal_embedder,
    )
    return multimodal_embedder
