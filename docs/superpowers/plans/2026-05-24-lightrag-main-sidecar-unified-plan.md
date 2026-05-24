# LightRAG Main Sidecar Unified Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace DlightRAG's RAGAnything/caption path and legacy page-render unified path with one PostgreSQL 18 + LightRAG-main sidecar pipeline that supports metadata, strict in-filtering, direct image embeddings, and BM25+RRF retrieval.

**Architecture:** `RAGService` owns one LightRAG instance per workspace, with PG18 storage only. Documents are ingested through LightRAG `main` parser pipeline using `teP` by default; native images and parser-extracted drawing/image assets are embedded directly through DlightRAG's multimodal embedding provider. Retrieval always uses LightRAG `mix`, adds PostgreSQL BM25, applies metadata candidate filters before every path, and fuses results with RRF.

**Tech Stack:** Python 3.12, LightRAG from `HKUDS/LightRAG@main`, PostgreSQL 18.4 baseline, pgvector, Apache AGE `PG18`, pg_textsearch, asyncpg, httpx, pytest.

---

## Verified Upstream Baseline

- LightRAG main target: `cfcba71` from `HKUDS/LightRAG` `origin/main`.
- PostgreSQL target: major `18`, current minor `18.4` as of 2026-05-24. Official references: [PostgreSQL 18.4 release announcement](https://www.postgresql.org/about/news/postgresql-184-1710-1614-1518-and-1423-released-3297/) and [PostgreSQL versioning policy](https://www.postgresql.org/support/versioning/).
- Apache AGE has a `PG18` branch.
- Docker Hub has `pgvector/pgvector:pg18` and `pgvector/pgvector:0.8.2-pg18` tags.

## PostgreSQL Extension Matrix

| Layer | Extension | Why |
|---|---|---|
| LightRAG vector storage | `vector` / pgvector | Required by `PGVectorStorage` for vector columns and HNSW search. |
| LightRAG graph storage | `age` / Apache AGE | Required by `PGGraphStorage` for the knowledge graph. |
| DlightRAG BM25 hybrid | `pg_textsearch` | Required by `PostgresBM25` for `USING bm25(content)` and `to_bm25query`. |

The commonly remembered LightRAG PG requirement is PG18 plus two core extensions: `vector` and `age`. DlightRAG's architecture adds `pg_textsearch` for BM25. Metadata filtering is LLM-assisted but deterministic at the matching layer: normalized exact fields backed by `LOWER(field)` btree expression indexes, date ranges, JSONB containment, and explicit filename patterns only.

## File Structure

Create:

- `src/dlightrag/storage/postgres_version.py` validates PostgreSQL server version.
- `src/dlightrag/core/lightrag_stores.py` centralizes LightRAG private storage access and precomputed vector writes.
- `src/dlightrag/storage/document_artifacts.py` stores document-level sidecar artifact locations in PostgreSQL.
- `src/dlightrag/storage/chunk_metadata.py` stores chunk-level sidecar and modality provenance in PostgreSQL.
- `src/dlightrag/models/multimodal_embedding.py` provides text/image embedding over the existing provider registry.
- `src/dlightrag/core/ingestion/sidecar.py` normalizes LightRAG/MinerU/Docling sidecar files.
- `src/dlightrag/core/ingestion/direct_image.py` builds direct-image chunk specs and vectors.
- `src/dlightrag/core/ingestion/engine.py` becomes the single ingestion orchestrator.
- `src/dlightrag/core/retrieval/bm25.py` implements PostgreSQL BM25 search.
- `src/dlightrag/core/retrieval/fusion.py` implements RRF.
- `src/dlightrag/core/retrieval/retriever.py` becomes the single retrieval orchestrator.
- Tests listed in each task.

Modify:

- `pyproject.toml`
- `config.yaml`
- `.env.example`
- `postgres/Dockerfile`
- `postgres/init.sql`
- `docker-compose.yml`
- `src/dlightrag/config.py`
- `src/dlightrag/core/service.py`
- `src/dlightrag/core/compat_guard.py`
- `src/dlightrag/core/retrieval/filtered_vdb.py`
- `src/dlightrag/core/retrieval/metadata_path.py`
- `src/dlightrag/core/retrieval/models.py`
- `src/dlightrag/core/retrieval/metadata_fields.py`
- `src/dlightrag/core/ingestion/hash_index.py`
- `src/dlightrag/core/ingestion/cleanup.py`
- `src/dlightrag/core/reset.py`
- `src/dlightrag/api/routes/status.py`
- `src/dlightrag/__init__.py`
- Existing tests that assert RAGAnything, JSON metadata, filesystem composition, or non-PG storage behavior.

Delete after replacements are green:

- `src/dlightrag/captionrag/`
- `src/dlightrag/unifiedrepresent/engine.py`
- `src/dlightrag/unifiedrepresent/retriever.py`
- `src/dlightrag/unifiedrepresent/lifecycle.py`
- `src/dlightrag/storage/json_metadata_index.py`
- tests that only validate removed non-PG or old-mode behavior.

Keep and move where useful:

- multimodal embedding provider code from `src/dlightrag/models/providers/`
- `VisualEmbedder` logic from `src/dlightrag/unifiedrepresent/embedder.py`
- image query helper ideas from `src/dlightrag/unifiedrepresent/retriever.py`
- metadata field registry and PG metadata index structure

---

### Task 1: PostgreSQL 18, Dependency, And Config Gates

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.yaml`
- Modify: `.env.example`
- Modify: `postgres/Dockerfile`
- Modify: `postgres/init.sql`
- Modify: `docker-compose.yml`
- Modify: `src/dlightrag/config.py`
- Create: `src/dlightrag/storage/postgres_version.py`
- Modify: `src/dlightrag/core/service.py`
- Test: `tests/unit/test_dependency_constraints.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_postgres_version.py`

- [ ] **Step 1: Write failing dependency policy tests**

Replace the RAGAnything tests in `tests/unit/test_dependency_constraints.py` with:

```python
def test_lightrag_dependency_uses_github_main() -> None:
    dependencies = _dependencies()
    lightrag_deps = [dep for dep in dependencies if dep.startswith("lightrag-hku")]

    assert lightrag_deps == [
        "lightrag-hku @ git+https://github.com/HKUDS/LightRAG.git@main"
    ]


def test_raganything_dependency_removed() -> None:
    dependencies = _dependencies()

    assert not any(dep.startswith("raganything") for dep in dependencies)
```

- [ ] **Step 2: Run dependency tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_dependency_constraints.py -v
```

Expected: two failures showing the current `lightrag-hku>=1.5.0rc1` and `raganything[all]>=1.3.0` dependency policy.

- [ ] **Step 3: Write failing PG-only config tests**

Add these tests to `tests/unit/test_config.py`:

```python
def test_storage_backends_are_postgres_only() -> None:
    cfg = DlightragConfig(
        embedding=EmbeddingConfig(
            model="qwen3-vl-embedding-2b",
            api_key="sk-test",
            capabilities={"multimodal": True},
        ),
    )

    assert cfg.vector_storage == "PGVectorStorage"
    assert cfg.graph_storage == "PGGraphStorage"
    assert cfg.kv_storage == "PGKVStorage"
    assert cfg.doc_status_storage == "PGDocStatusStorage"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vector_storage", "QdrantVectorDBStorage"),
        ("vector_storage", "MilvusVectorDBStorage"),
        ("graph_storage", "Neo4JStorage"),
        ("graph_storage", "NetworkXStorage"),
        ("kv_storage", "JsonKVStorage"),
        ("doc_status_storage", "JsonDocStatusStorage"),
    ],
)
def test_non_postgres_storage_rejected(field: str, value: str) -> None:
    kwargs = {
        "embedding": EmbeddingConfig(
            model="qwen3-vl-embedding-2b",
            api_key="sk-test",
            capabilities={"multimodal": True},
        ),
        field: value,
    }
    with pytest.raises(ValidationError):
        DlightragConfig(**kwargs)
```

- [ ] **Step 4: Add PostgreSQL version parser tests**

Create `tests/unit/test_postgres_version.py`:

```python
import pytest

from dlightrag.storage.postgres_version import parse_server_version_num, validate_postgres_major


@pytest.mark.parametrize(
    ("server_version_num", "expected"),
    [
        ("180004", 18),
        ("180000", 18),
        ("170010", 17),
        (180004, 18),
    ],
)
def test_parse_server_version_num(server_version_num, expected) -> None:
    assert parse_server_version_num(server_version_num) == expected


def test_validate_postgres_major_accepts_pg18() -> None:
    validate_postgres_major("180004", required_major=18)


def test_validate_postgres_major_rejects_pg17() -> None:
    with pytest.raises(RuntimeError, match="PostgreSQL 18"):
        validate_postgres_major("170010", required_major=18)
```

- [ ] **Step 5: Run config/version tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_config.py::test_storage_backends_are_postgres_only tests/unit/test_config.py::test_non_postgres_storage_rejected tests/unit/test_postgres_version.py -v
```

Expected: failures because `EmbeddingConfig.capabilities` and `postgres_version.py` do not exist yet, and non-PG storage literals still validate.

- [ ] **Step 6: Update dependencies**

Edit `pyproject.toml`:

```toml
dependencies = [
    # Core RAG
    "lightrag-hku @ git+https://github.com/HKUDS/LightRAG.git@main",
    # LLM
    "openai>=2.29.0",
```

Remove this line:

```toml
"raganything[all]>=1.3.0",
```

- [ ] **Step 7: Add PostgreSQL version validation helper**

Create `src/dlightrag/storage/postgres_version.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL server version validation."""

from __future__ import annotations

from typing import Any


def parse_server_version_num(value: str | int) -> int:
    """Return PostgreSQL major version from SHOW server_version_num."""
    number = int(value)
    return number // 10000


def validate_postgres_major(value: str | int, *, required_major: int = 18) -> None:
    """Raise if the connected server is not the required PostgreSQL major."""
    actual = parse_server_version_num(value)
    if actual < required_major:
        raise RuntimeError(
            f"DlightRAG requires PostgreSQL {required_major}.x or newer in the "
            f"{required_major}.x line; connected server reports major {actual}."
        )


async def ensure_postgres_major(conn: Any, *, required_major: int = 18) -> None:
    """Validate an asyncpg connection against the required PostgreSQL major."""
    value = await conn.fetchval("SHOW server_version_num")
    validate_postgres_major(value, required_major=required_major)
```

- [ ] **Step 8: Make config PG18-only and multimodal-required**

In `src/dlightrag/config.py`, add:

```python
class EmbeddingCapabilities(BaseModel):
    """Embedding model capability declaration."""

    multimodal: bool = False
```

Update `EmbeddingConfig`:

```python
class EmbeddingConfig(BaseModel):
    """Embedding-specific configuration."""

    provider: str | None = None
    model: str = "qwen3-vl-embedding-2b"
    api_key: str | None = None
    base_url: str | None = None
    dim: int = 1024
    max_token_size: int = 8192
    capabilities: EmbeddingCapabilities = Field(default_factory=EmbeddingCapabilities)
    startup_probe: bool = True
```

Replace storage literals:

```python
vector_storage: Literal["PGVectorStorage"] = Field(default="PGVectorStorage")
graph_storage: Literal["PGGraphStorage"] = Field(default="PGGraphStorage")
kv_storage: Literal["PGKVStorage"] = Field(default="PGKVStorage")
doc_status_storage: Literal["PGDocStatusStorage"] = Field(default="PGDocStatusStorage")
postgres_required_major: int = Field(default=18)
postgres_min_minor: str = Field(default="18.4")
```

Remove Neo4j, Milvus, Qdrant, Redis, Mongo, Json storage config blocks and remove non-PG env bridging in `model_post_init`.

Add an embedding validator:

```python
@model_validator(mode="after")
def _validate_embedding_capabilities(self):
    if not self.embedding.capabilities.multimodal:
        raise ValueError("embedding.capabilities.multimodal=true is required")
    return self
```

- [ ] **Step 9: Validate PG18 during service initialization**

In `src/dlightrag/core/service.py`, import:

```python
from dlightrag.storage.postgres_version import ensure_postgres_major
```

In `_initialize_with_pg_lock()`, immediately after the `asyncpg.connect` call succeeds and returns `conn`:

```python
await ensure_postgres_major(conn, required_major=self.config.postgres_required_major)
```

Remove the branch that proceeds without the distributed lock when PostgreSQL is unavailable. In PG-only mode, connection failure should raise:

```python
except Exception as e:
    raise RuntimeError(
        "PostgreSQL 18 is required for DlightRAG startup; connection or version check failed"
    ) from e
```

- [ ] **Step 10: Update Docker PostgreSQL image**

Edit `postgres/Dockerfile`:

```dockerfile
# PostgreSQL 18 with pgvector + Apache AGE + pg_textsearch for DlightRAG
FROM pgvector/pgvector:pg18

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates postgresql-server-dev-18 \
    libreadline-dev flex bison

RUN git clone --branch PG18 --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age && make install && cd / && rm -rf /tmp/age

RUN git clone --depth 1 https://github.com/timescale/pg_textsearch.git /tmp/pg_textsearch \
    && cd /tmp/pg_textsearch && make install && cd / && rm -rf /tmp/pg_textsearch

RUN apt-get purge -y build-essential git postgresql-server-dev-18 libreadline-dev flex bison \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY init.sql /docker-entrypoint-initdb.d/
```

Edit `postgres/init.sql`:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS age;
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

SET search_path = ag_catalog, "$user", public;
```

Edit `docker-compose.yml` PostgreSQL command:

```yaml
- "-c"
- "shared_preload_libraries=age,pg_textsearch"
- "-c"
- "io_method=worker"
```

- [ ] **Step 11: Update sample config**

In `config.yaml`, replace storage comments:

```yaml
vector_storage: PGVectorStorage
graph_storage: PGGraphStorage
kv_storage: PGKVStorage
doc_status_storage: PGDocStatusStorage
postgres_required_major: 18
postgres_min_minor: "18.4"
```

Replace embedding sample:

```yaml
embedding:
  provider: openai
  model: qwen3-vl-embedding-2b
  base_url: http://host.docker.internal:1234/v1
  dim: 2048
  capabilities:
    multimodal: true
```

- [ ] **Step 12: Run Task 1 tests**

Run:

```bash
uv run pytest tests/unit/test_dependency_constraints.py tests/unit/test_config.py tests/unit/test_postgres_version.py -v
```

Expected: all selected tests pass.

- [ ] **Step 13: Commit Task 1**

```bash
git add pyproject.toml config.yaml .env.example postgres/Dockerfile postgres/init.sql docker-compose.yml src/dlightrag/config.py src/dlightrag/storage/postgres_version.py src/dlightrag/core/service.py tests/unit/test_dependency_constraints.py tests/unit/test_config.py tests/unit/test_postgres_version.py
git commit -m "chore: require postgres 18 and lightrag main"
```

---

### Task 2: Remove RAGAnything And Old Mode Selection

**Files:**
- Modify: `src/dlightrag/core/service.py`
- Modify: `src/dlightrag/__init__.py`
- Modify: `src/dlightrag/api/routes/status.py`
- Modify: `src/dlightrag/core/servicemanager.py`
- Delete: `src/dlightrag/captionrag/`
- Delete or stop importing old mode-specific tests
- Test: `tests/unit/test_service.py`
- Test: `tests/unit/test_dependency_constraints.py`

- [ ] **Step 1: Add import policy test**

Append to `tests/unit/test_dependency_constraints.py`:

```python
def test_runtime_imports_do_not_reference_raganything() -> None:
    source_files = list(Path("src/dlightrag").rglob("*.py"))
    offenders = [
        path
        for path in source_files
        if "raganything" in path.read_text(encoding="utf-8").lower()
    ]

    assert offenders == []
```

- [ ] **Step 2: Run import policy test and verify it fails**

Run:

```bash
uv run pytest tests/unit/test_dependency_constraints.py::test_runtime_imports_do_not_reference_raganything -v
```

Expected: failure listing `src/dlightrag/core/service.py`, `src/dlightrag/captionrag/*`, and any docstring references.

- [ ] **Step 3: Remove RAGAnything service imports**

In `src/dlightrag/core/service.py`, remove imports from `raganything` and `dlightrag.captionrag`. Keep LightRAG imports local in initialization. The top of the file should not import `raganything`.

Replace the `_do_initialize()` branch with:

```python
async def _do_initialize(self) -> None:
    """Create one LightRAG-backed unified pipeline."""
    from dlightrag.core._lightrag_patches import apply as apply_lightrag_patches

    apply_lightrag_patches()
    await self._do_initialize_unified()
```

Delete `config.rag_mode` logging and the caption-mode construction block.

- [ ] **Step 4: Remove `rag_mode` from public status**

In `src/dlightrag/api/routes/status.py`, remove `rag_mode` from response payloads. In `src/dlightrag/core/servicemanager.py`, remove logic that switches behavior on `config.rag_mode`.

- [ ] **Step 5: Update package description**

Change `src/dlightrag/__init__.py` docstring to:

```python
"""DlightRAG: PostgreSQL-backed multimodal RAG built on LightRAG main."""
```

- [ ] **Step 6: Delete caption path**

Remove:

```bash
git rm -r src/dlightrag/captionrag
```

Also remove tests whose only subject is caption/RAGAnything behavior. Keep tests for shared metadata, retrieval models, provider code, and API schema.

- [ ] **Step 7: Run import policy and service tests**

Run:

```bash
uv run pytest tests/unit/test_dependency_constraints.py::test_runtime_imports_do_not_reference_raganything tests/unit/test_service.py -v
```

Expected: import policy passes; service tests that still encode old mode behavior fail and must be rewritten to the single unified service.

- [ ] **Step 8: Rewrite old mode service tests**

In `tests/unit/test_service.py`, replace assertions about `rag_mode` with assertions that `_do_initialize()` delegates to `_do_initialize_unified()`:

```python
async def test_do_initialize_uses_unified_path(monkeypatch):
    service = RAGService(config=_make_config())
    called = False

    async def fake_unified():
        nonlocal called
        called = True

    monkeypatch.setattr(service, "_do_initialize_unified", fake_unified)

    await service._do_initialize()

    assert called is True
```

- [ ] **Step 9: Run Task 2 tests**

Run:

```bash
uv run pytest tests/unit/test_dependency_constraints.py tests/unit/test_service.py tests/unit/test_api_server.py tests/unit/test_servicemanager.py -v
```

Expected: selected tests pass after removing old-mode expectations.

- [ ] **Step 10: Commit Task 2**

```bash
git add src/dlightrag tests pyproject.toml
git commit -m "refactor: remove raganything mode"
```

---

### Task 3: Multimodal Embedding Provider For Text And Images

**Files:**
- Create: `src/dlightrag/models/multimodal_embedding.py`
- Modify: `src/dlightrag/models/llm.py`
- Modify: `src/dlightrag/unifiedrepresent/embedder.py` or move logic into the new file
- Test: `tests/unit/test_multimodal_embedding.py`
- Test: `tests/unit/test_embedding_func.py`

- [ ] **Step 1: Write multimodal embedding tests**

Create `tests/unit/test_multimodal_embedding.py`:

```python
from PIL import Image

from dlightrag.models.multimodal_embedding import MultimodalEmbedder
from dlightrag.models.providers.embed_providers import OpenAICompatEmbedProvider, VoyageEmbedProvider


def test_openai_compat_image_payload_is_data_uri() -> None:
    embedder = MultimodalEmbedder(
        model="qwen3-vl-embedding-2b",
        base_url="http://localhost:1234/v1",
        api_key="",
        dim=3,
        provider=OpenAICompatEmbedProvider(),
    )
    image = Image.new("RGB", (1, 1), "white")

    payload = embedder.build_image_payload_for_test(image)

    assert payload["model"] == "qwen3-vl-embedding-2b"
    assert payload["input"].startswith("data:image/png;base64,")
    assert payload["encoding_format"] == "float"


def test_voyage_image_payload_uses_multimodal_inputs() -> None:
    embedder = MultimodalEmbedder(
        model="voyage-multimodal-3.5",
        base_url="https://api.voyageai.com/v1",
        api_key="key",
        dim=3,
        provider=VoyageEmbedProvider(),
    )
    image = Image.new("RGB", (1, 1), "white")

    payload = embedder.build_image_payload_for_test(image)

    assert payload["inputs"][0]["content"][0]["type"] == "image_base64"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_multimodal_embedding.py -v
```

Expected: import failure because `dlightrag.models.multimodal_embedding` does not exist.

- [ ] **Step 3: Create unified multimodal embedder**

Create `src/dlightrag/models/multimodal_embedding.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Text and image embedding over DlightRAG's multimodal provider registry."""

from __future__ import annotations

import asyncio
import base64
import io
import logging

import httpx
from PIL import Image

from dlightrag.models.providers.embed_base import EmbedProvider
from dlightrag.models.providers.embed_providers import VoyageEmbedProvider

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """Embed text and images into one shared multimodal vector space."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dim: int,
        provider: EmbedProvider,
        batch_size: int = 4,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else "https://api.openai.com"
        self.dim = dim
        self.provider = provider
        self.batch_size = batch_size
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            transport=httpx.AsyncHTTPTransport(retries=2),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self.provider.build_payload(self.model, texts)
        data = await self._post(payload)
        vectors = self.provider.parse_response(data)
        self._validate_vectors(vectors)
        return vectors

    async def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        if not images:
            return []
        sem = asyncio.Semaphore(self.batch_size)

        async def one(image: Image.Image) -> list[float]:
            async with sem:
                payload = self._build_image_payload(image)
                data = await self._post(payload)
                vectors = self.provider.parse_response(data)
                self._validate_vectors(vectors)
                return vectors[0]

        return await asyncio.gather(*(one(image) for image in images))

    async def probe_image_embedding(self) -> None:
        await self.embed_images([Image.new("RGB", (1, 1), "white")])

    def build_image_payload_for_test(self, image: Image.Image) -> dict:
        return self._build_image_payload(image)

    async def _post(self, payload: dict) -> dict:
        url = f"{self.base_url}{self.provider.endpoint}"
        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def _build_image_payload(self, image: Image.Image) -> dict:
        image_uri = self._image_to_data_uri(image)
        if isinstance(self.provider, VoyageEmbedProvider):
            return {
                "model": self.model,
                "inputs": [{"content": [{"type": "image_base64", "image_base64": image_uri}]}],
            }
        return {"model": self.model, "input": image_uri, "encoding_format": "float"}

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        if vectors and len(vectors[0]) != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {len(vectors[0])}")

    @staticmethod
    def _image_to_data_uri(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
```

- [ ] **Step 4: Route LightRAG text embeddings through the multimodal embedder**

In `src/dlightrag/models/llm.py`, update `get_embedding_func()` to create one `MultimodalEmbedder` and use `embed_texts`:

```python
from dlightrag.models.multimodal_embedding import MultimodalEmbedder

cfg = config.embedding
embed_provider = detect_embed_provider(
    model=cfg.model,
    provider=cfg.provider,
    base_url=cfg.base_url,
)
embedder = MultimodalEmbedder(
    model=cfg.model,
    api_key=cfg.api_key or "",
    base_url=cfg.base_url or "",
    dim=cfg.dim,
    provider=embed_provider,
    batch_size=config.embedding_func_max_async,
    timeout=float(config.embedding_request_timeout),
)

async def embed_func(texts: list[str]) -> np.ndarray:
    result = await embedder.embed_texts(texts)
    return np.array(result)
```

Expose a factory for direct-image ingestion:

```python
def get_multimodal_embedder(config: DlightragConfig) -> MultimodalEmbedder:
    from dlightrag.models.providers.embed_providers import detect_embed_provider

    cfg = config.embedding
    return MultimodalEmbedder(
        model=cfg.model,
        api_key=cfg.api_key or "",
        base_url=cfg.base_url or "",
        dim=cfg.dim,
        provider=detect_embed_provider(cfg.model, provider=cfg.provider, base_url=cfg.base_url),
        batch_size=config.embedding_func_max_async,
        timeout=float(config.embedding_request_timeout),
    )
```

- [ ] **Step 5: Add startup probe in service**

In `src/dlightrag/core/service.py`, after building the multimodal embedder:

```python
if config.embedding.startup_probe:
    await multimodal_embedder.probe_image_embedding()
```

When adding service tests for this branch, patch `MultimodalEmbedder.probe_image_embedding` with `AsyncMock(return_value=None)` before calling `RAGService.initialize()`.

- [ ] **Step 6: Run Task 3 tests**

Run:

```bash
uv run pytest tests/unit/test_multimodal_embedding.py tests/unit/test_embedding_func.py tests/unit/test_embed_providers.py -v
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add src/dlightrag/models tests/unit/test_multimodal_embedding.py tests/unit/test_embedding_func.py
git commit -m "feat: require multimodal embedding provider"
```

---

### Task 4: LightRAG Store Adapter And PG Provenance Stores

**Files:**
- Create: `src/dlightrag/core/lightrag_stores.py`
- Create: `src/dlightrag/storage/document_artifacts.py`
- Create: `src/dlightrag/storage/chunk_metadata.py`
- Modify: `src/dlightrag/core/compat_guard.py`
- Test: `tests/unit/test_lightrag_stores.py`
- Test: `tests/unit/test_document_artifacts.py`
- Test: `tests/unit/test_chunk_metadata.py`

- [ ] **Step 1: Write adapter contract tests**

Create `tests/unit/test_lightrag_stores.py`:

```python
import pytest

from dlightrag.core.lightrag_stores import LightRAGStores


class FakeLightRAG:
    chunks_vdb = object()
    text_chunks = object()
    full_docs = object()
    doc_status = object()
    entities_vdb = object()
    relationships_vdb = object()
    chunk_entity_relation_graph = object()
    full_entities = object()
    full_relations = object()
    entity_chunks = object()
    relation_chunks = object()
    llm_response_cache = object()

    def _build_global_config(self):
        return {"ok": True}


def test_lightrag_stores_validates_required_surfaces() -> None:
    stores = LightRAGStores(FakeLightRAG())

    assert stores.text_chunks is FakeLightRAG.text_chunks
    assert stores.build_global_config() == {"ok": True}


def test_lightrag_stores_reports_missing_surfaces() -> None:
    class Broken:
        chunks_vdb = object()

    with pytest.raises(RuntimeError, match="missing"):
        LightRAGStores(Broken())
```

- [ ] **Step 2: Write PG store shape tests**

Create `tests/unit/test_chunk_metadata.py` with pure validation tests:

```python
import pytest

from dlightrag.storage.chunk_metadata import validate_chunk_metadata_record


def test_chunk_metadata_accepts_native_image() -> None:
    validate_chunk_metadata_record(
        {
            "chunk_id": "chunk-native-1",
            "full_doc_id": "doc-1",
            "embedding_input_kind": "image",
            "sidecar_type": "native_image",
        }
    )


def test_chunk_metadata_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="embedding_input_kind"):
        validate_chunk_metadata_record(
            {
                "chunk_id": "chunk-1",
                "full_doc_id": "doc-1",
                "embedding_input_kind": "audio",
            }
        )
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_lightrag_stores.py tests/unit/test_chunk_metadata.py -v
```

Expected: import failures for the new modules.

- [ ] **Step 4: Create `LightRAGStores`**

Create `src/dlightrag/core/lightrag_stores.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Single boundary for LightRAG private storage access."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

from lightrag.utils import EmbeddingFunc


class LightRAGStores:
    """Typed accessor for LightRAG storage attributes DlightRAG writes directly."""

    _REQUIRED: ClassVar[frozenset[str]] = frozenset(
        {
            "chunks_vdb",
            "text_chunks",
            "full_docs",
            "doc_status",
            "entities_vdb",
            "relationships_vdb",
            "chunk_entity_relation_graph",
            "full_entities",
            "full_relations",
            "entity_chunks",
            "relation_chunks",
            "llm_response_cache",
            "_build_global_config",
        }
    )

    def __init__(self, lightrag: Any) -> None:
        missing = sorted(name for name in self._REQUIRED if not hasattr(lightrag, name))
        if missing:
            raise RuntimeError(f"LightRAGStores missing required surface(s): {missing}")
        self.raw = lightrag
        self._vector_write_lock = asyncio.Lock()

    def __getattr__(self, name: str) -> Any:
        if name in self._REQUIRED:
            return getattr(self.raw, name)
        raise AttributeError(name)

    def build_global_config(self) -> dict[str, Any]:
        return self.raw._build_global_config()

    async def upsert_chunks_with_vectors(
        self,
        rows: dict[str, dict[str, Any]],
        vectors: dict[str, list[float]],
        *,
        embedding_dim: int,
        max_token_size: int,
    ) -> None:
        """Write chunk rows while reusing precomputed image vectors."""
        if not rows:
            return

        async def cached_embed(texts: list[str]) -> list[list[float]]:
            return [vectors[text] for text in texts]

        async with self._vector_write_lock:
            original = self.raw.chunks_vdb.embedding_func
            self.raw.chunks_vdb.embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=max_token_size,
                func=cached_embed,
            )
            try:
                await self.raw.chunks_vdb.upsert(rows)
            finally:
                self.raw.chunks_vdb.embedding_func = original
```

- [ ] **Step 5: Create PG document artifacts store**

Create `src/dlightrag/storage/document_artifacts.py` using the DDL from the spec. The class must expose these methods with working PostgreSQL implementations:

```python
class PGDocumentArtifacts:
    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._pool = None

    async def initialize(self) -> None:
        from dlightrag.storage.pool import pg_pool

        self._pool = await pg_pool.get()

    async def upsert(self, record: Mapping[str, Any]) -> None:
        full_doc_id = str(record["full_doc_id"])
        if not full_doc_id:
            raise ValueError("full_doc_id is required")

    async def get(self, full_doc_id: str) -> dict[str, Any] | None:
        if not full_doc_id:
            raise ValueError("full_doc_id is required")
        return None

    async def delete_doc(self, full_doc_id: str) -> dict[str, Any] | None:
        if not full_doc_id:
            raise ValueError("full_doc_id is required")
        return None

    async def clear(self) -> None:
        if self._pool is None:
            raise RuntimeError("PGDocumentArtifacts is not initialized")
```

Replace the method bodies above with the SQL-backed body in the same step. Use `dlightrag.storage.pool.pg_pool` for the pool and table name `dlightrag_document_artifacts`.

- [ ] **Step 6: Create PG chunk metadata store**

Create `src/dlightrag/storage/chunk_metadata.py` with constants:

```python
EMBEDDING_INPUT_KINDS = frozenset({"text", "image", "multimodal"})
SIDECAR_TYPES = frozenset({"block", "drawing", "table", "equation", "native_image"})
```

Expose:

```python
def validate_chunk_metadata_record(record: Mapping[str, Any]) -> None:
    kind = record.get("embedding_input_kind")
    if kind not in EMBEDDING_INPUT_KINDS:
        raise ValueError(f"embedding_input_kind must be one of {sorted(EMBEDDING_INPUT_KINDS)}")
    sidecar_type = record.get("sidecar_type")
    if sidecar_type is not None and sidecar_type not in SIDECAR_TYPES:
        raise ValueError(f"sidecar_type must be one of {sorted(SIDECAR_TYPES)}")
    if not record.get("chunk_id"):
        raise ValueError("chunk_id is required")
    if not record.get("full_doc_id"):
        raise ValueError("full_doc_id is required")

class PGChunkMetadata:
    def __init__(self, workspace: str = "default") -> None:
        self._workspace = workspace
        self._pool = None

    async def initialize(self) -> None:
        from dlightrag.storage.pool import pg_pool

        self._pool = await pg_pool.get()

    async def upsert_many(self, records: list[Mapping[str, Any]]) -> None:
        for record in records:
            validate_chunk_metadata_record(record)

    async def get_batch(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        return {}

    async def chunk_ids_for_docs(self, doc_ids: list[str]) -> list[str]:
        return []

    async def delete_doc(self, full_doc_id: str) -> None:
        if not full_doc_id:
            raise ValueError("full_doc_id is required")

    async def clear(self) -> None:
        if self._pool is None:
            raise RuntimeError("PGChunkMetadata is not initialized")
```

- [ ] **Step 7: Wire stores into service**

In `src/dlightrag/core/service.py`, add instance fields:

```python
self._lightrag_stores: LightRAGStores | None = None
self._document_artifacts: PGDocumentArtifacts | None = None
self._chunk_metadata: PGChunkMetadata | None = None
```

After `await lightrag.initialize_storages()`:

```python
self._lightrag_stores = LightRAGStores(lightrag)
self._document_artifacts = PGDocumentArtifacts(workspace=config.workspace)
self._chunk_metadata = PGChunkMetadata(workspace=config.workspace)
await self._document_artifacts.initialize()
await self._chunk_metadata.initialize()
```

- [ ] **Step 8: Run Task 4 tests**

Run:

```bash
uv run pytest tests/unit/test_lightrag_stores.py tests/unit/test_document_artifacts.py tests/unit/test_chunk_metadata.py tests/unit/test_compat_guard.py -v
```

Expected: selected tests pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add src/dlightrag/core/lightrag_stores.py src/dlightrag/storage/document_artifacts.py src/dlightrag/storage/chunk_metadata.py src/dlightrag/core/service.py src/dlightrag/core/compat_guard.py tests/unit/test_lightrag_stores.py tests/unit/test_document_artifacts.py tests/unit/test_chunk_metadata.py
git commit -m "feat: add lightrag store adapter and provenance stores"
```

---

### Task 5: Unified Sidecar Ingestion

**Files:**
- Create: `src/dlightrag/core/ingestion/sidecar.py`
- Create: `src/dlightrag/core/ingestion/direct_image.py`
- Create: `src/dlightrag/core/ingestion/engine.py`
- Modify: `src/dlightrag/core/service.py`
- Modify: `src/dlightrag/core/ingestion/hash_index.py`
- Test: `tests/unit/test_sidecar.py`
- Test: `tests/unit/test_unified_ingestion_engine.py`
- Test: `tests/unit/test_direct_image_ingest.py`

- [ ] **Step 1: Write sidecar normalization tests**

Create `tests/unit/test_sidecar.py`:

```python
import json

from dlightrag.core.ingestion.sidecar import collect_sidecar_units


def test_collects_drawing_table_equation_units(tmp_path) -> None:
    artifact_dir = tmp_path / "doc"
    artifact_dir.mkdir()
    (artifact_dir / "drawings.json").write_text(
        json.dumps([{"id": "fig-1", "asset_path": "media/fig.png", "page": 2}]),
        encoding="utf-8",
    )
    (artifact_dir / "tables.json").write_text(
        json.dumps([{"id": "table-1", "llm_analyze_result": {"status": "success"}}]),
        encoding="utf-8",
    )
    (artifact_dir / "equations.json").write_text(
        json.dumps([{"id": "eq-1", "llm_analyze_result": {"status": "success"}}]),
        encoding="utf-8",
    )

    units = collect_sidecar_units(artifact_dir)

    assert [(u.sidecar_type, u.sidecar_id) for u in units] == [
        ("drawing", "fig-1"),
        ("table", "table-1"),
        ("equation", "eq-1"),
    ]
```

- [ ] **Step 2: Write LightRAG enqueue contract test**

Create `tests/unit/test_unified_ingestion_engine.py`:

```python
from pathlib import Path
from unittest.mock import AsyncMock

from dlightrag.core.ingestion.engine import UnifiedIngestionEngine


async def test_document_ingest_uses_lightrag_pending_parse_teP(tmp_path: Path) -> None:
    lightrag = AsyncMock()
    lightrag.apipeline_enqueue_documents.return_value = "track-1"
    lightrag.doc_status.get_by_id.return_value = {"chunks_list": ["chunk-a"]}
    stores = AsyncMock()
    metadata = AsyncMock()
    artifacts = AsyncMock()
    chunks = AsyncMock()
    hash_index = AsyncMock()
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4")

    engine = UnifiedIngestionEngine(
        lightrag=lightrag,
        stores=stores,
        metadata_index=metadata,
        document_artifacts=artifacts,
        chunk_metadata=chunks,
        hash_index=hash_index,
        multimodal_embedder=AsyncMock(),
        workspace="default",
        parser_engine="mineru",
        process_options="teP",
    )

    await engine.aingest_file(source, replace=False)

    kwargs = lightrag.apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert kwargs["parse_engine"] == "mineru"
    assert kwargs["process_options"] == "teP"
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_sidecar.py tests/unit/test_unified_ingestion_engine.py -v
```

Expected: import failures for the new ingestion modules.

- [ ] **Step 4: Create sidecar normalization module**

Create `src/dlightrag/core/ingestion/sidecar.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Normalize LightRAG parser sidecars into DlightRAG sidecar units."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SidecarUnit:
    sidecar_type: str
    sidecar_id: str
    asset_path: Path | None = None
    page_number: int | None = None
    bbox: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None


def collect_sidecar_units(artifact_dir: Path) -> list[SidecarUnit]:
    units: list[SidecarUnit] = []
    for sidecar_type, filename in (
        ("drawing", "drawings.json"),
        ("table", "tables.json"),
        ("equation", "equations.json"),
    ):
        path = artifact_dir / filename
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            items = data.get("items", [])
        else:
            items = data
        for index, item in enumerate(items):
            sidecar_id = str(item.get("id") or item.get("uid") or f"{sidecar_type}-{index}")
            raw_asset = item.get("asset_path") or item.get("image_path")
            units.append(
                SidecarUnit(
                    sidecar_type=sidecar_type,
                    sidecar_id=sidecar_id,
                    asset_path=(artifact_dir / raw_asset).resolve() if raw_asset else None,
                    page_number=item.get("page") or item.get("page_number"),
                    bbox=item.get("bbox"),
                    payload=item,
                )
            )
    return units
```

- [ ] **Step 5: Create direct image helper**

Create `src/dlightrag/core/ingestion/direct_image.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Direct image embedding chunk creation for native and extracted images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from dlightrag.core.ingestion.sidecar import SidecarUnit


def direct_image_chunk_id(workspace: str, full_doc_id: str, unit: SidecarUnit) -> str:
    return f"{workspace}:{full_doc_id}:sidecar:{unit.sidecar_type}:{unit.sidecar_id}"


async def build_direct_image_chunk(
    *,
    workspace: str,
    full_doc_id: str,
    unit: SidecarUnit,
    embedder: Any,
    text_content: str,
) -> tuple[str, dict[str, Any], list[float]]:
    if unit.asset_path is None:
        raise ValueError(f"{unit.sidecar_type}:{unit.sidecar_id} has no asset_path")
    image = Image.open(unit.asset_path).convert("RGB")
    vector = (await embedder.embed_images([image]))[0]
    chunk_id = direct_image_chunk_id(workspace, full_doc_id, unit)
    row = {
        "content": text_content,
        "file_path": str(unit.asset_path),
    }
    return chunk_id, row, vector


def native_image_unit(path: Path) -> SidecarUnit:
    return SidecarUnit(
        sidecar_type="native_image",
        sidecar_id=path.stem,
        asset_path=path.resolve(),
        page_number=None,
        bbox=None,
        payload={"source_path": str(path)},
    )
```

- [ ] **Step 6: Create unified ingestion engine**

Create `src/dlightrag/core/ingestion/engine.py` with `UnifiedIngestionEngine.aingest_file()`, `UnifiedIngestionEngine._ingest_document()`, and `UnifiedIngestionEngine._ingest_native_image()`. `aingest_file()` must dispatch by suffix and return a dictionary containing `doc_id`, `source_kind`, `chunks`, and `ingest_strategy`.

The document enqueue call must be:

```python
from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE

track_id = await self._lightrag.apipeline_enqueue_documents(
    input="",
    ids=[doc_id],
    file_paths=[str(file_path)],
    docs_format=FULL_DOCS_FORMAT_PENDING_PARSE,
    lightrag_document_paths=[str(file_path)],
    parse_engine=self._parser_engine,
    process_options=self._process_options,
)
await self._lightrag.apipeline_process_enqueue_documents()
```

After processing:

```python
doc_status = await self._lightrag.doc_status.get_by_id(doc_id)
light_chunks = doc_status.get("chunks_list", []) if doc_status else []
await self._chunk_metadata.upsert_many(
    [
        {
            "chunk_id": chunk_id,
            "full_doc_id": doc_id,
            "embedding_input_kind": "text",
            "sidecar_type": "block",
        }
        for chunk_id in light_chunks
    ]
)
```

For drawing units, use `build_direct_image_chunk()` and `stores.upsert_chunks_with_vectors()`.

- [ ] **Step 7: Wire engine into service**

In `src/dlightrag/core/service.py`, replace the `UnifiedRepresentEngine` construction with:

```python
from dlightrag.core.ingestion.engine import UnifiedIngestionEngine

self._backend = UnifiedIngestionEngine(
    lightrag=lightrag,
    stores=self._lightrag_stores,
    metadata_index=self._metadata_index,
    document_artifacts=self._document_artifacts,
    chunk_metadata=self._chunk_metadata,
    hash_index=self._hash_index,
    multimodal_embedder=multimodal_embedder,
    workspace=config.workspace,
    parser_engine=config.parser,
    process_options=config.parser_process_options,
)
```

Rename config `parser_process_options` if the implementation chooses a flat field; keep default `"teP"`.

- [ ] **Step 8: Run Task 5 tests**

Run:

```bash
uv run pytest tests/unit/test_sidecar.py tests/unit/test_unified_ingestion_engine.py tests/unit/test_direct_image_ingest.py -v
```

Expected: selected tests pass.

- [ ] **Step 9: Commit Task 5**

```bash
git add src/dlightrag/core/ingestion src/dlightrag/core/service.py tests/unit/test_sidecar.py tests/unit/test_unified_ingestion_engine.py tests/unit/test_direct_image_ingest.py
git commit -m "feat: ingest through lightrag sidecars"
```

---

### Task 6: Metadata Resolution And Strict PG In-Filtering

**Files:**
- Modify: `src/dlightrag/core/retrieval/filtered_vdb.py`
- Modify: `src/dlightrag/core/retrieval/metadata_path.py`
- Modify: `src/dlightrag/core/retrieval/models.py`
- Modify: `src/dlightrag/storage/pg_metadata_index.py`
- Test: `tests/unit/test_filtered_vdb.py`
- Test: `tests/unit/test_metadata_path.py`
- Test: `tests/unit/test_metadata_filter.py`

- [ ] **Step 1: Write strict empty-filter tests**

Create or update `tests/unit/test_filtered_vdb.py`:

```python
from dlightrag.core.retrieval.filtered_vdb import _active_filter, metadata_filter_scope


async def test_empty_candidate_set_is_active_filter() -> None:
    async with metadata_filter_scope(set()):
        assert _active_filter.get() == set()


async def test_none_candidate_set_is_no_filter() -> None:
    async with metadata_filter_scope(None):
        assert _active_filter.get() is None
```

- [ ] **Step 2: Write metadata path chunk metadata test**

Update `tests/unit/test_metadata_path.py`:

```python
from dlightrag.core.retrieval.metadata_path import metadata_retrieve
from dlightrag.core.retrieval.models import MetadataFilter


async def test_metadata_retrieve_uses_chunk_metadata() -> None:
    metadata_index = AsyncMock()
    metadata_index.query.return_value = ["doc-1"]
    chunk_metadata = AsyncMock()
    chunk_metadata.chunk_ids_for_docs.return_value = ["chunk-a", "chunk-b"]

    result = await metadata_retrieve(
        metadata_index=metadata_index,
        chunk_metadata=chunk_metadata,
        filters=MetadataFilter(filename="x.pdf"),
    )

    assert result == ["chunk-a", "chunk-b"]
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_filtered_vdb.py tests/unit/test_metadata_path.py -v
```

Expected: failures because empty set is treated as no filter and `metadata_retrieve` still accepts `rag_mode`.

- [ ] **Step 4: Make filter scope strict**

In `src/dlightrag/core/retrieval/filtered_vdb.py`, replace:

```python
if not candidate_ids:
    yield
    return
```

with:

```python
if candidate_ids is None:
    yield
    return
```

Remove Milvus/Qdrant branches and post-filter behavior. If backend is not `PGVectorStorage`, raise:

```python
raise RuntimeError(f"Filtered vector search requires PGVectorStorage, got {self._backend}")
```

In `fetch_missing_chunks()`, replace `if not active_ids:` with:

```python
if active_ids is None:
    return []
```

- [ ] **Step 5: Remove `rag_mode` metadata filter**

In `src/dlightrag/core/retrieval/models.py`, remove:

```python
rag_mode: str | None = None
```

In `src/dlightrag/storage/pg_metadata_index.py`, remove `rag_mode` from DDL/upsert/query and add:

```python
ingest_strategy TEXT
parse_engine TEXT
process_options TEXT
artifact_status TEXT
```

Update upsert values to store these fields.

- [ ] **Step 6: Rewrite metadata chunk resolution**

Replace `metadata_retrieve()` signature:

```python
async def metadata_retrieve(
    *,
    metadata_index: PGMetadataIndex,
    chunk_metadata: PGChunkMetadata,
    filters: MetadataFilter,
) -> list[str]:
    doc_ids = await metadata_index.query(filters)
    if not doc_ids:
        return []
    return await chunk_metadata.chunk_ids_for_docs(doc_ids)
```

- [ ] **Step 7: Run Task 6 tests**

Run:

```bash
uv run pytest tests/unit/test_filtered_vdb.py tests/unit/test_metadata_path.py tests/unit/test_metadata_filter.py tests/unit/test_metadata_index.py -v
```

Expected: selected tests pass.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/dlightrag/core/retrieval src/dlightrag/storage/pg_metadata_index.py tests/unit/test_filtered_vdb.py tests/unit/test_metadata_path.py tests/unit/test_metadata_filter.py tests/unit/test_metadata_index.py
git commit -m "fix: enforce strict postgres metadata filtering"
```

---

### Task 7: BM25 And RRF Retrieval

**Files:**
- Create: `src/dlightrag/core/retrieval/bm25.py`
- Create: `src/dlightrag/core/retrieval/fusion.py`
- Create or replace: `src/dlightrag/core/retrieval/retriever.py`
- Modify: `src/dlightrag/core/service.py`
- Modify: `src/dlightrag/core/compat_guard.py`
- Test: `tests/unit/test_bm25.py`
- Test: `tests/unit/test_retrieval_fusion.py`
- Test: `tests/unit/test_retriever.py`

- [ ] **Step 1: Write RRF tests**

Create `tests/unit/test_retrieval_fusion.py`:

```python
from dlightrag.core.retrieval.fusion import rrf_fuse


def test_rrf_fuse_deduplicates_by_chunk_id() -> None:
    semantic = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    bm25 = [{"chunk_id": "b"}, {"chunk_id": "c"}]

    fused = rrf_fuse([semantic, bm25], k=60)

    assert [row["chunk_id"] for row in fused] == ["b", "a", "c"]
    assert fused[0]["score"] > fused[1]["score"]
```

- [ ] **Step 2: Write BM25 SQL tests**

Create `tests/unit/test_bm25.py`:

```python
from dlightrag.core.retrieval.bm25 import build_bm25_sql


def test_bm25_sql_filters_candidates() -> None:
    sql = build_bm25_sql(candidate_ids={"chunk-a"}, limit=20)

    assert "id = ANY" in sql
    assert "to_bm25query" in sql
    assert "idx_lightrag_doc_chunks_bm25" in sql
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_bm25.py tests/unit/test_retrieval_fusion.py -v
```

Expected: import failures for new modules.

- [ ] **Step 4: Implement RRF**

Create `src/dlightrag/core/retrieval/fusion.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Retrieval score fusion."""

from __future__ import annotations

from typing import Any


def _chunk_id(row: dict[str, Any]) -> str:
    return str(row.get("chunk_id") or row.get("id"))


def rrf_fuse(rankings: list[list[dict[str, Any]]], *, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    best: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        for rank, row in enumerate(ranking):
            cid = _chunk_id(row)
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(cid, dict(row))
    fused = []
    for cid, row in best.items():
        row["chunk_id"] = cid
        row["score"] = scores[cid]
        fused.append(row)
    return sorted(fused, key=lambda row: row["score"], reverse=True)
```

- [ ] **Step 5: Implement BM25 PG search**

Create `src/dlightrag/core/retrieval/bm25.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL BM25 search over LightRAG document chunks."""

from __future__ import annotations

from typing import Any


BM25_INDEX = "idx_lightrag_doc_chunks_bm25"


def build_bm25_sql(*, candidate_ids: set[str] | None, limit: int) -> str:
    candidate_clause = "AND id = ANY($3)" if candidate_ids is not None else ""
    return f"""
        SELECT id, content, file_path,
               -(content <@> to_bm25query($1, '{BM25_INDEX}')) AS score
        FROM LIGHTRAG_DOC_CHUNKS
        WHERE workspace = $2
        {candidate_clause}
        ORDER BY content <@> to_bm25query($1, '{BM25_INDEX}')
        LIMIT {int(limit)}
    """


class PostgresBM25:
    def __init__(self, *, pool: Any, workspace: str, top_k: int = 40) -> None:
        self._pool = pool
        self._workspace = workspace
        self._top_k = top_k

    async def ensure_index(self, *, text_config: str = "simple") -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {BM25_INDEX} "
                "ON LIGHTRAG_DOC_CHUNKS USING bm25(content) "
                "WITH (text_config = $1)",
                text_config,
            )

    async def search(
        self,
        query: str,
        *,
        candidate_ids: set[str] | None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if candidate_ids is not None and len(candidate_ids) == 0:
            return []
        limit = top_k or self._top_k
        sql = build_bm25_sql(candidate_ids=candidate_ids, limit=limit)
        async with self._pool.acquire() as conn:
            if candidate_ids is None:
                rows = await conn.fetch(sql, query, self._workspace)
            else:
                rows = await conn.fetch(sql, query, self._workspace, list(candidate_ids))
        return [
            {
                "chunk_id": row["id"],
                "content": row["content"],
                "file_path": row["file_path"],
                "score": float(row["score"]),
            }
            for row in rows
        ]
```

If asyncpg rejects `$1` inside `WITH`, change `ensure_index()` to validate `text_config` against `{"simple", "english"}` and interpolate the sanitized literal.

- [ ] **Step 6: Create unified retriever**

Create `src/dlightrag/core/retrieval/retriever.py` with one public method:

```python
class UnifiedRetriever:
    async def aretrieve(
        self,
        query: str,
        *,
        metadata_filter: MetadataFilter | None = None,
        query_images: list[bytes] | None = None,
        top_k: int = 25,
    ) -> RetrievalResult:
        if metadata_filter is not None and not metadata_filter.is_empty():
            candidate_ids = set(
                await metadata_retrieve(
                    metadata_index=self._metadata_index,
                    chunk_metadata=self._chunk_metadata,
                    filters=metadata_filter,
                )
            )
        else:
            candidate_ids = None
        if candidate_ids is not None and not candidate_ids:
            return RetrievalResult(chunks=[], entities=[], relationships=[])
        return await self._retrieve_with_candidates(
            query,
            candidate_ids=candidate_ids,
            query_images=query_images or [],
            top_k=top_k,
        )
```

Core flow:

```python
candidate_ids = None
if metadata_filter is not None and not metadata_filter.is_empty():
    candidate_ids = set(
        await metadata_retrieve(
            metadata_index=self._metadata_index,
            chunk_metadata=self._chunk_metadata,
            filters=metadata_filter,
        )
    )
if candidate_ids is not None and not candidate_ids:
    return RetrievalResult(chunks=[], entities=[], relationships=[])

async with metadata_filter_scope(candidate_ids):
    lightrag_task = asyncio.create_task(self._lightrag.aquery_data(query, param=query_param))
    bm25_task = asyncio.create_task(self._bm25.search(query, candidate_ids=candidate_ids))
    visual_task = asyncio.create_task(self._direct_visual(query_images, candidate_ids))
    lightrag_result, bm25_chunks, visual_chunks = await asyncio.gather(
        lightrag_task, bm25_task, visual_task
    )

chunks = rrf_fuse([lightrag_chunks, bm25_chunks, visual_chunks], k=self._rrf_k)[:top_k]
```

Set `query_param = QueryParam(mode="mix", top_k=self._top_k, chunk_top_k=self._chunk_top_k)` and never expose LightRAG `hybrid` mode.

- [ ] **Step 7: Wire BM25 into service startup**

After LightRAG storages initialize:

```python
from dlightrag.core.retrieval.bm25 import PostgresBM25
from dlightrag.storage.pool import pg_pool

pool = await pg_pool.get()
self._bm25 = PostgresBM25(pool=pool, workspace=config.workspace, top_k=config.bm25_top_k)
await self._bm25.ensure_index(text_config=config.bm25_text_config)
```

Add config fields:

```python
bm25_enabled: bool = True
bm25_top_k: int = 40
bm25_text_config: Literal["simple", "english"] = "simple"
rrf_k: int = 60
direct_visual_top_k: int = 20
```

- [ ] **Step 8: Run Task 7 tests**

Run:

```bash
uv run pytest tests/unit/test_bm25.py tests/unit/test_retrieval_fusion.py tests/unit/test_retriever.py tests/unit/test_filtered_vdb.py -v
```

Expected: selected tests pass.

- [ ] **Step 9: Commit Task 7**

```bash
git add src/dlightrag/core/retrieval src/dlightrag/core/service.py src/dlightrag/config.py tests/unit/test_bm25.py tests/unit/test_retrieval_fusion.py tests/unit/test_retriever.py
git commit -m "feat: add postgres bm25 retrieval fusion"
```

---

### Task 8: Deletion, Reset, And Old Module Cleanup

**Files:**
- Modify: `src/dlightrag/core/ingestion/cleanup.py`
- Modify: `src/dlightrag/core/reset.py`
- Modify: `src/dlightrag/core/service.py`
- Delete: old unified modules no longer imported
- Delete or rewrite tests tied only to removed modules
- Test: `tests/unit/test_cleanup.py`
- Test: `tests/unit/test_service_reset.py`
- Test: `tests/unit/test_manager_reset.py`
- Test: `tests/unit/test_dependency_constraints.py`

- [ ] **Step 1: Write deletion cascade test**

Update `tests/unit/test_cleanup.py`:

```python
async def test_delete_cascades_to_artifacts_and_chunk_metadata() -> None:
    service = _make_service(kv_storage="PGKVStorage")
    service._document_artifacts = AsyncMock()
    service._chunk_metadata = AsyncMock()
    service._hash_index = AsyncMock()
    service._metadata_index = AsyncMock()
    service._lightrag = AsyncMock()
    service._lightrag.doc_status.get_by_id.return_value = {"chunks_list": ["chunk-a"]}

    await delete_files(service, ["doc-1"])

    service._chunk_metadata.delete_doc.assert_awaited_with("doc-1")
    service._document_artifacts.delete_doc.assert_awaited_with("doc-1")
```

- [ ] **Step 2: Run cleanup tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_cleanup.py::test_delete_cascades_to_artifacts_and_chunk_metadata -v
```

Expected: failure because cleanup does not know about the new stores.

- [ ] **Step 3: Update cleanup cascade**

In `src/dlightrag/core/ingestion/cleanup.py`, delete in this order for each full document id:

```python
if service._chunk_metadata is not None:
    await service._chunk_metadata.delete_doc(doc_id)
if service._document_artifacts is not None:
    artifact = await service._document_artifacts.delete_doc(doc_id)
else:
    artifact = None
await _delete_lightrag_document(service.lightrag, doc_id)
if service._metadata_index is not None:
    await service._metadata_index.delete(doc_id)
if service._hash_index is not None:
    await service._hash_index.remove_by_doc_id(doc_id)
```

If `PGHashIndex` lacks `remove_by_doc_id`, add it and update `HashIndexProtocol`.

- [ ] **Step 4: Update reset**

In `src/dlightrag/core/reset.py`, include:

```python
("document_artifacts", service._document_artifacts),
("chunk_metadata", service._chunk_metadata),
```

in the store clear loop. Remove JSON metadata/hash fallback reset branches.

- [ ] **Step 5: Remove old unified modules**

Delete modules that are no longer imported:

```bash
git rm src/dlightrag/unifiedrepresent/engine.py
git rm src/dlightrag/unifiedrepresent/retriever.py
git rm src/dlightrag/unifiedrepresent/lifecycle.py
```

Keep `embedder.py` only if Task 3 still imports it; otherwise delete it too after `rg "unifiedrepresent.embedder"` returns no matches.

- [ ] **Step 6: Remove JSON metadata implementation**

Delete:

```bash
git rm src/dlightrag/storage/json_metadata_index.py
git rm tests/unit/test_json_metadata_index.py
```

Update `src/dlightrag/storage/protocols.py` docstrings to name PG implementations and test doubles only.

- [ ] **Step 7: Run cleanup and import tests**

Run:

```bash
uv run pytest tests/unit/test_cleanup.py tests/unit/test_service_reset.py tests/unit/test_manager_reset.py tests/unit/test_dependency_constraints.py -v
```

Expected: selected tests pass and no removed modules are imported.

- [ ] **Step 8: Commit Task 8**

```bash
git add src/dlightrag tests
git commit -m "refactor: remove legacy ingestion storage paths"
```

---

### Task 9: End-To-End Verification Slice

**Files:**
- Test: `tests/integration/test_lightrag_sidecar_unified.py`
- Test: `tests/integration/test_pg_storage.py`
- Modify: `docs/PG.md`
- Modify: `README.md`

- [ ] **Step 1: Add integration test for required extensions**

Create `tests/integration/test_lightrag_sidecar_unified.py`:

```python
import pytest

from dlightrag.storage.pool import pg_pool
from dlightrag.storage.postgres_version import validate_postgres_major


pytestmark = pytest.mark.integration


async def test_postgres18_extensions_available() -> None:
    pool = await pg_pool.get()
    async with pool.acquire() as conn:
        version = await conn.fetchval("SHOW server_version_num")
        validate_postgres_major(version, required_major=18)
        extensions = {
            row["extname"]
            for row in await conn.fetch(
                "SELECT extname FROM pg_extension WHERE extname = ANY($1)",
                ["vector", "age", "pg_textsearch"],
            )
        }

    assert {"vector", "age", "pg_textsearch"}.issubset(extensions)
```

- [ ] **Step 2: Add document/native image fixture test**

Use mocks for LightRAG parser execution in unit tests and run one integration test only when a real PG18 container is available. The integration assertion must cover:

```python
assert result["source_kind"] in {"document", "native_image"}
assert result["ingest_strategy"] == "lightrag_sidecar_unified"
```

For the native image path, create a 2x2 PNG fixture with Pillow and assert `PGChunkMetadata.get_batch()` returns `embedding_input_kind == "image"`.

- [ ] **Step 3: Update docs**

In `docs/PG.md`, document:

```markdown
## PostgreSQL Version

DlightRAG requires PostgreSQL 18. The Docker image tracks the current PG18
minor release; the baseline verified during the LightRAG-main refactor was
PostgreSQL 18.4.

Required extensions:

- vector
- age
- pg_textsearch

LightRAG's PostgreSQL core uses `vector` and `age`. DlightRAG additionally
requires `pg_textsearch` for BM25 hybrid retrieval. LLM-driven intent-aware
metadata filtering must produce normalized exact/range/JSONB conditions or
explicit filename patterns; the database layer must not add fuzzy matching.
```

In `README.md`, remove RAGAnything references and document LightRAG main dependency plus multimodal embedding requirement.

- [ ] **Step 4: Run full verification**

Run unit tests:

```bash
uv run pytest tests/unit -v
```

Run integration tests with a PG18 container:

```bash
docker compose build postgres
docker compose up -d postgres
uv run pytest tests/integration -v
```

Run import scan:

```bash
rg -n "raganything|RAGAnything|rag_mode|JsonMetadataIndex|Milvus|Qdrant|Neo4J|NetworkX" src tests pyproject.toml config.yaml
```

Expected: unit tests pass; integration tests pass when Docker is available; import scan returns no runtime references except allowed migration notes in docs.

- [ ] **Step 5: Commit Task 9**

```bash
git add tests/integration docs README.md
git commit -m "test: verify lightrag sidecar unified path"
```

---

## Self-Review Checklist

- Spec section 1, dependency and PG18 decisions: Task 1.
- Spec section 4, clean module boundaries: Tasks 2, 4, 5, 7, 8.
- Spec section 5, unified ingestion flow: Task 5.
- Spec section 6, metadata and sidecar provenance: Tasks 4, 6, 8.
- Spec section 7, multimodal embedding requirement: Tasks 1 and 3.
- Spec section 8, retrieval architecture: Tasks 6 and 7.
- Spec section 9, config changes: Task 1.
- Spec section 10, deletion lifecycle: Task 8.
- Spec section 12, test strategy: Tasks 1 through 9.

No implementation task should preserve a product runtime path for RAGAnything, text-only embeddings, non-PostgreSQL storage, LightRAG non-`mix` query modes, or old page-render-first unified ingestion.
