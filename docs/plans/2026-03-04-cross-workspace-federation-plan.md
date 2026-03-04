# Cross-Workspace Federated Retrieval Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable cross-workspace federated retrieval so users can search across multiple LightRAG workspaces in a single query, with round-robin result merging and RBAC hook for future access control.

**Architecture:** Extend `pool.py` into a `WorkspacePool` that manages multiple `RAGService` instances keyed by workspace name. Add a `FederatedRetriever` in `retrieval/federation.py` that orchestrates parallel queries and merges results via round-robin interleaving. Update API/MCP endpoints to accept an optional `workspaces` parameter.

**Tech Stack:** asyncio (stdlib), existing RAGService, existing RetrievalResult dataclass

---

### Task 1: Add WorkspacePool to pool.py

**Files:**
- Modify: `src/dlightrag/pool.py`
- Modify: `tests/unit/test_pool.py`

**Step 1: Add WorkspacePool class and helpers to pool.py**

Add after `reset_shared_rag_service()` (line 186), before `__all__`:

```python
# ---------------------------------------------------------------------------
# WorkspacePool — manages multiple RAGService instances keyed by workspace
# ---------------------------------------------------------------------------

_workspace_services: dict[str, RAGService] = {}
_workspace_lock: asyncio.Lock | None = None


def _get_workspace_lock() -> asyncio.Lock:
    """Get or create the workspace pool lock (must be created within event loop)."""
    global _workspace_lock
    if _workspace_lock is None:
        _workspace_lock = asyncio.Lock()
    return _workspace_lock


async def get_workspace_service(
    workspace: str,
    base_config: DlightragConfig | None = None,
    enable_vlm: bool = True,
    cancel_checker: Callable[[], Awaitable[bool]] | None = None,
    url_transformer: Callable[[str], str] | None = None,
) -> RAGService:
    """Get or create a RAGService for a specific workspace.

    Services are cached per workspace name. All services share the same
    PG connection pool (LightRAG's ClientManager is a process-level singleton).
    """
    if workspace in _workspace_services:
        return _workspace_services[workspace]

    lock = _get_workspace_lock()
    async with lock:
        # Double-check after acquiring lock
        if workspace in _workspace_services:
            return _workspace_services[workspace]

        from dlightrag.config import get_config

        config = base_config or get_config()
        ws_config = config.model_copy(update={"postgres_workspace": workspace})
        svc = await RAGService.create(
            config=ws_config,
            enable_vlm=enable_vlm,
            cancel_checker=cancel_checker,
            url_transformer=url_transformer,
        )
        _workspace_services[workspace] = svc
        logger.info("Created RAGService for workspace '%s'", workspace)
        return svc


async def list_available_workspaces(
    base_config: DlightragConfig | None = None,
) -> list[str]:
    """Query PG for all distinct workspaces that have ingested data.

    Falls back to returning just the default workspace name if PG is not used.
    """
    from dlightrag.config import get_config

    config = base_config or get_config()

    if not config.kv_storage.startswith("PG"):
        return [config.postgres_workspace]

    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=config.postgres_host,
            port=config.postgres_port,
            user=config.postgres_user,
            password=config.postgres_password,
            database=config.postgres_database,
        )
        try:
            rows = await conn.fetch(
                "SELECT DISTINCT workspace FROM dlightrag_file_hashes ORDER BY workspace"
            )
            workspaces = [row["workspace"] for row in rows]
            return workspaces if workspaces else [config.postgres_workspace]
        finally:
            await conn.close()
    except Exception as exc:
        logger.warning("Failed to list workspaces from PG: %s", exc)
        return [config.postgres_workspace]


async def close_workspace_services() -> None:
    """Close all workspace-specific RAGService instances."""
    global _workspace_services
    for ws, svc in _workspace_services.items():
        try:
            await svc.close()
        except Exception:
            logger.warning("Failed to close workspace service '%s'", ws, exc_info=True)
    _workspace_services = {}


def reset_workspace_pool() -> None:
    """Reset workspace pool state. Useful for testing."""
    global _workspace_services, _workspace_lock
    _workspace_services = {}
    _workspace_lock = None
```

Then update `__all__` to add the new public functions:

```python
__all__ = [
    "RAGServiceUnavailableError",
    "close_shared_rag_service",
    "close_workspace_services",
    "get_rag_error_info",
    "get_shared_rag_service",
    "get_workspace_service",
    "is_rag_service_initialized",
    "list_available_workspaces",
    "reset_shared_rag_service",
    "reset_workspace_pool",
]
```

**Step 2: Run existing pool tests to verify no regression**

Run: `uv run python -m pytest tests/unit/test_pool.py -v`
Expected: All 12 tests PASS

**Step 3: Add WorkspacePool tests**

Add to `tests/unit/test_pool.py`, at the end:

```python
from dlightrag.pool import (
    get_workspace_service,
    close_workspace_services,
    list_available_workspaces,
    reset_workspace_pool,
)


@pytest.fixture(autouse=True)
def _reset_workspace_pool():
    """Reset workspace pool state before and after each test."""
    reset_workspace_pool()
    yield
    reset_workspace_pool()


class TestWorkspacePool:
    """Test workspace-keyed RAGService pool."""

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_creates_service_for_workspace(self, mock_create) -> None:
        mock_service = AsyncMock()
        mock_create.return_value = mock_service

        result = await get_workspace_service("project-a")

        assert result is mock_service
        # Verify config was copied with correct workspace
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["config"].postgres_workspace == "project-a"

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_caches_service_per_workspace(self, mock_create) -> None:
        mock_service = AsyncMock()
        mock_create.return_value = mock_service

        svc1 = await get_workspace_service("ws-1")
        svc2 = await get_workspace_service("ws-1")

        assert svc1 is svc2
        assert mock_create.await_count == 1

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_different_workspaces_get_different_services(self, mock_create) -> None:
        mock_create.side_effect = [AsyncMock(), AsyncMock()]

        svc1 = await get_workspace_service("ws-a")
        svc2 = await get_workspace_service("ws-b")

        assert svc1 is not svc2
        assert mock_create.await_count == 2

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_concurrent_creates_only_init_once(self, mock_create) -> None:
        mock_service = AsyncMock()

        async def slow_create(**kwargs):
            await asyncio.sleep(0.05)
            return mock_service

        mock_create.side_effect = slow_create

        results = await asyncio.gather(
            get_workspace_service("ws-x"),
            get_workspace_service("ws-x"),
            get_workspace_service("ws-x"),
        )

        assert mock_create.await_count == 1
        assert all(r is mock_service for r in results)

    @patch("dlightrag.pool.RAGService.create", new_callable=AsyncMock)
    async def test_close_workspace_services(self, mock_create) -> None:
        import dlightrag.pool as pool

        svc_a = AsyncMock()
        svc_b = AsyncMock()
        pool._workspace_services = {"a": svc_a, "b": svc_b}

        await close_workspace_services()

        svc_a.close.assert_awaited_once()
        svc_b.close.assert_awaited_once()
        assert pool._workspace_services == {}

    async def test_list_workspaces_non_pg_returns_default(self) -> None:
        """Non-PG storage returns just the configured workspace."""
        from dlightrag.config import DlightragConfig, set_config

        cfg = DlightragConfig(  # type: ignore[call-arg]
            kv_storage="JsonKVStorage",
            postgres_workspace="myws",
            openai_api_key="test",
        )
        set_config(cfg)

        result = await list_available_workspaces()
        assert result == ["myws"]
```

**Step 4: Run all pool tests**

Run: `uv run python -m pytest tests/unit/test_pool.py -v`
Expected: All tests PASS (existing + new)

**Step 5: Commit**

```bash
git add src/dlightrag/pool.py tests/unit/test_pool.py
git commit -m "feat: add WorkspacePool for multi-workspace RAGService management"
```

---

### Task 2: Add FederatedRetriever with round-robin merge

**Files:**
- Create: `src/dlightrag/retrieval/federation.py`
- Create: `tests/unit/test_federation.py`

**Step 1: Create retrieval/federation.py**

Create `src/dlightrag/retrieval/federation.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Federated retrieval across multiple workspaces.

Orchestrates parallel queries to multiple RAGService instances (one per
workspace) and merges results via round-robin interleaving to ensure
fair representation from each workspace.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any, Literal

from dlightrag.retrieval.engine import RetrievalResult

logger = logging.getLogger(__name__)

# Type alias for RBAC hook: given requested workspaces, return accessible subset
WorkspaceFilter = Callable[[list[str]], Awaitable[list[str]]]


def merge_results(
    results: list[RetrievalResult],
    workspaces: list[str],
    chunk_top_k: int | None = None,
) -> RetrievalResult:
    """Merge multiple RetrievalResults via round-robin interleaving.

    Each chunk/entity/relation is tagged with ``_workspace`` to identify
    its source. Results are interleaved: ws_a[0], ws_b[0], ws_a[1], ws_b[1]...
    then truncated to ``chunk_top_k``.

    Args:
        results: One RetrievalResult per workspace (same order as workspaces).
        workspaces: Workspace names corresponding to each result.
        chunk_top_k: Maximum number of chunks in merged output. None = no limit.
    """
    # Tag and collect chunks from each workspace
    per_ws_chunks: list[list[dict[str, Any]]] = []
    for result, ws in zip(results, workspaces):
        chunks = result.contexts.get("chunks", [])
        tagged = []
        for chunk in chunks:
            c = dict(chunk)
            c["_workspace"] = ws
            tagged.append(c)
        per_ws_chunks.append(tagged)

    # Round-robin interleave
    merged_chunks: list[dict[str, Any]] = []
    max_len = max((len(cs) for cs in per_ws_chunks), default=0)
    for i in range(max_len):
        for ws_chunks in per_ws_chunks:
            if i < len(ws_chunks):
                merged_chunks.append(ws_chunks[i])

    # Truncate
    if chunk_top_k is not None:
        merged_chunks = merged_chunks[:chunk_top_k]

    # Merge sources with workspace tag
    merged_sources: list[dict[str, Any]] = []
    for result, ws in zip(results, workspaces):
        for source in result.raw.get("sources", []):
            s = dict(source)
            s["_workspace"] = ws
            merged_sources.append(s)

    # Merge media with workspace tag
    merged_media: list[dict[str, Any]] = []
    for result, ws in zip(results, workspaces):
        for media in result.raw.get("media", []):
            m = dict(media)
            m["_workspace"] = ws
            merged_media.append(m)

    # Merge entities/relations (round-robin, same as chunks)
    merged_entities = _round_robin_merge_key(results, workspaces, "entities")
    merged_relations = _round_robin_merge_key(results, workspaces, "relationships")

    # Build merged answer (concatenate non-None answers)
    answers = [r.answer for r in results if r.answer]
    merged_answer = "\n\n---\n\n".join(answers) if answers else None

    return RetrievalResult(
        answer=merged_answer,
        contexts={
            "chunks": merged_chunks,
            "entities": merged_entities,
            "relationships": merged_relations,
        },
        raw={
            "sources": merged_sources,
            "media": merged_media,
            "workspaces": workspaces,
        },
    )


def _round_robin_merge_key(
    results: list[RetrievalResult],
    workspaces: list[str],
    key: str,
) -> list[dict[str, Any]]:
    """Round-robin merge a specific context key across results."""
    per_ws: list[list[dict[str, Any]]] = []
    for result, ws in zip(results, workspaces):
        items = result.contexts.get(key, [])
        tagged = []
        for item in items:
            d = dict(item)
            d["_workspace"] = ws
            tagged.append(d)
        per_ws.append(tagged)

    merged: list[dict[str, Any]] = []
    max_len = max((len(items) for items in per_ws), default=0)
    for i in range(max_len):
        for ws_items in per_ws:
            if i < len(ws_items):
                merged.append(ws_items[i])
    return merged


async def federated_retrieve(
    query: str,
    workspaces: list[str],
    get_service: Callable[[str], Awaitable[Any]],
    *,
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    workspace_filter: WorkspaceFilter | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated retrieval across multiple workspaces.

    Args:
        query: The search query.
        workspaces: List of workspace names to search.
        get_service: Async callable that returns a RAGService for a workspace name.
        mode: LightRAG query mode.
        top_k: Per-workspace top_k for vector search.
        chunk_top_k: Final merged chunk count limit.
        workspace_filter: Optional RBAC filter — given requested workspaces,
            returns the accessible subset. Default: all workspaces accessible.
        **kwargs: Additional kwargs passed to each RAGService.aretrieve().
    """
    # Apply RBAC filter if provided
    if workspace_filter is not None:
        workspaces = await workspace_filter(workspaces)

    if not workspaces:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={"sources": [], "media": [], "workspaces": []},
        )

    # Single workspace — no federation overhead
    if len(workspaces) == 1:
        svc = await get_service(workspaces[0])
        result = await svc.aretrieve(
            query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
        )
        # Tag chunks with workspace
        for chunk in result.contexts.get("chunks", []):
            chunk["_workspace"] = workspaces[0]
        for source in result.raw.get("sources", []):
            source["_workspace"] = workspaces[0]
        result.raw["workspaces"] = workspaces
        return result

    # Parallel queries
    async def _query_workspace(ws: str) -> RetrievalResult | Exception:
        try:
            svc = await get_service(ws)
            return await svc.aretrieve(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
            )
        except Exception as exc:
            logger.warning("Federated query failed for workspace '%s': %s", ws, exc)
            return exc

    raw_results = await asyncio.gather(*[_query_workspace(ws) for ws in workspaces])

    # Filter out failed workspaces
    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    for ws, result in zip(workspaces, raw_results):
        if isinstance(result, Exception):
            continue
        successful_results.append(result)
        successful_workspaces.append(ws)

    if not successful_results:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={"sources": [], "media": [], "workspaces": [], "errors": [str(r) for r in raw_results]},
        )

    return merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)


async def federated_answer(
    query: str,
    workspaces: list[str],
    get_service: Callable[[str], Awaitable[Any]],
    *,
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix",
    top_k: int | None = None,
    chunk_top_k: int | None = None,
    workspace_filter: WorkspaceFilter | None = None,
    **kwargs: Any,
) -> RetrievalResult:
    """Execute federated answer (retrieve + LLM) across multiple workspaces.

    Same as federated_retrieve but calls aanswer() on each service.
    """
    if workspace_filter is not None:
        workspaces = await workspace_filter(workspaces)

    if not workspaces:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={"sources": [], "media": [], "workspaces": []},
        )

    if len(workspaces) == 1:
        svc = await get_service(workspaces[0])
        result = await svc.aanswer(
            query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
        )
        for chunk in result.contexts.get("chunks", []):
            chunk["_workspace"] = workspaces[0]
        for source in result.raw.get("sources", []):
            source["_workspace"] = workspaces[0]
        result.raw["workspaces"] = workspaces
        return result

    async def _query_workspace(ws: str) -> RetrievalResult | Exception:
        try:
            svc = await get_service(ws)
            return await svc.aanswer(
                query=query, mode=mode, top_k=top_k, chunk_top_k=chunk_top_k, **kwargs
            )
        except Exception as exc:
            logger.warning("Federated answer failed for workspace '%s': %s", ws, exc)
            return exc

    raw_results = await asyncio.gather(*[_query_workspace(ws) for ws in workspaces])

    successful_results: list[RetrievalResult] = []
    successful_workspaces: list[str] = []
    for ws, result in zip(workspaces, raw_results):
        if isinstance(result, Exception):
            continue
        successful_results.append(result)
        successful_workspaces.append(ws)

    if not successful_results:
        return RetrievalResult(
            answer=None,
            contexts={"chunks": [], "entities": [], "relationships": []},
            raw={"sources": [], "media": [], "workspaces": [], "errors": [str(r) for r in raw_results]},
        )

    return merge_results(successful_results, successful_workspaces, chunk_top_k=chunk_top_k)
```

**Step 2: Update retrieval/__init__.py exports**

Add to `src/dlightrag/retrieval/__init__.py`:

```python
from dlightrag.retrieval.federation import (
    WorkspaceFilter,
    federated_answer,
    federated_retrieve,
    merge_results,
)
```

And add them to `__all__`:

```python
__all__ = [
    "EnhancedRAGAnything",
    "RetrievalResult",
    "WorkspaceFilter",
    "augment_retrieval_result",
    "federated_answer",
    "federated_retrieve",
    "merge_results",
]
```

**Step 3: Create tests/unit/test_federation.py**

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for cross-workspace federated retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.retrieval.engine import RetrievalResult
from dlightrag.retrieval.federation import (
    federated_answer,
    federated_retrieve,
    merge_results,
)


def _make_result(
    chunks: list[dict] | None = None,
    sources: list[dict] | None = None,
    answer: str | None = None,
) -> RetrievalResult:
    """Helper to create a RetrievalResult with given data."""
    return RetrievalResult(
        answer=answer,
        contexts={
            "chunks": chunks or [],
            "entities": [],
            "relationships": [],
        },
        raw={
            "sources": sources or [],
            "media": [],
        },
    )


class TestMergeResults:
    """Test round-robin merge logic."""

    def test_round_robin_interleaves_chunks(self) -> None:
        r1 = _make_result(chunks=[{"id": "a1"}, {"id": "a2"}, {"id": "a3"}])
        r2 = _make_result(chunks=[{"id": "b1"}, {"id": "b2"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])
        chunk_ids = [c["id"] for c in merged.contexts["chunks"]]

        assert chunk_ids == ["a1", "b1", "a2", "b2", "a3"]

    def test_chunks_tagged_with_workspace(self) -> None:
        r1 = _make_result(chunks=[{"id": "c1"}])
        r2 = _make_result(chunks=[{"id": "c2"}])

        merged = merge_results([r1, r2], ["legal", "finance"])

        assert merged.contexts["chunks"][0]["_workspace"] == "legal"
        assert merged.contexts["chunks"][1]["_workspace"] == "finance"

    def test_chunk_top_k_truncates(self) -> None:
        r1 = _make_result(chunks=[{"id": f"a{i}"} for i in range(10)])
        r2 = _make_result(chunks=[{"id": f"b{i}"} for i in range(10)])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"], chunk_top_k=5)

        assert len(merged.contexts["chunks"]) == 5

    def test_sources_merged_and_tagged(self) -> None:
        r1 = _make_result(sources=[{"id": "s1", "title": "Doc A"}])
        r2 = _make_result(sources=[{"id": "s2", "title": "Doc B"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])
        sources = merged.raw["sources"]

        assert len(sources) == 2
        assert sources[0]["_workspace"] == "ws-a"
        assert sources[1]["_workspace"] == "ws-b"

    def test_answers_concatenated(self) -> None:
        r1 = _make_result(answer="Answer from A")
        r2 = _make_result(answer="Answer from B")

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert "Answer from A" in merged.answer
        assert "Answer from B" in merged.answer

    def test_none_answers_skipped(self) -> None:
        r1 = _make_result(answer=None)
        r2 = _make_result(answer="Only B answered")

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert merged.answer == "Only B answered"

    def test_empty_results(self) -> None:
        merged = merge_results([], [])
        assert merged.contexts["chunks"] == []
        assert merged.raw["sources"] == []
        assert merged.answer is None

    def test_workspaces_recorded_in_raw(self) -> None:
        r1 = _make_result()
        r2 = _make_result()

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert merged.raw["workspaces"] == ["ws-a", "ws-b"]


class TestFederatedRetrieve:
    """Test federated_retrieve orchestration."""

    async def test_single_workspace_no_federation(self) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = _make_result(
            chunks=[{"id": "c1"}], sources=[{"id": "s1"}]
        )

        async def get_svc(ws: str):
            return mock_svc

        result = await federated_retrieve("test query", ["ws-only"], get_svc)

        mock_svc.aretrieve.assert_awaited_once()
        assert result.contexts["chunks"][0]["_workspace"] == "ws-only"
        assert result.raw["workspaces"] == ["ws-only"]

    async def test_multi_workspace_parallel(self) -> None:
        svc_a = AsyncMock()
        svc_a.aretrieve.return_value = _make_result(chunks=[{"id": "a1"}])
        svc_b = AsyncMock()
        svc_b.aretrieve.return_value = _make_result(chunks=[{"id": "b1"}])

        services = {"ws-a": svc_a, "ws-b": svc_b}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_retrieve("query", ["ws-a", "ws-b"], get_svc)

        assert len(result.contexts["chunks"]) == 2
        assert result.contexts["chunks"][0]["_workspace"] == "ws-a"
        assert result.contexts["chunks"][1]["_workspace"] == "ws-b"

    async def test_failed_workspace_excluded(self) -> None:
        svc_ok = AsyncMock()
        svc_ok.aretrieve.return_value = _make_result(chunks=[{"id": "ok1"}])

        svc_fail = AsyncMock()
        svc_fail.aretrieve.side_effect = RuntimeError("DB down")

        services = {"ws-ok": svc_ok, "ws-fail": svc_fail}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_retrieve("query", ["ws-ok", "ws-fail"], get_svc)

        assert len(result.contexts["chunks"]) == 1
        assert result.contexts["chunks"][0]["_workspace"] == "ws-ok"

    async def test_all_workspaces_fail(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.side_effect = RuntimeError("fail")

        async def get_svc(ws: str):
            return svc

        result = await federated_retrieve("query", ["ws-a", "ws-b"], get_svc)

        assert result.contexts["chunks"] == []
        assert "errors" in result.raw

    async def test_workspace_filter_rbac(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.return_value = _make_result(chunks=[{"id": "c1"}])

        async def get_svc(ws: str):
            return svc

        async def only_allow_a(requested: list[str]) -> list[str]:
            return [ws for ws in requested if ws == "ws-a"]

        result = await federated_retrieve(
            "query", ["ws-a", "ws-b"], get_svc, workspace_filter=only_allow_a
        )

        # Only ws-a should be queried (single workspace fast path)
        svc.aretrieve.assert_awaited_once()
        assert result.contexts["chunks"][0]["_workspace"] == "ws-a"

    async def test_workspace_filter_denies_all(self) -> None:
        svc = AsyncMock()

        async def get_svc(ws: str):
            return svc

        async def deny_all(requested: list[str]) -> list[str]:
            return []

        result = await federated_retrieve(
            "query", ["ws-a"], get_svc, workspace_filter=deny_all
        )

        svc.aretrieve.assert_not_awaited()
        assert result.contexts["chunks"] == []

    async def test_empty_workspaces_list(self) -> None:
        async def get_svc(ws: str):
            raise AssertionError("Should not be called")

        result = await federated_retrieve("query", [], get_svc)

        assert result.contexts["chunks"] == []


class TestFederatedAnswer:
    """Test federated_answer orchestration."""

    async def test_single_workspace_uses_aanswer(self) -> None:
        mock_svc = AsyncMock()
        mock_svc.aanswer.return_value = _make_result(
            chunks=[{"id": "c1"}], answer="The answer"
        )

        async def get_svc(ws: str):
            return mock_svc

        result = await federated_answer("query", ["ws-only"], get_svc)

        mock_svc.aanswer.assert_awaited_once()
        assert result.answer == "The answer"

    async def test_multi_workspace_merges_answers(self) -> None:
        svc_a = AsyncMock()
        svc_a.aanswer.return_value = _make_result(answer="Answer A")
        svc_b = AsyncMock()
        svc_b.aanswer.return_value = _make_result(answer="Answer B")

        services = {"a": svc_a, "b": svc_b}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_answer("query", ["a", "b"], get_svc)

        assert "Answer A" in result.answer
        assert "Answer B" in result.answer
```

**Step 4: Run federation tests**

Run: `uv run python -m pytest tests/unit/test_federation.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/dlightrag/retrieval/federation.py src/dlightrag/retrieval/__init__.py tests/unit/test_federation.py
git commit -m "feat: add FederatedRetriever with round-robin merge"
```

---

### Task 3: Update API endpoints for federated retrieval

**Files:**
- Modify: `src/dlightrag/api/server.py`

**Step 1: Add `workspaces` to request models and update handlers**

In `src/dlightrag/api/server.py`:

Add import at top (after existing imports):

```python
from dlightrag.pool import get_workspace_service, list_available_workspaces
from dlightrag.retrieval.federation import federated_answer, federated_retrieve
```

Update `RetrieveRequest` (line 94):

```python
class RetrieveRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    workspaces: list[str] | None = None
```

Update `AnswerRequest` (line 101):

```python
class AnswerRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    top_k: int | None = None
    chunk_top_k: int | None = None
    conversation_history: list[dict[str, str]] | None = None
    workspaces: list[str] | None = None
```

Update the `/retrieve` handler (line 161) to support federated mode:

```python
@app.post("/retrieve", dependencies=[Depends(_verify_auth)])
async def retrieve(body: RetrieveRequest) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    if body.workspaces and len(body.workspaces) > 1:
        result = await federated_retrieve(
            query=body.query,
            workspaces=body.workspaces,
            get_service=get_workspace_service,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
        )
    elif body.workspaces and len(body.workspaces) == 1:
        service = await get_workspace_service(body.workspaces[0])
        result = await service.aretrieve(
            query=body.query,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
        )
    else:
        service = await _get_rag_service()
        result = await service.aretrieve(
            query=body.query,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
        )

    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "raw": result.raw,
    }
```

Update the `/answer` handler (line 180) similarly:

```python
@app.post("/answer", dependencies=[Depends(_verify_auth)])
async def answer(body: AnswerRequest) -> dict[str, Any]:
    """RAG query with LLM-generated answer and structured results."""
    kwargs: dict[str, Any] = {}
    if body.conversation_history:
        kwargs["conversation_history"] = body.conversation_history

    if body.workspaces and len(body.workspaces) > 1:
        result = await federated_answer(
            query=body.query,
            workspaces=body.workspaces,
            get_service=get_workspace_service,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            **kwargs,
        )
    elif body.workspaces and len(body.workspaces) == 1:
        service = await get_workspace_service(body.workspaces[0])
        result = await service.aanswer(
            query=body.query,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            **kwargs,
        )
    else:
        service = await _get_rag_service()
        result = await service.aanswer(
            query=body.query,
            mode=body.mode,
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            **kwargs,
        )

    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "raw": result.raw,
    }
```

Add the `/workspaces` endpoint (after `/health` handler):

```python
@app.get("/workspaces", dependencies=[Depends(_verify_auth)])
async def workspaces() -> dict[str, Any]:
    """List all available workspaces."""
    ws_list = await list_available_workspaces()
    return {"workspaces": ws_list}
```

Also update `close_shared_rag_service` import to include `close_workspace_services`, and add cleanup in the lifespan:

```python
from dlightrag.pool import (
    RAGServiceUnavailableError,
    close_shared_rag_service,
    close_workspace_services,
    get_shared_rag_service,
    get_workspace_service,
    is_rag_service_initialized,
    list_available_workspaces,
)
```

In the lifespan `asynccontextmanager`, add `await close_workspace_services()` alongside the existing `await close_shared_rag_service()`.

**Step 2: Run existing API tests**

Run: `uv run python -m pytest tests/unit/test_api_server.py -v`
Expected: All existing tests PASS (workspaces=None → unchanged behavior)

**Step 3: Commit**

```bash
git add src/dlightrag/api/server.py
git commit -m "feat: add workspaces param and /workspaces endpoint to API"
```

---

### Task 4: Update MCP server for federated retrieval

**Files:**
- Modify: `src/dlightrag/mcp/server.py`

**Step 1: Add `workspaces` to MCP tool schemas and handlers**

In `src/dlightrag/mcp/server.py`:

Add import:

```python
from dlightrag.pool import get_workspace_service, list_available_workspaces
from dlightrag.retrieval.federation import federated_answer, federated_retrieve
```

Update the `retrieve` tool schema (in `list_tools()`) to add `workspaces`:

```python
Tool(
    name="retrieve",
    description="...",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "mode": {"type": "string", "enum": [...], "default": "mix"},
            "top_k": {"type": "integer", "description": "..."},
            "workspaces": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Workspace names to search. Omit to search default workspace.",
            },
        },
        "required": ["query"],
    },
)
```

Do the same for the `answer` tool schema.

Add a new `list_workspaces` tool:

```python
Tool(
    name="list_workspaces",
    description="List all available workspaces with ingested data.",
    inputSchema={"type": "object", "properties": {}},
)
```

Update the `retrieve` handler in `call_tool()`:

```python
if name == "retrieve":
    ws_list = arguments.get("workspaces")
    if ws_list and len(ws_list) > 1:
        result = await federated_retrieve(
            query=arguments["query"],
            workspaces=ws_list,
            get_service=get_workspace_service,
            mode=arguments.get("mode", "mix"),
            top_k=arguments.get("top_k"),
        )
    elif ws_list and len(ws_list) == 1:
        service = await get_workspace_service(ws_list[0])
        result = await service.aretrieve(
            query=arguments["query"],
            mode=arguments.get("mode", "mix"),
            top_k=arguments.get("top_k"),
        )
    else:
        result = await service.aretrieve(
            query=arguments["query"],
            mode=arguments.get("mode", "mix"),
            top_k=arguments.get("top_k"),
        )
    # ... same response formatting
```

Apply the same pattern to the `answer` handler.

Add the `list_workspaces` handler:

```python
if name == "list_workspaces":
    import json
    ws_list = await list_available_workspaces()
    return [TextContent(type="text", text=json.dumps({"workspaces": ws_list}, indent=2))]
```

**Step 2: Run full test suite**

Run: `uv run python -m pytest tests/unit/ -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/dlightrag/mcp/server.py
git commit -m "feat: add workspaces param and list_workspaces tool to MCP"
```

---

### Task 5: Run full test suite and verify backward compatibility

**Step 1: Run all unit tests**

Run: `uv run python -m pytest tests/unit/ -v`
Expected: All tests PASS

**Step 2: Verify backward compatibility**

Verify these scenarios manually or via tests:
- `POST /retrieve {"query": "test"}` → works exactly as before (no `workspaces` field)
- `POST /retrieve {"query": "test", "workspaces": null}` → same as above
- `POST /retrieve {"query": "test", "workspaces": ["default"]}` → single workspace, no federation
- `POST /retrieve {"query": "test", "workspaces": ["a", "b"]}` → federated retrieval

**Step 3: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix: address review feedback for federated retrieval"
```
