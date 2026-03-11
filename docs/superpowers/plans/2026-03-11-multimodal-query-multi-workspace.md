# Multimodal Query + Multi-Workspace Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multimodal image+text queries (both Caption and Unified RAG modes), image upload UX in the web chat composer, and multi-workspace selection with federated retrieval via the web frontend.

**Architecture:** Backend extends two retrieval paths: Caption mode transparently pipes images through RAGAnything's existing `multimodal_content` parameter; Unified mode implements a dual-path strategy (VLM-enhanced text retrieval + direct visual embedding search) with round-robin merge. Frontend adds a `+` button with drag-and-drop/clipboard paste for images, plus topbar workspace chips backed by slide-out panel for multi-select.

**Tech Stack:** FastAPI, RAGAnything, LightRAG vector stores, PIL/Pillow, Jinja2, htmx, vanilla JS, SSE streaming

**Spec:** `docs/superpowers/specs/2026-03-11-multimodal-query-multi-workspace-design.md`

**Lint config:** Ruff `target-version = "py312"`, `line-length = 100`, `select = ["E", "F", "W", "I", "UP", "B"]`, `ignore = ["E501", "B008"]`. Run `ruff check` and `ruff format --check` after each Python task.

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `src/dlightrag/unifiedrepresent/multimodal_query.py` | `enhance_query_with_images()` — VLM describes images, combines with query for text-path retrieval |
| `tests/test_multimodal_query.py` | Unit tests for `enhance_query_with_images()` |
| `tests/test_visual_retriever_multimodal.py` | Unit tests for `query_by_visual_embedding()` and dual-path merge |
| `tests/test_routes_multimodal.py` | Unit tests for web route multimodal + workspace wiring |

### Modified Files

| File | Changes |
|------|---------|
| `src/dlightrag/unifiedrepresent/retriever.py` | Add `query_by_visual_embedding()`, modify `retrieve()` for dual-path, add `images` param to `answer()`/`answer_stream()` |
| `src/dlightrag/unifiedrepresent/engine.py` | Wire `multimodal_content` through `aretrieve()`/`aanswer()`/`aanswer_stream()`, add `_extract_image_bytes()` helper |
| `src/dlightrag/web/routes.py` | Extract images (base64) + workspaces from request body, construct `multimodal_content`, temp file cleanup in `finally` |
| `src/dlightrag/api/server.py` | Add `multimodal_content` field to `AnswerRequest` with `field_validator`, wire through to manager |
| `src/dlightrag/web/static/citation.js` | Image state management, drag/drop/paste handlers, thumbnail strip rendering, workspace chip state, send images+workspaces in POST |
| `src/dlightrag/web/static/style.css` | `+` button, thumbnail strip, drop overlay, workspace chips, message image thumbnails |
| `src/dlightrag/web/templates/index.html` | Add `+` button to composer, workspace chips in topbar, drop zone overlay, hidden file input |
| `src/dlightrag/web/templates/partials/workspace_list.html` | Convert from form-submit "Switch" to checkbox toggle items for multi-select |

---

## Chunk 1: Backend — Unified Mode Multimodal Query

### Task 1: `enhance_query_with_images()` module

**Files:**
- Create: `src/dlightrag/unifiedrepresent/multimodal_query.py`
- Test: `tests/test_multimodal_query.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_multimodal_query.py`:

```python
"""Tests for enhance_query_with_images()."""

from __future__ import annotations

import pytest

from dlightrag.unifiedrepresent.multimodal_query import enhance_query_with_images


@pytest.mark.asyncio
async def test_single_image():
    """Single image query produces enhanced text with description."""
    calls = []

    async def mock_vlm(prompt, *, image_data=None):
        calls.append({"prompt": prompt, "image_data": image_data})
        return "A bar chart showing Q3 revenue"

    result = await enhance_query_with_images(
        query="Analyze this chart",
        images=[b"fake_png_bytes"],
        vision_model_func=mock_vlm,
    )
    assert "Analyze this chart" in result
    assert "A bar chart showing Q3 revenue" in result
    assert len(calls) == 1
    assert calls[0]["image_data"] == b"fake_png_bytes"


@pytest.mark.asyncio
async def test_multiple_images():
    """Multiple images each get VLM descriptions, all combined."""
    call_count = 0

    async def mock_vlm(prompt, *, image_data=None):
        nonlocal call_count
        call_count += 1
        return f"Description for image {call_count}"

    result = await enhance_query_with_images(
        query="Compare these",
        images=[b"img1", b"img2", b"img3"],
        vision_model_func=mock_vlm,
    )
    assert call_count == 3
    assert "Image 1 content:" in result
    assert "Image 2 content:" in result
    assert "Image 3 content:" in result


@pytest.mark.asyncio
async def test_with_conversation_context():
    """Conversation context is included in enhanced query."""

    async def mock_vlm(prompt, *, image_data=None):
        return "Chart description"

    result = await enhance_query_with_images(
        query="Follow up question",
        images=[b"img"],
        vision_model_func=mock_vlm,
        conversation_context="Previous discussion about Q3 earnings",
    )
    assert "Follow up question" in result
    assert "Previous discussion about Q3 earnings" in result
    assert "Chart description" in result


@pytest.mark.asyncio
async def test_empty_images_returns_query():
    """Empty images list returns original query unchanged."""

    async def mock_vlm(prompt, *, image_data=None):
        raise AssertionError("Should not be called")

    result = await enhance_query_with_images(
        query="Just text query",
        images=[],
        vision_model_func=mock_vlm,
    )
    assert result == "Just text query"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/test_multimodal_query.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `multimodal_query.py`**

Create `src/dlightrag/unifiedrepresent/multimodal_query.py`:

```python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multimodal query enhancement for unified representational RAG.

Uses VLM to describe uploaded images and combines descriptions with
the original text query for enhanced text-path retrieval.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


async def enhance_query_with_images(
    query: str,
    images: list[bytes],
    vision_model_func: Callable[..., Any],
    conversation_context: str | None = None,
) -> str:
    """Enhance a text query with VLM descriptions of uploaded images.

    Calls vision_model_func for each image to generate a text description,
    then combines all descriptions with the original query and optional
    conversation context into an enhanced query string for text-path retrieval.

    This mirrors RAGAnything's ``_process_multimodal_query_content()`` logic
    but is an independent implementation with no dependency on RAGAnything.

    Note: Conversation context truncation (min(last 5 turns, 50K tokens) per spec)
    is the caller's responsibility. This function receives pre-truncated context.
    The web route and API server apply truncation before calling this function.

    Args:
        query: Original text query from user.
        images: List of raw image bytes (max 3).
        vision_model_func: Async VLM callable accepting (prompt, image_data=bytes).
        conversation_context: Optional pre-truncated conversation history string.

    Returns:
        Enhanced query string combining original query, context, and image descriptions.
    """
    if not images:
        return query

    descriptions: list[str] = []
    for img_bytes in images:
        try:
            desc = await vision_model_func(
                "Describe this image in detail for document retrieval.",
                image_data=img_bytes,
            )
            descriptions.append(str(desc))
        except Exception:
            logger.warning("VLM image description failed", exc_info=True)
            descriptions.append("[Image description unavailable]")

    parts: list[str] = [query]
    if conversation_context:
        parts.append(f"Conversation context: {conversation_context}")
    for i, desc in enumerate(descriptions):
        parts.append(f"Image {i + 1} content: {desc}")

    return "\n\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/test_multimodal_query.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint check**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff check src/dlightrag/unifiedrepresent/multimodal_query.py tests/test_multimodal_query.py && ruff format --check src/dlightrag/unifiedrepresent/multimodal_query.py tests/test_multimodal_query.py`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/unifiedrepresent/multimodal_query.py tests/test_multimodal_query.py
git commit -m "feat: add enhance_query_with_images() for unified mode text-path"
```

---

### Task 2: `query_by_visual_embedding()` on VisualRetriever

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/retriever.py`
- Test: `tests/test_visual_retriever_multimodal.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_visual_retriever_multimodal.py`:

```python
"""Tests for VisualRetriever multimodal query capabilities."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from dlightrag.unifiedrepresent.retriever import VisualRetriever


def _make_retriever(*, embedder=None, chunks_vdb=None):
    """Create a VisualRetriever with mocked dependencies."""
    lightrag = MagicMock()
    if chunks_vdb is not None:
        lightrag.chunks_vdb = chunks_vdb
    else:
        lightrag.chunks_vdb = AsyncMock()

    visual_chunks = AsyncMock()
    config = MagicMock()

    retriever = VisualRetriever(
        lightrag=lightrag,
        visual_chunks=visual_chunks,
        config=config,
    )
    if embedder is not None:
        retriever.embedder = embedder
    return retriever


@pytest.mark.asyncio
async def test_query_by_visual_embedding_single_image():
    """Single image produces embedding query against chunks_vdb."""
    mock_embedder = AsyncMock()
    mock_embedder.embed_pages = AsyncMock(
        return_value=np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    )

    mock_vdb = AsyncMock()
    mock_vdb.query = AsyncMock(return_value=[
        {"id": "chunk-1", "distance": 0.9},
        {"id": "chunk-2", "distance": 0.8},
    ])

    retriever = _make_retriever(embedder=mock_embedder, chunks_vdb=mock_vdb)

    # Create a minimal fake image (1x1 PNG)
    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    fake_png = buf.getvalue()

    results = await retriever.query_by_visual_embedding([fake_png], top_k=5)

    mock_embedder.embed_pages.assert_called_once()
    mock_vdb.query.assert_called_once()
    call_kwargs = mock_vdb.query.call_args
    assert call_kwargs.kwargs.get("query_embedding") is not None
    assert len(results) == 2


@pytest.mark.asyncio
async def test_query_by_visual_embedding_multiple_images_round_robin():
    """Multiple images produce round-robin merged results."""
    call_count = 0

    async def mock_embed(images):
        return np.array([[0.1, 0.2, 0.3]] * len(images), dtype=np.float32)

    mock_embedder = AsyncMock()
    mock_embedder.embed_pages = mock_embed

    async def mock_query(query="", top_k=5, query_embedding=None):
        nonlocal call_count
        call_count += 1
        return [{"id": f"chunk-{call_count}-{i}", "distance": 0.9 - i * 0.1} for i in range(2)]

    mock_vdb = AsyncMock()
    mock_vdb.query = mock_query

    retriever = _make_retriever(embedder=mock_embedder, chunks_vdb=mock_vdb)

    import io
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="PNG")
    fake_png = buf.getvalue()

    results = await retriever.query_by_visual_embedding([fake_png, fake_png], top_k=5)

    assert call_count == 2
    # Round-robin: should interleave results from both images
    assert len(results) == 4


@pytest.mark.asyncio
async def test_query_by_visual_embedding_empty_images():
    """Empty images list returns empty results."""
    retriever = _make_retriever()
    results = await retriever.query_by_visual_embedding([], top_k=5)
    assert results == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/test_visual_retriever_multimodal.py -v`
Expected: FAIL (no `query_by_visual_embedding` method / no `embedder` attribute)

- [ ] **Step 3: Implement `query_by_visual_embedding()` and add `embedder` to `VisualRetriever`**

Modify `src/dlightrag/unifiedrepresent/retriever.py`:

3a. Add imports at top (after line 14):

```python
import io

from PIL import Image
```

3b. Add `embedder` parameter to `__init__` (after `path_resolver` param, line 40):

```python
    def __init__(
        self,
        lightrag: LightRAG,
        visual_chunks: Any,
        config: Any,
        vision_model_func: Callable | None = None,
        rerank_model: str | None = None,
        rerank_base_url: str | None = None,
        rerank_api_key: str | None = None,
        rerank_backend: str | None = None,
        path_resolver: PathResolver | None = None,
        embedder: Any | None = None,  # VisualEmbedder for image queries
    ) -> None:
        # ... existing assignments (lines 42-50) unchanged ...
        self.path_resolver = path_resolver
        self.embedder = embedder  # Add this line after self.path_resolver (line 50)
```

3c. Add `query_by_visual_embedding()` method (before the internal helpers section, around line 249):

```python
    async def query_by_visual_embedding(
        self,
        images: list[bytes],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query chunks_vdb using visual embeddings of uploaded images.

        Each image is embedded via VisualEmbedder and queried against the
        chunks vector database directly. Results from multiple images are
        merged via round-robin interleaving.

        Args:
            images: List of raw image bytes.
            top_k: Number of results per image.

        Returns:
            Round-robin merged list of chunk dicts from all images.
        """
        if not images or self.embedder is None:
            return []

        per_image_results: list[list[dict[str, Any]]] = []
        for img_bytes in images:
            try:
                pil_img = Image.open(io.BytesIO(img_bytes))
                vec = await self.embedder.embed_pages([pil_img])
                embedding = vec[0].tolist()

                chunks = await self.lightrag.chunks_vdb.query(
                    query="", top_k=top_k, query_embedding=embedding,
                )
                per_image_results.append(chunks if chunks else [])
            except Exception:
                logger.warning("Visual embedding query failed for image", exc_info=True)
                per_image_results.append([])

        # Round-robin merge across images
        merged: list[dict[str, Any]] = []
        max_len = max((len(r) for r in per_image_results), default=0)
        for i in range(max_len):
            for img_results in per_image_results:
                if i < len(img_results):
                    merged.append(img_results[i])
        return merged
```

3d. Update engine.py to pass `embedder` to retriever (line 81-91):

```python
        self.retriever = VisualRetriever(
            lightrag=lightrag,
            visual_chunks=visual_chunks,
            config=config,
            vision_model_func=vision_model_func,
            rerank_model=(config.effective_rerank_model if config.enable_rerank else None),
            rerank_base_url=(config.rerank_base_url if config.enable_rerank else None),
            rerank_api_key=(config.effective_rerank_api_key if config.enable_rerank else None),
            rerank_backend=(config.rerank_backend if config.enable_rerank else None),
            path_resolver=path_resolver,
            embedder=self.embedder,  # NEW: pass embedder for visual embedding queries
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/test_visual_retriever_multimodal.py -v`
Expected: 3 passed

- [ ] **Step 5: Lint check**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff check src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py && ruff format --check src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py tests/test_visual_retriever_multimodal.py
git commit -m "feat: add query_by_visual_embedding() for unified mode image-path"
```

---

### Task 3: Dual-path wiring in retriever + engine

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/retriever.py` (lines 56-62, 186-192, 218-224)
- Modify: `src/dlightrag/unifiedrepresent/engine.py` (lines 203-262)

- [ ] **Step 1: Modify `VisualRetriever.retrieve()` for dual-path**

In `retriever.py`, modify the `retrieve()` method signature (line 56-62) to accept `images`:

```python
    async def retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
    ) -> dict:
```

Add dual-path logic at the beginning of `retrieve()`, right after the method signature docstring (before "Phase 1: LightRAG retrieval", line 69):

```python
        # Dual-path: if images provided, run VLM-enhanced text query + visual embedding
        if images:
            from dlightrag.unifiedrepresent.multimodal_query import enhance_query_with_images

            if self.vision_model_func:
                enhanced_query = await enhance_query_with_images(
                    query=query,
                    images=images,
                    vision_model_func=self.vision_model_func,
                )
            else:
                enhanced_query = query
                logger.warning("No vision_model_func; using original query for text path")

            # Text path: standard retrieval with enhanced query
            text_result = await self._text_retrieve(
                enhanced_query, mode, top_k, chunk_top_k,
            )

            # Image path: visual embedding direct query
            visual_chunks = await self.query_by_visual_embedding(images, top_k=chunk_top_k)

            # Round-robin merge text chunks + visual chunks
            text_chunks = text_result.get("contexts", {}).get("chunks", [])
            merged_chunks: list[dict[str, Any]] = []
            max_len = max(len(text_chunks), len(visual_chunks))
            for i in range(max_len):
                if i < len(text_chunks):
                    merged_chunks.append(text_chunks[i])
                if i < len(visual_chunks):
                    merged_chunks.append(visual_chunks[i])

            # Cap at chunk_top_k to maintain consistent result count with text-only queries
            text_result["contexts"]["chunks"] = merged_chunks[:chunk_top_k]
            return text_result
```

Rename the existing `retrieve()` body (Phase 1-3 logic) to `_text_retrieve()`:

Move lines 69-184 of the current `retrieve()` method (from `# Phase 1: LightRAG retrieval` through the final `return` statement) verbatim into `_text_retrieve()`, with no other changes:

```python
    async def _text_retrieve(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
    ) -> dict:
        """Run Phase 1-3: LightRAG retrieval -> visual resolution -> reranking."""
        # Phase 1: LightRAG retrieval
        param = QueryParam(
            mode=cast(QueryMode, mode),
            only_need_context=True,
            top_k=top_k,
            enable_rerank=False,
        )
        result = await self.lightrag.aquery_data(query, param=param)

        # ... lines 79-184 of current retrieve() are moved here verbatim ...
        # (Extract chunk_ids, Phase 2 visual resolution, Phase 3 reranking,
        #  build return dict — all unchanged)
        data = result.get("data", {})
        chunk_text: dict[str, str] = {}
        chunk_ids: set[str] = set()
        for chunk in data.get("chunks", []):
            cid = chunk.get("chunk_id")
            if cid:
                chunk_ids.add(cid)
                chunk_text[cid] = chunk.get("content", "")
        for entity in data.get("entities", []):
            source_id = entity.get("source_id", "")
            if source_id:
                for cid in source_id.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)
        for rel in data.get("relationships", []):
            source_id = rel.get("source_id", "")
            if source_id:
                for cid in source_id.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid:
                        chunk_ids.add(cid)
        # Phase 2: Visual resolution
        chunk_id_list = list(chunk_ids)
        logger.info("[Visual Resolve] Looking up %d chunk_ids", len(chunk_id_list))
        visual_data = await self.visual_chunks.get_by_ids(chunk_id_list)
        resolved: dict[str, dict] = {}
        for cid, vd in zip(chunk_id_list, visual_data, strict=False):
            if vd is None:
                continue
            if isinstance(vd, str):
                try:
                    vd = json.loads(vd)
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Skipping unparseable visual_chunk %s", cid)
                    continue
            if isinstance(vd, dict):
                resolved[cid] = vd
        logger.info("[Visual Resolve] Resolved %d/%d", len(resolved), len(chunk_id_list))
        # Phase 3: Visual reranking (unchanged)
        if self.rerank_backend == "llm" and self.vision_model_func and resolved:
            resolved = await self._llm_visual_rerank(query, resolved, chunk_top_k, chunk_text)
        elif self.rerank_base_url and self.rerank_model and resolved:
            resolved = await self._visual_rerank(query, resolved, chunk_top_k, chunk_text)
        else:
            resolved = dict(list(resolved.items())[:chunk_top_k])
        # Build return dict (unchanged from current lines 143-184)
        sources: dict[str, dict] = {}
        media: list[dict] = []
        for cid, vd in resolved.items():
            doc_id = vd.get("full_doc_id", "")
            if doc_id not in sources:
                raw_path = vd.get("file_path", "")
                sources[doc_id] = {
                    "doc_id": doc_id,
                    "title": vd.get("doc_title", ""),
                    "author": vd.get("doc_author", ""),
                    "path": (
                        self.path_resolver.resolve(raw_path)
                        if self.path_resolver and raw_path
                        else raw_path
                    ),
                }
            media.append({
                "chunk_id": cid,
                "page_index": vd.get("page_index"),
                "image_data": vd.get("image_data"),
                "relevance_score": vd.get("relevance_score"),
            })
        return {
            "contexts": {
                "entities": data.get("entities", []),
                "relationships": data.get("relationships", []),
                "chunks": [
                    {
                        "reference_id": cid,
                        "page_index": vd.get("page_index"),
                        "relevance_score": vd.get("relevance_score"),
                        "content": chunk_text.get(cid, ""),
                    }
                    for cid, vd in resolved.items()
                ],
            },
            "raw": {
                "sources": list(sources.values()),
                "media": media,
            },
        }
```

Update `retrieve()` to delegate to `_text_retrieve()` for text-only queries:

```python
        # Text-only: unchanged existing path
        return await self._text_retrieve(query, mode, top_k, chunk_top_k)
```

- [ ] **Step 2: Add `images` parameter to `answer()` and `answer_stream()`**

In `answer()` (line 186-216):

```python
    async def answer(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
    ) -> dict:
        """Run Phase 1-4: retrieve + VLM answer generation."""
        retrieval = await self.retrieve(query, mode, top_k, chunk_top_k, images=images)
        # ... rest unchanged ...
```

In `answer_stream()` (line 218-248):

```python
    async def answer_stream(
        self,
        query: str,
        mode: str = "mix",
        top_k: int = 60,
        chunk_top_k: int = 10,
        images: list[bytes] | None = None,
    ) -> tuple[dict, dict, AsyncIterator[str] | None]:
        """Run Phase 1-3 batch + Phase 4 streaming."""
        retrieval = await self.retrieve(query, mode, top_k, chunk_top_k, images=images)
        # ... rest unchanged ...
```

- [ ] **Step 3: Wire `multimodal_content` through engine methods**

In `engine.py`, add a helper method to the class (before `aretrieve`, after `__init__`):

```python
    @staticmethod
    def _extract_image_bytes(
        multimodal_content: list[dict[str, Any]] | None,
    ) -> list[bytes] | None:
        """Convert multimodal_content dicts to raw image bytes."""
        if not multimodal_content:
            return None
        images: list[bytes] = []
        for item in multimodal_content:
            if item.get("type") != "image":
                continue
            if item.get("img_path"):
                images.append(Path(item["img_path"]).read_bytes())
            elif item.get("data"):
                images.append(base64.b64decode(item["data"]))
        return images or None
```

Update `aretrieve()` (lines 203-223):

```python
    async def aretrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        multimodal_content: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve relevant visual chunks (Phases 1-3)."""
        images = self._extract_image_bytes(multimodal_content)
        result = await self.retriever.retrieve(
            query=query,
            mode=mode or self.config.default_mode,
            top_k=top_k or self.config.top_k,
            chunk_top_k=chunk_top_k or self.config.chunk_top_k,
            images=images,
        )
        return RetrievalResult(
            answer=None,
            contexts=result.get("contexts", {}),
            raw=result.get("raw", {}),
        )
```

Update `aanswer()` (lines 225-245) similarly:

```python
    async def aanswer(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        multimodal_content: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> RetrievalResult:
        """Retrieve and generate answer (Phases 1-4)."""
        images = self._extract_image_bytes(multimodal_content)
        result = await self.retriever.answer(
            query=query,
            mode=mode or self.config.default_mode,
            top_k=top_k or self.config.top_k,
            chunk_top_k=chunk_top_k or self.config.chunk_top_k,
            images=images,
        )
        return RetrievalResult(
            answer=result.get("answer"),
            contexts=result.get("contexts", {}),
            raw=result.get("raw", {}),
        )
```

Update `aanswer_stream()` (lines 247-262) similarly:

```python
    async def aanswer_stream(
        self,
        query: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        multimodal_content: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any], AsyncIterator[str] | None]:
        """Retrieve and stream answer (Phases 1-3 batch + Phase 4 streaming)."""
        images = self._extract_image_bytes(multimodal_content)
        return await self.retriever.answer_stream(
            query=query,
            mode=mode or self.config.default_mode,
            top_k=top_k or self.config.top_k,
            chunk_top_k=chunk_top_k or self.config.chunk_top_k,
            images=images,
        )
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/test_multimodal_query.py tests/test_visual_retriever_multimodal.py -v`
Expected: All pass

- [ ] **Step 5: Lint check**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff check src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py && ruff format --check src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py`
Expected: All passed

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py
git commit -m "feat: wire dual-path multimodal retrieval through engine and retriever"
```

---

## Chunk 2: Backend — Web Route + API Server Wiring

### Task 4: Web route multimodal + multi-workspace support

**Files:**
- Modify: `src/dlightrag/web/routes.py` (lines 87-185)
- Test: `tests/test_routes_multimodal.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_routes_multimodal.py`:

```python
"""Tests for web route multimodal and multi-workspace wiring."""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image


def _make_fake_image_b64() -> str:
    """Create a minimal 1x1 PNG as base64."""
    buf = io.BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_image_b64_decode_roundtrip():
    """Verify base64 encode/decode produces valid image bytes."""
    b64 = _make_fake_image_b64()
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    assert img.size == (1, 1)


def test_multimodal_content_construction():
    """Verify multimodal_content list is built from base64 images."""
    import tempfile
    from pathlib import Path

    b64 = _make_fake_image_b64()
    images_b64 = [b64, b64]

    multimodal_content = []
    tmp_paths = []
    for b64_str in images_b64[:3]:
        img_bytes = base64.b64decode(b64_str)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        tmp_paths.append(tmp.name)
        multimodal_content.append({"type": "image", "img_path": tmp.name})

    assert len(multimodal_content) == 2
    assert all(item["type"] == "image" for item in multimodal_content)
    assert all(Path(item["img_path"]).exists() for item in multimodal_content)

    # Cleanup
    for p in tmp_paths:
        Path(p).unlink(missing_ok=True)


def test_image_count_limit():
    """Only first 3 images are processed."""
    b64 = _make_fake_image_b64()
    images_b64 = [b64] * 5
    limited = images_b64[:3]
    assert len(limited) == 3
```

- [ ] **Step 2: Run tests to verify they pass** (these are helper tests)

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/test_routes_multimodal.py -v`
Expected: All pass (these test the data construction logic, not the route itself)

- [ ] **Step 3: Modify `web/routes.py` answer_stream endpoint**

Add `import base64` to imports (line 5, after `import json`):

```python
import base64
```

Modify `answer_stream()` (lines 87-185). The key changes:

3a. After `query` extraction (line 94), add image and workspace extraction:

```python
    # Extract images (base64 from frontend)
    images_b64: list[str] = body.get("images", [])
    multimodal_content: list[dict[str, Any]] | None = None
    tmp_paths: list[str] = []
    if images_b64:
        multimodal_content = []
        for b64 in images_b64[:3]:  # enforce 3-image limit
            try:
                img_bytes = base64.b64decode(b64)
                if len(img_bytes) > 10 * 1024 * 1024:  # 10MB server-side limit
                    logger.warning("Skipping oversized image (%d bytes)", len(img_bytes))
                    continue
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                tmp.write(img_bytes)
                tmp.close()
                tmp_paths.append(tmp.name)
                multimodal_content.append({"type": "image", "img_path": tmp.name})
            except Exception:
                logger.warning("Failed to decode uploaded image", exc_info=True)
        if not multimodal_content:
            multimodal_content = None

    # Extract workspaces (multi-select from frontend)
    workspaces: list[str] | None = body.get("workspaces")
```

3b. Modify the `manager.aanswer_stream()` call (lines 108-112) to pass multimodal_content and workspaces:

```python
            contexts, raw, token_iter = await manager.aanswer_stream(
                query=query,
                workspace=workspace,
                workspaces=workspaces,
                multimodal_content=multimodal_content,
                **kwargs,
            )
```

3c. Add temp file cleanup in the `event_generator()` function. Wrap the entire try/except in another try/finally (at the end of event_generator, after the outer except):

```python
    async def event_generator() -> AsyncIterator[str]:
        full_answer = ""
        kwargs: dict[str, Any] = {}
        if conversation_history:
            kwargs["conversation_history"] = conversation_history
        try:
            # ... existing streaming logic unchanged ...
        except Exception:
            logger.exception("Answer streaming failed")
            yield "event: error\ndata: Service error. Please try again.\n\n"
        finally:
            # Clean up temp image files
            for p in tmp_paths:
                Path(p).unlink(missing_ok=True)
```

Note: The `tmp_paths` variable is captured by closure from the outer function scope.

- [ ] **Step 4: Run lint check**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff check src/dlightrag/web/routes.py && ruff format --check src/dlightrag/web/routes.py`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/web/routes.py tests/test_routes_multimodal.py
git commit -m "feat: add multimodal image + multi-workspace support to web route"
```

---

### Task 5: API server `AnswerRequest` multimodal_content field

**Files:**
- Modify: `src/dlightrag/api/server.py` (lines 97-104, 182-239)

- [ ] **Step 1: Add `multimodal_content` field to `AnswerRequest`**

Add import at top of file (after existing imports around line 15):

```python
from pydantic import BaseModel, field_validator
```

(Replace existing `from pydantic import BaseModel` import.)

Modify `AnswerRequest` (lines 97-104):

```python
class AnswerRequest(BaseModel):
    query: str
    mode: Literal["local", "global", "hybrid", "naive", "mix"] = "mix"
    stream: bool = False
    top_k: int | None = None
    chunk_top_k: int | None = None
    conversation_history: list[dict[str, str]] | None = None
    workspaces: list[str] | None = None
    multimodal_content: list[dict[str, Any]] | None = None

    @field_validator("multimodal_content")
    @classmethod
    def validate_image_count(cls, v: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if v and len(v) > 3:
            raise ValueError("Maximum 3 multimodal items per query")
        return v
```

- [ ] **Step 2: Wire `multimodal_content` through `/answer` endpoint**

In the `answer()` function (lines 182-239), add `multimodal_content` to kwargs:

```python
    kwargs: dict[str, Any] = {}
    if body.conversation_history:
        kwargs["conversation_history"] = body.conversation_history
    if body.multimodal_content:
        kwargs["multimodal_content"] = body.multimodal_content
```

This applies to both the non-streaming (`manager.aanswer()`) and streaming (`manager.aanswer_stream()`) paths. The `**kwargs` already passes them through.

- [ ] **Step 3: Lint check**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff check src/dlightrag/api/server.py && ruff format --check src/dlightrag/api/server.py`
Expected: All passed

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/api/server.py
git commit -m "feat: add multimodal_content to AnswerRequest with validation"
```

---

## Chunk 3: Frontend — Image Upload UX + Multi-Workspace

### Task 6: HTML structure changes (index.html)

**Files:**
- Modify: `src/dlightrag/web/templates/index.html`

- [ ] **Step 1: Add `+` button to composer, workspace chips to topbar, drop zone, and hidden file input**

Replace the current `index.html` content block with:

```html
{% extends "base.html" %}
{% block content %}
<div class="app">
    <header class="topbar">
        <div class="workspace-chips" id="workspace-chips">
            <span class="ws-chip" data-ws="{{ workspace }}">
                {{ workspace }}
            </span>
            <button class="ws-add-btn" id="ws-add-btn"
                    hx-get="/web/workspaces"
                    hx-target="#panel-content"
                    hx-swap="innerHTML">+ workspace</button>
        </div>
        <div class="topbar-spacer"></div>
        <button class="topbar-btn" id="files-btn"
                hx-get="/web/files"
                hx-target="#panel-content"
                hx-swap="innerHTML">Files</button>
    </header>

    <main class="chat-area" id="chat-area">
        <div id="chat-messages" class="chat-messages">
            <div class="welcome" id="welcome">
                <div class="welcome-brand">DlightRAG</div>
                <div class="welcome-sub">Ask anything about your documents</div>
            </div>
        </div>
    </main>

    <!-- Drop zone overlay (hidden by default) -->
    <div class="drop-overlay" id="drop-overlay">
        <div class="drop-overlay-content">
            Drop images here.<br>Up to 3 images (PNG, JPG, WebP)
        </div>
    </div>

    <div class="composer" id="composer">
        <div class="composer-inner">
            <!-- Thumbnail strip (outside, above composer) -->
            <div class="thumbnail-strip" id="thumbnail-strip"></div>
            <form id="query-form" class="composer-form">
                <button type="button" class="composer-plus" id="composer-plus" aria-label="Attach image">+</button>
                <textarea name="query" placeholder="Inquiry anything" class="composer-input" rows="1" autocomplete="off"></textarea>
                <button type="submit" class="composer-send" aria-label="Send">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
                </button>
            </form>
        </div>
    </div>

    <!-- Hidden file input for image upload -->
    <input type="file" id="image-input" accept="image/*" multiple style="display:none">

    <aside class="panel" id="panel">
        <div class="panel-header">
            <span id="panel-title">SOURCES</span>
            <button class="panel-close" onclick="closePanel()">&#10005;</button>
        </div>
        <div id="panel-content" class="panel-content"></div>
    </aside>
</div>
{% endblock %}
```

Key changes from current:
- Replaced single `workspace-btn` with `workspace-chips` div containing chip spans + `ws-add-btn`
- Added `drop-overlay` div for drag-and-drop visual feedback
- Added `thumbnail-strip` div above the `composer-form`
- Added `composer-plus` button inside the form (before textarea)
- Added hidden `image-input` file input

- [ ] **Step 2: Commit**

```bash
git add src/dlightrag/web/templates/index.html
git commit -m "feat: add image upload button, workspace chips, and drop zone to index.html"
```

---

### Task 7: CSS styles for new components

**Files:**
- Modify: `src/dlightrag/web/static/style.css`

- [ ] **Step 1: Add CSS for composer plus button, thumbnail strip, drop overlay, workspace chips, and message image thumbnails**

Add the following CSS blocks at the end of `style.css` (before the scrollbar section at line 533):

```css
/* === Composer + Button === */

.composer-plus {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    background: transparent;
    border: none;
    color: var(--text-500);
    font-size: 20px;
    line-height: 1;
    cursor: pointer;
    flex-shrink: 0;
    margin-bottom: 2px;
    transition: color 0.15s;
}

.composer-plus:hover {
    color: var(--text-300);
}

/* === Thumbnail Strip === */

.thumbnail-strip {
    display: flex;
    gap: 8px;
    padding: 0 8px 8px 8px;
}

.thumbnail-strip:empty {
    display: none;
}

.thumbnail-item {
    position: relative;
    flex-shrink: 0;
}

.thumbnail-img {
    width: 56px;
    height: 56px;
    border-radius: 10px;
    object-fit: cover;
    border: 1px solid rgba(120, 113, 108, 0.15);
}

.thumbnail-remove {
    position: absolute;
    top: -5px;
    right: -5px;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: var(--bg-elevated);
    border: 1px solid rgba(120, 113, 108, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 9px;
    color: var(--text-400);
    line-height: 1;
    padding: 0;
    transition: color 0.15s, background 0.15s;
}

.thumbnail-remove:hover {
    color: var(--text-100);
    background: var(--bg-surface);
}

/* === Drop Overlay === */

.drop-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(87, 74, 36, 0.08);
    border: 2px dashed var(--gold-400);
    z-index: 150;
    align-items: center;
    justify-content: center;
}

.drop-overlay.active {
    display: flex;
}

.drop-overlay-content {
    color: var(--gold-200);
    font-size: 15px;
    text-align: center;
    line-height: 1.8;
}

/* === Workspace Chips === */

.workspace-chips {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}

.ws-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(87, 74, 36, 0.3);
    border: 1px solid rgba(210, 182, 97, 0.25);
    border-radius: 8px;
    padding: 4px 8px 4px 10px;
    font-size: 12px;
    color: var(--gold-200);
    font-weight: 500;
}

.ws-chip-remove {
    background: none;
    border: none;
    color: var(--gold-400);
    font-size: 10px;
    cursor: pointer;
    padding: 0 2px;
    line-height: 1;
    transition: color 0.15s;
}

.ws-chip-remove:hover {
    color: var(--gold-100);
}

.ws-add-btn {
    display: inline-flex;
    align-items: center;
    background: transparent;
    border: 1px dashed rgba(120, 113, 108, 0.25);
    border-radius: 8px;
    padding: 4px 8px;
    font-size: 12px;
    color: var(--text-500);
    cursor: pointer;
    transition: color 0.15s, border-color 0.15s;
}

.ws-add-btn:hover {
    color: var(--text-300);
    border-color: rgba(120, 113, 108, 0.4);
}

/* === Message Image Thumbnails === */

.message-images {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
    margin-bottom: 6px;
}

.message-img {
    width: 72px;
    height: 52px;
    border-radius: 10px;
    object-fit: cover;
    border: 1px solid rgba(120, 113, 108, 0.12);
}
```

- [ ] **Step 2: Update composer-form CSS to accommodate the + button**

The current `.composer-form` padding-left is `16px` (line 146). Change to `4px` to make room for the + button:

```css
.composer-form {
    position: relative;
    display: flex;
    align-items: flex-end;
    background: var(--bg-surface);
    border: 1px solid rgba(120, 113, 108, 0.2);
    border-radius: var(--radius);
    padding: 4px 4px 4px 4px;
    transition: border-color 0.2s, box-shadow 0.2s;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/dlightrag/web/static/style.css
git commit -m "feat: add CSS for image upload, thumbnails, drop overlay, workspace chips"
```

---

### Task 8: JavaScript — image state, drag/drop/paste, workspace chips, submission

**Files:**
- Modify: `src/dlightrag/web/static/citation.js`

- [ ] **Step 1: Add image state management and helper functions**

After the conversation state section (after line 17), add:

```javascript
// === Image Upload State ===

const MAX_IMAGES = 3;
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB
const pendingImages = []; // [{file, base64, objectUrl}]

function addImage(file) {
    if (pendingImages.length >= MAX_IMAGES) return;
    if (!file.type.startsWith('image/')) return;
    if (file.size > MAX_IMAGE_SIZE) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const base64 = e.target.result.split(',')[1]; // strip data:...;base64,
        const objectUrl = URL.createObjectURL(file);
        pendingImages.push({file, base64, objectUrl});
        renderThumbnails();
    };
    reader.readAsDataURL(file);
}

function removeImage(index) {
    const removed = pendingImages.splice(index, 1);
    if (removed.length > 0) URL.revokeObjectURL(removed[0].objectUrl);
    renderThumbnails();
}

function clearImages() {
    pendingImages.forEach(function(img) { URL.revokeObjectURL(img.objectUrl); });
    pendingImages.length = 0;
    renderThumbnails();
}

function renderThumbnails() {
    const strip = document.getElementById('thumbnail-strip');
    if (!strip) return;
    strip.replaceChildren();
    pendingImages.forEach(function(img, idx) {
        var item = document.createElement('div');
        item.className = 'thumbnail-item';

        var imgEl = document.createElement('img');
        imgEl.className = 'thumbnail-img';
        imgEl.src = img.objectUrl;
        imgEl.alt = 'Attached image';
        item.appendChild(imgEl);

        var btn = document.createElement('button');
        btn.className = 'thumbnail-remove';
        btn.textContent = 'x';
        btn.setAttribute('data-idx', idx);
        btn.addEventListener('click', function() { removeImage(idx); });
        item.appendChild(btn);

        strip.appendChild(item);
    });
}
```

- [ ] **Step 2: Add workspace state management**

After the image upload state section, add:

```javascript
// === Workspace State ===

let activeWorkspaces = [];

function initWorkspaces() {
    // Initialize from the server-rendered workspace chip
    const chips = document.querySelectorAll('#workspace-chips .ws-chip');
    activeWorkspaces = [];
    chips.forEach(function(chip) {
        var ws = chip.getAttribute('data-ws');
        if (ws) activeWorkspaces.push(ws);
    });
    if (activeWorkspaces.length === 0) activeWorkspaces = ['default'];
}

function toggleWorkspace(ws) {
    var idx = activeWorkspaces.indexOf(ws);
    if (idx >= 0) {
        // Don't remove last workspace
        if (activeWorkspaces.length <= 1) return;
        activeWorkspaces.splice(idx, 1);
    } else {
        activeWorkspaces.push(ws);
    }
    renderWorkspaceChips();
    updateWorkspacePanelCheckboxes();
}

function renderWorkspaceChips() {
    var container = document.getElementById('workspace-chips');
    if (!container) return;

    // Preserve the add button
    var addBtn = document.getElementById('ws-add-btn');

    container.replaceChildren();
    activeWorkspaces.forEach(function(ws) {
        var chip = document.createElement('span');
        chip.className = 'ws-chip';
        chip.setAttribute('data-ws', ws);
        chip.textContent = ws;

        // Only show remove button if more than 1 workspace
        if (activeWorkspaces.length > 1) {
            var removeBtn = document.createElement('button');
            removeBtn.className = 'ws-chip-remove';
            removeBtn.textContent = 'x';
            removeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                toggleWorkspace(ws);
            });
            chip.appendChild(removeBtn);
        }
        container.appendChild(chip);
    });

    if (addBtn) container.appendChild(addBtn);
}

function updateWorkspacePanelCheckboxes() {
    var items = document.querySelectorAll('#panel-content .workspace-check-item');
    items.forEach(function(item) {
        var ws = item.getAttribute('data-ws');
        var isActive = activeWorkspaces.indexOf(ws) >= 0;
        item.classList.toggle('selected', isActive);
        var checkbox = item.querySelector('.ws-checkbox');
        if (checkbox) checkbox.classList.toggle('checked', isActive);
    });
}
```

- [ ] **Step 3: Replace the entire `submitQuery()` function**

Replace the entire `submitQuery()` function (lines 128-229 of current citation.js) with this complete replacement:

```javascript
async function submitQuery(query) {
    const chatMessages = document.getElementById('chat-messages');
    const chatArea = document.getElementById('chat-area');

    // Switch to chat mode (bottom input bar)
    activateChatMode();

    // Add user message with optional image thumbnails
    var userWrapper = document.createElement('div');
    userWrapper.style.marginLeft = 'auto';
    userWrapper.style.maxWidth = '85%';

    if (pendingImages.length > 0) {
        var msgImages = document.createElement('div');
        msgImages.className = 'message-images';
        pendingImages.forEach(function(img) {
            var imgEl = document.createElement('img');
            imgEl.className = 'message-img';
            imgEl.src = img.objectUrl;
            imgEl.alt = 'Attached image';
            msgImages.appendChild(imgEl);
        });
        userWrapper.appendChild(msgImages);
    }

    var userDiv = document.createElement('div');
    userDiv.className = 'user-message';
    userDiv.textContent = query;
    userWrapper.appendChild(userDiv);
    chatMessages.appendChild(userWrapper);

    // Add AI message container
    const aiDiv = document.createElement('div');
    aiDiv.className = 'ai-message';

    const headerDiv = document.createElement('div');
    headerDiv.className = 'ai-message-header';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.textContent = '\u25CF';
    headerDiv.appendChild(dot);
    headerDiv.appendChild(document.createTextNode(' DlightRAG'));
    aiDiv.appendChild(headerDiv);

    const contentDiv = document.createElement('div');
    contentDiv.className = 'ai-message-content';
    const streamingDot = document.createElement('span');
    streamingDot.className = 'streaming-dot';
    contentDiv.appendChild(streamingDot);
    aiDiv.appendChild(contentDiv);

    chatMessages.appendChild(aiDiv);
    chatArea.scrollTop = chatArea.scrollHeight;

    // Build history window using server-driven budget
    const historyWindow = getHistoryWindow();

    // Capture image data before clearing
    var imageData = pendingImages.map(function(img) { return img.base64; });
    clearImages();

    let fullAnswer = '';
    let success = false;

    try {
        const response = await fetch('/web/answer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                images: imageData,
                workspaces: activeWorkspaces,
                conversation_history: historyWindow,
            }),
        });

        if (!response.ok) {
            contentDiv.textContent = 'Service error. Please try again.';
            contentDiv.style.color = '#c44';
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let firstToken = true;

        while (true) {
            const result = await reader.read();
            if (result.done) break;

            buffer += decoder.decode(result.value, { stream: true });

            // Parse SSE events from buffer
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            let eventType = '';
            for (let i = 0; i < lines.length; i++) {
                const line = lines[i];
                if (line.startsWith('event: ')) {
                    eventType = line.slice(7).trim();
                } else if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (eventType === 'token') {
                        fullAnswer += data;
                    }
                    handleSSEData(eventType, data, contentDiv, aiDiv, chatArea, firstToken);
                    if (eventType === 'token' && firstToken) firstToken = false;
                    eventType = '';
                }
            }
        }

        success = true;
    } catch (err) {
        contentDiv.textContent = 'Connection error. Please try again.';
        contentDiv.style.color = '#c44';
    }

    // Only record in history on success
    if (success && fullAnswer) {
        conversationHistory.push({role: 'user', content: query});
        conversationHistory.push({role: 'assistant', content: fullAnswer});
    }
}
```

- [ ] **Step 4: Add drag/drop and paste event handlers**

In the `DOMContentLoaded` handler, add after the existing event listeners (before the closing `});`):

```javascript
    // Image upload: + button
    var plusBtn = document.getElementById('composer-plus');
    var imageInput = document.getElementById('image-input');
    if (plusBtn && imageInput) {
        plusBtn.addEventListener('click', function() {
            imageInput.click();
        });
        imageInput.addEventListener('change', function() {
            var files = Array.from(imageInput.files || []);
            files.forEach(function(f) { addImage(f); });
            imageInput.value = '';
        });
    }

    // Drag and drop
    var chatArea = document.getElementById('chat-area');
    var dropOverlay = document.getElementById('drop-overlay');
    var dragCounter = 0;

    document.addEventListener('dragenter', function(e) {
        e.preventDefault();
        if (e.dataTransfer && e.dataTransfer.types.indexOf('Files') >= 0) {
            dragCounter++;
            if (dropOverlay) dropOverlay.classList.add('active');
        }
    });

    document.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            if (dropOverlay) dropOverlay.classList.remove('active');
        }
    });

    document.addEventListener('dragover', function(e) {
        e.preventDefault();
    });

    document.addEventListener('drop', function(e) {
        e.preventDefault();
        dragCounter = 0;
        if (dropOverlay) dropOverlay.classList.remove('active');
        var files = Array.from(e.dataTransfer.files || []);
        files.forEach(function(f) {
            if (f.type.startsWith('image/')) addImage(f);
        });
    });

    // Clipboard paste
    document.addEventListener('paste', function(e) {
        var items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (var i = 0; i < items.length; i++) {
            if (items[i].type.startsWith('image/')) {
                var file = items[i].getAsFile();
                if (file) addImage(file);
            }
        }
    });

    // Workspace: + workspace button opens panel
    var wsAddBtn = document.getElementById('ws-add-btn');
    if (wsAddBtn) {
        wsAddBtn.addEventListener('htmx:afterRequest', function() {
            openPanel('WORKSPACES');
            updateWorkspacePanelCheckboxes();
        });
    }

    // Panel auto-close on chat area click
    if (chatArea) {
        chatArea.addEventListener('click', function() {
            closePanel();
        });
    }

    // Panel auto-close on Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closePanel();
    });

    // Initialize workspace state
    initWorkspaces();
```

Also update the form submit handler to close panel:

In the existing form submit handler (around line 301-308), add `closePanel()`:

```javascript
    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const query = textarea.value.trim();
        if (!query) return;
        textarea.value = '';
        textarea.style.height = 'auto';
        closePanel();
        submitQuery(query);
    });
```

- [ ] **Step 5: Remove old workspace button handler**

Remove the old workspace button event listener (lines 319-324):

```javascript
    // REMOVE this block:
    const wsBtn = document.getElementById('workspace-btn');
    if (wsBtn) {
        wsBtn.addEventListener('htmx:afterRequest', function () {
            openPanel('WORKSPACES');
        });
    }
```

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/web/static/citation.js
git commit -m "feat: add image upload, drag/drop/paste, workspace chips to frontend JS"
```

---

### Task 9: Workspace list template for multi-select

**Files:**
- Modify: `src/dlightrag/web/templates/partials/workspace_list.html`

- [ ] **Step 1: Convert from form-submit to checkbox toggle items**

Replace the entire content of `workspace_list.html` with the simplified version below. The active/selected state is managed entirely client-side by JS (`updateWorkspacePanelCheckboxes()` runs after htmx loads the list), so no `active_workspaces` template variable is needed:

```html
{% for ws in workspaces %}
<div class="workspace-check-item" data-ws="{{ ws }}" onclick="toggleWorkspace('{{ ws }}')">
    <div class="ws-checkbox"></div>
    <span class="ws-check-name">{{ ws }}</span>
</div>
{% endfor %}
```

- [ ] **Step 2: Add CSS for workspace check items**

In `style.css`, replace the existing workspace list styles (lines 440-488) with:

```css
/* === Workspace List (Multi-Select) === */

.workspace-check-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: 10px;
    margin-bottom: 6px;
    cursor: pointer;
    transition: background 0.15s;
}

.workspace-check-item:hover {
    background: var(--bg-elevated);
}

.workspace-check-item.selected {
    background: rgba(87, 74, 36, 0.12);
    border: 1px solid rgba(210, 182, 97, 0.2);
}

.workspace-check-item:not(.selected) {
    border: 1px solid transparent;
}

.ws-checkbox {
    width: 18px;
    height: 18px;
    border-radius: 5px;
    border: 1.5px solid rgba(120, 113, 108, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 11px;
    color: transparent;
    transition: all 0.15s;
}

.ws-checkbox.checked {
    background: rgba(87, 74, 36, 0.5);
    border-color: var(--gold-200);
    color: var(--gold-200);
}

.ws-check-name {
    color: var(--text-300);
    font-size: 13px;
}

.workspace-check-item.selected .ws-check-name {
    color: var(--gold-200);
    font-weight: 500;
}
```

- [ ] **Step 3: Verify web route needs no changes**

The `workspace_list()` function in `web/routes.py` (lines 262-274) needs no changes. It already serves the workspace list via htmx. The JS `updateWorkspacePanelCheckboxes()` runs after htmx loads the content (via `htmx:afterRequest` handler) and applies the correct selected/checked classes client-side.

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/web/templates/partials/workspace_list.html src/dlightrag/web/static/style.css
git commit -m "feat: convert workspace list to multi-select checkboxes"
```

---

## Chunk 4: Final Integration + Lint

### Task 10: Full lint, format check, and integration verification

**Files:** All modified files

- [ ] **Step 1: Run ruff check on all Python files**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff check src/dlightrag/unifiedrepresent/multimodal_query.py src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py src/dlightrag/web/routes.py src/dlightrag/api/server.py`
Expected: All passed

- [ ] **Step 2: Run ruff format check on all Python files**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && ruff format --check src/dlightrag/unifiedrepresent/multimodal_query.py src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py src/dlightrag/web/routes.py src/dlightrag/api/server.py`
Expected: All passed

- [ ] **Step 3: Run all tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/ -v`
Expected: All pass

- [ ] **Step 4: Fix any lint or test failures**

If any failures, fix and re-run until all pass.

- [ ] **Step 5: Final commit (if any fixes)**

```bash
git add -u
git commit -m "fix: resolve lint and test issues from multimodal integration"
```
