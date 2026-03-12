# Raw Elimination & Sources Panel Redesign — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `raw` field from `RetrievalResult`, unify data flow via shared `SourceBuilder`, and redesign Sources panel with hierarchical accordion navigation.

**Architecture:** Each mode's retriever fills `contexts` completely. A new `build_sources()` utility groups chunks into document-level `SourceReference` objects. `CitationProcessor` is simplified to only filter cited sources and extract highlights. Frontend uses accordion document→chunk hierarchy.

**Tech Stack:** Python (FastAPI, Pydantic), Jinja2, vanilla JS, CSS

**Spec:** `docs/superpowers/specs/2026-03-12-raw-elimination-sources-panel-design.md`

---

## Chunk 1: Core Data Layer

### Task 1: Create SourceBuilder

**Files:**
- Create: `src/dlightrag/citations/source_builder.py`
- Test: `tests/unit/test_source_builder.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_source_builder.py
"""Tests for build_sources() shared utility."""
from __future__ import annotations

import pytest

from dlightrag.citations.schemas import ChunkSnippet, SourceReference
from dlightrag.citations.source_builder import build_sources


def _chunk(
    chunk_id: str,
    reference_id: str,
    file_path: str = "/data/report.pdf",
    content: str = "text",
    image_data: str | None = None,
    page_idx: int | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "reference_id": reference_id,
        "file_path": file_path,
        "content": content,
        "image_data": image_data,
        "page_idx": page_idx,
        "metadata": {"file_name": file_path.rsplit("/", 1)[-1]},
    }


class TestBuildSources:
    def test_groups_by_reference_id(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_idx=1),
                _chunk("c2", "ref-1", page_idx=2),
                _chunk("c3", "ref-2", file_path="/data/chart.xlsx"),
            ],
            "entities": [],
            "relationships": [],
        }
        sources = build_sources(contexts)

        assert len(sources) == 2
        assert sources[0].id == "ref-1"
        assert len(sources[0].chunks) == 2
        assert sources[1].id == "ref-2"
        assert len(sources[1].chunks) == 1

    def test_chunks_sorted_by_page_idx(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c2", "ref-1", page_idx=3),
                _chunk("c1", "ref-1", page_idx=1),
                _chunk("c3", "ref-1", page_idx=2),
            ],
        }
        sources = build_sources(contexts)

        page_order = [c.page_idx for c in sources[0].chunks]
        assert page_order == [1, 2, 3]

    def test_source_title_from_filename(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", file_path="/long/path/report.pdf")]}
        sources = build_sources(contexts)

        assert sources[0].title == "report.pdf"
        assert sources[0].path == "/long/path/report.pdf"

    def test_preserves_image_data(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", image_data="base64data", page_idx=1)]}
        sources = build_sources(contexts)

        assert sources[0].chunks[0].image_data == "base64data"

    def test_empty_contexts(self) -> None:
        assert build_sources({}) == []
        assert build_sources({"chunks": []}) == []

    def test_source_order_by_first_appearance(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-2"),
                _chunk("c2", "ref-1"),
                _chunk("c3", "ref-2"),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].id == "ref-2"
        assert sources[1].id == "ref-1"

    def test_chunk_idx_assigned_sequentially(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_idx=1),
                _chunk("c2", "ref-1", page_idx=2),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].chunks[0].chunk_idx == 0
        assert sources[0].chunks[1].chunk_idx == 1

    def test_none_page_idx_sorted_last(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_idx=None),
                _chunk("c2", "ref-1", page_idx=1),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].chunks[0].page_idx == 1
        assert sources[0].chunks[1].page_idx is None

    def test_with_path_resolver(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", file_path="/data/report.pdf")]}

        class FakeResolver:
            def resolve(self, path: str) -> str:
                return f"https://cdn.example.com/{path}"

        sources = build_sources(contexts, path_resolver=FakeResolver())

        assert sources[0].url == "https://cdn.example.com//data/report.pdf"

    def test_skips_chunks_without_reference_id(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", ""),
                _chunk("c2", "ref-1"),
            ],
        }
        sources = build_sources(contexts)

        assert len(sources) == 1
        assert sources[0].id == "ref-1"
```

- [ ] **Step 2: Run tests — verify they fail**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_source_builder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dlightrag.citations.source_builder'`

- [ ] **Step 3: Implement build_sources**

```python
# src/dlightrag/citations/source_builder.py
"""Shared utility for building document-level SourceReferences from contexts.

Groups chunks by reference_id into SourceReference objects with ChunkSnippets.
Called by both web routes and API server after retrieval.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .schemas import ChunkSnippet, SourceReference
from .utils import filter_content_for_display

if TYPE_CHECKING:
    from dlightrag.core.retrieval.path_resolver import PathResolver


def build_sources(
    contexts: dict[str, Any],
    path_resolver: PathResolver | None = None,
) -> list[SourceReference]:
    """Group chunks by reference_id into document-level SourceReferences.

    Args:
        contexts: Retrieval contexts dict with "chunks" key.
        path_resolver: Optional PathResolver with .resolve(path) -> url method.

    Returns:
        List of SourceReference, ordered by first chunk appearance.
        Chunks within each source are sorted by page_idx.
    """
    chunks = contexts.get("chunks", [])
    if not chunks:
        return []

    # Group by reference_id, preserving first-appearance order
    groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        ref_id = str(chunk.get("reference_id", ""))
        if not ref_id:
            continue
        groups.setdefault(ref_id, []).append(chunk)

    sources: list[SourceReference] = []
    for ref_id, group in groups.items():
        first = group[0]
        file_path = first.get("file_path", "")
        metadata = first.get("metadata", {})

        # Sort by page_idx (None last)
        sorted_chunks = sorted(group, key=lambda c: (c.get("page_idx") is None, c.get("page_idx") or 0))

        snippets = [
            ChunkSnippet(
                chunk_id=c["chunk_id"],
                chunk_idx=idx,
                page_idx=c.get("page_idx"),
                content=filter_content_for_display(c.get("content", "")),
                image_data=c.get("image_data"),
            )
            for idx, c in enumerate(sorted_chunks)
        ]

        url = None
        if path_resolver and file_path:
            try:
                url = path_resolver.resolve(file_path)
            except Exception:
                pass

        sources.append(
            SourceReference(
                id=ref_id,
                title=metadata.get("file_name") or (Path(file_path).name if file_path else None),
                path=file_path,
                type=metadata.get("file_type"),
                url=url,
                chunks=snippets,
            )
        )

    return sources
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_source_builder.py -v`
Expected: ALL PASS

- [ ] **Step 5: Export from citations package**

Add `build_sources` to `src/dlightrag/citations/__init__.py` exports.

- [ ] **Step 6: Commit**

```bash
git add -f src/dlightrag/citations/source_builder.py tests/unit/test_source_builder.py src/dlightrag/citations/__init__.py
git commit -m "feat(citations): add shared build_sources() utility"
```

---

### Task 2: Update RetrievalResult and Protocol

**Files:**
- Modify: `src/dlightrag/core/retrieval/protocols.py`
- Modify: `tests/unit/test_retrieval_protocol.py`

- [ ] **Step 1: Remove `raw` from RetrievalResult**

In `src/dlightrag/core/retrieval/protocols.py:17`, delete:
```python
    raw: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: Update `aanswer_stream` Protocol signature to 2-tuple**

In `src/dlightrag/core/retrieval/protocols.py:44-52`, change return type:
```python
    async def aanswer_stream(
        self,
        query: str,
        *,
        mode: str = "mix",
        top_k: int | None = None,
        chunk_top_k: int | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], AsyncIterator[str] | None]: ...
```

- [ ] **Step 3: Update test_retrieval_protocol.py**

Remove any assertions on `result.raw`. Update any test constructing `RetrievalResult(raw=...)`.

- [ ] **Step 4: Run protocol tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_retrieval_protocol.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/retrieval/protocols.py tests/unit/test_retrieval_protocol.py
git commit -m "refactor(protocols): remove raw field, update aanswer_stream to 2-tuple"
```

---

### Task 3: Update Citation Processor

**Files:**
- Modify: `src/dlightrag/citations/processor.py:54-105`

- [ ] **Step 1: Rename `_build_sources` to `_filter_cited_sources`**

The method now filters the pre-built `available_sources` to only cited ones and attaches chunk snippets + highlights, rather than constructing SourceReferences from scratch.

Logic stays nearly identical — it already receives `available_sources` and looks up chunks from `_enriched_contexts`. The rename signals the changed role: filtering, not building.

```python
    def _filter_cited_sources(self, cited_chunks: dict[str, list[str]]) -> list[SourceReference]:
        """Filter available sources to only those cited, attaching chunk details."""
        # ... (same body as current _build_sources)
```

Update the call in `process()` method at line 46:
```python
        sources = self._filter_cited_sources(cited_chunks)
```

- [ ] **Step 2: Run existing citation tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/ -k citation -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/dlightrag/citations/processor.py
git commit -m "refactor(citations): rename _build_sources to _filter_cited_sources"
```

---

## Chunk 2: Backend Pipeline Changes

### Task 4: Update Unified Mode (retriever + engine)

**Files:**
- Modify: `src/dlightrag/unifiedrepresent/retriever.py:332-380`
- Modify: `src/dlightrag/unifiedrepresent/engine.py:287-307`
- Modify: `tests/unit/test_visual_retriever.py`

- [ ] **Step 1: Update VisualRetriever.answer_stream() to return 2-tuple**

In `src/dlightrag/unifiedrepresent/retriever.py`, change `answer_stream()` (line 340):
```python
    ) -> tuple[dict, AsyncIterator[str] | None]:
```

Line 358: change `return contexts, {}, None` to `return contexts, None`
Line 380: change `return contexts, {}, token_iterator` to `return contexts, token_iterator`

- [ ] **Step 2: Update UnifiedRepresentEngine methods**

In `src/dlightrag/unifiedrepresent/engine.py`:

**`aretrieve()` and `aanswer()`:** Remove `raw` from `RetrievalResult` construction:
```python
    return RetrievalResult(answer=..., contexts=contexts)  # no raw=
```

**`aanswer_stream()` (line 296):** Change return type to 2-tuple:
```python
    ) -> tuple[dict[str, Any], AsyncIterator[str] | None]:
```

And line 300-307: relay 2-tuple from retriever (drop raw from unpack and return).

- [ ] **Step 3: Update test_visual_retriever.py**

Update any tests that unpack 3-tuple to 2-tuple. The `contexts, {}, token_iter` pattern becomes `contexts, token_iter`.

- [ ] **Step 4: Run unified tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_visual_retriever.py tests/unit/test_unified_retriever_schema.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/unifiedrepresent/retriever.py src/dlightrag/unifiedrepresent/engine.py tests/unit/test_visual_retriever.py
git commit -m "refactor(unified): update aanswer_stream to 2-tuple, drop raw"
```

---

### Task 5: Update Caption Mode Retriever

**Files:**
- Modify: `src/dlightrag/captionrag/retrieval.py`
- Modify: `tests/unit/test_retrieval_engine.py`

- [ ] **Step 1: Remove `raw` construction from aretrieve()**

In `aretrieve()` (line 42-100):
- Line 58: change `return RetrievalResult(answer=None, contexts={}, raw={"warning": "..."})` to `return RetrievalResult(answer=None, contexts={})`
- Line 91: delete `raw: dict[str, Any] = retrieval_data if isinstance(retrieval_data, dict) else {}`
- Line 93: change `result = RetrievalResult(answer=None, contexts=contexts, raw=raw)` to `result = RetrievalResult(answer=None, contexts=contexts)`
- Lines 95-100: replace `augment_retrieval_result()` call with inline `page_idx` lookup only

Inline page_idx lookup (extracted from `augment_retrieval_result` lines 393-408):
```python
        # Attach page_idx from text_chunks KV store
        chunk_contexts = contexts.get("chunks", [])
        if lightrag and hasattr(lightrag, "text_chunks") and chunk_contexts:
            chunk_ids = [ctx.get("chunk_id") for ctx in chunk_contexts if ctx.get("chunk_id")]
            if chunk_ids:
                try:
                    chunk_data_list = await lightrag.text_chunks.get_by_ids(chunk_ids)
                    for cid, cdata in zip(chunk_ids, chunk_data_list, strict=True):
                        if cdata and cdata.get("page_idx") is not None:
                            for ctx in chunk_contexts:
                                if ctx.get("chunk_id") == cid:
                                    ctx["page_idx"] = cdata["page_idx"] + 1  # 0→1-based
                except Exception as e:
                    logger.warning(f"Failed to load page_idx: {e}")

        return RetrievalResult(answer=None, contexts=contexts)
```

- [ ] **Step 2: Apply same changes to aanswer()**

Same pattern: remove raw, remove augment call, inline page_idx lookup.

- [ ] **Step 3: Update aanswer_stream() to return 2-tuple**

Change return type (line 177):
```python
    ) -> tuple[dict[str, Any], AsyncIterator[str]]:
```

Replace augment call with inline page_idx lookup. Return `(contexts, token_iter)` instead of `(augmented.contexts, augmented.raw, token_iter)`.

- [ ] **Step 4: Delete unused functions**

Delete from the file:
- `augment_retrieval_result()` (lines 373-425)
- `build_sources_and_media_from_contexts()` (lines 294-370)
- `_to_download_url()` (lines 267-291)
- `_extract_rag_relative()` (lines 247-264)

Update `__all__` to remove deleted functions.

Remove import of `filter_content_for_snippet` from `dlightrag.utils.content_filters` (line 22) if no longer used.
Remove import of `PathResolver` (line 20) if no longer used in this file.

- [ ] **Step 5: Update tests**

In `tests/unit/test_retrieval_engine.py`, remove any assertions on `result.raw`, update any 3-tuple unpacking.

- [ ] **Step 6: Run caption tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_retrieval_engine.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/dlightrag/captionrag/retrieval.py tests/unit/test_retrieval_engine.py
git commit -m "refactor(caption): remove raw, delete augment_retrieval_result, inline page_idx"
```

---

### Task 6: Update Federation

**Files:**
- Modify: `src/dlightrag/core/federation.py`
- Modify: `tests/unit/test_federation.py`

- [ ] **Step 1: Remove raw merging from merge_results()**

In `src/dlightrag/core/federation.py`, delete lines 63-77 (sources + media merging) and line 94-98 (raw in return):

```python
    return RetrievalResult(
        answer=merged_answer,
        contexts={
            "chunks": merged_chunks,
            "entities": merged_entities,
            "relationships": merged_relations,
        },
    )
```

- [ ] **Step 2: Remove raw from federated_retrieve() and federated_answer()**

In `federated_retrieve()`:
- Lines 158-160: empty result returns `RetrievalResult(answer=None, contexts={...})` — no raw
- Lines 171-173: single-workspace path — remove `raw.sources` tagging and `raw.workspaces` assignment
- Lines 198-208: all-fail path — remove raw with errors

Same for `federated_answer()`.

- [ ] **Step 3: Update test_federation.py**

In `tests/unit/test_federation.py`:
- `_make_result()`: remove `raw` parameter, construct without raw
- Delete `test_sources_merged_and_tagged` (line 67-76)
- Delete `test_workspaces_recorded_in_raw` (line 101-107)
- Update `test_empty_results` (line 95-99): remove `raw` assertion
- Update `test_all_workspaces_fail` (line 166-176): remove `raw` assertion
- Update `test_single_workspace_no_federation` (line 114-127): remove `raw` assertion

- [ ] **Step 4: Run federation tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_federation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/federation.py tests/unit/test_federation.py
git commit -m "refactor(federation): remove raw merging, contexts-only"
```

---

### Task 7: Update Service and ServiceManager Relay

**Files:**
- Modify: `src/dlightrag/core/service.py:811-832`
- Modify: `src/dlightrag/core/servicemanager.py:207-230`
- Modify: `tests/unit/test_service.py`

- [ ] **Step 1: Update RAGService.aanswer_stream() signature AND body**

In `src/dlightrag/core/service.py:819`:
```python
    ) -> tuple[dict[str, Any], AsyncIterator[str]]:
```

Also update the internal relay body — change any 3-tuple unpacking:
```python
# Before: contexts, raw, token_iter = await self._engine.aanswer_stream(...)
# After:  contexts, token_iter = await self._engine.aanswer_stream(...)
```

And return 2-tuple:
```python
# Before: return contexts, raw, token_iter
# After:  return contexts, token_iter
```

Also update `aretrieve()` and `aanswer()` to construct `RetrievalResult` without `raw=`.

- [ ] **Step 2: Update RAGServiceManager.aanswer_stream() signature AND body**

In `src/dlightrag/core/servicemanager.py:214`:
```python
    ) -> tuple[dict[str, Any], AsyncIterator[str]]:
```

Also update the internal relay body — change any 3-tuple unpacking:
```python
# Before: contexts, raw, token_iter = await service.aanswer_stream(...)
# After:  contexts, token_iter = await service.aanswer_stream(...)
```

And return 2-tuple. Remove any `raw` references in the method body.

- [ ] **Step 3: Update tests**

Fix any 3-tuple unpacking in `tests/unit/test_service.py`.

- [ ] **Step 4: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/core/service.py src/dlightrag/core/servicemanager.py tests/unit/test_service.py
git commit -m "refactor(service): update aanswer_stream to 2-tuple relay"
```

---

## Chunk 3: Consumer Layer Changes

### Task 8: Update API Server

**Files:**
- Modify: `src/dlightrag/api/server.py:172-249`
- Modify: `tests/unit/test_api_server.py`

- [ ] **Step 1: Add build_sources import**

```python
from dlightrag.citations.source_builder import build_sources
```

- [ ] **Step 2: Update /retrieve endpoint (lines 172-187)**

```python
@app.post("/retrieve", dependencies=[Depends(_verify_auth)])
async def retrieve(body: RetrieveRequest, request: Request) -> dict[str, Any]:
    manager = _get_manager(request)
    result = await manager.aretrieve(
        body.query,
        workspaces=body.workspaces,
        mode=body.mode,
        top_k=body.top_k,
        chunk_top_k=body.chunk_top_k,
    )
    sources = build_sources(result.contexts)
    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "sources": [s.model_dump() for s in sources],
    }
```

- [ ] **Step 3: Update /answer endpoint (lines 190-249)**

Non-streaming:
```python
    result = await manager.aanswer(...)
    sources = build_sources(result.contexts)
    return {
        "answer": result.answer,
        "contexts": result.contexts,
        "sources": [s.model_dump() for s in sources],
    }
```

Streaming: unpack 2-tuple, build sources before streaming starts:
```python
    contexts, token_iter = await manager.aanswer_stream(...)
    sources = build_sources(contexts)

    async def event_generator():
        yield f"data: {json.dumps({'type': 'context', 'data': contexts, 'sources': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"
        ...
```

- [ ] **Step 4: Update tests**

Fix 3-tuple → 2-tuple and `raw` → `sources` assertions in `tests/unit/test_api_server.py`.

- [ ] **Step 5: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/test_api_server.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/api/server.py tests/unit/test_api_server.py
git commit -m "refactor(api): use build_sources, return sources instead of raw"
```

---

### Task 9: Update Web Routes

**Files:**
- Modify: `src/dlightrag/web/routes.py`
- Modify: `tests/unit/web/test_routes.py`

- [ ] **Step 1: Replace helper functions with build_sources**

Delete `_contexts_to_flat_list()` (lines 35-41) and `_extract_available_sources()` (lines 44-64).

Add import:
```python
from dlightrag.citations.source_builder import build_sources
```

- [ ] **Step 2: Update answer_stream() event_generator**

Unpack 2-tuple:
```python
            contexts, token_iter = await manager.aanswer_stream(
                query=query,
                workspace=workspace,
                workspaces=workspaces,
                multimodal_content=multimodal_content,
                **kwargs,
            )
```

Replace citation processing section (lines 159-166):
```python
            # Build sources from contexts
            flat_contexts = []
            for items in contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)

            sources = build_sources(contexts)

            processor = CitationProcessor(
                contexts=flat_contexts,
                available_sources=sources,
            )
            result = processor.process(full_answer)
```

- [ ] **Step 3: Update tests**

Fix any mocked 3-tuple returns in `tests/unit/web/test_routes.py`.

- [ ] **Step 4: Run tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/web/test_routes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/web/routes.py tests/unit/web/test_routes.py
git commit -m "refactor(web): use build_sources, delete _extract_available_sources"
```

---

## Chunk 4: Frontend Redesign

### Task 10: Hierarchical Sources Panel Template

**Files:**
- Modify: `src/dlightrag/web/templates/partials/source_panel.html`

- [ ] **Step 1: Replace flat list with document→chunk hierarchy**

```html
{% for src in sources %}
<div class="source-doc" data-ref="{{ src.id }}">
  <div class="source-doc-header" onclick="toggleDoc(this)">
    <span class="collapse-icon">▶</span>
    <span class="source-doc-title">{{ src.title or src.path }}</span>
    <span class="source-doc-badge">[{{ src.id }}]</span>
    <span class="source-doc-count">{{ src.chunks|length }}</span>
  </div>
  <div class="source-doc-chunks" style="display:none">
    {% for chunk in src.chunks %}
    <div class="source-chunk" data-ref="{{ src.id }}" data-chunk="{{ chunk.chunk_idx }}">
      <div class="source-chunk-header">
        <span class="source-chunk-page">
          {% if chunk.page_idx %}p.{{ chunk.page_idx }}{% else %}#{{ chunk.chunk_idx }}{% endif %}
        </span>
      </div>
      {% if chunk.image_data %}
      <div class="source-chunk-image">
        <img src="data:image/png;base64,{{ chunk.image_data }}"
             alt="Page {{ chunk.page_idx or '' }}" loading="lazy">
      </div>
      {% elif chunk.content %}
      <div class="source-chunk-content">
        {{ chunk.content | highlight_content(chunk.highlight_phrases) }}
      </div>
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>
{% endfor %}
```

- [ ] **Step 2: Commit**

```bash
git add src/dlightrag/web/templates/partials/source_panel.html
git commit -m "feat(web): hierarchical source panel template"
```

---

### Task 11: Accordion JS Interaction

**Files:**
- Modify: `src/dlightrag/web/static/citation.js:187-243`

- [ ] **Step 1: Rewrite filterSource() for accordion behavior**

Replace current `filterSource()` (lines 187-231) with:

```javascript
function filterSource(badge) {
    const ref = badge.dataset.ref;
    const chunk = badge.dataset.chunk;
    const panelContent = document.getElementById('panel-content');

    // Find source data for current answer
    const answerEl = badge.closest('.ai-message');
    if (!answerEl) return;
    const sourceData = answerEl.querySelector('.source-data');
    if (!sourceData) return;

    // Clone into panel
    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    clone.style.display = '';
    while (clone.firstChild) {
        panelContent.appendChild(clone.firstChild);
    }

    // Accordion: expand matching doc, collapse others
    const docs = panelContent.querySelectorAll('.source-doc');
    docs.forEach(function(doc) {
        if (doc.dataset.ref === ref) {
            doc.classList.add('expanded');
            var chunksContainer = doc.querySelector('.source-doc-chunks');
            if (chunksContainer) chunksContainer.style.display = '';

            // If chunk-level, show only that chunk
            if (chunk) {
                var chunks = doc.querySelectorAll('.source-chunk');
                chunks.forEach(function(c) {
                    if (c.dataset.chunk === chunk) {
                        c.style.display = '';
                        c.classList.add('active');
                    } else {
                        c.style.display = 'none';
                    }
                });
            }
        } else {
            doc.classList.remove('expanded');
            var chunksContainer = doc.querySelector('.source-doc-chunks');
            if (chunksContainer) chunksContainer.style.display = 'none';
        }
    });

    openPanel('SOURCES');
}
```

- [ ] **Step 2: Add toggleDoc() function**

```javascript
function toggleDoc(header) {
    const doc = header.closest('.source-doc');
    if (!doc) return;
    const panelContent = doc.closest('#panel-content') || doc.parentElement;

    // Accordion: collapse all others
    var allDocs = panelContent.querySelectorAll('.source-doc');
    allDocs.forEach(function(d) {
        if (d !== doc) {
            d.classList.remove('expanded');
            var chunks = d.querySelector('.source-doc-chunks');
            if (chunks) chunks.style.display = 'none';
        }
    });

    // Toggle this doc
    var isExpanded = doc.classList.contains('expanded');
    doc.classList.toggle('expanded', !isExpanded);
    var chunksContainer = doc.querySelector('.source-doc-chunks');
    if (chunksContainer) {
        chunksContainer.style.display = isExpanded ? 'none' : '';
        // Show all chunks when toggling open
        if (!isExpanded) {
            chunksContainer.querySelectorAll('.source-chunk').forEach(function(c) {
                c.style.display = '';
                c.classList.remove('active');
            });
        }
    }
}
```

- [ ] **Step 3: Update showAllSources()**

```javascript
function showAllSources() {
    const panelContent = document.getElementById('panel-content');
    var docs = panelContent.querySelectorAll('.source-doc');
    docs.forEach(function(doc) {
        doc.classList.add('expanded');
        var chunks = doc.querySelector('.source-doc-chunks');
        if (chunks) {
            chunks.style.display = '';
            chunks.querySelectorAll('.source-chunk').forEach(function(c) {
                c.style.display = '';
                c.classList.remove('active');
            });
        }
    });
}
```

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/web/static/citation.js
git commit -m "feat(web): accordion filterSource + toggleDoc interaction"
```

---

### Task 12: Source Panel CSS

**Files:**
- Modify: `src/dlightrag/web/static/style.css`

- [ ] **Step 1: Add document-level hierarchy styles**

Add after existing `.source-chunk` styles:

```css
/* Document-level source grouping */
.source-doc {
    border-bottom: 1px solid rgba(120, 113, 108, 0.1);
}

.source-doc:last-child {
    border-bottom: none;
}

.source-doc-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    cursor: pointer;
    transition: background 0.15s ease;
}

.source-doc-header:hover {
    background: rgba(161, 98, 7, 0.08);
}

.collapse-icon {
    font-size: 10px;
    transition: transform 0.2s ease;
    color: var(--text-secondary, #a8a29e);
    flex-shrink: 0;
    width: 12px;
}

.source-doc.expanded .collapse-icon {
    transform: rotate(90deg);
}

.source-doc-title {
    flex: 1;
    font-size: 13px;
    font-weight: 500;
    color: var(--text-primary, #e7e5e4);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.source-doc-badge {
    font-size: 11px;
    color: var(--gold-400, #e6a817);
    font-weight: 600;
}

.source-doc-count {
    font-size: 11px;
    color: var(--text-secondary, #a8a29e);
    flex-shrink: 0;
}

.source-doc-count::after {
    content: ' chunks';
}

.source-doc-chunks {
    display: none;
    padding: 0 8px 8px;
}

.source-doc.expanded .source-doc-chunks {
    display: block;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/dlightrag/web/static/style.css
git commit -m "feat(web): accordion source panel CSS styles"
```

---

## Chunk 5: Final Integration

### Task 13: Full Integration Test

- [ ] **Step 1: Run all tests**

Run: `cd /Users/hanlianlyu/Github/DlightRAG && python -m pytest tests/unit/ -v --tb=short`
Expected: ALL PASS

- [ ] **Step 2: Check for leftover `raw` references**

Search for remaining `raw` usages — multiple patterns:
```bash
cd /Users/hanlianlyu/Github/DlightRAG

# Attribute access (.raw)
grep -rn '\.raw\b' src/dlightrag/ --include='*.py' | grep -v '__pycache__'

# Constructor keyword (raw=)
grep -rn '\braw=' src/dlightrag/ --include='*.py' | grep -v '__pycache__'

# Dict literal ("raw":)
grep -rn '"raw"' src/dlightrag/ --include='*.py' | grep -v '__pycache__'

# 3-tuple unpack patterns (contexts, raw, token)
grep -rn 'contexts.*raw.*token' src/dlightrag/ --include='*.py' | grep -v '__pycache__'
```

Fix any remaining references.

- [ ] **Step 3: Verify web/deps.py**

Spec section 9 lists `web/deps.py` as needing verification. Read `src/dlightrag/web/deps.py` and confirm:
- `_citation_badges()` still works — it reads from DOM data-ref/data-chunk attributes which haven't changed
- No references to `raw` field
- `highlight_content` filter is still registered and compatible with new template

No changes expected, but verify.

- [ ] **Step 4: Check imports are clean**

```bash
cd /Users/hanlianlyu/Github/DlightRAG && python -c "from dlightrag.citations.source_builder import build_sources; print('OK')"
cd /Users/hanlianlyu/Github/DlightRAG && python -c "from dlightrag.core.retrieval.protocols import RetrievalResult; r = RetrievalResult(); print(hasattr(r, 'raw'))"
```

Expected: "OK" and "False"

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git add src/dlightrag/ tests/
git commit -m "fix: clean up remaining raw references"
```
