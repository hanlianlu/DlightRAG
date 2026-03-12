# Raw Elimination & Sources Panel Redesign

## Overview

Eliminate the `raw` field from `RetrievalResult` across all modes (caption + unified), introduce a shared `SourceBuilder` utility for document-level source aggregation, and redesign the Sources panel with hierarchical document-level navigation.

**Breaking change**: API response schema changes — `raw` removed, `sources` added.

## Motivation

1. **`raw` is misnamed and inconsistent** — caption mode fills it with derived sources/media (not raw data); unified mode returns it empty. It's a dumping ground, not a contract.
2. **Scattered source-building logic** — caption mode builds sources in `augment_retrieval_result()`, web routes merge data from `raw`, CitationProcessor builds its own SourceReferences. Three places doing overlapping work.
3. **Federation broken for unified mode** — `merge_results()` reads `raw.sources` / `raw.media` which unified mode never populates.
4. **Flat Sources panel** — current UI is a flat chunk list with visibility filtering. No document hierarchy, no progressive disclosure.

## Design Constraints

- Each mode's retriever is self-contained: fills `contexts` with all chunk-level data
- `CitationProcessor` only handles citation parsing + highlight extraction — no data assembly
- Unified mode must support federation (multi-workspace queries)
- Breaking API change is acceptable (no backward compatibility shim)
- Sources panel: accordion-style document grouping, exclusive chunk display

---

## 1. Data Contract

### 1.1 RetrievalResult

```python
@dataclass
class RetrievalResult:
    answer: str | None = None
    contexts: dict[str, Any] = field(default_factory=dict)
    # raw field removed entirely
```

### 1.1b RetrievalBackend Protocol

The `RetrievalBackend` Protocol's `aanswer_stream()` signature changes from 3-tuple to 2-tuple:

```python
# Before
async def aanswer_stream(...) -> tuple[dict[str, Any], dict[str, Any], AsyncIterator[str] | None]: ...

# After
async def aanswer_stream(...) -> tuple[dict[str, Any], AsyncIterator[str] | None]: ...
```

All implementations (`RetrievalEngine`, `UnifiedRepresentEngine`) and relay layers (`RAGService`, `RAGServiceManager`) must match this signature.

### 1.2 Unified Contexts Schema

Both caption and unified mode retrievers return the same structure:

```python
contexts = {
    "chunks": [
        {
            "chunk_id": str,           # unique chunk identifier
            "reference_id": str,       # document-level ID (shared by chunks from same doc)
            "file_path": str,          # original file path
            "content": str,            # caption: text content; unified: VLM description
            "image_data": str | None,  # caption: None; unified: base64 PNG
            "page_idx": int | None,    # 1-based page number
            "relevance_score": float | None,
            "metadata": {              # optional, mode-specific
                "file_name": str,
                "file_type": str,
            }
        }
    ],
    "entities": [...]        # LightRAG KG entities (unchanged)
    "relationships": [...]   # LightRAG KG relationships (unchanged)
}
```

**Key invariants:**
- `image_data` naturally distinguishes modes: unified has base64 PNG, caption has None
- `reference_id` is the document grouping key used by SourceBuilder
- `page_idx` is 1-based; `None` = unknown (consistent with ChunkSnippet schema and template truthiness checks)

---

## 2. SourceBuilder — Shared Utility

### 2.1 Location

```
src/dlightrag/citations/source_builder.py
```

### 2.2 Interface

```python
def build_sources(
    contexts: dict[str, Any],
    path_resolver: PathResolver | None = None,
) -> list[SourceReference]:
    """Group chunks by reference_id into document-level SourceReferences.

    Chunks are sorted by page_idx within each document.
    SourceReferences are ordered by first appearance in chunks list.
    If path_resolver is provided, resolves file_path to download URLs for SourceReference.url.
    """
```

### 2.3 Logic

1. Iterate `contexts["chunks"]`, group by `reference_id`
2. For each group, build `SourceReference`:
   - `id` = reference_id
   - `title` = filename from first chunk's file_path
   - `path` = first chunk's file_path
   - `type` = first chunk's metadata.file_type (if available)
   - `url` = resolved via `path_resolver(path)` if provided, else `None`
   - `chunks` = list of `ChunkSnippet` sorted by page_idx

### 2.4 Call Sites

| Caller | Usage |
|--------|-------|
| `web/routes.py` | `sources = build_sources(contexts)` → pass to CitationProcessor + template |
| `api/server.py` | `sources = build_sources(contexts)` → return in API response |
| Federation | Not called — merge contexts first, then caller builds sources from merged result |

---

## 3. Retriever Layer Changes

### 3.1 Caption Mode (`captionrag/retrieval.py`)

**Delete:**
- `augment_retrieval_result()` function
- `build_sources_and_media_from_contexts()` function
- All `raw` construction logic
- Helper functions: `_to_download_url()`, `_extract_rag_relative()`, `filter_content_for_snippet()` (URL resolution moves to SourceBuilder via PathResolver)

**Modify `aretrieve()` / `aanswer()` / `aanswer_stream()`:**
- Return `RetrievalResult(answer=..., contexts=...)` without raw
- Ensure chunks in contexts carry: `chunk_id`, `reference_id`, `file_path`, `content`, `page_idx`
- `page_idx` lookup from `text_chunks` KV store (existing logic, relocate inline)
- Caption mode embedded images: the `Image Path:` / `Caption:` regex patterns currently extract image references from chunk content into `raw.media`. After refactor, these remain as inline text within `content` — the web frontend renders them as text. Image file serving continues via the existing `/api/files/` endpoint. No separate `image_data` field for caption mode (that's unified-mode only).

### 3.2 Unified Mode (`unifiedrepresent/retriever.py`)

**Already mostly done (prior session).** Verify:
- Chunks carry `image_data` (base64), `page_idx`, `reference_id`, `file_path`
- No `raw` in return values
- `_build_vlm_messages()` reads from `contexts["chunks"]`
- `answer_stream()` return type changes from `tuple[dict, dict, AsyncIterator]` to `tuple[dict, AsyncIterator]` (drop empty raw dict)

### 3.3 Unified Engine (`unifiedrepresent/engine.py`)

- `aretrieve()` / `aanswer()`: construct `RetrievalResult(answer, contexts)` — no raw
- `aanswer_stream()`: return `(contexts, token_iterator)` — drop raw from tuple

---

## 4. Federation Changes

### 4.1 `core/federation.py` — `merge_results()`

**Delete:**
- All `raw.sources` merging logic
- All `raw.media` merging logic
- `raw` field from merged RetrievalResult

**Keep:**
- Round-robin interleave of `contexts.chunks` (with `_workspace` tagging)
- Round-robin interleave of entities/relationships
- Answer concatenation

**Return:**
```python
RetrievalResult(
    answer=merged_answer,
    contexts={
        "chunks": merged_chunks,
        "entities": merged_entities,
        "relationships": merged_relations,
    },
)
```

Sources are built by the caller after merge via `build_sources(merged_contexts)`.

---

## 5. API Server Changes

### 5.1 Response Schema

**Before (breaking):**
```json
{"answer": "...", "contexts": {...}, "raw": {...}}
```

**After:**
```json
{"answer": "...", "contexts": {...}, "sources": [...]}
```

### 5.2 Endpoints

- `/retrieve`: call `build_sources(result.contexts)`, return `{"answer", "contexts", "sources"}`
- `/answer` (non-streaming): same as above
- `/answer` (streaming SSE): first event sends `{"type": "context", "data": contexts, "sources": [s.model_dump() for s in sources]}` (Pydantic models serialized via `.model_dump()`)

---

## 6. CitationProcessor Simplification

### 6.1 Constructor Change

```python
class CitationProcessor:
    def __init__(self, contexts: list[dict], available_sources: list[SourceReference]):
        ...
```

No change to constructor signature — but `available_sources` now comes from `build_sources()` instead of being built ad-hoc in routes.py.

### 6.2 Method Changes

- `_build_sources()` renamed to `_filter_cited_sources()` — filters the pre-built sources list to only those cited in the answer, and attaches highlight phrases
- No longer constructs SourceReference from scratch — receives complete ones from SourceBuilder

---

## 7. Sources Panel Frontend

### 7.1 Template: Hierarchical Structure

**File:** `src/dlightrag/web/templates/partials/source_panel.html`

```html
{% for src in sources %}
<div class="source-doc" data-ref="{{ src.id }}">
  <div class="source-doc-header" onclick="toggleDoc(this)">
    <span class="collapse-icon">▶</span>
    <span class="source-doc-title">{{ src.title or src.path }}</span>
    <span class="source-doc-count">{{ src.chunks|length }} chunks</span>
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

### 7.2 JS: Interaction Model

**`filterSource(badge)`:**
1. Extract `ref` and optional `chunk` from badge dataset
2. Clone source data into panel (existing pattern)
3. For each `.source-doc`:
   - If `data-ref` matches: expand (show `.source-doc-chunks`, rotate icon to ▼)
   - Else: collapse
4. Within expanded doc:
   - If `chunk` specified: show only that `.source-chunk`, hide others
   - Else (doc-level): show all chunks

**`toggleDoc(header)`:**
- Accordion behavior: expand clicked doc, collapse all others
- Show all chunks within expanded doc

**`showAllSources()`:**
- Expand all docs, show all chunks

### 7.3 CSS

- `.source-doc`: document container with bottom border separator
- `.source-doc-header`: clickable, flex row, hover highlight, cursor pointer
- `.collapse-icon`: inline ▶/▼ with CSS transform rotation transition
- `.source-doc-chunks`: hidden by default, slide-down transition on expand
- `.source-doc.expanded .collapse-icon`: rotate(90deg)
- `.source-doc.expanded .source-doc-chunks`: display block

---

## 8. aanswer_stream() Signature Change

Current unified mode returns 3-tuple `(contexts, raw, token_iterator)`.
Caption mode returns 3-tuple `(contexts, raw, token_iterator)`.

**After:**
Both modes return 2-tuple `(contexts, token_iterator)`.

All layers updated (full call chain):
- `core/retrieval/protocols.py`: `RetrievalBackend.aanswer_stream()` signature → 2-tuple
- `captionrag/retrieval.py`: `RetrievalEngine.aanswer_stream()` → return 2-tuple
- `unifiedrepresent/retriever.py`: `VisualRetriever.answer_stream()` → return 2-tuple
- `unifiedrepresent/engine.py`: `UnifiedRepresentEngine.aanswer_stream()` → return 2-tuple
- `core/service.py`: `RAGService.aanswer_stream()` → relay 2-tuple
- `core/servicemanager.py`: `RAGServiceManager.aanswer_stream()` → relay 2-tuple
- `web/routes.py`: `contexts, token_iter = await manager.aanswer_stream(...)`
- `api/server.py`: `contexts, token_iter = await manager.aanswer_stream(...)`

---

## 9. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `core/retrieval/protocols.py` | Modify | Remove `raw` field from RetrievalResult; update `RetrievalBackend.aanswer_stream()` signature to 2-tuple |
| `citations/source_builder.py` | **Create** | `build_sources()` shared utility |
| `citations/processor.py` | Modify | `_build_sources` → `_filter_cited_sources`, receive pre-built sources |
| `captionrag/retrieval.py` | Modify | Delete `augment_retrieval_result()`, `build_sources_and_media_from_contexts()`, inline page_idx lookup, remove raw |
| `unifiedrepresent/retriever.py` | Modify | Verify schema conformance, confirm no raw |
| `unifiedrepresent/engine.py` | Modify | Drop raw from returns, 2-tuple for stream |
| `core/federation.py` | Modify | Remove raw merging logic |
| `core/service.py` | Modify | Update stream relay to 2-tuple, remove raw passthrough |
| `core/servicemanager.py` | Modify | Update `aanswer_stream()` relay to 2-tuple, remove raw references |
| `api/server.py` | Modify | Call `build_sources()`, return sources instead of raw, serialize via `.model_dump()` |
| `web/routes.py` | Modify | Call `build_sources()`, delete `_extract_available_sources()` and `_contexts_to_flat_list()`, update stream handling to 2-tuple |
| `web/templates/partials/source_panel.html` | Modify | Hierarchical document → chunks template |
| `web/static/citation.js` | Modify | Accordion filterSource + toggleDoc |
| `web/static/style.css` | Modify | Source-doc hierarchy styles |
| `web/deps.py` | Verify | Citation badge rendering (may need minor adjustment) |
| `tests/unit/test_visual_retriever.py` | Modify | Update assertions for no raw |
| `tests/` | Modify/Create | Tests for SourceBuilder, updated integration tests |
