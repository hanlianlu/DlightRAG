# Retrieval Content Pipeline Fix

**Date:** 2026-03-16
**Status:** Draft
**Scope:** Fix retrieval→answer pipeline so LLM receives complete chunk data (text + visual)

## Problem Statement

The current retrieval pipeline has several architectural gaps that cause incorrect or empty answers:

1. **AnswerEngine ignores chunk text content** — `_build_user_prompt()` only includes KG entity/relationship descriptions and a reference file list. The actual `chunk.content` text is never passed to the LLM. The system prompt assumes the LLM reads inlined page images, but this fails when images are missing or the model can't process them effectively.

2. **Unified mode drops non-visual chunks** — `_text_retrieve()` builds the final chunks list only from `resolved` (chunks found in `visual_chunks` KV store). Chunks with text content but no visual data are silently discarded.

3. **Multi-path supplementary chunks lack visual data** — `_resolve_chunk_contexts()` only queries `text_chunks`, never `visual_chunks`. In unified mode, supplementary chunks arrive at the AnswerEngine without `image_data`.

4. **MetadataPath returns all chunks without relevance filtering** — When a metadata filter matches a 100-page document, all 100 chunk_ids are returned with no semantic ranking.

5. **Step 2 VectorPath is redundant** — The global vector search in Step 2 duplicates LightRAG's internal vector search in Step 1, adding no value.

6. **Step 1 + Step 2 merge is unranked** — `_merge_supplementary_chunks()` appends non-duplicate chunks without any ranking consideration.

## Observed Symptom

Query "what is the key info from img 9551 png" returns an answer about PDF Parser Benchmarks (MinerU, Markitdown) instead of the actual image content (aio sandbox inline data analysis). Root cause: KG entities from unrelated documents dominated the prompt, and the correct document's chunk text content was never included.

## Design

### Architecture Overview

```
aretrieve()
    │
    ├── Step 1: LightRAG-based retrieval
    │   ├── Caption mode: LightRAG aquery_data (KG+Vector, internal rerank)
    │   │   → chunks with content, no image_data (caption doesn't store visual)
    │   │
    │   └── Unified mode: VisualRetriever._retrieve()
    │       ├── LightRAG aquery_data (KG+Vector, rerank=False)
    │       ├── Backfill missing chunk text from text_chunks
    │       ├── Visual resolution (visual_chunks → image_data)
    │       ├── Visual reranking (VLM image + text fallback)
    │       └── Include non-visual chunks (text-only, no image_data)
    │           → chunks with content + optional image_data (fully enriched)
    │
    ├── Step 2: MetadataPath supplementary
    │   ├── QueryAnalyzer → MetadataFilter
    │   ├── MetadataPath: filter → doc_ids → chunk_ids
    │   ├── Scoped vector search: native DB filtered search → top_k
    │   └── Resolve chunk contexts:
    │       ├── text_chunks → content
    │       └── unified mode: visual_chunks → image_data
    │           → chunks fully enriched (same quality as Step 1)
    │
    └── Step 3: Merge
        ├── Dedup by chunk_id (Step 1 wins on collision)
        ├── Round-robin interleave (Step1[0], Step2[0], Step1[1], ...)
        └── top_k cutoff
            │
            ▼
      AnswerEngine
      ├── _build_user_prompt():
      │   ├── KG Context (entities + relationships)
      │   ├── Document Excerpts (chunk.content text)  ← NEW
      │   ├── Reference List (filenames + page numbers)
      │   └── Question
      ├── _build_messages():
      │   ├── Inline image_data (if present)
      │   └── Text prompt
      └── LLM call → answer
```

### Key Design Decisions

**Dual-channel content (text + image):** In unified mode, the LLM receives BOTH chunk text content (in the text prompt) and page images (inlined as image_url). Text serves as fallback and supplement — even with perfect images, text content helps the LLM ground its answer and improves citation accuracy.

**Each step enriches its own output:** No separate enrichment step at the end. Step 1 and Step 2 each produce fully-enriched ChunkContext with content + image_data (unified mode). This keeps the pipeline clean — merge only needs to combine and rank.

**Dedup by chunk_id throughout:** LightRAG internally deduplicates its KG + vector dual paths. Step 2 produces its own chunk_ids. Step 3 deduplicates and merges via round-robin interleaving (not RRF — see Change 6 rationale).

**Round-robin merge, not RRF:** Step 2's value is document-level precision (finding the RIGHT file via metadata filter), not chunk-level semantic ranking. The semantic query extracted from a metadata-oriented question (e.g., "what is the key info" from "what is the key info from img 9551 png") is often vague and won't score well against the actual document content. RRF would unfairly penalize Step 2 chunks due to low semantic scores. Round-robin ensures both perspectives (KG/vector relevance from Step 1, document precision from Step 2) get equal representation for the LLM to judge.

### Change 1: Rename `_text_retrieve()` → `_retrieve()`

**File:** `src/dlightrag/unifiedrepresent/retriever.py`

The method performs KG retrieval + visual resolution + visual reranking. The name `_text_retrieve` is misleading. Rename to `_retrieve()`. Update the public `retrieve()` method's internal calls accordingly.

### Change 2: Don't Drop Non-Visual Chunks

**File:** `src/dlightrag/unifiedrepresent/retriever.py`

Current code builds the final chunks list only from `resolved` (chunks found in `visual_chunks`). Fix: build a unified candidate set from ALL chunk_ids:

```python
# Build unified candidate set for reranking
all_candidates: dict[str, dict] = {}

# Visual chunks (have image_data)
for cid, vd in resolved.items():
    all_candidates[cid] = vd

# Text-only chunks (not in visual_chunks but have content)
for cid in chunk_ids:
    if cid not in all_candidates and cid in chunk_text:
        all_candidates[cid] = {}  # No visual data, text-only

# Visual reranking handles both:
# - Chunks with image_data → VLM scores the image
# - Chunks without image_data → VLM scores the text (existing fallback)
```

The existing `_llm_visual_rerank()` already handles text-only fallback: when `img_data` is None, it sends the chunk text content in the prompt instead of an image. No reranker changes needed.

Final chunks list includes both visual and text-only chunks:

```python
"chunks": [
    {
        "chunk_id": cid,
        "content": chunk_text.get(cid, ""),
        "image_data": all_candidates[cid].get("image_data"),  # None for text-only
        "reference_id": ...,
        "file_path": ...,
        "page_idx": ...,
        "relevance_score": ...,
    }
    for cid in reranked_candidates
]
```

### Change 3: MetadataPath Scoped Vector Search

**File:** `src/dlightrag/core/retrieval/metadata_path.py` (+ new helper)

Replace the current "return all chunk_ids" with native DB filtered vector search. This bypasses LightRAG's `BaseVectorStorage.query()` (which doesn't support filters) and calls the underlying DB directly.

```python
async def metadata_retrieve(
    metadata_index, filters, lightrag, rag_mode,
    query: str,          # NEW: needed for scoped search
    chunks_vdb: Any,     # NEW: needed for scoped search
    top_k: int = 20,     # NEW: limit results
) -> list[str]:
    doc_ids = await metadata_index.query(filters)
    all_chunk_ids = []
    for doc_id in doc_ids:
        all_chunk_ids.extend(await _resolve_chunks_for_doc(...))

    if len(all_chunk_ids) <= top_k:
        return all_chunk_ids  # Small set, return all

    # Scoped vector search using native DB filtered search
    return await scoped_vector_search(
        query, all_chunk_ids, chunks_vdb, top_k
    )
```

#### Native Filtered Search Implementation

New helper: `src/dlightrag/core/retrieval/scoped_search.py`

```python
async def scoped_vector_search(
    query: str,
    chunk_ids: list[str],
    chunks_vdb: Any,
    top_k: int = 20,
) -> list[str]:
    """Rank chunk_ids by semantic relevance using native DB filtered search.

    Detects the backend type and uses native filtered vector search:
    - PostgreSQL: WHERE id = ANY($1) ORDER BY embedding <=> query_vec
    - Milvus: search(filter="id in [...]")
    - Qdrant: search(query_filter=HasIdCondition(...))

    Falls back to get_vectors_by_ids + manual cosine similarity
    if backend type is unrecognized.
    """
```

**Cross-modal embedding assumption:** In unified mode, `chunks_vdb` stores visual embeddings (page images embedded via `VisualEmbedder`). The scoped search embeds the text query and compares against these visual vectors. This relies on the multimodal embedding model (e.g., `jina-clip-v2`) producing aligned text/image embeddings in the same vector space. This is the same assumption LightRAG's internal vector search already makes — `_get_vector_context()` in `operate.py` queries `chunks_vdb` with a text query against visual embeddings. If the embedding model lacks cross-modal alignment, scoped search quality degrades but does not break (it falls back to effectively random ranking within the scoped set, which is still better than returning all chunks unranked).

**PostgreSQL implementation** (primary — we already have asyncpg pool):

The vector column name must be read from the PGVectorStorage instance's configuration at runtime rather than hardcoded. LightRAG's PG implementation uses SQL templates with a vector column — the implementation should query the table schema or reference the known attribute from the storage instance.

```sql
SELECT id
FROM {table_name}
WHERE workspace = $1 AND id = ANY($2)
ORDER BY {vector_column} <=> $3::vector
LIMIT $4
```

**Milvus implementation:**

```python
results = client.search(
    collection_name=namespace,
    data=[query_embedding],
    filter=f'id in {chunk_id_list}',
    limit=top_k,
    output_fields=["id"],
)
```

**Qdrant implementation:**

```python
results = client.query_points(
    collection_name=namespace,
    query=query_embedding,
    query_filter=models.Filter(
        must=[models.HasIdCondition(has_id=chunk_ids)]
    ),
    limit=top_k,
)
```

**Fallback (unknown backend):**

Uses `get_vectors_by_ids()` (available on all three LightRAG VDB backends: PGVectorStorage, MilvusVectorDBStorage, QdrantVectorDBStorage) and manual cosine similarity:

```python
vectors = await chunks_vdb.get_vectors_by_ids(chunk_ids)
# Embed query via the VDB's embedding function (EmbeddingFunc wrapper)
query_vec = await chunks_vdb.embedding_func.func([query])
scores = {cid: cosine_similarity(query_vec[0], vec) for cid, vec in vectors.items()}
return sorted(scores, key=scores.get, reverse=True)[:top_k]
```

Note: `chunks_vdb.embedding_func` is a LightRAG `EmbeddingFunc` wrapper — call `.func()` to access the underlying async callable.

### Change 4: Remove Redundant VectorPath from Orchestrator

**File:** `src/dlightrag/core/retrieval/orchestrator.py`

Remove the independent VectorPath dispatch (lines 64-72). Step 1's LightRAG already includes vector search internally. Step 2's MetadataPath now has its own scoped vector search.

The orchestrator simplifies to: run MetadataPath only → return ranked chunk_ids.

Delete `vector_path.py` and remove its import from `orchestrator.py`.

### Change 5: Resolve Visual Chunks in `_resolve_chunk_contexts()`

**File:** `src/dlightrag/core/service.py`

In unified mode, also query `visual_chunks` to populate `image_data`:

```python
async def _resolve_chunk_contexts(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
    # ... existing text_chunks lookup ...

    # Unified mode: also resolve visual data
    if self.config.rag_mode == "unified":
        visual_store = getattr(self.unified, "visual_chunks", None)
        if visual_store:
            visual_data = await visual_store.get_by_ids(chunk_ids)
            # Build lookup dict to avoid zip misalignment
            # (contexts may have fewer entries than chunk_ids if some had no text_chunks data)
            visual_lookup: dict[str, dict] = {}
            for cid, vd in zip(chunk_ids, visual_data):
                if vd and isinstance(vd, dict):
                    visual_lookup[cid] = vd
            for ctx in contexts:
                vd = visual_lookup.get(ctx["chunk_id"])
                if vd:
                    ctx["image_data"] = vd.get("image_data")
                    if vd.get("page_index") is not None:
                        ctx["page_idx"] = vd["page_index"] + 1

    return contexts
```

### Change 6: Merge via Round-Robin Instead of Append

**File:** `src/dlightrag/core/service.py`

Replace `_merge_supplementary_chunks()` (simple append) with round-robin interleaving:

**Rationale:** Step 2's value is document-level precision (metadata filter found the right file). The semantic query is often vague ("what is the key info") and won't match document content well. RRF would penalize Step 2 chunks due to low semantic scores. Round-robin gives both paths equal representation and lets the LLM decide which content is relevant.

```python
@staticmethod
def _merge_round_robin(
    primary: RetrievalResult,
    supplementary: list[dict[str, Any]],
    top_k: int = 20,
) -> None:
    primary_chunks = primary.contexts.get("chunks", [])

    # Dedup: collect Step 2 chunks not already in Step 1
    existing_ids = {c.get("chunk_id") for c in primary_chunks}
    new_supplementary = [c for c in supplementary if c.get("chunk_id") not in existing_ids]

    # Round-robin interleave
    merged: list[dict[str, Any]] = []
    max_len = max(len(primary_chunks), len(new_supplementary))
    for i in range(max_len):
        if i < len(primary_chunks):
            merged.append(primary_chunks[i])
        if i < len(new_supplementary):
            merged.append(new_supplementary[i])

    primary.contexts["chunks"] = merged[:top_k]
```

Note: `_merge_supplementary_chunks` is deleted and replaced by `_merge_round_robin`. The call site in `aretrieve()` (currently line 1090) must be updated accordingly.

### Change 7: Include Chunk Text in Answer Prompt

**File:** `src/dlightrag/core/answer.py`

Add a "Document Excerpts" section to `_build_user_prompt()`:

```python
def _build_user_prompt(self, query: str, contexts: RetrievalContexts) -> str:
    kg_context = self._format_kg_context(contexts)
    excerpts = self._format_chunk_excerpts(contexts)  # NEW
    indexer = self._build_citation_indexer(contexts)
    ref_list = indexer.format_reference_list()

    prompt_parts = [
        f"Knowledge Graph Context:\n{kg_context}",
        f"Document Excerpts:\n{excerpts}",  # NEW
        f"Reference Document List:\n{ref_list}",
        f"Question: {query}",
    ]
    return "\n\n".join(prompt_parts)

@staticmethod
def _format_chunk_excerpts(contexts: RetrievalContexts) -> str:
    """Format chunk text content for the LLM prompt."""
    chunks = contexts.get("chunks", [])
    if not chunks:
        return "No document excerpts available."

    parts: list[str] = []
    for chunk in chunks:
        content = chunk.get("content", "").strip()
        if not content:
            continue
        ref_id = chunk.get("reference_id", "")
        page_idx = chunk.get("page_idx")
        file_path = chunk.get("file_path", "")
        filename = Path(file_path).name if file_path else f"Source {ref_id}"

        if page_idx:
            label = f"[{filename}, Page {page_idx}]"
        else:
            label = f"[{filename}]"
        parts.append(f"{label}\n{content}")

    return "\n\n".join(parts) if parts else "No document excerpts available."
```

### Change 8: Update System Prompt

**File:** `src/dlightrag/unifiedrepresent/prompts.py`

Update `_ANSWER_CORE` to reflect that the LLM now receives text excerpts alongside images:

```
You will receive:
1. Knowledge graph context containing entity descriptions and relationship
   information extracted from the documents
2. Document text excerpts from the most relevant pages/chunks
3. One or more document page images (when available) that are most relevant
4. A hierarchical reference list mapping document sources and their pages/chunks
```

Update instructions to reference both text excerpts and images as sources.

## Files Changed

| File | Change | Lines (est.) |
|------|--------|------|
| `unifiedrepresent/retriever.py` | Rename `_text_retrieve` → `_retrieve`; don't drop non-visual chunks | ~30 |
| `core/retrieval/metadata_path.py` | Add query/chunks_vdb params; call scoped search | ~15 |
| `core/retrieval/scoped_search.py` | **New file**: native filtered vector search per backend | ~120 |
| `core/retrieval/orchestrator.py` | Remove VectorPath; pass query/vdb to MetadataPath | ~15 |
| `core/service.py` | `_resolve_chunk_contexts` visual enrichment; replace `_merge_supplementary_chunks` with `_merge_round_robin`; update `aretrieve()` call site | ~40 |
| `core/answer.py` | `_format_chunk_excerpts`; update `_build_user_prompt` | ~30 |
| `unifiedrepresent/prompts.py` | Update system prompt | ~10 |
| `core/retrieval/vector_path.py` | Delete | ~0 |
| Tests | Update existing + new tests for scoped search, enrichment, prompt | ~150 |

## Dedup Strategy

Deduplication by `chunk_id` occurs at three levels:

1. **Within Step 1**: LightRAG internally deduplicates its KG + vector dual paths
2. **Within Step 2**: MetadataPath produces unique chunk_ids per doc; scoped search preserves uniqueness
3. **Step 3 merge**: Round-robin deduplicates before interleaving — Step 2 chunks already present in Step 1 are excluded, then remaining chunks are interleaved. Chunks found by both paths appear once via Step 1's version (which has reranking scores from visual/LightRAG rerank)

## KG Entity Scoping (Out of Scope)

KG entities/relationships are globally scoped by LightRAG's design. A query about "img 9551 png" may return entities from unrelated documents matching keywords "img" or "png". This is a LightRAG framework limitation. With chunk text content now included in the prompt, the LLM has correct grounding data to override misleading KG entities. A proper fix would require LightRAG-level changes (entity provenance filtering) and is deferred.

## Caption Mode Impact

Caption mode also benefits from Change 7 (chunk text in prompt). Caption mode's `RetrievalEngine.aretrieve()` already populates `chunk.content` via LightRAG's `aquery_data()`. Since both modes flow through the same `AnswerEngine._build_user_prompt()`, caption mode answers will now include document text excerpts — a significant improvement since caption mode has no `image_data` to inline. Chunk ID formats differ between modes (caption: LightRAG's variable-length text chunks; unified: one chunk per page computed from `doc_id:page:N`) but dedup by `chunk_id` works identically for both.

## Testing Strategy

1. **Unit tests for scoped_search.py**: Mock each backend, verify filtered search is called correctly. Edge cases:
   - Empty `chunk_ids` list → return empty
   - `chunk_ids` fewer than `top_k` → return all without search
   - Backend returns fewer results than `top_k`
   - Stale chunk_ids (deleted from VDB) → gracefully skip
   - Large chunk_ids list (10,000+) → verify query batching if needed
2. **Unit tests for _format_chunk_excerpts**: Verify prompt includes chunk content; deduplicate by `(file_path, page_idx)` to avoid repeating same page content
3. **Update test_answer_engine.py**: Verify prompt contains "Document Excerpts" section
4. **Update test for _retrieve rename**: Verify non-visual chunks are included in output
5. **Integration test**: End-to-end query with known document, verify answer uses correct content
