# Retrieval & Answer Mechanism

DlightRAG supports two RAG modes: **unified** and **caption**. Both expose the
same public API (`aretrieve`, `aanswer`, `aanswer_stream`) via `RAGService`,
but their internal pipelines differ significantly.

---

## Common Pipeline

All queries share the same two-step pipeline at the `RAGService` level:

```
RAGService.aretrieve / aanswer(query, multimodal_content)
    │
    Step 1  Mode-specific primary retrieval
    │         unified → VisualRetriever
    │         caption → RetrievalEngine (RAGAnything → LightRAG)
    │
    Step 2  MetadataPath — supplementary retrieval
    │         QueryAnalyzer → filename filter → scoped vector search
    │         _resolve_chunk_contexts (text lookup + visual enrichment)
    │
    Merge   _merge_round_robin(Step 1, Step 2) → dedup → top_k
    │
    Answer  Mode-specific answer generation
```

**Step 2** runs identically for all query types and modes. It requires
PostgreSQL backend (`metadata_index`); with JSON/file backends, only Step 1
runs. Step 2 supplements Step 1 with document-level precision when metadata
filters (e.g. filename) are available or auto-detected from the query.

---

## Unified Mode

### Step 1: VisualRetriever

```
VisualRetriever._retrieve()
  Phase 1  LightRAG mix mode (KG traversal + vector search, internal round-robin)
  Phase 2  Visual resolution (chunk_id → page image via visual_chunks KV)
             Text-only chunks (no visual data) are kept with their text content
  Phase 3  Visual reranking (VLM pointwise or API reranker)
```

### Query Types

#### Pure Text Query

```
query → LightRAG mix mode (KG + vector search)
          │
          ├─ entities/relationships (source_id → chunk_ids)
          └─ chunks (text vector match)
                  │
             [Backfill] recover text for entity/relationship chunks
                  │
             Visual Resolve: chunk_id → page image (visual_chunks KV)
                  │
             Reranker: VLM scores "text query vs page image"
```

- No visual embedding search — text embedding already covers chunk retrieval.
- Visual resolve maps chunk_ids to page images for VLM consumption.

#### Image + Text Query (Dual-Path)

```
                     ┌─ Text Path ─────────────────────────┐
query + images ──┬──>│ VLM describes images → enhanced_query │──> LightRAG
                 │   └─────────────────────────────────────┘
                 │   ┌─ Visual Path ───────────────────────┐
                 └──>│ Image embedding → chunks_vdb search  │──> visual chunks
                     └─────────────────────────────────────┘
                                      │
                             Round-robin merge
                                      │
                             Visual Resolve + Rerank
```

- **Text path**: `enhance_query_with_images()` calls VLM to describe each
  image, concatenates descriptions with original query, feeds into LightRAG.
- **Visual path**: Images are embedded via `VisualEmbedder.embed_pages()` and
  queried directly against `chunks_vdb`. The embedding model is multimodal
  (text and images share the same vector space).
- **Merge**: Results from both paths are round-robin interleaved, capped at
  `chunk_top_k`, then visually resolved and reranked.
- **Reranker**: Receives `enhanced_query` (includes image descriptions).

#### Pure Image Query (No Text)

Same as image + text, but `query=""`. The `enhanced_query` consists solely of
VLM-generated image descriptions. Retrieval and reranking work normally;
answer generation uses the image descriptions as the question context.

### Reranking

Two backends, configured via `rerank_backend`:

| Backend | How it works |
|---------|-------------|
| `llm` | VLM pointwise scoring: each chunk scored 0-1 independently. Concurrency capped at `Semaphore(4)`. Uses `response_format=json_object` for structured output. |
| API (`rerank_base_url`) | Calls external OpenAI-compatible `/rerank` endpoint with query + page images. |

Post-rerank filtering removes chunks below `rerank_score_threshold` (default
0.5). Chunks without scores (unranked) are kept.

Score parsing (`_parse_rerank_score`) handles three formats:
1. Plain float: `0.82`
2. JSON: `{"score": 0.82}`
3. Free-form text with trailing number (think-mode models)

### Answer Generation

- **System prompt**: `_ANSWER_CORE` + structured suffix (JSON output for
  providers supporting `response_schema`) or core-only (freetext providers).
  Instructs the LLM to answer based on document text excerpts, page images
  (when available), and knowledge graph context.
- **User prompt**: KG context + document text excerpts + reference list +
  question. `FREETEXT_REMINDER` is appended at end for freetext providers
  to maximize instruction adherence.
- **Messages**: OpenAI multimodal format with inline base64 page images from
  retrieved chunks. The LLM receives both chunk text content (in the user
  prompt as "Document Excerpts") and page images (as `image_url` content
  blocks), providing dual-channel context.

Structured providers return `StructuredAnswer` (JSON with `answer` +
`references` array). Freetext providers return markdown with a
`### References` section parsed by the caller.

---

## Caption Mode

### Step 1: RetrievalEngine

```
RetrievalEngine (captionrag/retrieval.py)
  wraps RAGAnything → LightRAG
  Retrieval/reranking handled by LightRAG internally
```

### Query Types

#### Pure Text Query

```
query → LightRAG.aquery_data() / aquery_llm()
          │
          ├─ KG traversal (entities/relationships)
          └─ Chunk vector search (text embedding)
          │
          Reranking: handled by LightRAG (if rerank.enabled=true)
```

- No visual resolution — caption mode stores OCR captions as text chunks,
  not page images.
- Reranking is LightRAG's built-in reranker (not VLM-based).

#### Multimodal Query (Image + Text)

```
query + multimodal_content
          │
          RAGAnything._process_multimodal_query_content()
          │
          enhanced_query (text with image descriptions)
          │
          LightRAG retrieval (same as pure text)
```

- Images are converted to text descriptions by RAGAnything's internal VLM.
- No visual embedding path — only text-path retrieval.
- No dual-path merging.
- Answer uses `enhanced_query` for both retrieval and LLM generation.

---

## Key Differences

| Aspect | Unified | Caption |
|--------|---------|---------|
| Page representation | Original page images (base64) | OCR text captions |
| Visual vector search | Yes (multimodal embedding) | No |
| Visual resolution | chunk_id → page image via KV (text-only chunks kept) | N/A |
| Reranking | VLM pointwise or external API | LightRAG built-in (text) |
| Answer model | VLM (sees text excerpts + page images) | LLM (text-only) |
| Dual-path retrieval | Yes (text + visual embedding) | No (text only) |
| Structured output | Provider-dependent JSON schema | N/A (LightRAG controls) |
| Citation format | `[n-m]` page-level inline | LightRAG default |
| Step 2 (MetadataPath) | Always available | Requires PostgreSQL backend |
