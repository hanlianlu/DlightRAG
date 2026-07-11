# Retrieval And Answer

This page is for maintainers and advanced users who need to understand how
queries become contexts, answers, sources, and citations. It owns retrieval and
answer behavior. Interface payloads live in [interfaces.md](interfaces.md);
configuration fields live in [configuration.md](configuration.md);
runtime ownership lives in [architecture.md](architecture.md).

DlightRAG exposes one runtime path: LightRAG main is the base graph/vector
engine, always queried in `mix` mode, while DlightRAG adds metadata
management, optional direct multimodal image search, PostgreSQL BM25, RRF
fusion, reranking, citations, and answer generation.

## Ingestion Shape

```text
source file
  -> LightRAG parser/routing
       sidecar-backed text, tables, equations, and images when available;
       LightRAG raw parser route otherwise
  -> LightRAG ingest
       chunks, entities, relationships, graph, vectors, doc status
  -> image vector alignment when direct image embedding is active
       successful LightRAG drawing multimodal chunks keep their VLM text,
       sidecar provenance, BM25, and KG identity; DlightRAG overwrites the
       existing chunk vector with the raw image embedding
  -> DlightRAG metadata/BM25 layer
       declared metadata index, in-filter scope, pg_textsearch BM25
```

All source files that LightRAG can ingest, including native image files, go
through LightRAG parser/routing. Tables, equations, text, and document-derived
image sidecars stay aligned with the LightRAG document record.
LightRAG raw-route documents can have no sidecar artifacts; those documents
still participate in LightRAG text/KG/vector retrieval but do not receive
image-vector alignment.
With DlightRAG's default
`docx:native-iteP,md:native-iteP,textpack:native-iteP,*:mineru-iteP` rules,
this is a defensive path rather than the normal document parser route.

Successful drawing sidecars have one canonical chunk identity. LightRAG's
multimodal semantic chunk owns `llm_analyze_result` text and exposes it through
`text_chunks`, BM25, and KG extraction. DlightRAG then overwrites that existing
chunk's vector with the raw image embedding. It does not create a second
visual-only chunk, so the same VLM description is not exposed twice as independent
retrieved evidence. When `embedding.input_modality` resolves to text, this
overwrite is skipped and the LightRAG semantic visual chunk remains untouched.
In auto mode, a failed native image probe produces the same safe downgrade;
explicit multimodal mode instead treats probe failure as a startup error.

## Query Pipeline

```text
RAGService.aretrieve / aanswer(query, query_images, multimodal_content, filters)
  |
  |-- QueryPlanner
  |     declared metadata fields only
  |     explicit filters are strict
  |     LLM-inferred empty candidates fall back to unfiltered retrieval
  |
  |-- Query image preparation
  |     session image lookup
  |     VLM semantic descriptions for text/BM25/KG retrieval
  |     raw image payloads for direct multimodal embedding when active
  |
  |-- LightRAGMixBackend
  |     QueryParam(mode="mix")
  |     KG entities + relationships + text chunks
  |
  |-- Direct image path, when image embedding is active
  |     query images -> multimodal embedding -> image/vector chunks
  |
  |-- BM25 path
  |     pg_textsearch over candidate-scoped chunks
  |
  |-- RRF fusion + dedup + chunk candidate budget
  |
  |-- Provenance hydration
  |     page labels, visual sidecars, image bytes for fused chunks
  |
  |-- Final rerank
  |     multimodal listwise or external reranker over fused candidates
  |
  |-- Metadata enrichment + reference canonicalization
  |
  `-- AnswerEngine
        text excerpts, KG context, source metadata, optional images
```

LightRAG's `hybrid` mode is not used as a public downgrade path; the pipeline
above is the DlightRAG hybrid layer.

BM25 runs against the same LightRAG `LIGHTRAG_DOC_CHUNKS` rows through
DlightRAG-managed pg_textsearch profiles. During ingest, DlightRAG labels each
chunk with `dlightrag_bm25_language` using the shared Lingua-based classifier.
Primary startup creates one partial BM25 index per configured language profile
and one full-table `simple` fallback index. Query-time language detection routes
Chinese, English, German, Swedish, Spanish, French, Italian, Portuguese, Dutch,
Russian, Danish, and Finnish queries to the matching partial index;
unsupported, unknown, or ambiguous queries use the `simple`
fallback. `bm25_profiles`, `bm25_k1`, and `bm25_b` define the index signatures;
changing them requires a primary-role startup so DlightRAG can rebuild profile
indexes before query workers attach. Each non-fallback BM25 profile maps to
exactly one language; the fallback profile must not declare languages.

## Metadata In-Filtering

Metadata filtering is explicit-schema first:

- Declared fields are normalized and filterable.
- Undeclared metadata can be stored as JSONB enrichment, but is not filterable
  by default.
- User/API filters are strict. If they resolve to zero candidate documents or
  chunks, retrieval returns no matches.
- LLM-inferred filters include `filter_confidence` and evidence spans for
  observability. DlightRAG does not use hand-written fuzzy/static rules to
  invent or reject filters. If an inferred filter resolves to zero candidates
  or filtered retrieval returns no chunks, DlightRAG retries without that
  inferred filter because the planner may have over-inferred.
- Non-empty inferred candidate sets constrain semantic search and BM25 unless
  that inferred-filter retry path is needed.

For semantic search, `FilteredVectorDB` applies the candidate set before
ranking. Empty strict candidates return immediately. Small candidate sets use
exact vector scoring in a materialized candidate CTE; larger candidate sets
use pgvector HNSW with iterative scan settings.

## Multimodal Queries

Text queries go through LightRAG `mix`, BM25, fused-candidate hydration, and
reranking. Image-bearing queries add a direct image vector path only when the
configured `embedding.input_modality` resolves to multimodal and its startup
probe succeeds:

```text
query + images
  |-- text query -> LightRAG mix + BM25
  `-- images -> multimodal embedding(context="query") -> image chunks
```

Document and sidecar images are embedded with document context at ingestion.
Query images are embedded with query context when the provider supports
asymmetric embeddings. If the provider does not expose task-aware routing,
LightRAG's symmetric embedding mode is used.

Images also produce VLM semantic text through LightRAG's multimodal sidecar
path. That text feeds BM25 and KG extraction. For successful drawing chunks,
visual similarity search uses the same LightRAG chunk id after DlightRAG
overwrites its vector with the raw image embedding, preserving sidecar
provenance and avoiding duplicate VLM text exposure.

With `embedding.input_modality: text`, DlightRAG skips both image-vector
overwrite and query-image vector retrieval. Query images can still be described
by the VLM for text/BM25/KG retrieval, and document images still follow
LightRAG's native semantic multimodal path. Auto mode may make the same safe
downgrade after a failed native-provider probe; explicit multimodal mode fails
startup instead. See [Embedding configuration](configuration.md#embeddings) for
the provider and modality matrix.

## Reranking

`rerank.strategy` chooses the final ranker. DlightRAG does not pass
`rerank_model_func` into LightRAG; it disables LightRAG query reranking and
reranks the DlightRAG fused candidate set after provenance hydration. This lets
BM25-only hits, direct image matches, and LightRAG `mix` chunks compete in one
list with page/image data already attached.

| Strategy | How it works |
|---|---|
| `chat_llm_reranker` | Batched listwise scoring through the configured rerank model, or `llm.roles.vlm`, `llm.roles.query`, then `llm.default`. With `input_modality: auto`, the selected scoring model reuses the startup vision probe: vision-capable models get bounded image payloads plus text; non-vision models get VLM text only. |
| `jina_reranker` | Calls Jina `/v1/rerank`. The latest default is `jina-reranker-v3`, so auto mode sends chunk text; `jina-reranker-m0` auto mode sends bounded image documents when available. |
| `aliyun_reranker` | Calls Alibaba Model Studio rerank. `qwen3-rerank` uses the compatible text payload; `qwen3-vl-rerank` uses the DashScope multimodal payload. `base_url` must point at the matching workspace/region endpoint. |
| `local_reranker` | Calls a self-hosted `/rerank` endpoint. `input_modality: auto` is text-only; set `input_modality: multimodal` only when the endpoint accepts image document objects. |
| `voyage_reranker` | Calls Voyage AI `/v1/rerank` with text documents. |
| `cohere_reranker` | Calls Cohere `/v2/rerank` with text documents. |
| `azure_cohere` | Calls Azure AI Services Cohere rerank with text documents. Model endpoint roots use `/v1/rerank`; Foundry project roots use `/providers/cohere/v2/rerank`; a full `/rerank` URL is used as-is. |

When `rerank.score_threshold` is set, post-rerank filtering removes chunks below
that score. The threshold is hard: if every candidate in a workspace scores
below it, that workspace contributes no reranked chunks to federated round-robin
merge. When omitted, all strategies keep scored candidates before taking
`top_k`. If the reranker itself fails at request time, DlightRAG treats that as
infrastructure degradation and falls back to the pre-rerank fused order for that
request.

Configuration errors fail fast instead of falling back. For example, explicitly
choosing `voyage_reranker`, `cohere_reranker`, or another provider reranker
without the required API key prevents service initialization; DlightRAG does not
silently switch that configuration to `chat_llm_reranker`.

Reranking has an independent image budget because it runs after retrieval
hydration but before answer-context packing. `chat_llm_reranker` and
image-capable HTTP rerankers bound each request with `rerank.image_max_bytes`,
`rerank.image_max_total_bytes`, and the rerank-stage size/quality floors before
constructing model payloads. Visual chunks whose images cannot fit fall back to
their text, if present, rather than sending unbounded data URIs.

## Answer Generation

The answer prompt receives:

- chunk text excerpts
- KG entities and relationships from LightRAG `mix`
- LightRAG's doc-level `reference_id`/`references` mapping as the seed for
  source numbering
- document/source metadata
- quality-preserving bounded inline page or image previews when available
- user-supplied `query_images`, bounded by the independent user-image budget

### Answer LLM Input Shape

The answer model does not receive the raw `contexts` JSON. `AnswerEngine`
builds OpenAI-style messages with explicit evidence and task boundaries:

```python
[
    {"role": "system", "content": get_answer_system_prompt()},
    # optional conversation history
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "## User-attached images\n"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "## Document Excerpts"},
            {"type": "text", "text": "### Document [1]: report.pdf"},
            {"type": "text", "text": "[1-1] report.pdf, Page 3\nEvidence text..."},
            {"type": "text", "text": "[1-2] \"Doc Title\" Page 4"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": (
                "## Knowledge Graph Context\n..."
                "\n\n## Reference List\n..."
                "\n\n## Question\nWhat are the key findings?"
            )},
        ],
    },
]
```

The `## User-attached images` blocks are omitted when no current query images
fit the user-image budget. Conversation history, when present, is inserted as
prior messages before the current user message.

The sections are intentional:

- `## User-attached images` are part of the user's question, not retrieved evidence.
- `## Document Excerpts` contains retrieved text and page/image previews grouped by document.
- Excerpt labels such as `[1-1] report.pdf, Page 3` give the model the citation marker it must use.
- Retrieved document images are preceded by a text label, then sent as an `image_url` block only if they fit the RAG image budget.
- `## Knowledge Graph Context` gives entity/relationship facts, with source document tags when available.
- `## Reference List` maps citation IDs to documents.
- `## Question` is the actual user task and is placed last.

Answer generation has two image budgets. Retrieved visual chunks use
`answer.max_images` plus the answer image byte/quality limits. User-supplied
`query_images` and history images use `answer.max_user_images`; current-turn
`query_images` take those user slots before history images. The two budgets do
not compete, so an uploaded user image does not evict a retrieved page preview.
Budgeted JPEG, PNG, and WebP payloads are preserved as-is. When recompression is
needed, DlightRAG enforces both a long-edge floor and a JPEG quality floor; an
image that cannot fit within those limits is skipped instead of being degraded
into a low-quality preview.

`/retrieve` and `/answer` both accept an explicit `chunk_top_k` request to
override the configured chunk/visual candidate budget; otherwise DlightRAG uses
`config.chunk_top_k`. For `/answer`, retrieval deliberately over-fetches those
candidates, then the answer stage packs up to `answer.context_top_k` chunks.
That budget maps to LightRAG `QueryParam.chunk_top_k`; LightRAG `top_k` remains
the separate KG entity/relationship breadth. Retrieved
visual chunks are admitted in reranked order within the RAG context image
budget. Pure visual chunks whose image cannot be sent are removed from the
answer context and the packer backfills from later candidates; mixed text+image
chunks keep their text even if the image is skipped. KG entities and
relationships are filtered to the packed chunk ids, so citation indexes,
reference lists, streamed contexts, and returned sources describe the material
the answer model actually saw. Use `retrieve` when callers need the broader
pre-answer retrieval set.

DlightRAG does not use LightRAG `aquery_llm()` for final answer generation
because post-LightRAG context can include BM25 results, direct image matches,
federated chunks, and reranked multimodal pages.
Instead, it uses LightRAG `aquery_data()` as the base context and reference
seed, then validates inline `[n]` and `[n-m]` citations against the final
post-fusion context. The system prompt tells the model not to generate a
reference section; the output boundary still normalizes provider drift by
discarding generated bibliography tails and deriving `sources` deterministically
from validated inline markers. Returned `sources` contain only cited documents
and chunks.
Answer generation also derives `answer_images` and `answer_blocks` from those
validated cited sources before transport projection, so SDK, REST, MCP, and Web
expose the same image registry and insertion hints without trusting
model-generated Markdown image URLs.
Streaming callers receive tokens immediately and a final normalized answer plus
cited sources after validation.

## Semantic Highlights

Semantic highlights are answer-source enrichment, not retrieval. They run only
after answer finalization has validated inline citations and built `sources`.
The highlighter uses the finalized answer text plus cited source chunk content
to fill `sources[].chunks[].highlight_phrases`.

Web streaming attempts highlight enrichment by default after the answer and
source panel are finalized. SDK, REST, and MCP answer calls default to no
semantic highlights; pass `semantic_highlights=True` or
`semantic_highlights: true` on an answer request to opt in. `/retrieve` never
emits highlights because it has no finalized answer citations.

`citations.highlights.enabled` is the global kill switch. When enabled, the
highlighter uses the keyword LLM role, runs with its own timeout/concurrency
limits, and returns the original sources unchanged on timeout or failure.

## Multi-Workspace Retrieval

Federated retrieval plans the query once, then queries requested workspaces
concurrently. Each workspace runs the full single-workspace pipeline, including
metadata filtering, LightRAG `mix`, BM25 fusion, provenance hydration, and final
rerank thresholding. The federation layer then tags chunks with `_workspace`,
canonicalizes reference ids across workspaces, round-robin interleaves the
already-thresholded per-workspace lists, and truncates to `chunk_top_k`.

There is no cross-workspace global rerank. Round-robin is intentional: it keeps
workspace representation stable without assuming rerank scores from different
workspace/model calls are globally calibrated.
