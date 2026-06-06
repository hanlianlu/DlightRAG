# Retrieval & Answer Mechanism

DlightRAG exposes one runtime path: LightRAG main is the base graph/vector
engine, always queried in `mix` mode, while DlightRAG adds metadata
management, direct multimodal image search, PostgreSQL BM25, RRF fusion,
reranking, citations, and answer generation.

## Ingestion Shape

```text
source file
  -> LightRAG parser/routing
       sidecar-backed text, tables, equations, and images when available;
       LightRAG raw parser route otherwise
  -> LightRAG ingest
       chunks, entities, relationships, graph, vectors, doc status
  -> image vector alignment
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
With DlightRAG's default `docx:native-iteP,*:mineru-iteP` rules, this is a
defensive path rather than the normal document parser route.

Successful drawing sidecars have one canonical chunk identity. LightRAG's
multimodal semantic chunk owns `llm_analyze_result` text and exposes it through
`text_chunks`, BM25, and KG extraction. DlightRAG then overwrites that existing
chunk's vector with the raw image embedding. It does not create a second
visual-only chunk, so the same VLM description is not exposed twice as independent
retrieved evidence.

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
  |     raw image payloads for direct multimodal embedding
  |
  |-- LightRAGMixBackend
  |     QueryParam(mode="mix")
  |     KG entities + relationships + text chunks
  |
  |-- Direct image path
  |     query images -> multimodal embedding -> image/vector chunks
  |
  |-- BM25 path
  |     pg_textsearch over candidate-scoped chunks
  |
  |-- RRF fusion + dedup + answer.candidate_top_k
  |
  |-- Rerank
  |     multimodal listwise or external reranker
  |
  `-- AnswerEngine
        text excerpts, KG context, source metadata, optional images
```

LightRAG's `hybrid` mode is not used as a public downgrade path. The
DlightRAG hybrid layer is the combination of LightRAG `mix` retrieval,
pg_textsearch BM25, direct image retrieval, and RRF fusion.

BM25 runs against the same LightRAG `LIGHTRAG_DOC_CHUNKS` rows through the
configured pg_textsearch profiles. Chinese queries use the `public.jiebacfg`
profile from `pg_jieba`, English queries use the PostgreSQL `english` profile,
German queries use `german`, Swedish queries use `swedish`, Spanish queries use
`spanish`, and French queries use `french`. Unsupported, unknown, or ambiguous
queries use the `simple` fallback.
`bm25_profiles`, `bm25_k1`, and `bm25_b` define the index signatures; changing
them requires a primary-role startup so DlightRAG can rebuild profile indexes
before query workers attach.

## Metadata In-Filtering

Metadata filtering is explicit-schema first:

- Declared fields are normalized and filterable.
- Undeclared metadata can be stored as JSONB enrichment, but is not filterable
  by default.
- User/API filters are strict. If they resolve to zero candidate documents or
  chunks, retrieval returns no matches.
- LLM-inferred filters include `filter_confidence` and evidence spans for
  observability. DlightRAG does not use hand-written fuzzy/static rules to
  invent or reject filters. If an inferred filter resolves to zero candidates,
  DlightRAG retries without that inferred filter because the planner may have
  over-inferred.
- Non-empty inferred candidate sets constrain semantic search and BM25.

For semantic search, `FilteredVectorDB` applies the candidate set before
ranking. Empty strict candidates return immediately. Small candidate sets use
exact vector scoring in a materialized candidate CTE; larger candidate sets
use pgvector HNSW with iterative scan settings.

## Multimodal Queries

Text queries go through LightRAG `mix`, BM25, and reranking. Image-bearing
queries add a direct image vector path:

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

## Reranking

`rerank.strategy` chooses the post-fusion ranker:

| Strategy | How it works |
|---|---|
| `chat_llm_reranker` | Batched listwise scoring through the configured rerank model, or the default LLM when no rerank model is set. The selected model must support images when retrieved chunks include page/image data. |
| `jina_reranker` / `aliyun_reranker` / `local_reranker` | Calls an OpenAI-compatible `/rerank` endpoint with text documents. |
| `azure_cohere` | Calls Azure AI Services Cohere rerank with text documents. |

Post-rerank filtering removes chunks below `rerank.score_threshold`. If all
chunks fall below the threshold, DlightRAG keeps the top scored fallback set
instead of returning an accidental empty answer context.

Reranking has an independent image budget because it runs before answer-context
packing. `chat_llm_reranker` and image-capable HTTP rerankers bound each request
with `rerank.image_max_bytes`, `rerank.image_max_total_bytes`, and the
rerank-stage size/quality floors before constructing model payloads. Visual
chunks whose images cannot fit fall back to their text, if present, rather than
sending unbounded data URIs.

## Answer Generation

The answer prompt receives:

- chunk text excerpts
- KG entities and relationships from LightRAG `mix`
- LightRAG's doc-level `reference_id`/`references` mapping as the seed for
  source numbering
- document/source metadata
- quality-preserving bounded inline page or image previews when available
- user-supplied `query_images`, also bounded by the same shared image budget

The answer image budget preserves budgeted JPEG, PNG, and WebP payloads as-is.
When recompression is needed, DlightRAG enforces both a long-edge floor and a
JPEG quality floor; an image that cannot fit within those limits is skipped
instead of being degraded into a low-quality preview.

For `/answer`, retrieval deliberately over-fetches with
`answer.candidate_top_k`, then the answer stage packs up to
`answer.context_top_k` chunks. That over-fetch is a chunk/visual candidate
budget and maps to LightRAG `QueryParam.chunk_top_k`; LightRAG `top_k` remains
the separate KG entity/relationship breadth. User `query_images` consume the
shared image budget first, then retrieved visual chunks are admitted in reranked
order. Pure visual chunks whose image cannot be sent are removed from the
answer context and the packer backfills from later candidates; mixed text+image
chunks keep their text even if the image is skipped. KG entities and
relationships are filtered to the packed chunk ids, so citation indexes,
reference lists, streamed contexts, and returned sources describe the material
the answer model actually saw. Use `retrieve` when callers need the broader
pre-answer retrieval set.

DlightRAG does not use LightRAG `aquery_llm()` for final answer generation
because post-LightRAG context can include BM25 results, direct image matches,
metadata-path injections, federated chunks, and reranked multimodal pages.
Instead, it uses LightRAG `aquery_data()` as the base context and reference
seed, then validates inline `[n]` and `[n-m]` citations against the final
post-fusion context. The system prompt tells the model not to generate a
reference section; the output boundary still normalizes provider drift by
discarding generated bibliography tails and deriving `sources` deterministically
from validated inline markers. Returned `sources` contain only cited documents
and chunks.
Streaming callers receive tokens immediately and a final normalized answer plus
cited sources after validation.
