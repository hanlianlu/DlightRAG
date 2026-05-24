# Retrieval & Answer Mechanism

DlightRAG exposes one runtime path: LightRAG main is the base graph/vector
engine, always queried in `mix` mode, while DlightRAG adds metadata
management, direct multimodal image search, PostgreSQL BM25, RRF fusion,
reranking, citations, and answer generation.

## Ingestion Shape

```text
source file
  -> LightRAG parser sidecars
       text, tables, equations, document-derived images
  -> LightRAG ingest
       chunks, entities, relationships, graph, text vectors
  -> DlightRAG side tables
       metadata registry, document artifacts, chunk provenance
  -> direct image embedding
       native images and parser-extracted image sidecars
  -> visual semantic projection
       VLM text for entity and relationship extraction over images
```

Tables, equations, and document-derived image sidecars stay aligned with the
LightRAG document record. Native images use the same metadata and provenance
contract, but their visual similarity path stores direct image embeddings
instead of embedding VLM captions.

## Query Pipeline

```text
RAGService.aretrieve / aanswer(query, multimodal_content, filters)
  |
  |-- QueryPlanner
  |     declared metadata fields only
  |     explicit filters are strict
  |     LLM-inferred empty candidates fall back to unfiltered retrieval
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
  |-- RRF fusion + dedup + top_k
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

## Metadata In-Filtering

Metadata filtering is explicit-schema first:

- Declared fields are normalized and filterable.
- Undeclared metadata can be stored as JSONB enrichment, but is not filterable
  by default.
- User/API filters are strict. If they resolve to zero candidate documents or
  chunks, retrieval returns no matches.
- LLM-inferred filters include `filter_confidence` and evidence spans from the
  query. If an inferred filter resolves to zero candidates, DlightRAG retries
  without that inferred filter because the planner may have over-inferred.
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
  `-- images -> multimodal embedding(context="query") -> visual chunks
```

Document and sidecar images are embedded with document context at ingestion.
Query images are embedded with query context when the provider supports
asymmetric embeddings. If the provider does not expose task-aware routing,
LightRAG's symmetric fallback is used.

Images also produce VLM semantic text for entity and relationship extraction.
That semantic projection feeds the graph, but visual similarity search uses
direct image embeddings.

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

## Answer Generation

The answer prompt receives:

- chunk text excerpts
- KG entities and relationships from LightRAG `mix`
- LightRAG's doc-level `reference_id`/`references` mapping as the seed for
  source numbering
- document/source metadata
- inline page or image data when available
- user-supplied `query_images` when the answer model should reason over them

DlightRAG does not use LightRAG `aquery_llm()` for final answer generation
because post-LightRAG context can include BM25 results, direct image matches,
metadata-path injections, federated chunks, and reranked multimodal pages.
Instead, it uses LightRAG `aquery_data()` as the base context and reference
seed, then validates inline `[n]` and `[n-m]` citations against the final
post-fusion context. Generated `### References` tails are stripped at the
output boundary; returned `sources` contain only cited documents and chunks.
Streaming callers receive tokens immediately and a final normalized answer plus
cited sources after validation.
