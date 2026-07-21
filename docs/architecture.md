# Architecture

This page is for readers who need to understand DlightRAG's runtime boundaries.
It owns the product architecture, the LightRAG/DlightRAG responsibility split,
the storage topology, and the code-layering rule. Interface contracts live in
[interfaces.md](interfaces.md); retrieval internals live in
[retrieval-answer.md](retrieval-answer.md); PostgreSQL deployment details live
in [postgresql.md](postgresql.md).

<p align="center">
  <img src="architecture.svg" alt="DlightRAG Architecture" width="1080" />
</p>

## Runtime Ownership

```text
Clients
  -> REST / Web / MCP / SDK adapters
  -> RAGServiceManager
       workspace routing, user scope, federation, read-after-write barriers
  -> RAGService
       one workspace runtime, ingest, retrieve, answer, reset
  -> LightRAG main
       parser routing, staged ingest, chunks, doc status, KG, vectors
  -> DlightRAG PostgreSQL stores
       metadata index, BM25 indexes, workspace/job/Web conversation metadata
```

LightRAG remains the core RAG engine. It owns parser routing, staged ingest,
document chunks, document status, vector storage, and the knowledge graph.
DlightRAG adds product-layer source staging, metadata governance, durable ingest
jobs, PostgreSQL BM25, fused visual-vector alignment, answer orchestration,
citations, REST, Web, SDK, and MCP interfaces.

DlightRAG does not reimplement LightRAG parser sidecars, document status, KG
extraction, or LightRAG `mix` retrieval.

## Ingestion Flow

```text
source file or upload
  -> DlightRAG source staging and metadata normalization
  -> LightRAG parser routing
       native route for DOCX, Markdown, and textpack inputs
       MinerU-compatible route for configured sidecar document formats
  -> LightRAG staged ingest
       chunks, multimodal semantic text, KG entities/relations, vector rows
  -> DlightRAG post-ingest maintenance
       active fused text+image embedding overwrites canonical LightRAG drawing chunk vectors
       chunk language labels update BM25 partial indexes
       declared metadata updates filterable columns
```

Source files and parser-extracted images both go through LightRAG's multimodal
path. When the configured embedding provider is a unified multimodal model and
the startup probe succeeds, DlightRAG aligns the existing canonical LightRAG
visual chunk with a fused vector that interleaves the VLM description and the
image. With a text-only embedding model, this alignment is skipped and
LightRAG's semantic visual chunk remains the multimodal ingestion path.

## Web Composer Attachment Flow

Web Composer documents borrow processing capability from the first normalized
requested workspace, but they never become workspace documents:

```text
temporary Web document
  -> real LightRAG parser in a TemporaryDirectory
  -> cache-neutral LightRAG VLM/EXTRACT analysis over a strict proxy
  -> real LightRAG text chunkers and multimodal sidecar renderer
  -> shared RobustDocumentEmbedder
       fused image+text when the selected service has working image capability
       text-only when capability is absent or fused embedding fails
  -> web_conversation_attachment_chunks
       enriched content, image bytes, signatures, JSONB vectors
  -> exact Composer-local retrieval and answer packing
```

The selected `RAGService` remains the sole owner of provider clients, the
embedding executor, and the reranker. Composer borrows those resources; it does
not create or close duplicate clients. Results are isolated by
`principal_id + conversation_id` in the existing `web_conversation_*` tables.
No Composer path writes LightRAG `full_docs`, `doc_status`, chunk/vector, BM25,
LLM-cache, or KG rows, and no HNSW or other ANN index is created for Composer
vectors.

Manual conversation deletion and inactivity-TTL pruning cascade through parsed
chunks, VLM-derived content, image bytes, and vectors. Max-turn trimming removes
old turns only and preserves the conversation-level attachment cache. Identical
bytes may be reused within one conversation under matching signatures, but are
never reused across conversations or principals.

## Retrieval And Answer Flow

```text
query
  -> query planning and optional metadata filter inference
  -> strict metadata in-filtering when filters are explicit
  -> LightRAG mix retrieval
  -> direct image->image retrieval when fused visual embedding is active
  -> pg_textsearch BM25 over the same candidate scope
  -> RRF fusion
  -> provenance hydration and final rerank
  -> answer packing with citations, bounded images, and optional highlights
```

DlightRAG uses LightRAG `mix` as the base retrieval mode; the steps shown
around it form the DlightRAG hybrid layer.

Use [retrieval-answer.md](retrieval-answer.md) for the detailed retrieval,
filtering, reranking, citation, and multimodal-answer behavior.

## PostgreSQL Topology

DlightRAG has one application-level PostgreSQL endpoint. REST, Web, MCP, and
SDK surfaces all use the same configured write-capable endpoint, and LightRAG's
staged pipeline supports ingest and query in the same service process.

Core storage is PostgreSQL 18:

| Component | Backend |
|---|---|
| Vector store | `PGVectorStorage` with pgvector |
| Graph store | `PGGraphStorage` with Apache AGE |
| KV store | `PGKVStorage` |
| Document status | `PGDocStatusStorage` |
| BM25 | pg_textsearch |

If production infrastructure uses managed read replicas, keep that routing in
the PostgreSQL, proxy, or platform layer. DlightRAG does not expose separate
query/runtime database roles or read-replica credentials.

## Code Layering

Modules sit on a decreasing dependency stack: a module at a higher layer may
import from lower layers, but lower layers must not import higher ones.

```text
L9  api, mcp, web                                  interface adapters
L8  core.servicemanager                            multi-workspace coordinator
L7  core.{service, reset}                          per-workspace facade
L6  core orchestration                             ingest, retrieve, answer, visual assets
L5  LightRAG/store adapters                        patches, parser sidecar, BM25, filtered VDB
L4  models and shared retrieval helpers            embedding, LLM, rerank, metadata path
L3  providers, storage, sourcing, citations        external/domain implementations
L2  config, schemas, scope, protocols              shared contracts
L1  observability                                  Langfuse wrappers and no-op fallback
L0  prompts, utils                                 pure helpers
```

The layering checks are part of local and CI verification:

```bash
uv run lint-imports
```

`lint-imports` enforces four contracts: `api`/`mcp`/`web` stay out of the
internal packages; foundation packages never import domain code; the lower
stack is ordered `models → config → observability → prompts`/`utils`; and the
core coordination stack is ordered `servicemanager → service → reset`. Shared
contract modules (`core.retrieval.protocols`/`models`, `models.schemas`) are
imported across several layers, so the full table above is a design guide rather
than a single machine-checked chain.
