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
       workspace routing, user scope, federation, writer/reader role gating
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
     one internal wildcard derived from the configured MinerU or Docling block
     unsupported engine/suffix combinations use LightRAG native/legacy fallback
  -> LightRAG staged ingest
       chunks, multimodal semantic text, KG entities/relations, vector rows
  -> DlightRAG post-ingest maintenance
       active fused text+image embedding overwrites canonical LightRAG drawing chunk vectors
       chunk language labels update BM25 partial indexes
       caller metadata updates the document metadata row
```

Source files and parser-extracted images both go through LightRAG's multimodal
path. When the configured embedding provider is a unified multimodal model and
the startup probe succeeds, DlightRAG aligns the existing canonical LightRAG
visual chunk with a fused vector that interleaves the VLM description and the
image. With a text-only embedding model, this alignment is skipped and
LightRAG's semantic visual chunk remains the multimodal ingestion path.

Parser adapters converge on LightRAG's shared IR and sidecars. DlightRAG does
not branch on MinerU versus Docling after that boundary. That derived parser
policy drives durable workspace ingestion only. Tables and equations remain
structured text evidence, while successful drawings use the shared VLM and
fused-vector path. MinerU and Docling are the only durable ingestion parsers,
and the temporary Answer resource path never invokes them.

## Answer Resource Flow

Every answer runs through one `AnswerOrchestrator`. Public callers attach files
and HTTPS references as **answer attachments**; the same contract is used by
REST, the Python SDK, MCP, and the Web UI. Attachments become request-local
resources for the lifetime of one answer and are never promoted into a
workspace, LightRAG storage, or a durable cache:

```text
answer request (query + optional attachments)
  -> request-local ResourceRegistry
       inline bytes held in memory; HTTPS links fetched lazily and revalidated
       per read (HTTPS-only, SSRF guard, per/total byte and pixel limits)
  -> AnswerOrchestrator routes by capability
       fast path: no resources and no web-search key
         -> fixed knowledge-base retrieval -> one AnswerSynthesizer final answer
       research path: resources present or an Exa web-search key is set
         -> fixed initial retrieval + optional strict web-scope decision
         -> peer tools (search_knowledge_base, read_resource, inspect_resource,
            optional search_web) writing observations into the EvidenceLedger
         -> evidence-growth convergence
         -> one tools-disabled AnswerSynthesizer final answer
```

Resource reads are deterministic first. `read_resource` decodes UTF-8/CSV text
directly and converts HTML, PDF, DOCX, PPTX, and XLSX through selected MarkItDown
converters with plugins disabled and no network access; OOXML archives pass a
zip-bomb preflight before any converter opens them. `inspect_resource` performs
focused visual inspection through the VLM role (falling back to the default
LLM), rasterizing PDFs off the event loop and bounding images through the one
canonical image path. Every visual observation is marked as VLM-derived evidence
with its exact source/page/sheet/cell locator, so the model cites where a claim
came from and never treats a description as the final answer.

Full resource bytes never enter model context. Only bounded text windows, capped
tool observations, and budgeted image blocks do. One `AnswerCapacity` shares the
configured context window across evidence packing and final synthesis: evidence
is bounded to a fraction of the window, each tool observation is capped, and a
fixed final-generation reserve is input-packing headroom, not an output cap.

When `web_search.api_key` (Exa) is set, the research path may call Exa Search and
Contents as one more peer tool; passages belong to no workspace and are packed
beside corpus evidence. A missing key simply removes the capability.

## Web Attachment Storage

The browser channel wraps the same orchestrator with a principal-scoped
conversation lifecycle. Uploaded answer attachments are persisted verbatim in one
raw table, `web_conversation_attachments`, keyed by principal, conversation, and
turn; there is no parsed-chunk table and no vector cache. Historical attachments
are re-registered lazily as request-local resources when a follow-up turn needs
them, and browser thumbnails are derived on demand. Manual deletion and
inactivity-TTL pruning cascade attachment bytes through the owning turn and
conversation. Answer attachments are the only durable answer inputs the Web store
keeps; no answer-time parse or embedding artifact is retained.

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

DlightRAG uses one PostgreSQL endpoint per service process. A writer process
(the default) serves REST, Web, MCP, and SDK operations against a write-capable
endpoint. Optional reader processes serve stateless read/query APIs against an
infrastructure-provided read endpoint; they disable Web conversations and reject
mutations. LightRAG's staged pipeline supports ingest and query together in the
writer process.

Core storage is PostgreSQL 18:

| Component | Backend |
|---|---|
| Vector store | `PGVectorStorage` with pgvector |
| Graph store | `PGTableGraphStorage` (plain tables) |
| KV store | `PGKVStorage` |
| Document status | `PGDocStatusStorage` |
| BM25 | pg_textsearch |

Replica routing, credentials, lag, and failover remain infrastructure concerns;
DlightRAG exposes only the process-level `service_role: writer | reader`. See
[postgresql.md](postgresql.md#reader-role-and-read-replicas) for the complete
read-only attachment and deployment contract.

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
