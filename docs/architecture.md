# Architecture

This page is for readers who need to understand DlightRAG's runtime boundaries.
It owns the product architecture, the LightRAG/DlightRAG responsibility split,
the storage topology, and the code-layering rule. Canonical product terms live
in [domain-language.md](domain-language.md); interface contracts live in
[interfaces.md](interfaces.md); retrieval internals live in
[retrieval-answer.md](retrieval-answer.md); PostgreSQL deployment details live
in [postgresql.md](postgresql.md).

<p align="center">
  <img src="architecture.svg" alt="DlightRAG Architecture" width="1080" />
</p>

The figure keeps three views separate: compile-time package imports, runtime
call/provider flow, and persistence adapter wiring. Solid slate lines are
runtime calls, dashed open lines are dependencies or port implementation,
dotted magenta lines are AI-provider admission, and `↻` marks the three
independent concurrency owners.

## Runtime Ownership

```text
Clients
  -> REST / Web / MCP / SDK adapters
  -> Application
       eager composition and lifecycle
       -> AnswerService -> dlightrag.runtime RunCoordinator
            neutral lifecycle records, store port, leases, events, checkpoints
            -> AnswerExecutor -> PGAnswerRunStore adapter
       -> RetrievalService -> WorkspacePool -> WorkspaceRag
       -> CorpusAdmin -> WorkspacePool -> WorkspaceRag
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
            -> canonical knowledge-base retrieval (including RetrievalPlanner)
               -> one AnswerSynthesizer final answer
       research path: resources present or an Exa web-search key is set
               -> agent selects from peer tools (search_knowledge_base,
                    read, inspect, optional search_web, optional
                    delegate_research, optional path grep)
               -> a selected KB search invokes the same canonical retrieval;
                    RetrievalPlanner preserves the agent query and derives
                    lexical/filter/image hints inside that operation
               -> selected tools write observations into the EvidenceLedger
               -> each control turn replays the session episode's exchanges
         -> evidence-growth convergence
         -> one tools-disabled AnswerSynthesizer final answer
```

Resource reads are deterministic first. `read` decodes UTF-8/CSV text
directly and converts HTML, PDF, DOCX, PPTX, and XLSX through selected MarkItDown
converters with plugins disabled and no network access; OOXML archives pass a
zip-bomb preflight before any converter opens them. `inspect` performs
focused visual inspection through the VLM role (falling back to the default
LLM), rasterizing PDFs off the event loop and bounding images through the one
canonical image path. Every visual observation is marked as VLM-derived evidence
with its exact source/page/sheet/cell locator, so the model cites where a claim
came from and never treats a description as the final answer.

Current image attachments reach the research agent and final generation as
bounded image blocks, while the same verified bytes remain request-local
resources for optional focused evidence. If the agent selects a knowledge-base
search, one VLM description and the raw image feed that retrieval's text and
direct-visual legs; no KB call means no query-image planning work. A source-image
inspection sends the bounded whole image with a concrete focus; it does not crop
or zoom an arbitrary region.
Structural zoom-in is available for a selected PDF page or an extracted embedded
visual handle. The control prompt tells the model not to repeat inspection for a
general description when the current image is already visible.

Full resource bytes never enter model context. Only bounded text windows, capped
tool observations, and budgeted image blocks do. Every reachable model endpoint
has an immutable `ModelProfile`: context window (`C`), optional provider input
limit (`I`), optional output limit (`O`), and capability flags. Facts resolve by
normalized provider/model/endpoint identity from an explicit root override, a
trusted adapter, or the versioned AI catalog; an unknown identity fails closed.
The revisioned `ContextPolicy` derives one hard input limit
`L = min(I if known else C, floor(0.85C))`, proactive research compaction at
`floor(0.85L)`, and output allowance `min(O, C - input)` when `O` is known.
Evidence, resource windows, and tool observations consume the actual residual
of the model request rather than independent global token caps.

`RetrievalPlanner` is an internal node of the canonical retrieval operation; the
answer workflow never creates or injects a plan. It never receives attachment
bytes, converted attachment text, or resource manifests. Fast answers give
retrieval the bounded prior turns so the planner can resolve references. Public
retrieve calls are history-free. Research KB tool calls receive the one pinned
history projection, but `preserve_query` keeps their caller-chosen semantic query
unchanged while lexical terms, inferred metadata filters, and optional
current-image hints are derived. Explicit filters and BM25 terms remain
authoritative.

Workspace resolution stays at each interface's Access boundary. Retrieval starts cold
workspace initialization before retrieval planning for retrieve-only, fast-answer,
and research-answer requests; the later retrieval joins those same services.

Research control and final generation also use separate system prompts. Control
turns receive identity, tool-selection policy, trust boundaries, and stopping
rules, but not the answer/citation contract. The tools-disabled final call swaps
in the normal `answer_core` prompt while retaining the original request,
conversation history, resource manifest, latest native tool exchange, and final
citable evidence.

When `web_search.api_key` (Exa) is set, Exa Search is an optional peer
capability. Its passages belong to no workspace and are packed beside corpus
evidence; evidence-producing result URLs become inert request-local resource
handles that the model may deep-read with `read`. Exa Contents is not a
peer tool: it is a bounded internal fallback only when a selected public URL
cannot be read directly. A missing key removes both capabilities.

## Durable Answer Runs

Every answer — REST, MCP, Web, Python SDK, CLI, and evaluation — is one durable
run with one identifier and one lifecycle owned by PostgreSQL. `POST /answer`
validates, persists, and returns HTTP 202; the run outlives its creating request,
and a disconnected client only detaches.

```text
create (202)  -> dlightrag_answer_runs row + input blobs + references (one txn)
claim         -> FOR UPDATE SKIP LOCKED, fencing epoch++, lease heartbeat
execute       -> phase progress, coalesced token batches, per-turn checkpoint
finish        -> canonical result + exactly one terminal event, same txn
recover       -> expired lease reclaimed, resumed from the latest checkpoint
```

A process restart resumes from the latest completed control turn; generation
interrupted mid-stream emits `reset` and regenerates. Four tables own that state:
`dlightrag_answer_runs`, `dlightrag_answer_run_events`,
`dlightrag_blobs`/`dlightrag_blob_chunks`, and `dlightrag_answer_run_artifacts`. See
[durable-answer-runs.md](durable-answer-runs.md) for the full contract and
[postgresql.md](postgresql.md#durable-answer-run-state) for the schema.

`dlightrag.runtime` owns the storage-neutral records, store protocol,
subscription, coordinator, fenced session, checkpoint failures, and caller-wait
failures. It imports neither Answer policy nor PostgreSQL. The current Answer
executor classifies product errors into `RunExecutionError` before they cross
that boundary; `dlightrag.adapters.postgres.answer_runs.PGAnswerRunStore`
implements the runtime store port.

`dlightrag-rag-core` owns the coherent `WorkspaceCorpusBackend` bundle:
coordination and maintenance, durable ingest jobs, plus a runtime binder for
metadata, chunk, filtered-vector, and BM25 stores. The root PostgreSQL adapter
implements those ports and hides environment translation, server/version/
extension checks, advisory-lock lifetimes, reader attachment, catalog scans,
workspace maintenance, schema DDL, and SQL identifiers. Startup availability
failures are translated to corpus errors; operation-specific failures retain
their adapter context for the current product error policy.
The current `Application` composes the adapter; the independently installable RAG
package, Runtime, status routes, API, Web, and MCP never import it. Corpus and
operational pools remain separate even when they use the same endpoint.

## Web Conversation Boundary

The browser channel wraps the same durable runs with a principal-scoped
conversation lifecycle. A conversation owns navigation and history only: each
turn is one row linking to the Answer run that owns the request input, the
uploaded bytes, the streamed events, and the canonical result. The turn is
inserted inside the run's own creation transaction, so history exists before the
202 response.

Uploaded answer attachments are stored once as owner-scoped content-addressed
chunked blobs in `dlightrag_blobs`/`dlightrag_blob_chunks` and referenced by
`dlightrag_answer_run_artifacts`; there is no Web-owned raw attachment table, no
parsed-chunk table, and no vector cache. Historical attachments are re-registered
lazily as request-local resources on every follow-up, newest first up to the
available attachment-count limit. An attachment-bearing conversation therefore
remains on the research path. Browser thumbnails are derived on demand. Manual
deletion, conversation TTL pruning, and 30-day run retention all delete the
linked runs, cascade their events and references, and release blobs no surviving
run references.

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

DlightRAG uses one PostgreSQL endpoint per service process. A writer process (the
default) serves REST, Web, MCP, and SDK operations and owns schema migrations.
A `reader` process is **corpus-read-only, not process-read-only**: it may create
and execute answer runs and write DlightRAG operational state (runs, events,
artifacts, Web conversations), while `CorpusAdmin` rejects ingestion, workspace
creation/reset, metadata mutation, retry, and deletion. Both roles
therefore use the same primary endpoint; DlightRAG makes no physical-standby or
read-endpoint promise.

Core storage is PostgreSQL 18:

| Component | Backend |
|---|---|
| Vector store | `PGVectorStorage` with pgvector |
| Graph store | `PGTableGraphStorage` (plain tables) |
| KV store | `PGKVStorage` |
| Document status | `PGDocStatusStorage` |
| BM25 | pg_textsearch |

Every process serving KB images or source downloads must see the same POSIX
artifact tree at the same absolute `working_dir` path. See
[postgresql.md](postgresql.md#service-roles-and-shared-artifacts) for the
complete role, migration-order, and shared-artifact contract.

## Code Layering

The repository is one UV workspace with four lockstep distributions. Distinct
top-level Python packages make their import directions observable in source and
in built wheels:

```text
dlightrag-ai          immutable settings/fingerprints; fair provider admission;
     ↑       ↑        chat, tool, embedding, rerank and probe lifecycles
     │       │
dlightrag-agent-core  generic tool contracts and deterministic turn execution

dlightrag-rag-core    LightRAG chat/embedding adapters, rerank orchestration,
                      storage-neutral metadata records and score fusion

dlightrag             product composition, PostgreSQL, REST/Web/MCP/SDK
```

Agent and RAG core depend on AI; RAG core also owns its direct LightRAG API
dependency. The root product depends on all three cores and maps Pydantic input
configuration into immutable AI settings before composition. RAG imports
neither the root product nor Agent and never imports concrete PostgreSQL
adapters. Concrete provider SDKs are lazy AI extras, so importing
`dlightrag_ai` does not load OpenAI, Anthropic, or Gemini clients. Root
`LangfuseTelemetry` is injected into core model operations; standalone cores use
the explicit no-op adapter.

Inside the root product, modules still sit on a decreasing dependency stack: a
module at a higher layer may import from lower layers, but lower layers must not
import higher ones.

```text
L9  api, mcp, web                                  interface adapters
L8  application; services                         composition and use cases
L7  dlightrag_rag.WorkspacePool, WorkspaceRag      corpus runtime ownership
L6  answer                                         execution, lifecycle, source/media projection
L5  host and storage adapters                      PostgreSQL; LightRAG contract and lifecycle
L4  workspace cores and model adapters             AI; Agent; RAG retrieval, ingestion, sourcing
L3  product domain                                 access, requests, citations, conversations
L2c application, runtime                           health and durable lifecycle contracts
L2b model settings, schemas                        resolved foundation values
L2a config, scope, protocols                       shared configuration and contracts
L1  observability                                  Langfuse telemetry adapter
L0  prompts, utils                                 pure helpers
```

The layering checks are part of local and CI verification:

```bash
uv run lint-imports
```

`lint-imports` enforces contracts over all four roots: AI cannot import the
product, Agent, RAG, LightRAG, PostgreSQL, or transport packages; Agent may use
AI but not product/RAG/storage/transport code; and RAG may use AI and LightRAG
APIs but cannot import the product, Agent, PostgreSQL, or transport packages.
Existing root contracts continue to
keep `api`/`mcp`/`web` out of internal modules, order the foundation and core
coordination stacks, keep Runtime free of Answer/RAG/storage/transport code,
make status routes depend only on `ApplicationHealth`, and separate resources
from model-visible tool adapters.
The same checks run against installed wheel contents so an editable workspace
cannot hide an undeclared dependency; that artifact gate also rejects imports
of LightRAG's concrete PostgreSQL backend, whose external submodule path cannot
be represented by import-linter.
