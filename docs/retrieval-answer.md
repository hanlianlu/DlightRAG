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

`/retrieve` is a knowledge-base-only path: it accepts `query_images` for visual
search and every `top_k`/`chunk_top_k`/`direct_visual`/KG/BM25/RRF/rerank control.
`/answer` takes a query plus optional **attachments** and routes through one
`AnswerOrchestrator`. The public Answer Mode is ``auto | fast | research`` (omitted means auto).
Fast and Research are resolved from that selector and the Valid Mode Set;
configuring Web Search does not by itself force Research.

REST, MCP, and Python answer/retrieve calls require no client-managed conversation ID.
Each accepted Answer run durably pins its query, bounded caller-supplied history,
resources, search scope, and execution facts so recovery and server-owned
follow-up/fork remain equivalent. The Web conversation lifecycle is a
principal-scoped adapter around the same Answer pipeline. It loads server-owned
text history and run-owned attachments; its 100-turn snapshot is a read window,
while run retention follows the shared configured floor.

## Ingestion Shape

```text
source file
  -> LightRAG parser/routing
       sidecar-backed text, tables, equations, and images when available;
       LightRAG raw parser route otherwise
  -> LightRAG ingest
       chunks, entities, relationships, graph, vectors, doc status
  -> fused visual-vector alignment when direct image embedding is active
       successful LightRAG drawing multimodal chunks keep their VLM text,
       sidecar provenance, BM25, and KG identity; DlightRAG overwrites the
       existing chunk vector with one fused vector interleaving the VLM
       description and the image, so text queries still retrieve the figure
  -> DlightRAG metadata/BM25 layer
       metadata index, in-filter scope, pg_textsearch BM25
```

All source files that LightRAG can ingest, including native image files, go
through LightRAG parser/routing. DlightRAG derives one internal wildcard from
the configured MinerU or Docling sidecar block; if both blocks exist, MinerU is
effective. Tables, equations, text, and document-derived image sidecars stay
aligned with the LightRAG document record.
LightRAG raw-route documents can have no sidecar artifacts; those documents
still participate in LightRAG text/KG/vector retrieval but do not receive
fused visual-vector alignment.
The selected engine applies only to suffixes it supports. LightRAG routes other
suffixes to their native/default or legacy path without first calling the
external parser.

Successful drawing sidecars have one canonical chunk identity. LightRAG's
multimodal semantic chunk owns `llm_analyze_result` text and exposes it through
`text_chunks`, BM25, and KG extraction. DlightRAG then overwrites that existing
chunk's vector with a single fused vector that interleaves the VLM description
and the image. A bare image-crop vector reintroduces the text/image modality
gap, so text queries cannot reach the figure; fusing the description back into
the vector closes that gap while keeping the native multimodal alignment. It
does not create a second visual-only chunk, so the same VLM description is not
exposed twice as independent retrieved evidence. The overwrite is skipped --
leaving LightRAG's native VLM->text vector untouched -- when
`models.embedding.input_modality` resolves to text or the provider cannot fuse text and
image into one vector. In auto mode, a failed native image probe produces the
same safe downgrade; explicit multimodal mode instead treats probe failure as a
startup error.

## Query Pipeline

This pipeline is used by `/retrieve`. Durable Answer execution reuses the same
planner and workspace-set schema provider but owns its retrieval call and has no
inline request timeout. In the research path retrieval runs lazily only when the
agent selects `search_knowledge_base`. The agent's tool query remains the
semantic search query; `RetrievalPlanner` is an internal retrieval node that
contributes BM25 terms, metadata filters, and current-image retrieval context
without rewriting that query. Attachment resources are not planner inputs.

```text
RetrievalService.retrieve(RetrieveRequest(...))
  |
  |-- RetrievalPlanner
  |     built-in metadata fields plus custom_metadata keys
  |     explicit filters are strict
  |     LLM-inferred empty candidates fall back to unfiltered retrieval
  |
  |-- Query image preparation (retrieve only; query_images)
  |     current-request images only
  |     VLM semantic descriptions for text/BM25/KG retrieval
  |     raw image payloads for direct multimodal embedding when active
  |
  |-- LightRAGMixBackend
  |     QueryParam(mode="mix")
  |     KG entities + relationships + text chunks
  |
  |-- Direct image->image path, when fused visual embedding is active
  |     query images -> image embedding (batched) -> fused visual chunks
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
  `-- AnswerSynthesizer
        text excerpts, KG context, source metadata, optional images
```

Each interface resolves an authorized concrete workspace set before entering
`RetrievalService`, which starts cold-service warm-up. A multi-workspace request plans once over the selected
workspace set, then dispatches that one resolved retrieval request to each
workspace before round-robin merging.

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
fallback. `corpus.retrieval.bm25_profiles`, `corpus.retrieval.bm25_k1`, and
`corpus.retrieval.bm25_b` define the index signatures;
changing them for an existing corpus requires the offline workspace BM25
rebuild before query workers attach. Each non-fallback BM25 profile maps to
exactly one language; the fallback profile must not declare languages.
`corpus.retrieval.bm25_enabled` controls this workspace PostgreSQL lane only. The answer research
path reads attachments through request-local resources, whose lexical ranking is
in-memory and has no workspace index or shared configuration.

The semantic and workspace BM25 lanes degrade independently. A BM25 query
failure retains successful semantic results and records `bm25_error_type` in
trace; a semantic failure can return BM25-only results and records
`lightrag_error_type`. If both lanes fail, retrieval raises the semantic error
with the BM25 failure chained as its cause.

## Metadata In-Filtering

Metadata filtering is explicit-schema first:

- Named fields (`filename`, `file_extension`, `title`, `author`,
  `creation_date_from`/`creation_date_to`) are typed columns. Custom metadata is
  one JSONB column, matched by `filters.custom` under the same case-insensitive
  rule; no key needs declaring first.
- A named file is one filter field, `filename`. Callers and the planner write the
  name as it was said — complete or partial, with or without an extension — and
  retrieval resolves it: exact match against the stored name or its stem first,
  then a contains match if neither hits. Splitting that into separate exact,
  stem, and pattern fields asked the planner to choose a match operator for a
  corpus it cannot see, and the schema deliberately shows it column names rather
  than values because a workspace's document count is unbounded.
- User/API filters are strict. If they resolve to zero candidate documents or
  chunks, retrieval returns no matches.
- LLM-inferred filters include `filter_confidence` and evidence spans for
  observability. DlightRAG does not use hand-written fuzzy/static rules to
  invent or reject filters. If an inferred filter resolves to zero candidates
  or filtered retrieval returns no chunks, DlightRAG retries without that
  inferred filter because the planner may have over-inferred.
- Non-empty inferred candidate sets constrain semantic search and BM25 unless
  that inferred-filter retry path is needed.

A document scope has to reach every leg that contributes chunks, and LightRAG
`mix` has three. The semantic leg goes through `FilteredVectorStorage`, which
applies the candidate set before ranking: empty strict candidates return
immediately, small candidate sets use exact vector scoring in a materialized
candidate CTE, and larger ones use pgvector HNSW with iterative scan settings.
The entity and relation legs never vector-search for their chunks — they resolve
the chunk ids baked into graph nodes at ingest time by primary key — so
`FilteredChunkStore` scopes them at the `text_chunks` lookup instead, returning
the same `None` the storage already returns for ids it cannot resolve. Both
wrappers read one contextvar, so ingest and delete paths run unscoped and pass
through untouched. `metadata_kg_chunks_dropped` in the retrieval trace counts
what the graph legs asked for and did not get.

Filtering the vector lookup that *selects* those graph-referenced ids is not an
option: LightRAG reads a short result from `chunks_vdb.get_vectors_by_ids` as
storage corruption and falls back to an unfiltered ranking method. Scoping after
the fact costs recall — the selection budget is still spent on out-of-scope
chunks — but that budget is internal to LightRAG.

What a filter scopes is the evidence: every chunk the answer can quote comes
from a filtered leg. It does not scope the entity and relationship *summaries*
that `mix` also puts in the prompt. Those are LightRAG's own context sections,
and their descriptions are corpus-level syntheses — one entity's description is
merged across every document that mentioned it, so there is no per-document
share of it to keep or drop. Restricting them would not narrow the answer, it
would replace LightRAG's `mix` mode with a different retrieval semantic.
DlightRAG keeps the native one.

## Multimodal Queries

This is the `/retrieve` visual path (`query_images`). Text queries go through
LightRAG `mix`, BM25, fused-candidate hydration, and reranking. Image-bearing
queries add a direct image vector path only when the configured
`models.embedding.input_modality` resolves to multimodal and its startup probe
succeeds:

```text
query + images
  |-- text query -> LightRAG mix + BM25
  `-- images -> multimodal embedding(context="query") -> image chunks
```

Sidecar visual chunks are embedded as fused (VLM description + image) document
vectors at ingestion. The executor splits them at provider input, token, and
image-byte limits without changing order. Query images are embedded image-only.
Known retrieval models always receive their official task semantics: provider
query/document fields, Gemini text prefixes, or Cohere's image input type;
symmetric protocols remain symmetric.

Images also produce VLM semantic text through LightRAG's multimodal sidecar
path. That text feeds BM25 and KG extraction. For successful drawing chunks,
visual similarity search uses the same LightRAG chunk id after DlightRAG
overwrites its vector with a fused text+image embedding, preserving sidecar
provenance and avoiding duplicate VLM text exposure. The image->image query leg
is lossless where the VLM-description text path is lossy, so partial overlap
between the two is expected and resolved by RRF, dedup, and reranking.

With `models.embedding.input_modality: text`, DlightRAG skips both image-vector
overwrite and query-image vector retrieval. Query images can still be described
by the VLM for text/BM25/KG retrieval, and document images still follow
LightRAG's native semantic multimodal path. Auto mode may make the same safe
downgrade after a failed native-provider probe; explicit multimodal mode fails
startup instead. See [Embedding configuration](configuration.md#embeddings) for
the provider and modality matrix.

## Reranking

`models.rerank.strategy` chooses the final ranker. DlightRAG does not pass
`rerank_model_func` into LightRAG; it disables LightRAG query reranking and
reranks the DlightRAG fused candidate set after provenance hydration. This lets
BM25-only hits, direct image matches, and LightRAG `mix` chunks compete in one
list with page/image data already attached.

| Strategy | How it works |
|---|---|
| `chat_llm_reranker` | Batched listwise scoring through the configured rerank model, or `models.chat.default` when no rerank model is set. With `input_modality: auto`, the selected scoring model reuses the startup vision probe: vision-capable models get bounded image payloads plus text; non-vision models get VLM text only. |
| `jina_reranker` | Calls Jina `/v1/rerank`. Default model `jina-reranker-v3` (text). Set `input_modality: multimodal` with `jina-reranker-m0` to send bounded image documents when chunks have `image_data`. |
| `aliyun_reranker` | Calls Alibaba Model Studio rerank. `qwen3-rerank` uses the compatible text payload; `qwen3-vl-rerank` with `input_modality: multimodal` uses the DashScope multimodal payload. `base_url` must point at the matching workspace/region endpoint. |
| `local_reranker` | Generic entry for any standard `/rerank` endpoint (self-hosted or hosted) in the `{model, query, documents, top_n} -> {results}` shape. `auto` is text; set `input_modality: multimodal` when the endpoint accepts image documents. |
| `voyage_reranker` | Calls Voyage AI `/v1/rerank` with text documents. |
| `cohere_reranker` | Calls Cohere `/v2/rerank` with text documents. |
| `azure_cohere` | Calls Azure AI Services Cohere rerank with text documents. Model endpoint roots use `/v1/rerank`; Foundry project roots use `/providers/cohere/v2/rerank`; a full `/rerank` URL is used as-is. |

When `models.rerank.score_threshold` is set, post-rerank filtering removes chunks below
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
image-capable HTTP rerankers bound each request with fixed rerank-stage image
size, byte, and quality limits before constructing model payloads. Visual chunks
whose images cannot fit fall back to their text, if present, rather than sending
unbounded data URIs.

## Answer Orchestration

One `AnswerOrchestrator` executes the durable Resolved Mode:

- **Fast Answer** — requested or routed `fast`. Planning, KB retrieval, and one
  lightweight model invocation use shared Context Contribution, Evidence,
  citation, Profile Memory, model-call, and usage infrastructure. Fast creates
  no Agent Session, workspace, tools, or publication.
- **Research** — requested or routed `research`. The product-neutral AgentLoop
  projects the canonical linear journal as a parent-linked selected-head view and selects from the
  run-local ToolRegistry: KB/resource/Web tools, optional path tools, parent
  Profile Memory tools, progressive `load_skill`, outbound MCP tools, and
  `spawn_agent` plus child status/wait/cancel. Foreground children may run in
  parallel with selected context/model/inherited tools; adopted Evidence is
  parent-citable and persists before spawn settlement. A no-tool assistant turn
  ends Research and its text is the answer. There is no hidden finalizer model
  call. Optional `artifacts/report.md` publishes as a handle-only report.

Both paths use deterministic citation/source finalization and expose the same
result shape, including `sources`, `answer_images`, `usage`, and Evidence counts.
Resource reads are deterministic first: `read` decodes UTF-8/CSV
directly and converts HTML/PDF/DOCX/PPTX/XLSX through selected MarkItDown
converters (plugins disabled, no network, OOXML zip-bomb preflight).
`inspect` performs focused VLM inspection through the VLM role (or the
default LLM), and marks every visual observation as VLM-derived evidence with its
exact source/page/sheet/cell locator. Full resource bytes never enter model
context — only bounded text windows, capped tool observations, and budgeted image
blocks do.

Each model call uses the immutable profile pinned for its normalized endpoint.
The Context Policy reserves output, observations, safety, retained tail,
episodic continuation, and minimum input directly; a provider input limit is
respected independently rather than nested under percentages. Evidence,
resource reads, schemas, history, and parallel observations share the measured
request residual. Before parallel tools execute, the exact next-control residual
is divided by call count. Provider output is capped by its own limit and
physical remaining context. Unknown endpoint capacity fails before acceptance.

## Answer Generation

The answer prompt receives:

- chunk text excerpts
- KG entities and relationships from LightRAG `mix`
- LightRAG's doc-level `reference_id`/`references` mapping as the seed for
  source numbering
- document/source metadata
- quality-preserving bounded inline page or image previews when available
- attachment evidence: bounded text windows from `read` and VLM-derived
  observations from `inspect`, each carrying its source locator

### Answer LLM Input Shape

The answer model does not receive the raw `contexts` JSON. `AnswerSynthesizer`
builds OpenAI-style messages with explicit evidence and task boundaries:

```python
[
    {"role": "system", "content": get_answer_system_prompt()},
    # optional server-prepared Web text history
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "## User-attached images\n"},
            {"type": "image_url", "image_url": {"url": "..."}},
            {"type": "text", "text": "## User-attached documents"},
            {"type": "text", "text": "### Document [att-1]: upload.pdf"},
            {"type": "text", "text": "[att-1-1] upload.pdf\nAttachment evidence..."},
            {"type": "text", "text": "## Knowledge-base evidence"},
            {"type": "text", "text": "### Document [1]: report.pdf"},
            {"type": "text", "text": "[1-1] report.pdf, Page 3\nEvidence text..."},
            {"type": "text", "text": '[1-2] "Doc Title" Page 4'},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {
                "type": "text",
                "text": (
                    "## Knowledge Graph Context\n..."
                    "\n\n## Question\nWhat are the key findings?"
                ),
            },
        ],
    },
]
```

The `## User-attached images` blocks are omitted when the request has no image
attachments. An image resource that cannot fit its budget is skipped and its
VLM-derived text observation remains. Server-prepared Web text history, when
present, is inserted as prior messages before the current user message.

The sections are intentional:

- `## User-attached images` are part of the user's question, not retrieved evidence.
- `## User-attached documents` contains attachment evidence read through
  `read`. Each answer assigns compact document labels such as `att-1`;
  its chunk markers use the `[att-1-1]` form.
- `## Knowledge-base evidence` contains LightRAG excerpts and page/image previews.
- Excerpt labels such as `[1-1] report.pdf, Page 3` give the model the citation marker it must use.
- Retrieved document images are preceded by a text label, then sent as an `image_url` block only if they fit the answer image budget.
- `## Knowledge Graph Context` gives entity/relationship facts, with source document tags when available.
- `## Question` is the actual user task and is placed last.

Every citation marker is defined exactly once, on the evidence it labels: `[n]`
on the `### Document [n]` heading and `[n-m]` on the excerpt label line. There
is no separate reference list, so the model cannot pick a marker from a
content-free menu without reading the excerpt it points at.

Attachment `att-N` labels are answer-scoped citation identities for the
request-local resources, not durable IDs. They are computed per answer from the
resources registered for that request; nothing about them is persisted as a
parsed chunk or vector.

Answer generation uses one image transport budget for current attachment images
and retrieved workspace visuals, bounded by `answer.generation.max_images` and the answer
byte/geometry fields. Focused inspection is a separate VLM call; every inspection
uses the same byte/geometry fields as per-call limits without consuming the final
answer budget. Budgeted JPEG, PNG, and WebP payloads are preserved as-is. When
recompression is needed, DlightRAG
enforces both a long-edge floor and a JPEG quality floor; an image that cannot
fit within those limits is skipped instead of being degraded into a low-quality
preview, and its text observation remains.

`/retrieve` and `/answer` both accept an explicit `chunk_top_k` request to
override the configured chunk/visual candidate budget; otherwise DlightRAG uses
`config.chunk_top_k`. For `/answer`, retrieval deliberately over-fetches those
candidates, then the answer stage packs evidence into the resolved query
model's remaining input capacity. `chunk_top_k` maps to LightRAG
`QueryParam.chunk_top_k`; LightRAG `top_k` remains the separate KG
entity/relationship breadth. Retrieved
visual chunks are admitted in reranked order within the answer image
budget. Pure visual chunks whose image cannot be sent are removed from the
answer context and the packer backfills from later candidates; mixed text+image
chunks keep their text even if the image is skipped. KG entities and
relationships are filtered to the packed chunk ids, so citation indexes,
streamed contexts, and returned sources describe the material
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
from validated inline markers. Validation has two outcomes. A marker resolving
to no chunk is dropped. A `[n-m]` marker resolving to an excerpt of nothing but
Markdown headings degrades to its document marker `[n]`: that excerpt states no
fact, so the claim is credited to the document rather than to a passage the
model only guessed at. Returned `sources` contain only
cited documents and chunks.
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

`answer.citations.highlights.enabled` is the global kill switch. When enabled, the
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

## Answer Attachments And Resources

Answer attachments are read as request-local resources; they are never parsed
into workspace documents, never written to LightRAG `full_docs`, `doc_status`,
chunks, vectors, BM25, LLM cache, or KG rows, and never enter `/retrieve`. A
request-local `ResourceRegistry` owns every resource for the lifetime of one
answer: inline bytes stay in memory, HTTPS links are fetched lazily and
revalidated on every live read (HTTPS-only, SSRF guard, per/total byte and pixel
limits). Fetched bytes settled on an effect already passed that gate and are
replayed from the Blob store without another network request. Full bytes never enter model
context — only bounded text windows, capped observations, and budgeted image
blocks do.

`read` is deterministic. UTF-8 and CSV text decode directly; HTML, PDF,
DOCX, PPTX, and XLSX are converted through selected MarkItDown converters with
plugins disabled and no network access. A fresh converter is built per call, and
OOXML archives pass a central-directory zip-bomb preflight (entry-count,
per-entry, total-size, and expansion-ratio limits) before any converter opens
them. Continuation cursors are opaque, request-local tokens bound to a resource
and focus; they expose no path, offset, or provider locator and never cross
requests. A cursor is single-use. Its compact durable state records the original
focus-plan budget, current rank position, and absolute character offset; the
deterministic focus plan is cached in memory and rebuilt once after recovery.
Changing a later observation budget therefore neither skips nor repeats text,
and consumed cursors do not accumulate in the journal.

`inspect` performs focused visual inspection through the VLM role
(falling back to the default LLM). Images are bounded through the one canonical
image path and `ImagePayloadBudget`; PDFs are rasterized with pypdfium2 off the
event loop as a bounded low-resolution overview and, on request, one higher-
resolution page. Every result is marked `derived_by_vlm` and carries its exact
source/page/sheet/cell/visual locator, so the model can cite where a claim came
from and never treats a VLM description as the final answer.

For a current source image, automatic image description, image-aware planning,
direct visual retrieval, and bounded final-model visibility happen before the
agent decides whether more research is needed. `inspect` is therefore
optional: it re-examines the bounded whole image with a concrete focus and adds
the result as located, citable VLM evidence. It does not currently accept a
bounding box or crop arbitrary regions. PDF page locators and embedded visual
handles provide the narrower structural inspection paths.

When `answer.web_search.api_key` (Exa) is set, the research path can also call Exa
Search as a peer tool. Exa passages come back already scored against the query;
they belong to no workspace and are packed beside corpus evidence. Unique URLs
that produced evidence become inert request-local handles, and only an explicit
`read` call fetches one under the normal SSRF, redirect, and byte
limits. Exa Contents is a bounded internal fallback when direct extraction
fails or is empty, not a model-visible tool. It does not supply cookies,
authenticated sessions, or Playwright interaction. Login-gated content must be
provided by the caller as attachment bytes or a screenshot. When the key is
unset, both Web capabilities are removed and answers stay corpus-only. A
rejected or unpaid key parks the capability for a short window rather than
retrying every turn.

The Web channel stores uploaded answer attachments once as owner-scoped
content-addressed blobs owned by the durable Answer run, not by a Web-owned
table. There is no parsed-chunk table and no vector cache. Historical attachments
are re-registered as lazy request-local resources on every follow-up, newest
first up to the available attachment-count limit. An attachment-bearing
conversation therefore remains on the research path. Browser thumbnails are
derived on demand. Manual delete and the shared run-retention floor delete
linked runs and release blobs no surviving run references; nothing crosses a
conversation or principal boundary.
