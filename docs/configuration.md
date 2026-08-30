# Configuration

This page is for operators and in-process Application users deciding which settings to change. It
owns configuration precedence, public field groups, defaults, and advanced
overrides. Runtime architecture lives in [architecture.md](architecture.md);
auth and access-control guidance lives in [security.md](security.md);
interface payloads live in [interfaces.md](interfaces.md).

Root [config.yaml](../config.yaml) is intentionally curated. It should contain
the product and deployment choices most operators actually change. The typed
configuration model still supports advanced overrides through constructor
arguments, `DLIGHTRAG_*` environment variables, `.env`, or explicit additions
to `config.yaml` when a deployment has a concrete reason.

Precedence:

```text
constructor args > environment variables > .env > config.yaml > code defaults
```

## Public Configuration Boundary

DlightRAG 3.0 keeps exactly eight top-level sections: `deployment`, `storage`,
`models`, `corpus`, `answer`, `access`, `interfaces`, and `observability`.
Nested environment variables use the same path with `__` separators. Removed
1.x flat names are rejected rather than aliased.

Keep these in normal `config.yaml`:

- model/provider choices and endpoint facts: `models.catalogue`, `models.chat`,
  `models.embedding`, `models.rerank`
- parser sidecar endpoint and visual context controls: `corpus.sidecars`
- metadata schema: fixed; custom metadata needs no declaration
- domain entity guidance: `corpus.retrieval.kg_entity_types`,
  `corpus.extraction.entity_type_prompt_file`
- PostgreSQL endpoint, process role, and workspace identity:
  `storage.postgres`, `deployment.service_role`, `deployment.workspace`
- high-level concurrency raised above upstream defaults: AI-provider `models.max_concurrency`,
  `answer.runtime.answer_worker_concurrency`, `corpus.ingestion.pipeline.max_concurrency`,
  `models.embedding.max_concurrency`, `models.embedding.batch_size`
- retrieval/answer controls: `corpus.retrieval.top_k`, `corpus.retrieval.chunk_top_k`, `corpus.retrieval.bm25_enabled`, `corpus.retrieval.direct_visual_top_k`,
  `answer.*`
- auth and observability mode switches when they are not secret

Keep these out of normal `config.yaml` unless debugging or load-testing proves
they need to change:

- storage backend literals
- raw LightRAG parser rules (derived internally from the active sidecar)
- PostgreSQL retry/backoff internals
- per-stage ingest worker counts (`corpus.ingestion.pipeline.max_parallel_*`) that match LightRAG defaults
- queue sizes
- HNSW index internals
- BM25 language profile signatures and k1/b tuning
- RRF and exact-vector filter thresholds
- image compression budgets
- source highlight timeout/cache internals
- query-image semantic description limits
- visual thumbnail cache internals
- remote source URL signing expiry/region

## Parser Routing And Sidecars

The configured external sidecar selects the parser automatically. Configure
exactly one `docling` or `mineru` block; DlightRAG derives the internal LightRAG
wildcard. With neither block, the code default is Docling at
`http://127.0.0.1:5001`. If both are present, MinerU remains effective for
backward compatibility, so do not configure both. Docling and MinerU are durable
ingestion parsers only; answer attachments are decoded and converted
request-locally and never invoke them.

The checked-in Docker-first configuration consumes the host-native
Docling Serve MPS service rather than enabling the optional Compose profile:

```yaml
corpus:
  parser:
    chunk_options: {}
  extraction:
    use_json: true
  sidecars:
    vlm:
      enabled: true
      max_image_bytes: 5242880
      # DlightRAG default 80px, above LightRAG's native 64px minimum.
      min_image_pixel: 80
    docling:
      endpoint: http://host.docker.internal:5001
      code_formula_preset: granite_docling
      # force_ocr: false
```

`corpus.extraction.language` defaults to `English`. It is a free-form target
language inserted into LightRAG's prompts for generated entity and relation
descriptions, summaries, and retrieval keywords—not a switch that adds general
language support. The configured models, embeddings, parser/OCR, and lexical
retrieval must support the corpus language independently. Keep the default to
canonicalize a multilingual graph in English, or override it when another graph
output language is intentional. Changing it does not translate existing graph
data; reset and reingest the corpus for a consistent change.

Changing the selected parser affects new parses; it does not rewrite an
existing workspace's chunks, vectors, graph, or parser cache. Explicitly reset
and reingest a corpus when it must be rebuilt with Docling artifacts.

`corpus.sidecars.docling.do_formula_enrichment` transcribes detected formula
regions and defaults on, matching MinerU's `enable_formula`, so the parser
choice does not decide whether a corpus keeps its mathematics. Turning it off
drops formulas silently rather than erroring, so turn it off only on a corpus
without mathematics.

`corpus.sidecars.docling.code_formula_preset` names the model that transcribes
them. The code default matches the host-native Apple Silicon service; explicitly
set YAML `null` for other devices:

| Parser service device | `code_formula_preset` |
| --- | --- |
| CUDA, XPU, or CPU | `null` — use Docling's built-in `codeformulav2` |
| MPS | `granite_docling` |

Docling's default formula model cannot run on MPS, so enrichment fails on Apple
Silicon without `granite_docling`. The setup wizard asks for the service device
instead of inferring it from the endpoint. Repointing the preset invalidates the
Docling bundle cache, so affected documents re-parse on their own. The
`DOCLING_SERVE_ALLOWED_CODE_FORMULA_PRESETS` allowlist matters only if an
operator narrowed it; a stock docling-serve allows every preset.

DlightRAG always sends `do_pdf_heading_hierarchy`, which infers section-header
levels from the PDF bookmarks, outline numbering and font style. docling-serve
defaults it off, and without it Docling leaves every heading at level 1, so a
chunk's section breadcrumbs collapse to the document title alone and the
retrieved context loses the chapter it came from. There is no setting because
turning it off only degrades the corpus. It needs **docling-serve 1.30.0 or
newer** (docling-jobkit 3.3.0 is the first release that maps the field onto the
pipeline); an older service accepts the field and drops it silently, so verify
the service version.

DlightRAG does not select Docling's OCR engine or forward `ocr_lang`; that policy
belongs to the independently operated service. The checked-in MPS deployment
uses OCRMac with Simplified Chinese and English recognition, while the optional
CPU image retains its own OCR default.

`corpus.sidecars.docling.force_ocr` re-runs OCR over the whole page and discards
the PDF's embedded text layer. docling-serve defaults it off; DlightRAG defaults
it on because a PDF whose CID fonts carry no Unicode mapping — common in
Chinese typesetting output — renders correctly yet extracts as mojibake, and
that text reaches chunks, the KG and citations with nothing to flag it. Set it
to `false` for a corpus of well-encoded born-digital PDFs, where the embedded
text layer is lossless and OCR can only degrade it. Flipping it invalidates the
Docling bundle cache, so affected documents re-parse on their own.

Both external parser clients poll every 5 seconds for at most 1440 attempts, a
two-hour wait budget. **The parser service's HTTP keep-alive must exceed that
interval.** The clients reuse a pooled connection, so a keep-alive equal to the
poll interval makes every poll race the server's connection close, and a long
parse eventually dies with `Server disconnected without sending a response`
while the parser keeps working, unaware. docling-serve already ships 60 seconds;
`scripts/mineru/sitecustomize.py` raises the local MinerU sidecar to match. A
parser service DlightRAG does not launch must be configured the same way
(docling-serve: `--timeout-keep-alive`).

OCR, formula enrichment, output formats, referenced images, raw-bundle
validation, and retry semantics remain owned by LightRAG and the parser service.
Dockerized DlightRAG reaches a host-native service through
`host.docker.internal`; a native DlightRAG process uses `127.0.0.1` when the
parser runs on the same host. The optional Compose profile starts with
`docker compose --profile docling up -d`; point its block to
`http://docling:5001` and set `code_formula_preset: null`.

To use MinerU instead, remove the Docling block and configure only:

```yaml
corpus:
  sidecars:
    mineru:
      api_mode: local
      local_endpoint: http://host.docker.internal:8210
      language: ch
      backend: hybrid-engine
```

The local sidecar installer supports MinerU 3.4.5 through the reviewed 3.x API
range. For an existing checkout, remove or update an older `MINERU_VERSION` /
`MINERU_MIN_VERSION` override in `.env.mineru`, then upgrade and restart:

```bash
make mineru-service-stop
make mineru-install
make mineru-service-start
```

MinerU 3.4.5 corrects special-character preservation in DOCX tables and
supplementary-plane Unicode extraction from PDFs. Existing indexed documents do
not change when the sidecar package changes; reset and reingest affected sources
when those corrections matter.

`corpus.sidecars.vlm` owns figure understanding, and MinerU's own image analysis
is deliberately left off. MinerU extracts each figure as a crop; the VLM sidecar
then describes that crop together with the surrounding text. Enabling MinerU's
analysis would run a second VLM over the same image for roughly 58% more parse
time and largely duplicate content, so there is no setting for it. A parse
therefore emits zero `chart` blocks by design: figures arrive as `image` blocks
and become `drawing` chunks carrying the sidecar's description.

The sidecar only ever sees the figures MinerU cut, so the `hybrid-engine`
backend's effort setting decides what it gets. MinerU's own default, `medium`,
consumes precomputed layout boxes and can split a dense multi-panel figure into
fragments. `high` lets the VLM detect blocks itself, returning whole figures with
correctly bound captions, at roughly 5x the parse time. Set
`MINERU_HYBRID_EFFORT=high` in `.env.mineru` for figure-heavy corpora.

`corpus.sidecars.mineru.language` is MinerU's OCR language hint for scanned or
image-based documents. It is separate from `corpus.extraction.language`, which
controls the target language of LightRAG's generated graph content and retrieval
keywords.

`corpus.sidecars.mineru.backend` selects MinerU's parse engine and defaults to
`hybrid-engine`, MinerU's current VLM-assisted default. Accepted values are
`pipeline`, `vlm-engine`, and `hybrid-engine`. Use `pipeline` (MinerU's non-VLM
OCR engine) to avoid VLM transcription artifacts on difficult scans, at the cost
of weaker complex-layout and chart handling. DlightRAG always maps the selected
value privately to `MINERU_LOCAL_BACKEND`, avoiding LightRAG's legacy default.
Public environment overrides use the typed `DLIGHTRAG_CORPUS__SIDECARS__...`
form; raw MinerU/Docling/VLM variables are not independent configuration inputs.

DlightRAG does not expose MinerU-side image/chart analysis as a product setting;
LightRAG 1.5.5 defaults that parser-time path off, while LightRAG's separate
multimodal analyze stage handles images, tables, and equations after parse.

## Embeddings

Embedding configuration defines the one canonical vector space shared by
LightRAG ingestion and every DlightRAG retrieval leg. `provider` names a wire
protocol, not a company account or deployment type. DlightRAG never infers it
from a model name, URL, or port.

### Provider matrix

Each adapter owns its complete request URL, authentication, model capability
table, payload, response ordering, and usage extraction.

| `provider` | First-class model | Request route | Canonical visual vector | Retrieval task mapping | `dim` wire behavior |
|---|---|---|---|---|---|
| `openai` | `text-embedding-3-large` | `/embeddings` | Text only | Symmetric | `dimensions` for `text-embedding-3-*`; fixed models validate only |
| `openai_compatible` | Caller-defined | `/embeddings` | Text only | Symmetric | Never sent; response dimension is validated |
| `voyage` | `voyage-multimodal-3.5` | `/multimodalembeddings` | Native text+image fusion | `query` / `document` | `output_dimension` |
| `gemini` | `gemini-embedding-2` | `/models/{model}:embedContent` | Native content aggregation | Official query/document text prefixes | `outputDimensionality` |
| `jina` | `jina-embeddings-v4` | `/v1/embeddings` | Native text+image fusion | `retrieval.query` / `retrieval.passage` | `dimensions` |
| `cohere` | `embed-v4.0` | `/v2/embed` | Native mixed-input fusion | Text uses `search_query` / `search_document`; image-bearing input uses `image` | `output_dimension` |
| `azure_cohere` | `Cohere-embed-v4` | `/v1/embed` | Native mixed-input fusion | Text uses `search_query` / `search_document`; image-bearing input uses `image` | `output_dimension` |

Jina v4 is intentional: the newer Jina v5 Omni protocol exposes aligned image
and text inputs but does not document one native fused text+image output. A
DlightRAG multimodal provider must preserve the single canonical chunk-vector
invariant; it cannot add a second visual-only document representation.

Unknown model names are allowed for private deployments, but resolve
conservatively to text-only operation with no upstream dimension narrowing.
`openai_compatible` is deliberately minimal: model, text input, float response,
and optional Bearer authentication. It does not accept image data URIs or
invent vendor-specific dimension/task fields.

`openai` also supports the Azure OpenAI v1 root ending in `/openai/v1`; Azure
API-key endpoints receive the `api-key` header. `azure_cohere` accepts a full
official URL ending in `/embed` as-is, otherwise appends `/v1/embed` to a known
Azure Foundry deployment scoring root. Unknown Azure roots fail explicitly.

### Fields

| Field | Default | Meaning |
|---|---|---|
| `provider` | `voyage` | One protocol from the matrix. Unknown values fail configuration loading. |
| `model` | `voyage-multimodal-3.5` | Exact model or deployment identifier. |
| `api_key` | None | Provider credential. Prefer `DLIGHTRAG_MODELS__EMBEDDING__API_KEY` in `.env`. |
| `base_url` | Voyage v1 root | Protocol root or accepted complete endpoint. An omitted native URL uses the adapter default. |
| `dim` | `1024` | Final vector/schema dimension. Known adapters decide whether to send it upstream; every response is validated. |
| `max_token_size` | `8192` | Local per-input ceiling and LightRAG-advertised limit. The stricter provider/model limit always wins. |
| `input_modality` | `auto` | `auto`, `text`, or `multimodal`. This local policy is never serialized upstream. |
| `startup_probe` | `true` | When multimodal routing is active, verify both image-only query embedding and description+image fused document embedding. |
| `timeout` | `120` | Per-request timeout in seconds. |
| `max_concurrency` | `16` | Calls admitted through the process-wide model scheduler. |
| `batch_size` | `64` | Local maximum. Execution also splits at provider input-count and total-token limits. |

OpenAI known models use `tiktoken` for exact local budgets. Other adapters use
the shared estimator with a safety margin. An over-limit input fails locally;
it is never silently truncated. Batches auto-split on input count, total tokens,
and known combined inline-image limits (Cohere v4: 20 MB) while preserving input
and response order. OpenAI-shaped responses require a complete, unique `index`
cover and are reordered by that index.

Connection failures, HTTP 408/409/429, and 5xx responses receive at most two
retries. `Retry-After` wins over exponential backoff with jitter. Other 4xx,
invalid input, schema, index, dimension, and vector-value failures are never
retried. Numeric provider usage, request count, and retry count are attached to
embedding telemetry.

### Input modality

| First-class model capability | `auto` | `text` | `multimodal` |
|---|---|---|---|
| Native single-vector fusion (`voyage`, `gemini`, `jina` v4, `cohere`, `azure_cohere`) | Enable image-query retrieval and fused visual-vector overwrite | Disable both image paths | Require both paths; startup probe failure stops startup |
| Text-only or conservative unknown model (`openai`, `openai_compatible`, unknown names) | Text only | Text only | Fail configuration/runtime construction |

`text` leaves LightRAG's VLM-description chunk vector intact and disables the
raw-image query leg. `multimodal` upgrades that same canonical vector by
replacing it with one native description+image fused vector; it never creates a
second visual document vector. BM25, KG, provenance, chunk identity, filtering,
and citations continue to use the original canonical chunk.

In `auto`, a failed live fusion probe safely leaves the LightRAG text vector in
place. Explicit `multimodal` treats the same failure as fatal.

### Examples

Voyage native multimodal embeddings:

```yaml
models:
  embedding:
    provider: voyage
    model: voyage-multimodal-3.5
    base_url: https://api.voyageai.com/v1
    dim: 1024
    max_token_size: 8192
    input_modality: auto
    startup_probe: true
```

OpenAI official text embeddings:

```yaml
models:
  embedding:
    provider: openai
    model: text-embedding-3-large
    base_url: https://api.openai.com/v1
    dim: 3072
    input_modality: text
```

Cohere Embed v4 native fusion:

```yaml
models:
  embedding:
    provider: cohere
    model: embed-v4.0
    base_url: https://api.cohere.com
    dim: 1536
    input_modality: auto
```

LM Studio or another conservative OpenAI-compatible text server:

```yaml
models:
  embedding:
    provider: openai_compatible
    model: text-embedding-nomic-embed-text-v1.5
    base_url: http://127.0.0.1:1234/v1
    dim: 768
    input_modality: text
```

For local servers, `model` must match the identifier exposed by the running
embedding endpoint; loading a chat-only model is insufficient.

### Docker host access

When DlightRAG runs directly on the host, local services normally use
`127.0.0.1`. Inside this repository's Compose containers, `127.0.0.1` means the
container itself. Compose configures the `host.docker.internal` alias for
host-side services, so an LM Studio embedding endpoint on the host uses:

```yaml
base_url: http://host.docker.internal:1234/v1
```

### Changing the vector space

Do not mix vectors produced by different models, dimensions, or embedding
spaces in one workspace. Use a new workspace, or recreate/migrate the vector
schema as needed and perform a complete offline rebuild. See
[Operations](operations.md#offline-vector-storage-rebuild) for the rebuild
procedure and [PostgreSQL](postgresql.md#required-version) for the dimension
constraint.

## LLM Providers

`provider` names the API protocol and SDK family DlightRAG speaks — not the
vendor brand. It accepts exactly three values (case-insensitive):

| `provider`  | Transport               | Use for                                                                                                                | `base_url`      |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------- |
| `openai`    | OpenAI Chat Completions | OpenAI, DeepSeek, OpenRouter, Azure OpenAI, MiniMax, Qwen, Zhipu, vLLM/Ollama, and any other OpenAI-compatible endpoint | Vendor endpoint |
| `anthropic` | Anthropic native SDK    | Anthropic Claude                                                                                                       | Omit (native)   |
| `gemini`    | Google GenAI SDK        | Google Gemini                                                                                                          | Omit (native)   |

Pick the vendor through `base_url`, never through `provider`. DeepSeek and
OpenRouter are `provider: openai` plus their `base_url` — there is no
`provider: deepseek` or `provider: openrouter`, and any unknown value is
rejected when the config loads.

### Model catalogue and reasoning profiles

Each chat model resolves an endpoint-scoped profile containing its context
window (`C`), optional maximum input (`I`), optional maximum output (`O`), image
capability, and an optional reasoning profile. DlightRAG matches the normalized
`provider`, exact `model`, and normalized `base_url`; the same model name at a
different endpoint is a different profile. The built-in catalogue is ordered by
upstream vendor and model; each native endpoint immediately precedes its
OpenRouter counterpart.

Resolution is deterministic: a PostgreSQL runtime overlay has highest
precedence, followed by startup entries from `models.catalogue`, the versioned
built-in JSON catalogue, and finally a permissive fallback profile. Both startup
and runtime rows are complete profiles, not field patches. DlightRAG does not
probe model endpoints or persist inferred limits after a provider rejection.

Startup entries are suitable for version-controlled deployment configuration:

```yaml
models:
  catalogue:
    - provider: openai
      model: vendor/new-model
      base_url: https://api.vendor.example/v1
      profile:
        context_window_tokens: 262144
        max_input_tokens: null
        max_output_tokens: 32768
        supports_images: true
        reasoning:
          format: openai
          levels:
            off: none
            minimal: null
            low: low
            medium: medium
            high: high
            xhigh: null
            max: null
  chat:
    default:
      provider: openai
      model: vendor/new-model
      base_url: https://api.vendor.example/v1
```

Startup entries are installed before the composed application can start or
resolve model profiles and require a restart to change. PostgreSQL runtime
updates are hot, publish atomically with
an optimistic revision and `NOTIFY`, and clear every process's capability
cache. Startup and reconnect synchronize the current runtime value. Invalid
updates, stale revisions, and updates that would make a configured role invalid
are rejected before publication.

Administrators can read and edit the effective runtime catalogue through:

- REST: `GET|PUT|DELETE /models/catalogue`
- Web: Settings → Runtime Model Catalogue
- MCP: `get_model_catalogue`, `upsert_model_catalogue_entry`, and
  `remove_model_catalogue_entry`

A write sends the revision returned by the preceding read (`If-Match` for
HTTP, `expected_revision` for MCP). Each PUT contains `provider`, `model`,
optional `base_url`, and one complete `profile`. A DELETE removes only the
runtime overlay; the endpoint then resolves to its startup-configured profile,
or to its built-in profile when no startup entry exists. Catalogue writes use
the admin-only `model_catalogue.write` action and are disabled entirely on
reader-only deployments.

A reasoning profile is either `null` (unsupported) or a request `format` plus
an explicit mapping for all seven levels: `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max`. Each mapping value is the provider-native value or
`null` when that level is unavailable. `off: null` means reasoning cannot be
disabled. A non-off request is deterministically clamped to the nearest
supported level; an impossible `off` request is a configuration error.
Supported request formats are `openrouter`, `openai`, `deepseek`,
`anthropic`, and `gemini`; an unknown format is rejected rather than silently
dropping a reasoning control.

An uncatalogued endpoint has no verified effort map. Typed reasoning therefore
uses a best-effort protocol mapping inferred from the provider and endpoint:
OpenRouter, native Anthropic, native Gemini, DeepSeek, or otherwise OpenAI.
DlightRAG sends the requested level without local clamping and surfaces any
provider rejection; it never silently drops the requested control. Internal
compaction does not opt into an unverified reasoning mapping.

Model configuration uses only the typed, provider-independent levels:

```yaml
models:
  chat:
    default:
      reasoning: max
    roles:
      query:
        reasoning: "off"
        agentic_reasoning: high
```

`agentic_reasoning` inherits `reasoning` when omitted. Compaction requests the
cheapest supported level rather than assuming every endpoint accepts `off`.
The selected profile format is the only owner of translation to provider-native
request kwargs.

Capacity arithmetic is owned by one immutable, revisioned policy. It applies
explicit output (16,384), dynamic-context (40,000), safety (1,024), retained-tail
(20,000), episodic-summary (8,000), and minimum-input (1,024) token reserves.
Research may clamp its dynamic reserve for small profiles; Fast requires the
full 40,000-token reserve and is removed from automatic routing (or rejected
when explicitly requested) when its fixed envelope cannot preserve it. The
provider input limit and physical context remain independent ceilings; a known
output limit is capped by physical context left. Capacity is not configured
under `answer`, and there are no nested percentages or independent evidence
ratios.

### LLM Structured Output

Planner and other small control-plane calls pass a `StructuredOutput` contract
through the shared LLM factory. Model configuration decides which provider
request format is used:

```yaml
models:
  chat:
    roles:
      extract:
        provider: openai
        model: deepseek-v4-flash
        base_url: https://api.deepseek.com
        structured_output: json_object
```

`structured_output` defaults to `auto`, which requests the strongest
schema-constrained output path for OpenAI and OpenAI-compatible endpoints,
Anthropic native `output_config.format`, and Gemini native `response_schema`.
For `provider: openai`, a failed strict JSON Schema request is retried once as
`json_object` with the required JSON instruction. An explicit `json_schema` is
therefore normally redundant. Set `structured_output: json_object` only for an
endpoint already known not to support strict schemas, avoiding a predictably
failed first request. Anthropic native does not support the lower-confidence
`json_object` mode; use `auto` or `json_schema`.

`model_kwargs` apply to ordinary calls. `agentic_model_kwargs` are a shallow
top-level overlay used by research control and final calls. They remain an
explicit escape hatch for provider options that DlightRAG does not type.
Reasoning parameters have one owner: when typed `reasoning` (or inherited/typed
`agentic_reasoning`) is configured, raw reasoning keys such as `reasoning`,
`reasoning_effort`, `thinking`, `thinking_config`, `enable_thinking`, and
`chat_template_kwargs` are rejected at config load. When no typed reasoning is
configured, those raw keys remain available for an unknown endpoint.

Research final generation starts with the agentic overlay. If the provider
finishes without user-visible text, DlightRAG retries once with ordinary
`model_kwargs`; a second empty response fails instead of storing an empty
answer. If `roles.query` is absent or incomplete, both sets of options come from
`models.chat.default` through the normal role fallback.

## Remote Source URLs

`source_uri` identifies the source; `download_uri` tells DlightRAG how to
retrieve the original bytes when no local copy is retained. The two values are
independent: connector-specific identities such as `bynder://asset/...` are
valid provenance but are not download locations.

By default, Azure Blob, S3, URL, and SDK connector files are not copied into
DlightRAG storage. A non-retained document therefore needs a durable S3, Azure,
or queryless public HTTPS `download_uri`. Set
`corpus.ingestion.retain_remote_source_files: true` to keep fetched files under the workspace
input root by default, or pass `retain_source_file=true` on one SDK/REST/MCP
ingest call. Retained sources use that local copy for download instead.

Query- or fragment-bearing signed HTTPS fetch URLs are ephemeral. Use
`retain_source_file=true` or provide a separate queryless `download_uri`; the
signed token is never persisted as an implicit locator. A non-retained custom
`AsyncDataSource` must set `SourceDocument.download_uri` or provide
`download_uri_for_key`. Invalid or missing locators are rejected before parser
materialization. DlightRAG never silently retains bytes to rescue an invalid
request.

REST `GET /files/raw/{document_id}` and Web
`GET /web/api/files/raw/{document_id}` are separate authenticated projections. Each
resolves the exact workspace metadata row server-side, then serves a retained
local file or redirects through a supported provider locator. Azure uses
`DLIGHTRAG_CORPUS__SOURCES__BLOB_CONNECTION_STRING`. S3 uses the standard AWS credential chain
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_REGION`/`AWS_DEFAULT_REGION`, IAM role, or shared AWS config).
REST/MCP `source_type="url"` accepts public HTTP or HTTPS URLs, does not rewrite
schemes, does not follow HTTPS to HTTP or redirects to private hosts, and caps each download with
`corpus.ingestion.url_max_bytes`. SaaS APIs that require auth headers must stage content
through a supported local, Azure Blob, or S3 source, or expose a public HTTP(S)
fetch URL. `source_uri`/`source_uris` set stable identity; they do not substitute
for the durable locator required by a non-retained signed fetch.
Set `corpus.ingestion.url_private_host_allowlist` only for trusted enterprise hosts that
must be fetched by REST/MCP URL ingest. Entries are host/IP patterns such as
`docs.corp.example`, `*.corp.example`, or `10.0.0.5`.

Remote prefix ingest streams provider listings into bounded local staging
windows. It uses the same ingest job substrate as local and single-object ingest,
while keeping source ownership in the cloud provider. DlightRAG delete/reset
operations remove DlightRAG metadata, LightRAG storage, and local parser
artifacts only; they do not delete Azure Blob, S3, or URL source objects.

Advanced signing defaults:

```yaml
corpus:
  ingestion:
    retain_remote_source_files: false
    url_max_bytes: 104857600
    url_private_host_allowlist: []
  sources:
    azure_sas_expiry: 3600
    s3_presign_expiry: 3600
    s3_region:
```

## PostgreSQL

Core storage is PostgreSQL 18 only. The backend literals are code defaults and
should normally stay out of `config.yaml`:

```yaml
storage:
  lightrag:
    vector_storage: PGVectorStorage
    graph_storage: PGTableGraphStorage
    kv_storage: PGKVStorage
    doc_status_storage: PGDocStatusStorage
```

Advanced PostgreSQL and index tuning:

```yaml
storage:
  lightrag:
    vector_index_type: HNSW_HALFVEC
    hnsw_m: 32
    hnsw_ef_construction: 256
    hnsw_ef_search: 256
  postgres:
    lightrag_pool_max_size: 16
    pool_min_size: 2
    pool_max_size: 16
    session_settings: {}
    statement_cache_size:
    connection_retries: 10
    connection_retry_backoff: 3.0
    connection_retry_backoff_max: 30.0
    pool_close_timeout: 5.0
```

`storage.postgres.pool_max_size` sizes the DlightRAG domain-store pool (BM25, metadata,
conversations, jobs, answer runs); `storage.postgres.lightrag_pool_max_size` sizes the
LightRAG backend pool. Each process opens up to the sum of the two, so multiply
by the worker count and keep the total under PostgreSQL `max_connections`. Raise
`storage.postgres.pool_max_size` for high single-worker concurrency; lower it when
running many workers.

### Process role (writer / reader)

`deployment.service_role` selects what a process may do with its single PostgreSQL endpoint:

```yaml
deployment:
  service_role: writer   # default: ingest + all APIs, and owns schema migrations
```

- `writer` (default) provisions schema, ingests, and serves every API.
- `reader` is **corpus-read-only, not process-read-only**. It creates and
  executes durable answer runs, writes DlightRAG operational state (runs, events,
  artifacts, Web conversations), and serves the bundled Web surface. Its LightRAG
  pool keeps `default_transaction_read_only=on` and the no-DDL attach path, the
  LightRAG response cache stays disabled, and ingestion, workspace
  creation/reset, metadata mutation, retry, and deletion are still rejected with
  HTTP 403.

Both roles point at the **same primary endpoint**; DlightRAG makes no
physical-standby or read-endpoint promise. A reader validates the migrated schema
at startup and issues no DDL, so apply writer migrations first and then roll
readers. An operator-set `default_transaction_read_only=on` on the domain session
fails `/ready` for both roles. See
[postgresql.md](postgresql.md#service-roles-and-shared-artifacts) for the full
deployment and shared-artifact contract.

Multi-host deployments must mount one shared POSIX `deployment.working_dir` at the **same
absolute path** in every process that serves KB images or retained source
downloads.

Use [postgresql.md](postgresql.md) for production sizing, SSL, shared memory, and extension
notes.

## Ingestion Concurrency And Queues

`config.yaml` keeps only the high-level AI, Runtime, and RAG concurrency knobs.
`models.max_concurrency` bounds all provider requests through the process-wide fair AI
scheduler. `answer.runtime.answer_worker_concurrency` bounds claimed durable Answer
runs executed by one process. `corpus.ingestion.pipeline.max_concurrency` bounds each workspace's
LightRAG pipeline width; its provider requests still pass through the AI
scheduler. `models.embedding.max_concurrency` and `models.embedding.batch_size` shape LightRAG's
embedding work without changing either worker admission or the global provider
cap. The
per-stage worker counts below already match LightRAG's defaults, so they are
omitted from `config.yaml` and follow DlightRAG's code defaults; set them
explicitly (in `config.yaml` or via `DLIGHTRAG_*` env) only when a deployment
needs different parallelism:

```yaml
corpus:
  ingestion:
    pipeline:
      max_parallel_insert: 3
      max_parallel_parse_native: 5
      max_parallel_parse_mineru: 2
      max_parallel_parse_docling: 2
      max_parallel_analyze: 5
```

Queue sizes are internal backpressure settings and should only change after
measuring parser/analyze/insert pressure:

```yaml
corpus:
  ingestion:
    pipeline:
      queue_size_parse: 20
      queue_size_analyze: 100
      queue_size_insert: 4
models:
  embedding:
    timeout: 120
```

`models.embedding.batch_size` is a local upper bound, not a promise that one
HTTP request carries that many inputs. The embedding executor automatically
splits at the selected model's input-count and total-token limits (for example,
Voyage accepts up to 1000 inputs, OpenAI 2048, Cohere 96, and synchronous
Gemini Embedding 2 one). Result vectors are reassembled in original input order.

## BM25

BM25 is part of the supported DlightRAG retrieval path. BM25 candidate breadth
follows the configured chunk candidate budget. `/retrieve` does not re-cap
fused chunks after the LightRAG mix/BM25 merge; `/answer` packs final prompt
evidence against the resolved query model's remaining input capacity. Retrieval
logs call the pre-fusion LightRAG lane `lightrag_mix_chunks`, and the trace
reports `lightrag_mix_chunk_count`; neither name changes a retrieval budget.
Language profiles and scoring constants are advanced index signatures.

Defaults:

```yaml
corpus:
  retrieval:
    bm25_enabled: true
    bm25_profiles:
      - name: zh
        text_config: public.jiebacfg
        languages: ["zh"]
      - name: en
        text_config: english
        languages: ["en"]
      - name: de
        text_config: german
        languages: ["de"]
      - name: sv
        text_config: swedish
        languages: ["sv"]
      - name: es
        text_config: spanish
        languages: ["es"]
      - name: fr
        text_config: french
        languages: ["fr"]
      - name: it
        text_config: italian
        languages: ["it"]
      - name: pt
        text_config: portuguese
        languages: ["pt"]
      - name: nl
        text_config: dutch
        languages: ["nl"]
      - name: ru
        text_config: russian
        languages: ["ru"]
      - name: da
        text_config: danish
        languages: ["da"]
      - name: fi
        text_config: finnish
        languages: ["fi"]
      - name: simple
        text_config: simple
        fallback: true
    bm25_k1: 1.2
    bm25_b: 0.75
```

`corpus.retrieval.bm25_enabled` controls workspace PostgreSQL BM25 indexing, ingest-time
language labels, and query fusion. It applies to the workspace knowledge-base
lane only; the answer research path reads attachments through request-local
resources that never touch workspace indexes.

Changing profile names, text configs, languages, `corpus.retrieval.bm25_k1`, or `corpus.retrieval.bm25_b`
changes the expected pg_textsearch index signature. Enabling BM25 for an
existing corpus or changing profile languages also requires relabeling existing
chunks; restarting alone does not rewrite historical labels. Use the offline
workspace BM25 rebuild described in [operations.md](operations.md#workspace-bm25-rebuild).

## Fusion And Filtering

Advanced retrieval scoring:

```yaml
corpus:
  retrieval:
    rrf_k: 60
    metadata_filter_exact_vector_threshold: 8192
```

`corpus.retrieval.metadata_filter_exact_vector_threshold` controls when DlightRAG can use exact
vector scoring inside a small metadata candidate set.

## Image Budgets

`answer.generation.max_images` and the answer byte/geometry fields define one image
transport budget for every answer, across REST, MCP, Web, and in-process Application. That single
budget covers current attachment images and retrieved workspace visuals.
Focused VLM inspection is a separate model call: each inspection applies the
same byte/geometry limits independently and does not consume the final answer
transport budget.
At startup the configured shape is clamped to the query-role model's discovered
image capability. Compression budgets are advanced model transport limits:

`chat_llm_reranker` can use its own `models.rerank.provider` and `models.rerank.model`. When
those are omitted, it reuses `models.chat.default`.

Voyage's text reranker is available with `strategy: voyage_reranker`,
`model: rerank-2.5` or `rerank-2.5-lite`, and `DLIGHTRAG_MODELS__RERANK__API_KEY`.
Cohere's public text reranker is available with `strategy: cohere_reranker`,
`model: rerank-v4.0-pro` or `rerank-v4.0-fast`, and the same API key env var.
When a provider reranker is explicitly selected, missing credentials are a
configuration error and fail service initialization rather than falling back to
`chat_llm_reranker`.

`models.rerank.input_modality` defaults to `auto`. For `chat_llm_reranker`, auto
reuses the startup vision probe for the selected scoring model: vision-capable
models receive bounded image data plus text, and non-vision models receive VLM
text only. HTTP rerankers have no reliable capability probe (the API returns a
relevance score whether or not it read the image), so DlightRAG does not guess
per model -- `input_modality` is the whole signal: `auto` resolves to `text`,
and `multimodal` is an explicit opt-in. Set `input_modality: multimodal` for an
image-capable rerank protocol (`jina-reranker-m0`, `qwen3-vl-rerank`, or a
self-hosted `/rerank` endpoint that accepts image documents). A text-only
strategy (`voyage_reranker`, `cohere_reranker`, `azure_cohere`) rejects
`multimodal` at startup rather than sending images its API cannot read.

```yaml
models:
  rerank:
    strategy: chat_llm_reranker
    input_modality: auto
    max_concurrency: 8
    batch_size: 8
answer:
  generation:
    max_images: 12
    max_attachments: 6
    max_attachment_bytes: 104857600
    max_total_attachment_bytes: 134217728
    image_max_bytes: 3000000
    image_max_total_bytes: 24000000
    image_max_px: 1536
    image_max_pixels: 40000000
    image_min_px: 1024
    image_quality: 89
    image_min_quality: 79
```

`answer.generation.max_attachments` (6),
`answer.generation.max_attachment_bytes` (100 MiB), and `answer.generation.max_total_attachment_bytes`
(128 MiB) bound answer attachment admission. `query_images` remains the
retrieve-only current-image path.

`answer.generation.image_max_pixels` rejects source images whose decoded dimensions exceed
the limit before RGB conversion or resizing. The Web upload validator,
request-local resource inspection, retrieve query-image description, and final
answer transport use the same ceiling.

## Answer Attachments And Web Conversations

Answer public inputs are **attachments**, not query images. REST, the Python
REST, MCP, Web, and in-process Application attach files and HTTPS references that become
request-local resources for the lifetime of one answer. `max_attachments`,
`answer.generation.max_attachment_bytes`, and `answer.generation.max_total_attachment_bytes` (above) bound admission
on every channel. Attachments are read on demand — deterministic UTF-8/CSV
decoding and MarkItDown conversion of HTML/PDF/DOCX/PPTX/XLSX first, then focused
VLM inspection of figures — and their full bytes never enter model context.

`query_images` is a separate, retrieve-only current-image path. `/retrieve`
accepts at most three current-request query images, a fixed public
contract shared by REST and MCP. Those images are described with the VLM for
text retrieval and embedded directly for the visual retrieval leg. They do not
share an answer budget.

Answer images arrive only as attachments/resources. `answer.generation.max_attachment_bytes`
governs original upload admission, `answer.generation.max_images` is capability-clamped at
runtime, and the `answer.generation.image_*` fields bound the compressed payload sent to a
model. Public REST, MCP, CLI, and Python calls require no client-managed
conversation ID, but each accepted Answer run durably pins its bounded history
and resources for recovery and follow-up/fork. Web additionally owns the
principal-scoped conversation index:

```yaml
corpus:
  visual_assets:
    thumb_max_px: 300
    thumb_cache_size: 256
```

Web conversations have no retention knobs of their own. Every terminal Answer
run — conversation-linked or not — is reclaimed once after the shared
`answer.runtime.answer_run_retention_days` floor (default 365 days) counted from
`finished_at`; the turn cascade empties the conversation, and a lightweight
hourly task then reclaims conversations that have no turns left. The history
endpoint returns recent turns in bounded signed keyset pages (40 by default,
maximum 100); older turns stay durable until retention reclaims them. Cleanup
releases blobs no surviving
run references and deletes an Agent Session when no remaining run routing row
names it. Empty Web Conversation rows do not extend model-history retention;
Session trees shared by another routed run are preserved. Cleanup never touches
ingest documents, chunks, vectors, graph data,
source files, visual assets, or jobs.

Uploaded answer attachments are stored once as owner-scoped content-addressed
blobs owned by the durable run, not by a Web-owned table, and the newest
historical attachments that fit the attachment-count limit are re-registered as
lazy request-local resources on every follow-up. Consequently, a Web conversation
that contains an attachment remains on the research path. `corpus.visual_assets`
controls browser thumbnails derived on demand from those attachments. There is no
answer-time parse cache, no attachment chunk table, and no vector cache; the
research path reads every resource fresh from its stored bytes.

Durable Answer run state has one operator knob: `answer.runtime.answer_run_retention_days`
(default 365) is the retention floor for terminal run rows, their event logs,
and superseded profile-memory history. The sweep is a best-effort hourly task
and may reclaim later, never earlier. Lease duration, heartbeat and sweep
cadence, batch sizes, token coalescing, and the crash-recovery bound are fixed
internal constants.

### Agent Artifact publication and browser preview

Agent-authored Artifacts use independent working-set, publication, image, and
active-preview limits:

```yaml
answer:
  agent:
    publication:
      max_artifacts: 20
      max_file_bytes: 31457280          # 30 MiB per published Artifact
      max_total_bytes: 104857600        # 100 MiB per Answer run
      workspace_max_bytes: 1073741824   # 1 GiB; maximum configurable value is 5 GiB
      preview_image_max_pixels: 16000000
      preview_image_max_edge: 4096
      original_image_max_pixels: 64000000
      original_image_max_edge: 8000
      active_html_max_bytes: 20971520    # 20 MiB
  conversations:
    active_html_preview_enabled: true
```

Publication rejects an over-limit Artifact whole; it does not truncate or
archive it. `workspace_max_bytes` counts the Agent Workspace working set, while
`max_file_bytes`, `max_total_bytes`, and `max_artifacts` govern only explicitly
referenced published output. Preview and original image limits are separate so
a valid download may still be unavailable for inline preview.

`answer.conversations.active_html_preview_enabled` defaults to `true`. Set it
to `false` to remove the interactive opt-in: HTML Artifacts remain in an
isolated script-disabled frame, with Source and Download still available. The
environment equivalents follow the nested settings names, for example
`DLIGHTRAG_ANSWER__AGENT__PUBLICATION__MAX_ARTIFACTS` and
`DLIGHTRAG_ANSWER__CONVERSATIONS__ACTIVE_HTML_PREVIEW_ENABLED`.

## Web Search (optional)

The answer research path can call the open web when an Exa key is set:

```yaml
answer:
  web_search:
    api_key: null  # set DLIGHTRAG_ANSWER__WEB_SEARCH__API_KEY in .env to enable
```

The key's presence is the whole capability toggle. When set, the orchestrator
may call Exa Search as a peer tool. Evidence-producing result URLs become
request-local resource handles for optional `read` calls; Exa Contents
is a bounded internal fallback only after direct reading fails or returns no
text. When unset, both Web capabilities are removed and answers stay
corpus-only. Neither path supplies cookies or browser automation for login- or
interaction-gated pages.

## REST API

REST binds to loopback by default for local development:

```yaml
interfaces:
  api:
    host: 127.0.0.1
    port: 8100
access:
  auth_mode: none
```

Set `interfaces.api.host: 0.0.0.0` only when the server is behind a trusted network or
`access.auth_mode` is explicitly enabled.

Use [security.md](security.md) for `simple`, static JWT, and JWKS/OIDC issuer
guidance, and its [authorization model](security.md#authorization-model) for
Workspace Access Rules and enterprise non-goals. The related config fields are
`access.auth_mode`, `access.api_token`, `access.jwt_verification_key`,
`access.jwt_jwks_url`, `access.jwt_issuer`, `access.jwt_audience`,
`access.jwt_algorithm`, `access.cors_allow_origins`, `access.control`, and
`access.web_identity` (edge-asserted Web identity). Example:

```yaml
access:
  auth_mode: jwt
  web_identity:
    edge: cloudflare            # cloudflare | azure | aws
    issuer: https://<team>.cloudflareaccess.com
    audience: <application-aud-tag>
    # jwks_url: ...             # required for aws; derived for cloudflare/azure
```

See security.md for the per-edge credential shapes and the required
`jwks_url` on `aws`.

The CLI and evaluation HTTP helper resolve their target from `DLIGHTRAG_API_URL` and an
optional bearer from `DLIGHTRAG_API_TOKEN` (or the deployment-only
`DLIGHTRAG_ACCESS__API_TOKEN`). `DLIGHTRAG_CLIENT_TIMEOUT` bounds one caller-owned
HTTP request and defaults to 120 seconds. It is independent of the server's
inline retrieval timeout, configured separately as
`DLIGHTRAG_CORPUS__RETRIEVAL__TIMEOUT`.

## MCP Streamable HTTP

DlightRAG's HTTP MCP server uses the current Streamable HTTP transport on a
single `/mcp` endpoint (it does not expose the deprecated HTTP+SSE `/sse` +
`/messages` pair). It binds to loopback by default:

```yaml
interfaces:
  mcp:
    transport: streamable-http
    host: 127.0.0.1
    port: 8101
```

To expose MCP beyond loopback, set `interfaces.mcp.host`, explicitly allow the public Host
and Origin with `interfaces.mcp.allowed_hosts` / `interfaces.mcp.allowed_origins`, and enable
`access.auth_mode`. Browser clients must also be allowed by `access.cors_allow_origins`. JWT
mode continues to accept direct bearer tokens. To additionally enable MCP 2.0
OAuth discovery, set the public endpoint URL:

```yaml
interfaces:
  mcp:
    resource_server_url: https://rag.example.com/mcp
```

This one optional field makes the MCP server publish RFC 9728 Protected Resource
Metadata using the existing `access.jwt_issuer`. It is not needed for stdio, simple
bearer auth, or directly supplied static-key JWTs. See [security.md](security.md).

## Observability

Tracing is off until both `observability.langfuse_public_key` and `observability.langfuse_secret_key` are
set; those are secrets and belong in `.env`. The rest is non-secret and lives in
`config.yaml`:

| Field | Default | Meaning |
| --- | --- | --- |
| `observability.langfuse_host` | `https://cloud.langfuse.com` | Where traces are sent. Set per run mode — see [operations.md](operations.md#trace-endpoint-address) |
| `observability.langfuse_trace_sensitive_data` | `true` | `false` suppresses raw query/answer text, LLM prompts/responses, raw error text, and raw IDs while retaining bounded structural metadata, usage, cost, status, and timing |
| `observability.langfuse_export_external_spans` | `false` | Also export third-party OTEL spans. DlightRAG records model calls itself, so leaving this off avoids double counting |
| `observability.langfuse_environment` | unset | Environment label on every trace |
| `observability.langfuse_release` | unset | Release label on every trace |
| `observability.langfuse_sample_rate` | `1.0` | Fraction of traces exported |
| `observability.langfuse_timeout` | SDK default | Export request timeout, seconds |
| `observability.langfuse_flush_at` | SDK default | Events buffered before a flush |
| `observability.langfuse_flush_interval` | SDK default | Seconds between flushes |

Running the bundled local stack is covered in
[operations.md](operations.md#local-langfuse-observability).

Memory recall contributes usage accounting only: `memory_recall_record_count`
and `memory_recall_chars` in the answer trace, with no record bodies in logs.
The exact rendered Profile facts are committed in the Session request state before a Research model call so
recovery is replay-stable. JWT owners and the stable local single-user owner
(`auth_mode: none`) are eligible; shared simple-auth callers are not. Forget is
idempotent and leaves a tombstone. Hosts may form a proposal and commit it
separately with a stable proposal id.

## Citations

Citation validation is always part of answer finalization. Web Inspector Sources
semantic highlights are enabled by default and use the keyword LLM role after
the answer has already been streamed/finalized. REST, MCP, and in-process Application answer calls
default to no semantic highlights; pass `semantic_highlights=True` in Python or
`semantic_highlights: true` in JSON on one answer request to include
`sources[].chunks[].highlight_phrases`.

Advanced highlight controls:

```yaml
answer:
  citations:
    highlights:
      enabled: true
      timeout: 10.0
      max_concurrency: 8
      batch_size: 8
      max_input_chars: 4096
      cache_size: 500
```

Set `answer.citations.highlights.enabled: false` to disable semantic highlight
extraction for every interface.

## Conversation And Upload Limits

```yaml
corpus:
  ingestion:
    max_upload_bytes: 104857600
    timeout: null
  retrieval:
    timeout: 300
interfaces:
  max_upload_size_mb: 512
```

```yaml
answer:
  agent:
    execution_environment: trust   # disabled | trust | sandbox
    workspace_root: null
    outbound_mcp: []
```

The shipped default is `trust`. `trust` runs the rooted local adapter as the
service user; rooted file tools prevent traversal but Bash can reach the
host/container filesystem and network. Set `execution_environment: disabled` to
expose no path, Bash, spill, or publication tools. This distribution ships no
sandbox backend: selecting `sandbox` fails explicitly and never downgrades to
trust.

For `trust` or `sandbox`, an omitted root defaults to
`~/.dlightrag/agent_workspaces`. An explicit root must be absolute, must not
overlap `deployment.working_dir`, and must be the same shared RWX path on every
worker. Compose mounts `/app/dlightrag_agent_workspaces` as that root.

Outbound MCP is explicit and allowlisted:

```yaml
answer:
  agent:
    outbound_mcp:
      - name: analytics
        transport: streamable-http
        url: https://mcp.example.com/mcp
        tools: [lookup_metric]
      - name: local_helper
        transport: stdio
        command: uvx
        args: [helper-mcp]
        tools: [read_catalog]
```

Each remote call owns a foreground MCP session and closes it before returning.
There is no endpoint discovery, registry, marketplace, OAuth service, or MCP
management plane. Configure endpoint authentication outside this thin adapter.

Research discovers Agent Skills from `~/.agents/skills/` and the active Agent
Workspace's `.agents/skills/`, with workspace precedence. Initial context gets
name, description, and global/workspace source metadata only. `load_skill` reads `SKILL.md` or a contained
reference on demand and never executes Skill code.

There is no in-process extension wrapper or plugin discovery. First-party tools,
Skills metadata, and deployment-allowlisted outbound MCP tools are composed into
one run-local closed ToolRegistry; the accepted AgentRunPlan pins their contracts.
Context is assembled by the typed Answer Host ContextAssembler, and execution
policy is enforced by concrete rooted tools.

Research ends when the model writes the answer and makes no tool call, or when
the run is cancelled or the provider fails. There is no turn cap.

Conversation history is projected once when a run is accepted. The projector
keeps the newest contiguous complete user/assistant pairs that fit every
reachable planner, Research, and Fast invocation after fixed input and explicit
reserves. Older omitted pairs become a separate extractive episodic
continuation. The projected tail, episodic continuation, and model profiles are
pinned, so planner and generation cannot disagree about prior turns. Research
KB tool queries receive that pinned history for lexical/filter inference while
`preserve_query` keeps the Agent's semantic query unchanged.

Transport and retention limits answer different questions.
`MAX_HISTORY_MESSAGES` and `MAX_HISTORY_CONTENT_CHARS` are transport contracts
that also size the JSON body limit, so they are a security bound rather than a
memory policy. How many turns stay durable in PostgreSQL is decided by the
shared retention floor, not a per-conversation window. The pinned model profiles
and context policy decide how much reaches a model.

`corpus.ingestion.max_upload_bytes` is the per-file cap for REST multipart ingest
and Web workspace/folder uploads. It also supplies the tighter receive-layer cap
for `/ingest/blob`, with fixed multipart framing allowance. URL ingestion has its
own `corpus.ingestion.url_max_bytes` download cap. Answer attachments use the
separate `answer.generation.max_attachment_bytes` (100 MiB) per-attachment
ceiling, not this ingest cap. `interfaces.max_upload_size_mb` is the general
receive-layer cap for multipart uploads and the per-request total cap for
multi-file Web workspace uploads. Answer routes use their tighter answer
attachment policy instead. `corpus.ingestion.timeout` limits how long the
in-process `CorpusAdmin.ingest()` convenience method waits for its durable job. When it expires, the job keeps running and the method returns its
current row instead of cancelling it. REST, Web, and MCP start jobs immediately
and are not governed by this wait setting.

Ingest job state is stored in `dlightrag_ingest_jobs` as operational state, not
user-facing configuration. A sweeper runs every 30 minutes and on startup: it
fails jobs whose owner stopped renewing the 5-minute lease an hour ago, and
deletes finished jobs older than 7 days. Startup also recovers recent
queued/running jobs, and workspace reset cancels active in-process jobs before
deleting that workspace's job rows. Remote prefix recovery resumes
from the next unfinished source window; single-document internals remain owned by
LightRAG's document status pipeline.

## LightRAG KG Internals

```yaml
corpus:
  ingestion:
    chunk_token_size: 2000
  retrieval:
    kg_chunk_pick_method: VECTOR
    max_entity_tokens: 6000
    max_relation_tokens: 8000
    max_total_tokens: 40000
storage:
  lightrag:
    vector_db_kwargs: {}
```

`corpus.retrieval.kg_entity_types` is public because it shapes domain extraction,
but it is empty by default so DlightRAG defers to LightRAG's built-in general
taxonomy (Person/Organization/Location/Event/Concept/Method/Content/Data/
Artifact/NaturalObject/...). Set a domain list only to bias extraction toward a
specific corpus. For stronger domain control, use
`corpus.extraction.entity_type_prompt_file` with a file under
`prompts/entity_type/`.
