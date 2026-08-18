# Configuration

This page is for operators and SDK users deciding which settings to change. It
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

Keep these in normal `config.yaml`:

- model/provider choices: `llm`, `embedding`, `rerank.enabled`, `rerank.strategy`
- parser sidecar endpoint and visual context controls: `parser_sidecars`
- metadata schema: fixed; custom metadata needs no declaration
- domain entity guidance: `kg_entity_types`, `extraction.entity_type_prompt_file`
- PostgreSQL endpoint, process role, and workspace identity: `workspace`,
  `service_role`, `postgres_*`
- high-level concurrency raised above upstream defaults: AI-provider `max_async`,
  `runtime.answer_worker_concurrency`, `rag_pipeline_max_async`,
  `embedding_func_max_async`, `embedding_batch_num`
- retrieval/answer controls: `top_k`, `chunk_top_k`, `bm25_enabled`, `direct_visual_top_k`,
  `answer.*`
- auth and observability mode switches when they are not secret

Keep these out of normal `config.yaml` unless debugging or load-testing proves
they need to change:

- storage backend literals
- raw LightRAG parser rules (derived internally from the active sidecar)
- PostgreSQL retry/backoff internals
- per-stage ingest worker counts (`max_parallel_*`) that match LightRAG defaults
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

The configured external sidecar selects the parser automatically. Configure a
`mineru` block or a `docling` block; DlightRAG derives the internal LightRAG
wildcard. With neither block, the code default is MinerU. If both are present,
only MinerU is effective. MinerU and Docling are durable ingestion parsers only;
answer attachments are decoded and converted request-locally and never invoke
them.

Advanced parser fields with code defaults:

```yaml
parser:
  chunk_options: {}

extraction:
  use_json: true

parser_sidecars:
  vlm:
    enabled: true
    max_image_bytes: 5242880
    # DlightRAG default 80px, above LightRAG's native 64px minimum: sub-80px
    # crops are treated as decorative (icons/separators/ornaments) and skipped.
    # Set 64 explicitly to use LightRAG's native threshold.
    min_image_pixel: 80
  mineru:
    api_mode: local
    local_endpoint: http://host.docker.internal:8210
    language: ch
    backend: hybrid-engine
```

`parser_sidecars.vlm` owns figure understanding, and MinerU's own image
analysis is deliberately left off. MinerU extracts each figure as a crop; the
VLM sidecar then describes that crop together with the surrounding text.
Enabling MinerU's analysis would run a second VLM over the same image for
roughly 58% more parse time and largely duplicate content, so there is no
setting for it.

A parse therefore emits zero `chart` blocks by design — the figures arrive as
`image` blocks and become `drawing` chunks carrying the sidecar's description.
That is the expected shape, not a missing feature.

The sidecar only ever sees the figures MinerU cut, so the `hybrid-engine`
backend's effort setting decides what it gets. MinerU's own default, `medium`,
consumes precomputed layout boxes and can split a dense multi-panel figure into
fragments. `high` lets the VLM detect blocks itself, returning whole figures with
correctly bound captions, at roughly 5x the parse time. Set
`MINERU_HYBRID_EFFORT=high` in `.env.mineru` for figure-heavy corpora.

To use Docling instead, remove/comment the MinerU block and configure only:

```yaml
parser_sidecars:
  docling:
    endpoint: http://docling:5001
    # code_formula_preset: granite_docling
    # force_ocr: false
```

`parser_sidecars.docling.do_formula_enrichment` transcribes detected formula
regions and defaults on, matching MinerU's `enable_formula`, so the parser
choice does not decide whether a corpus keeps its mathematics. Turning it off
drops formulas silently rather than erroring, so turn it off only on a corpus
without mathematics.

`parser_sidecars.docling.code_formula_preset` names the model that transcribes
them. Leave it unset unless the parser service runs on Apple Silicon:

| Parser service device | `code_formula_preset` |
| --- | --- |
| CUDA, XPU, or CPU | Unset — Docling's built-in `codeformulav2` is used |
| MPS | `granite_docling` |

Docling's default model cannot run on MPS, so enrichment fails on Apple Silicon
until the preset is set. Repointing the preset invalidates the Docling bundle
cache, so affected documents re-parse on their own. The
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

The OCR engine needs no configuration, and `ocr_lang` has no effect: the CPU
image resolves to a single engine that reads Han and Latin from one table.

`parser_sidecars.docling.force_ocr` re-runs OCR over the whole page and discards
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
The optional local profile starts with `docker compose --profile docling up -d`;
an external deployment supplies its own reachable endpoint. Native DlightRAG
processes use `127.0.0.1` endpoints when their parser runs on the same host.

`parser_sidecars.mineru.language` is MinerU's OCR language hint for scanned or
image-based documents. It is separate from `extraction.language`, which controls
LightRAG's KG extraction prompt language.

`parser_sidecars.mineru.backend` selects MinerU's parse engine and defaults to
`hybrid-engine`, MinerU's current VLM-assisted default. Accepted values are
`pipeline`, `vlm-engine`, and `hybrid-engine`. Use `pipeline` (MinerU's non-VLM
OCR engine) to avoid VLM transcription artifacts on difficult scans, at the cost
of weaker complex-layout and chart handling. DlightRAG always maps the selected
value privately to `MINERU_LOCAL_BACKEND`, avoiding LightRAG's legacy default.
Public environment overrides use the typed `DLIGHTRAG_PARSER_SIDECARS__...`
form; raw MinerU/Docling/VLM variables are not independent configuration inputs.

DlightRAG does not expose MinerU-side image/chart analysis as a product setting;
LightRAG 1.5.5 defaults that parser-time path off, while LightRAG's separate
multimodal analyze stage handles images, tables, and equations after parse.

## Embeddings

Embedding configuration defines the vector space shared by ingestion and
retrieval. `provider` selects an API protocol; `input_modality` independently
controls whether DlightRAG may send raw images through that protocol. Provider
selection is always explicit and is never inferred from a model name, URL, or
port.

### Provider matrix

`base_url` is the root before the endpoint shown below. DlightRAG appends the
endpoint itself.

| `provider` | Endpoint appended to `base_url` | Image policy | Asymmetric | Authentication | `dim` behavior |
|---|---|---|---|---|---|
| `voyage` | `/multimodalembeddings` | Native | Yes | Bearer token | Sent as `output_dimension`; returned vectors are validated |
| `gemini` | `/models/{model}:embedContent` | Native | No | `x-goog-api-key` | Sent as `output_dimensionality`; returned vectors are validated |
| `jina` | `/v1/embeddings` | Native | Yes | Bearer token | Sent as `dimensions`; returned vectors are validated |
| `openai_compatible` | `/embeddings` | Text by default; explicit image opt-in | No | Optional bearer token | Sent as `dimensions`; returned vectors are validated |
| `ollama` | `/api/embed` | Text only | No | None | Not sent; returned vectors are still validated |

The supported provider values are exactly the five names above. For example,
LM Studio is `openai_compatible` because it exposes an OpenAI-style
`/v1/embeddings` API. Ollama has a native provider because its embedding API is
`/api/embed`, not `/v1/embeddings`.

### Fields

| Field | Default | Meaning |
|---|---|---|
| `provider` | Required | One transport from the matrix above. Unknown values fail configuration loading. |
| `model` | Required | The exact model identifier expected by the remote or local server. |
| `api_key` | None | Provider credential. Prefer `DLIGHTRAG_EMBEDDING__API_KEY` in `.env`; omit it for unauthenticated local servers. |
| `base_url` | OpenAI API root | API root before the appended endpoint. Include `/v1` only when that protocol expects it; configure this explicitly for non-OpenAI transports. |
| `dim` | `1024` | Expected vector length. It is sent when the protocol supports a dimension parameter and is always checked against every returned vector. |
| `max_token_size` | `8192` | Maximum input size advertised to LightRAG's embedding pipeline; it does not change the model's real context limit. |
| `input_modality` | `auto` | Local routing policy: `auto`, `text`, or `multimodal`. It is never included in an upstream request. |
| `asymmetric` | `auto` | `auto` enables query/document hints when supported; `require` fails for unsupported providers; `disable` forces symmetric embeddings. |
| `startup_probe` | `true` | When image routing is active, send one in-memory 1x1 image at startup to verify the selected endpoint/model. The probe writes no storage or files. |

DlightRAG does not guess whether an arbitrary model accepts a particular
dimension. Set `dim` to the model's real output size; a mismatch fails when a
response is validated.

### Input modality

DlightRAG only pairs with unified multimodal embedding models, which embed text
and images into one shared space AND fuse interleaved text+image into a single
vector. `input_modality` is the whole capability signal -- there is no separate
per-provider fusion table to maintain, and the startup probe verifies the live
endpoint actually embeds an image.

| Provider capability | `auto` | `text` | `multimodal` |
|---|---|---|---|
| Native multimodal (`voyage`, `gemini`, `jina`) | Enable both image paths (image->image query retrieval AND the fused visual-vector overwrite); run the startup probe | Disable both locally | Require image embedding; probe failure stops startup |
| Native text-only (`ollama`) | Text only | Text only | Fail before service initialization |
| OpenAI-compatible extension (`openai_compatible`) | Conservative text only | Text only | Opt into the data-URI image payload; probe failure stops startup |

`text` guarantees the embedding provider receives text only. It disables both
the document visual-vector overwrite and image->image query retrieval. Images,
tables, and equations may still be described by the VLM; those descriptions
remain ordinary text in LightRAG's semantic, BM25, and KG paths, and current
query images are still described by the VLM to shape the query plan.

Both DlightRAG image paths turn on together from this one signal. The
**image->image query leg** embeds the query image and matches the index in the
provider's shared text-image space, complementing the VLM-description text path.
The **document visual-vector overwrite** replaces a drawing chunk's vector with
one fused text+image vector, so the figure stays reachable by text queries.
Because every supported multimodal model fuses, no provider is left with one
path but not the other.

`multimodal` is a capability assertion, not a hint. DlightRAG fails fast when
the configured adapter cannot serialize images or when the live startup probe
rejects them. In `auto`, a native multimodal provider may safely downgrade to
the semantic text path if its live probe fails. `startup_probe: false` skips
only the live request and trusts the resolved provider/modality combination;
static mismatches such as `ollama + multimodal` still fail.

### Examples

Voyage native multimodal embeddings:

```yaml
embedding:
  provider: voyage
  model: voyage-multimodal-3.5
  base_url: https://api.voyageai.com/v1
  dim: 1024
  max_token_size: 8192
  input_modality: auto
  asymmetric: auto
  startup_probe: true
```

Keep the Voyage key in `.env`:

```dotenv
DLIGHTRAG_EMBEDDING__API_KEY=pa-...
```

Ollama's native text embedding endpoint:

```yaml
embedding:
  provider: ollama
  model: nomic-embed-text
  base_url: http://127.0.0.1:11434
  dim: 768
  max_token_size: 8192
  input_modality: auto
  asymmetric: disable
```

LM Studio or another OpenAI-compatible text embedding server:

```yaml
embedding:
  provider: openai_compatible
  model: text-embedding-nomic-embed-text-v1.5
  base_url: http://127.0.0.1:1234/v1
  dim: 768
  max_token_size: 8192
  input_modality: text
  asymmetric: disable
```

An OpenAI-compatible endpoint serving a multimodal Qwen3-VL embedding model
uses the same provider and opts into images explicitly:

```yaml
embedding:
  provider: openai_compatible
  model: qwen3-vl-embedding-2b
  base_url: http://127.0.0.1:1234/v1
  dim: 2048
  max_token_size: 8192
  input_modality: multimodal
  asymmetric: disable
  startup_probe: true
```

For LM Studio, `model` must match the identifier exposed by the running local
server. The model itself must implement `/v1/embeddings`; loading a chat-only
model is not sufficient.

### Docker host access

When DlightRAG runs directly on the host, local services normally use
`127.0.0.1`. Inside this repository's Compose containers, `127.0.0.1` means the
container itself. Compose configures the `host.docker.internal` alias for
host-side services, so use:

```yaml
# Ollama from Compose
base_url: http://host.docker.internal:11434

# LM Studio from Compose
base_url: http://host.docker.internal:1234/v1
```

Notice that Ollama has no `/v1`, while LM Studio's OpenAI-compatible root does.

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

### Model capacity profiles

Each chat model resolves an endpoint-scoped capacity profile containing its
context window (`C`), optional maximum input (`I`), optional maximum output
(`O`), and image/tool/reasoning capabilities. DlightRAG matches the normalized
`provider`, exact `model`, and normalized `base_url`; the same model name at a
different endpoint is a different profile. Resolution precedence is an
explicit root override, trusted provider-adapter facts, then the versioned
catalog shipped by `dlightrag-ai`. Unknown endpoints fail closed instead of
inheriting a global default or being probed.

Adapter facts are an optional, static class-level declaration: DlightRAG asks
the selected adapter without constructing a client or making a network call.
An adapter that publishes no model facts returns no profile and resolution
continues to the catalog. The currently cataloged endpoints remain in the
shared catalog so runtime and the stdlib-only setup wizard use identical facts.

The setup wizard reads that same catalog. It asks for an override only when a
selected endpoint is unknown. For a manually configured private or newly
released model, provide one complete override at the root:

```yaml
model_capacity_overrides:
  - provider: openai
    model: private-model
    base_url: http://localhost:8888/v1
    context_window_tokens: 262144
    max_input_tokens: 200000
    max_output_tokens: 32768
    supports_images: true
    supports_tools: true
    supports_reasoning: false
```

`context_window_tokens` is required. `max_input_tokens` and
`max_output_tokens` may be omitted only when the endpoint does not publish
those separate limits. Capability flags default to `false`; set them from
trusted endpoint documentation. Duplicate normalized endpoint identities and
an input limit greater than the context window are configuration errors.

Capacity arithmetic is owned by one immutable, revisioned policy. Its hard
input limit is `L = min(I if known else C, floor(0.85C))`; research compacts
proactively at `floor(0.85L)`. A request with known `O` receives at most
`min(O, C - input)` output tokens. Capacity is not configured under `answer`;
there is no evidence ratio, fixed generation reserve, or fixed tool-observation
token cap.

### LLM Structured Output

Planner and other small control-plane calls pass a `StructuredOutput` contract
through the shared LLM factory. Model configuration decides which provider
request format is used:

```yaml
llm:
  roles:
    extract:
      provider: openai
      model: deepseek-v4-flash
      base_url: https://api.deepseek.com
      structured_output: json_object
```

`structured_output` defaults to `auto`. Auto uses schema-constrained output for
providers with a native schema path: OpenAI's default endpoint, Anthropic
native `output_config.format`, and Gemini native `response_schema`.
OpenAI-compatible endpoints with a custom `base_url` default to `json_object`
because feature parity is provider-specific. Set `structured_output` to
`json_schema` only for a custom OpenAI-compatible endpoint known to support
strict JSON schema response formats. Anthropic native does not support the
lower-confidence `json_object` mode; use `auto` or `json_schema`.

`model_kwargs` apply to ordinary calls. `agentic_model_kwargs` are a shallow
top-level overlay used by research control and final calls. This keeps fast-path
answers inexpensive while allowing explicit provider-native thinking for
research turns without guessing a cross-provider flag:

```yaml
llm:
  default:
    model_kwargs:
      reasoning: {enabled: false}
    agentic_model_kwargs:
      reasoning: {enabled: true}
  roles:
    query:
      model_kwargs:
        reasoning: {enabled: false}
      agentic_model_kwargs:
        reasoning: {enabled: true}
```

The overlay is unconditional key merging, not fallback selection. DlightRAG
copies `model_kwargs` and then replaces any same-named top-level key supplied by
`agentic_model_kwargs`. With the example above, ordinary calls receive
`reasoning.enabled: false`, while research control/final calls receive
`reasoning.enabled: true`; unrelated ordinary options remain present.
The explicit `query` block follows the same shape because it replaces the
default role as a complete model configuration rather than deep-merging with it.

Research final generation starts with the agentic overlay. If the provider
finishes without user-visible text, DlightRAG retries once with `model_kwargs`;
a second empty response fails instead of storing an empty answer. Use the
endpoint's actual reasoning switch in the ordinary options. For OpenRouter
reasoning models such as MiMo and GLM, that switch is
`reasoning: {enabled: false}`; `thinking: {type: disabled}` does not disable
their reasoning tokens.

Self-hosted Unsloth, llama.cpp, or vLLM deployments commonly expose the switch
through the chat template instead. Configure the field the endpoint actually
supports:

```yaml
llm:
  roles:
    query:
      model_kwargs:
        chat_template_kwargs: {enable_thinking: false}
      agentic_model_kwargs:
        chat_template_kwargs: {enable_thinking: true}
```

If `roles.query` is absent or incomplete, both sets of options come from
`llm.default` through the normal role fallback.

## Remote Source URLs

`source_uri` identifies the source; `download_uri` tells DlightRAG how to
retrieve the original bytes when no local copy is retained. The two values are
independent: connector-specific identities such as `bynder://asset/...` are
valid provenance but are not download locations.

By default, Azure Blob, S3, URL, and SDK connector files are not copied into
DlightRAG storage. A non-retained document therefore needs a durable S3, Azure,
or queryless public HTTPS `download_uri`. Set
`retain_remote_source_files: true` to keep fetched files under the workspace
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
`GET /web/files/raw/{document_id}` are separate authenticated projections. Each
resolves the exact workspace metadata row server-side, then serves a retained
local file or redirects through a supported provider locator. Azure uses
`DLIGHTRAG_BLOB_CONNECTION_STRING`. S3 uses the standard AWS credential chain
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_REGION`/`AWS_DEFAULT_REGION`, IAM role, or shared AWS config).
REST/MCP `source_type="url"` accepts public or signed HTTPS URLs only, does not
follow redirects to private hosts, and caps each download with
`url_ingest_max_bytes`. SaaS APIs that require auth headers must stage content
through a supported local, Azure Blob, or S3 source, or expose a public HTTPS
fetch URL. `source_uri`/`source_uris` set stable identity; they do not substitute
for the durable locator required by a non-retained signed fetch.
Set `url_ingest_private_host_allowlist` only for trusted enterprise hosts that
must be fetched by REST/MCP URL ingest. Entries are host/IP patterns such as
`docs.corp.example`, `*.corp.example`, or `10.0.0.5`.

Remote prefix ingest streams provider listings into bounded local staging
windows. It uses the same ingest job substrate as local and single-object ingest,
while keeping source ownership in the cloud provider. DlightRAG delete/reset
operations remove DlightRAG metadata, LightRAG storage, and local parser
artifacts only; they do not delete Azure Blob, S3, or URL source objects.

Advanced signing defaults:

```yaml
retain_remote_source_files: false
url_ingest_max_bytes: 104857600
url_ingest_private_host_allowlist: []
azure_sas_expiry: 3600
s3_presign_expiry: 3600
s3_region:
```

## PostgreSQL

Core storage is PostgreSQL 18 only. The backend literals are code defaults and
should normally stay out of `config.yaml`:

```yaml
vector_storage: PGVectorStorage
graph_storage: PGTableGraphStorage
kv_storage: PGKVStorage
doc_status_storage: PGDocStatusStorage
```

Advanced PostgreSQL and index tuning:

```yaml
pg_vector_index_type: HNSW_HALFVEC
pg_hnsw_m: 32
pg_hnsw_ef_construction: 256
pg_hnsw_ef_search: 256
postgres_lightrag_pool_max_size: 16
postgres_pool_min_size: 2
postgres_pool_max_size: 16
postgres_session_settings: {}
postgres_statement_cache_size:
postgres_connection_retries: 10
postgres_connection_retry_backoff: 3.0
postgres_connection_retry_backoff_max: 30.0
postgres_pool_close_timeout: 5.0
```

`postgres_pool_max_size` sizes the DlightRAG domain-store pool (BM25, metadata,
conversations, jobs, checkpoints); `postgres_lightrag_pool_max_size` sizes the
LightRAG backend pool. Each process opens up to the sum of the two, so multiply
by the worker count and keep the total under PostgreSQL `max_connections`. Raise
`postgres_pool_max_size` for high single-worker concurrency; lower it when
running many workers.

### Process role (writer / reader)

`service_role` selects what a process may do with its single PostgreSQL endpoint:

```yaml
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

Multi-host deployments must mount one shared POSIX `working_dir` at the **same
absolute path** in every process that serves KB images or retained source
downloads.

Use [postgresql.md](postgresql.md) for production sizing, SSL, shared memory, and extension
notes.

## Ingestion Concurrency And Queues

`config.yaml` keeps only the high-level AI, Runtime, and RAG concurrency knobs.
`max_async` bounds all provider requests through the process-wide fair AI
scheduler. `runtime.answer_worker_concurrency` bounds claimed durable Answer
runs executed by one process. `rag_pipeline_max_async` bounds each workspace's
LightRAG pipeline width; its provider requests still pass through the AI
scheduler. `embedding_func_max_async` and `embedding_batch_num` shape LightRAG's
embedding work without changing either worker admission or the global provider
cap. The
per-stage worker counts below already match LightRAG's defaults, so they are
omitted from `config.yaml` and follow DlightRAG's code defaults; set them
explicitly (in `config.yaml` or via `DLIGHTRAG_*` env) only when a deployment
needs different parallelism:

```yaml
max_parallel_insert: 3        # insert workers (code/LightRAG default 3)
max_parallel_parse_native: 5  # native + legacy parser workers (default 5)
max_parallel_parse_mineru: 2  # MinerU parser workers (default 2)
max_parallel_parse_docling: 2 # Docling parser workers (default 2)
max_parallel_analyze: 5       # VLM analysis workers (default 5)
```

Queue sizes are internal backpressure settings and should only change after
measuring parser/analyze/insert pressure:

```yaml
queue_size_parse: 20
queue_size_analyze: 100
queue_size_insert: 4
embedding_request_timeout: 120
```

`embedding_batch_num` is the number of texts sent per embedding provider
request. Raise it to match your provider's per-request cap (for example, Voyage
accepts up to 1000 inputs and OpenAI up to 2048); a value too high for the
configured provider surfaces as a request error during ingest, so lower it then.

## BM25

BM25 is part of the supported DlightRAG retrieval path. BM25 candidate breadth
follows the configured chunk candidate budget. `/retrieve` does not re-cap
fused chunks after semantic/BM25 merge; `/answer` packs final prompt evidence
against the resolved query model's remaining input capacity. Language profiles
and scoring constants are advanced index signatures.

Defaults:

```yaml
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

`bm25_enabled` controls workspace PostgreSQL BM25 indexing, ingest-time
language labels, and query fusion. It applies to the workspace knowledge-base
lane only; the answer research path reads attachments through request-local
resources that never touch workspace indexes.

Changing profile names, text configs, languages, `bm25_k1`, or `bm25_b`
changes the expected pg_textsearch index signature. Enabling BM25 for an
existing corpus or changing profile languages also requires relabeling existing
chunks; restarting alone does not rewrite historical labels. Use the offline
workspace BM25 rebuild described in [operations.md](operations.md#workspace-bm25-rebuild).

## Fusion And Filtering

Advanced retrieval scoring:

```yaml
rrf_k: 60
metadata_filter_exact_vector_threshold: 8192
```

`metadata_filter_exact_vector_threshold` controls when DlightRAG can use exact
vector scoring inside a small metadata candidate set.

## Image Budgets

`answer.max_images` and the answer byte/geometry fields define one image
transport budget for every answer, across REST, SDK, MCP, and Web. That single
budget covers current attachment images and retrieved workspace visuals.
Focused VLM inspection is a separate model call: each inspection applies the
same byte/geometry limits independently and does not consume the final answer
transport budget.
At startup the configured shape is clamped to the query-role model's discovered
image capability. Compression budgets are advanced model transport limits:

`chat_llm_reranker` can use its own `rerank.provider` and `rerank.model`. When
those are omitted, it reuses `llm.default`.

Voyage's text reranker is available with `strategy: voyage_reranker`,
`model: rerank-2.5` or `rerank-2.5-lite`, and `DLIGHTRAG_RERANK__API_KEY`.
Cohere's public text reranker is available with `strategy: cohere_reranker`,
`model: rerank-v4.0-pro` or `rerank-v4.0-fast`, and the same API key env var.
When a provider reranker is explicitly selected, missing credentials are a
configuration error and fail service initialization rather than falling back to
`chat_llm_reranker`.

`rerank.input_modality` defaults to `auto`. For `chat_llm_reranker`, auto
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
rerank:
  strategy: chat_llm_reranker
  input_modality: auto
  # Optional. Omitted keeps all scored candidates before top_k.
  # score_threshold: 0.5
  max_concurrency: 8
  batch_size: 8

answer:
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

`answer.max_attachments` (6),
`answer.max_attachment_bytes` (100 MiB), and `answer.max_total_attachment_bytes`
(128 MiB) bound answer attachment admission. `query_images` remains the
retrieve-only current-image path.

`answer.image_max_pixels` rejects source images whose decoded dimensions exceed
the limit before RGB conversion or resizing. The Web upload validator,
request-local resource inspection, retrieve query-image description, and final
answer transport use the same ceiling.

## Answer Attachments And Web Conversations

Answer public inputs are **attachments**, not query images. REST, the Python
SDK, MCP, and the Web UI attach files and HTTPS references that become
request-local resources for the lifetime of one answer. `max_attachments`,
`max_attachment_bytes`, and `max_total_attachment_bytes` (above) bound admission
on every channel. Attachments are read on demand — deterministic UTF-8/CSV
decoding and MarkItDown conversion of HTML/PDF/DOCX/PPTX/XLSX first, then focused
VLM inspection of figures — and their full bytes never enter model context.

`query_images` is a separate, retrieve-only current-image path. `/retrieve`
accepts at most three current-request query images, a fixed public
contract shared by REST and MCP. Those images are described with the VLM for
text retrieval and embedded directly for the visual retrieval leg. They do not
share an answer budget.

Answer images arrive only as attachments/resources. `answer.max_attachment_bytes`
governs original upload admission, `answer.max_images` is capability-clamped at
runtime, and the `answer.image_*` fields bound the compressed payload sent to a
model. Public REST, MCP, CLI, and Python answer/retrieve calls remain stateless;
durable conversation attachments belong only to the Web conversation store:

```yaml
web_conversations:
  max_turns: 100
  ttl_days: 30

visual_assets:
  thumb_max_px: 300
  thumb_cache_size: 256
```

`web_conversations` applies only to the principal-scoped Web-only conversation
lifecycle. It keeps at most 100 complete turns and uses 30-day inactivity
retention; expired conversations are hidden immediately and reclaimed in
skip-locked batches by a lightweight hourly task. Listing conversations also
removes expired rows for the active principal. Cleanup deletes the linked answer
runs, which cascades their events and artifact references and releases blobs no
surviving run references, without touching ingest documents, chunks, vectors,
graph data, source files, visual assets, or jobs.

Uploaded answer attachments are stored once as owner-scoped content-addressed
blobs owned by the durable run, not by a Web-owned table, and the newest
historical attachments that fit the attachment-count limit are re-registered as
lazy request-local resources on every follow-up. Consequently, a Web conversation
that contains an attachment remains on the research path. `visual_assets`
controls browser thumbnails derived on demand from those attachments. There is no
answer-time parse cache, no attachment chunk table, and no vector cache; the
research path reads every resource fresh from its stored bytes.

Durable Answer run state has no operator knobs. Terminal run rows and every
terminal run's event log expire 30 days after the run finished, except a
succeeded run a committed Web turn still references. Lease duration, heartbeat
and sweep cadence, retention cadence, batch sizes, token coalescing, and the
crash-recovery bound are fixed internal constants.

## Web Search (optional)

The answer research path can call the open web when an Exa key is set:

```yaml
web_search:
  api_key: null  # set DLIGHTRAG_WEB_SEARCH__API_KEY in .env to enable
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
api_host: 127.0.0.1
api_port: 8100
auth_mode: none
```

Set `api_host: 0.0.0.0` only when the server is behind a trusted network or
`auth_mode` is explicitly enabled.

Use [security.md](security.md) for `simple`, static JWT, JWKS/OIDC issuer, and
access-control deployment guidance. The related config fields are `auth_mode`,
`api_auth_token`, `jwt_verification_key`, `jwt_jwks_url`, `jwt_issuer`,
`jwt_audience`, `jwt_algorithm`, `cors_allow_origins`, and `access_control`.

The Python SDK and CLI resolve their HTTP target from `DLIGHTRAG_API_URL` and an
optional bearer from `DLIGHTRAG_API_TOKEN` (or the deployment-only
`DLIGHTRAG_API_AUTH_TOKEN`). `DLIGHTRAG_CLIENT_TIMEOUT` bounds one caller-owned
HTTP request and defaults to 120 seconds. It is independent of the server's
inline retrieval timeout, configured separately as
`DLIGHTRAG_RETRIEVAL_TIMEOUT`.

## MCP Streamable HTTP

DlightRAG's HTTP MCP server uses the current Streamable HTTP transport on a
single `/mcp` endpoint (it does not expose the deprecated HTTP+SSE `/sse` +
`/messages` pair). It binds to loopback by default:

```yaml
mcp_transport: streamable-http
mcp_host: 127.0.0.1
mcp_port: 8101
```

To expose MCP beyond loopback, set `mcp_host`, explicitly allow the public Host
and Origin with `mcp_allowed_hosts` / `mcp_allowed_origins`, and enable
`auth_mode`. Browser clients must also be allowed by `cors_allow_origins`. JWT
mode continues to accept direct bearer tokens. To additionally enable MCP 2.0
OAuth discovery, set the public endpoint URL:

```yaml
mcp_resource_server_url: https://rag.example.com/mcp
```

This one optional field makes the MCP server publish RFC 9728 Protected Resource
Metadata using the existing `jwt_issuer`. It is not needed for stdio, simple
bearer auth, or directly supplied static-key JWTs. See [security.md](security.md).

## Observability

Tracing is off until both `langfuse_public_key` and `langfuse_secret_key` are
set; those are secrets and belong in `.env`. The rest is non-secret and lives in
`config.yaml`:

| Field | Default | Meaning |
| --- | --- | --- |
| `langfuse_host` | `https://cloud.langfuse.com` | Where traces are sent. Set per run mode — see [operations.md](operations.md#trace-endpoint-address) |
| `langfuse_trace_sensitive_data` | `true` | `false` suppresses raw query/answer text, LLM prompts/responses, raw error text, and raw IDs while retaining bounded structural metadata, usage, cost, status, and timing |
| `langfuse_export_external_spans` | `false` | Also export third-party OTEL spans. DlightRAG records model calls itself, so leaving this off avoids double counting |
| `langfuse_environment` | unset | Environment label on every trace |
| `langfuse_release` | unset | Release label on every trace |
| `langfuse_sample_rate` | `1.0` | Fraction of traces exported |
| `langfuse_timeout` | SDK default | Export request timeout, seconds |
| `langfuse_flush_at` | SDK default | Events buffered before a flush |
| `langfuse_flush_interval` | SDK default | Seconds between flushes |

Running the bundled local stack is covered in
[operations.md](operations.md#local-langfuse-observability).

## Citations

Citation validation is always part of answer finalization. Web source-panel
semantic highlights are enabled by default and use the keyword LLM role after
the answer has already been streamed/finalized. SDK, REST, and MCP answer calls
default to no semantic highlights; pass `semantic_highlights=True` in Python or
`semantic_highlights: true` in JSON on one answer request to include
`sources[].chunks[].highlight_phrases`.

Advanced highlight controls:

```yaml
citations:
  highlights:
    enabled: true
    timeout: 10.0
    max_concurrency: 8
    batch_size: 8
    max_input_chars: 4096
    cache_size: 500
```

Set `citations.highlights.enabled: false` to disable semantic highlight
extraction for every interface.

## Conversation And Upload Limits

```yaml
max_upload_bytes: 104857600
max_upload_size_mb: 512
ingest_timeout:
retrieval_timeout: 300
max_agent_turns: 50
```

```yaml
agent:
  execution_environment: disabled   # or local_trusted
  workspace_root: null              # required absolute path when local_trusted
```

Path tools, Bash, and private spill are absent unless `execution_environment` is
`local_trusted`. That value is an operator assertion, not a sandbox. The
workspace root must be absolute, must not overlap `working_dir`, and must have
headroom for one 2 GiB epoch copy. Compose mounts
`/app/dlightrag_agent_workspaces`. Environment overrides use
`DLIGHTRAG_AGENT__EXECUTION_ENVIRONMENT` and `DLIGHTRAG_AGENT__WORKSPACE_ROOT`.

`max_agent_turns` is a safety cap, not a tuning knob: research normally ends when
the agent calls no tool or a tool batch adds no evidence. The cap bounds a run
that keeps finding new evidence -- an open-web question can always find one more
page -- and answers from what it already has instead of failing.

Conversation history is projected once when an answer run is accepted. The
projector keeps the newest contiguous complete user/assistant pairs that fit
every reachable planner, research, and synthesis request. For each target, its
allowance is the smaller of 20 percent of that model's hard input limit and the
residual after fixed input; research targets use their proactive compaction
threshold. The resulting history and resolved model profiles are pinned in the
durable run, so planner, research control, and final generation cannot disagree
about which prior turns exist. Research KB tool queries are already formulated
by the agent, so their internal RetrievalPlanner receives the pinned history for
lexical/filter inference while `preserve_query` keeps the agent's semantic query
unchanged.

Transport and retention limits answer different questions.
`MAX_HISTORY_MESSAGES` and `MAX_HISTORY_CONTENT_CHARS` are transport contracts
that also size the JSON body limit, so they are a security bound rather than a
memory policy. `web_conversations.max_turns` decides how many turns are retained
in PostgreSQL. The pinned model profiles and context policy decide how much
reaches a model.

`max_upload_bytes` is the per-file cap for REST multipart ingest and Web
workspace/folder uploads. It also supplies the tighter receive-layer cap for
`/ingest/blob`, with fixed multipart framing allowance. URL ingestion has its own
`url_ingest_max_bytes` download cap. Answer attachments use the separate
`answer.max_attachment_bytes` (100 MiB) per-attachment ceiling, not this ingest
cap. `max_upload_size_mb` is the general receive-layer cap for multipart uploads
and the per-request total cap for multi-file Web workspace uploads. Answer routes
use their tighter answer attachment policy instead. `ingest_timeout` limits how
long the in-process `CorpusAdmin.ingest()` convenience method waits for its
durable job. When it expires, the job keeps running and the method returns its
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
chunk_p_token_size: 2000
kg_chunk_pick_method: VECTOR
max_entity_tokens: 6000
max_relation_tokens: 8000
max_total_tokens: 40000
vector_db_kwargs: {}
```

`kg_entity_types` is public because it shapes domain extraction, but it is empty
by default so DlightRAG defers to LightRAG's built-in general taxonomy
(Person/Organization/Location/Event/Concept/Method/Content/Data/Artifact/
NaturalObject/...). Set a domain list only to bias extraction toward a specific
corpus. For stronger domain control, use `extraction.entity_type_prompt_file`
with a file under `prompts/entity_type/`.
