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
- metadata schema: `metadata.fields`
- domain entity guidance: `kg_entity_types`, `extraction.entity_type_prompt_file`
- PostgreSQL endpoint and workspace identity: `workspace`, `postgres_*`
- high-level concurrency raised above upstream defaults: `max_async`,
  `embedding_func_max_async`, `embedding_batch_num`
- retrieval/answer breadth: `top_k`, `chunk_top_k`, `direct_visual_top_k`,
  `answer.*`
- auth and observability mode switches when they are not secret

Keep these out of normal `config.yaml` unless debugging or load-testing proves
they need to change:

- storage backend literals
- parser routing rules
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

DlightRAG defaults to LightRAG native parsing for DOCX, Markdown, and textpack
bundles, and a MinerU-compatible external parser endpoint for other supported
document formats. Change parser routing only when validating a new LightRAG
parser strategy.

Advanced parser fields with code defaults:

```yaml
parser:
  rules: "docx:native-iteP,md:native-iteP,textpack:native-iteP,*:mineru-iteP"
  chunk_options: {}

extraction:
  use_json: true

parser_sidecars:
  vlm:
    enabled: true
    max_image_bytes: 5242880
    # Unset by default: DlightRAG defers to LightRAG's native minimum (64px).
    # Set an explicit integer only to impose a different minimum image side.
    # min_image_pixel: 64
  mineru:
    api_mode: local
    local_endpoint: http://127.0.0.1:8210
    language: ch
    # backend:            # unset ⇒ LightRAG default (hybrid-auto-engine)
    max_polls: 3600
    auxiliary_block_policy: conservative
```

`parser_sidecars.mineru.language` is MinerU's OCR language hint for scanned or
image-based documents. It is separate from `extraction.language`, which controls
LightRAG's KG extraction prompt language.

`parser_sidecars.mineru.backend` selects MinerU's parse engine. Leave it unset to
inherit LightRAG's default (`hybrid-auto-engine`, VLM-assisted). Accepted values:
`pipeline`, `vlm-engine`, `hybrid-engine`, `vlm-auto-engine`, `hybrid-auto-engine`.
Use `pipeline` (MinerU's non-VLM OCR engine) to avoid VLM transcription artifacts
on difficult scans, at the cost of weaker complex-layout and chart handling. It
maps to `MINERU_LOCAL_BACKEND`, so a deployment may also set that env directly.

DlightRAG does not expose MinerU-side image/chart analysis as a product setting;
LightRAG 1.5.4 defaults that parser-time path off, while LightRAG's separate
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

### LLM Structured Output

Planner and other small control-plane calls pass a `StructuredOutput` contract
through the shared LLM factory. Model configuration decides which provider
request format is used:

```yaml
llm:
  roles:
    keyword:
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
`url_ingest_max_bytes`. SaaS APIs that require auth headers should be wrapped by
an SDK `AsyncDataSource` connector and ingested with
`RAGServiceManager.aingest_source()`. `source_uri`/`source_uris` set stable
identity; they do not substitute for the durable locator required by a
non-retained signed fetch.
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
graph_storage: PGGraphStorage
kv_storage: PGKVStorage
doc_status_storage: PGDocStatusStorage
```

Advanced PostgreSQL and index tuning:

```yaml
postgres_required_major: 18
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

Use [postgresql.md](postgresql.md) for production sizing, SSL, shared memory, and extension
notes.

## Ingestion Concurrency And Queues

`config.yaml` keeps only the concurrency knobs raised above LightRAG's upstream
defaults (`max_async`, `embedding_func_max_async`, `embedding_batch_num`). The
per-stage worker counts below already match LightRAG's defaults, so they are
omitted from `config.yaml` and follow DlightRAG's code defaults; set them
explicitly (in `config.yaml` or via `DLIGHTRAG_*` env) only when a deployment
needs different parallelism:

```yaml
max_parallel_insert: 3        # insert workers (code/LightRAG default 3)
max_parallel_parse_native: 5  # native + legacy parser workers (default 5)
max_parallel_parse_mineru: 2  # MinerU parser workers (default 2)
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
fused chunks after semantic/BM25 merge; `/answer` still packs final prompt
chunks with `answer.context_top_k`. Language profiles and scoring constants
are advanced index signatures.

Defaults:

```yaml
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

Changing profile names, text configs, languages, `bm25_k1`, or `bm25_b`
changes the expected pg_textsearch index signature. Restart DlightRAG against
its configured PostgreSQL endpoint so it can rebuild or verify indexes before
serving traffic.

## Fusion And Filtering

Advanced retrieval scoring:

```yaml
rrf_k: 60
metadata_filter_exact_vector_threshold: 8192
```

`metadata_filter_exact_vector_threshold` controls when DlightRAG can use exact
vector scoring inside a small metadata candidate set.

## Image Budgets

`answer.max_images` and the answer byte/geometry fields define one transport
budget shape. REST, SDK, and MCP use one instance for their request-local
current/history/RAG visuals. Web Composer turns create two independent instances
with that same shape:

- Composer: current direct images, selected history images, and visuals parsed
  from current/history Composer documents;
- RAG: LightRAG visual evidence only.

Each Web lane receives the full configured count and byte ceiling. The two
instances do not borrow or consume each other's remaining capacity. At startup
the configured shape is clamped to the query-role model's discovered image
capability. `answer.max_images` must remain `>= query_images.max_current_images`
so current direct uploads fit inside the Composer budget. Compression budgets
are advanced model transport limits:

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
  image_max_bytes: 1500000
  image_max_total_bytes: 8000000
  image_max_px: 1280
  image_min_px: 768
  image_quality: 86
  image_min_quality: 76

answer:
  max_images: 8
  image_max_bytes: 3000000
  image_max_total_bytes: 24000000
  image_max_px: 1536
  image_max_pixels: 40000000
  image_min_px: 1024
  image_quality: 89
  image_min_quality: 79
```

`answer.image_max_pixels` rejects source images whose decoded dimensions exceed
the limit before RGB conversion or resizing. The Web upload validator,
cache-neutral Composer analysis transport, query-image description, and final
answer transport use the same ceiling.

## Query Images And Visual Assets

In Web/UI terminology, **query attachments** are query images plus Composer
documents. Current-request images are described with the VLM before retrieval
when a VLM is configured. Public REST, MCP, CLI, and Python answer/retrieve calls are
stateless; durable conversation images belong only to the Web conversation
store. `max_upload_bytes` is the decoded per-image Web admission limit, not the
answer-model compression budget. Its default is 15 MiB and the backend and
browser consume the same value:

```yaml
query_images:
  max_current_images: 3
  max_upload_bytes: 15728640  # 15 MiB per current Web image

web_conversations:
  max_turns: 100
  ttl_days: 30

visual_assets:
  thumb_max_px: 300
  thumb_cache_size: 256
```

`web_conversations` applies only to the principal-scoped Web-only conversation
lifecycle. It keeps at most 100 complete turns and uses 30-day inactivity
retention; expired conversations are hidden immediately and pruned without
touching ingest documents, chunks, vectors, graph data, source files, visual
assets, or jobs. Current-turn images always have priority; historical images
that miss a transport slot contribute their stored text descriptions. These
retention and upload controls do not change that behavior.

Web Composer documents use the configured parser routing and
`parser_sidecars.vlm.enabled` switch. On a cache miss they run the real parser,
cache-neutral LightRAG VLM/EXTRACT analysis, and LightRAG multimodal renderer in
temporary storage. After parse/chunk work or a cache load, each current or
planner-selected historical Composer document is routed once by its total estimated
chunk tokens. Documents at or below 24,576 tokens use full mode: every chunk enters
the Answer Composer document context directly, with no document embedding, vector
read, exact retrieval, BM25/RRF, or reranking. Documents above 24,576 tokens
use retrieval mode: document vectors are embedded and cached, exact dense and
lexical/structural retrieval share one candidate pool, and each document is
packed independently to at most 24,576 tokens.

`RAGServiceManager` lazily creates, owns, and closes one cache-neutral
`ComposerModelBundle` for the VLM and EXTRACT roles. The bundle's VLM callable
is shared with `QueryImageDescriber` and sidecar analysis; the manager's
direct-LLM semaphore is their common concurrency bound rather than a
per-request or per-workspace limit.

The first normalized requested workspace deterministically selects one
`RAGService`. That service owns and provides its initialized LightRAG
parser/multimodal renderer, shared `RobustDocumentEmbedder` plus resolved image
capability, and reranker. Composer only borrows them, does not close them, and
does not add another resource pool: parser/MM work remains request-local,
retrieval-mode embedding uses the shared embedder's own semaphore, and
retrieval-mode reranking uses the service-owned callable. In retrieval mode,
fused image+text document embedding is used when that resolved capability is
active; otherwise, or after a fused provider failure, the document vector falls
back to text-only embedding.

Provider and processing-resource sharing does not share results. Enriched
chunks and JSONB vectors remain owned only by the Web PostgreSQL store in
`web_conversation_attachment_chunks`, scoped by principal and conversation.
Manual delete and TTL pruning cascade those rows; `max_turns` trimming preserves
them until the conversation itself is deleted or expires. There is no
cross-conversation reuse, workspace RAG write, HNSW/ANN index, or Composer
vector configuration. Composer dense ranking is exact blockwise `float32`
cosine with a fixed internal inclusion threshold of `>= 0.5`. The Composer
current/history visual budget described above remains independent from the
workspace RAG visual budget.

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

## MCP Streamable HTTP

DlightRAG's HTTP MCP server uses the current Streamable HTTP transport on a
single `/mcp` endpoint (it does not expose the deprecated HTTP+SSE `/sse` +
`/messages` pair). It binds to loopback by default:

```yaml
mcp_transport: streamable-http
mcp_host: 127.0.0.1
mcp_port: 8101
```

To expose MCP beyond loopback, set `mcp_host` and enable `auth_mode`. See
[security.md](security.md).

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
max_conversation_turns: 50
max_conversation_tokens: 81920
max_upload_bytes: 104857600
max_upload_size_mb: 512
ingest_timeout:
request_timeout: 300
```

`max_conversation_tokens` caps recent text history supplied to the query
planner. Web planning additionally reserves current-request context: current
Composer documents share an 8192-token, structure-aware digest budget; prior
image/document catalogs share 4096 tokens; and the complete rendered planner
input is bounded to 102400 estimated tokens. When the envelope is full, the
oldest text history and catalog entries yield before current Composer documents. These
are internal planner safety budgets rather than additional public settings.

`max_upload_bytes` applies to REST multipart ingest; `max_upload_size_mb`
applies to Web uploads. `ingest_timeout` limits how long the SDK convenience
method `RAGServiceManager.aingest()` waits for its durable job. When it expires,
the job keeps running and the method returns its current row instead of
cancelling it. REST, Web, and MCP start jobs immediately and are not governed by
this wait setting.

Ingest job state is stored in `dlightrag_ingest_jobs`. DlightRAG keeps this as
operational state rather than user-facing configuration: recent queued/running
jobs are recovered automatically on startup, completed jobs are pruned after 14
days, queued/running jobs that have not updated for 24 hours are marked failed
on job-store initialization, and workspace reset cancels active in-process jobs
before deleting the matching workspace's job rows. Remote prefix recovery resumes
from the next unfinished source window; single-document internals remain owned by
LightRAG's document status pipeline.

## LightRAG KG Internals

```yaml
chunk_p_token_size: 1024
kg_chunk_pick_method: VECTOR
max_entity_tokens: 6000
max_relation_tokens: 8000
max_total_tokens: 40000
vector_db_kwargs: {}
```

`kg_entity_types` is intentionally public because it shapes domain extraction.
For stronger domain control, use `extraction.entity_type_prompt_file` with a
file under `prompts/entity_type/`.
