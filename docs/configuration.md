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
    min_image_pixel: 100
  mineru:
    api_mode: local
    local_endpoint: http://127.0.0.1:8210
    language: ch
    max_polls: 1200
    auxiliary_block_policy: conservative
```

`parser_sidecars.mineru.language` is MinerU's OCR language hint for scanned or
image-based documents. It is separate from `extraction.language`, which controls
LightRAG's KG extraction prompt language. DlightRAG does not expose
MinerU-side image/chart analysis as a product setting; LightRAG 1.5.4 defaults
that parser-time path off, while LightRAG's separate multimodal analyze stage
handles images, tables, and equations after parse.

## Embeddings

The root config keeps the embedding provider/model visible because it defines
the LightRAG vector schema. DlightRAG prefers unified multimodal embeddings for
direct image retrieval, but text-only providers are valid. At startup,
`embedding.startup_probe` checks image-embedding capability with a small
in-memory image through the provider-specific payload serializer and does not
write to PostgreSQL, LightRAG storage, or local files. Generic
`openai_compatible` is treated as text-only because OpenAI-compatible
embedding APIs do not define a standard image payload. A Qwen3-VL embedding
model on a non-DashScope OpenAI-compatible endpoint is routed to
`qwen_openai_compatible`; DashScope-hosted Qwen uses `dashscope_qwen`. If the
provider is text-only or the probe fails, DlightRAG automatically skips direct
image vector overwrite and query-image vector retrieval while leaving LightRAG's
semantic multimodal path enabled.

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

```yaml
llm:
  default:
    provider: openai                    # protocol family, not the vendor
    model: deepseek-v4-flash
    base_url: https://api.deepseek.com   # ← the vendor is selected here
```

## LLM Structured Output

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

By default, Azure Blob, S3, and URL source files are not copied into DlightRAG
storage. Retrieved sources point back to `azure://...`, `s3://...`, URL
provenance, or a caller-supplied stable URI such as `bynder://asset/...`.
Set `retain_remote_source_files: true` to keep fetched remote files under the
workspace input root by default, or pass `retain_source_file` on one SDK/REST/MCP
ingest call to override that default. When retention is enabled, stored metadata
`file_path` points at the retained workspace-local file, not the remote URI.
`GET /files/raw/{file_path}` redirects only when the stored source is a direct
provider/source URL. Azure uses `DLIGHTRAG_BLOB_CONNECTION_STRING`. S3 uses the
standard AWS credential chain (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
`AWS_SESSION_TOKEN`, `AWS_REGION`/`AWS_DEFAULT_REGION`, IAM role, or shared AWS
config).
REST/MCP `source_type="url"` accepts public or signed HTTPS URLs only, does not
follow redirects to private hosts, and caps each download with
`url_ingest_max_bytes`. SaaS APIs that require auth headers should be wrapped by
an SDK `AsyncDataSource` connector and ingested with
`RAGServiceManager.aingest_source()`. URL ingest stores fetch URLs without query
or fragment by default; pass `source_uri` or `source_uris` when a stable source
identity should be persisted instead.
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
postgres_pool_max_size: 10
postgres_session_settings: {}
postgres_statement_cache_size:
postgres_connection_retries: 10
postgres_connection_retry_backoff: 3.0
postgres_connection_retry_backoff_max: 30.0
postgres_pool_close_timeout: 5.0
```

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

Root config exposes `answer.max_images` for retrieved RAG context images and
`answer.max_user_images` for user-supplied `query_images` plus history images.
Those slot budgets are independent; user images do not evict retrieved page
previews. Compression budgets are advanced model transport limits:

`chat_llm_reranker` can use its own `rerank.provider` and `rerank.model`. When
those are omitted, it reuses the configured chat model roles in this order:
`llm.roles.vlm`, `llm.roles.query`, then `llm.default`.

Voyage's text reranker is available with `strategy: voyage_reranker`,
`model: rerank-2.5` or `rerank-2.5-lite`, and `DLIGHTRAG_RERANK__API_KEY`.
Cohere's public text reranker is available with `strategy: cohere_reranker`,
`model: rerank-v4.0-pro` or `rerank-v4.0-fast`, and the same API key env var.

`rerank.input_modality` defaults to `auto`. For `chat_llm_reranker`, auto
reuses the startup vision probe for the selected scoring model: vision-capable
models receive bounded image data plus text, and non-vision models receive VLM
text only. HTTP rerankers follow their API contract: latest text rerankers such
as `jina-reranker-v3`, `qwen3-rerank`, Voyage `rerank-2.5` family, and Cohere
`rerank-v4.0` family receive chunk VLM text content, while `jina-reranker-m0`
and `qwen3-vl-rerank` receive bounded image documents when chunks have
`image_data`. Use
`input_modality: text` to force text-only reranking, or
`input_modality: multimodal` for a self-hosted/new model that accepts image
document objects.

```yaml
rerank:
  strategy: chat_llm_reranker
  input_modality: auto
  # Optional. Omitted defaults to 0.5 for chat_llm_reranker.
  # Provider rerankers keep all scored candidates before top_k when omitted.
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
  max_images: 6
  max_user_images: 3
  image_max_bytes: 3000000
  image_max_total_bytes: 24000000
  image_max_px: 1536
  image_min_px: 1024
  image_quality: 89
  image_min_quality: 79
```

## Query Images And Visual Assets

User-attached chat images are always described with the VLM before retrieval
when a VLM is configured. Session, description, and thumbnail cache settings
are internal resource limits. Session image memory uses
`checkpoint_session_ttl_days`, so checkpoint-restorable image references expire
with the Web checkpoint history they belong to:

```yaml
query_images:
  max_described_images: 3
  session_max_images: 50
  session_max_sessions: 100

visual_assets:
  thumb_max_px: 300
  thumb_cache_size: 256
```

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
max_conversation_tokens: 150000
max_upload_bytes: 104857600
max_upload_size_mb: 512
ingest_timeout:
request_timeout: 300
```

`max_upload_bytes` applies to REST multipart ingest; `max_upload_size_mb`
applies to Web uploads. `ingest_timeout` limits how long synchronous
SDK/REST/MCP ingest calls wait. When it expires, the ingest job keeps running
and callers receive/read the job status instead of cancelling the ingest.

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
