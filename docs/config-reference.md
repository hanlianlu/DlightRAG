# DlightRAG Configuration Reference

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
- high-level concurrency: `max_async`, `embedding_func_max_async`,
  `max_parallel_*`
- retrieval/answer breadth: `top_k`, `chunk_top_k`, `bm25_top_k`,
  `direct_visual_top_k`, `answer.*`
- auth and observability mode switches when they are not secret

Keep these out of normal `config.yaml` unless debugging or load-testing proves
they need to change:

- storage backend literals
- parser routing rules
- PostgreSQL retry/backoff internals
- queue sizes
- HNSW index internals
- BM25 language profile signatures and k1/b tuning
- RRF and exact-vector filter thresholds
- image compression budgets
- source highlight timeout/cache internals
- query-image semantic description limits
- visual thumbnail cache internals

## Parser And MinerU

DlightRAG defaults to LightRAG native parsing for DOCX and MinerU for other
supported document formats. Change parser routing only when validating a new
LightRAG parser strategy.

Advanced parser fields with code defaults:

```yaml
parser:
  rules: "docx:native-iteP,*:mineru-iteP"
  chunk_options: {}

extraction:
  use_json: true

parser_sidecars:
  vlm:
    enabled: true
    max_image_bytes: 5242880
    min_image_pixel: 100
  mineru:
    local_backend: hybrid-auto-engine
    local_parse_method: auto
    local_image_analysis: true
    enable_table: true
    enable_formula: true
    poll_interval_seconds: 2
    max_polls: 180
    language: ch
    page_ranges:
    model_version: vlm
    is_ocr: false
```

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

Use [PG.md](PG.md) for production sizing, SSL, shared memory, and extension
notes.

## Staged Ingest Queues

The root config exposes worker concurrency. Queue sizes are internal backpressure
settings and should only change after measuring parser/analyze/insert pressure.

```yaml
queue_size_parse: 20
queue_size_analyze: 32
queue_size_insert: 4
embedding_batch_num: 10
embedding_request_timeout: 120
```

## BM25

BM25 is part of the supported DlightRAG retrieval path. Root config exposes
`bm25_top_k`; language profiles and scoring constants are advanced index
signatures.

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

Root config exposes `answer.max_images`. Compression budgets are advanced model
transport limits:

```yaml
rerank:
  score_threshold: 0.5
  max_concurrency: 4
  batch_size: 7
  image_max_bytes: 1500000
  image_max_total_bytes: 8000000
  image_max_px: 1280
  image_min_px: 768
  image_quality: 86
  image_min_quality: 76

answer:
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
are internal resource limits:

```yaml
query_images:
  max_described_images: 3
  session_max_images: 50
  session_max_sessions: 100
  session_ttl_seconds: 3600

visual_assets:
  thumb_max_px: 300
  thumb_cache_size: 256
```

## Citations

Citation validation is always part of answer finalization. Web source-panel
semantic highlights are enabled by default and use the keyword LLM role after
the answer has already been streamed/finalized. REST and MCP answer payloads do
not run this highlight enrichment path.

Advanced highlight controls:

```yaml
citations:
  highlights:
    enabled: true
    timeout: 10.0
    max_concurrency: 8
    max_input_chars: 4096
    cache_size: 500
```

Disable highlights only when Web UI follow-up source enrichment cost is more
important than highlighted source snippets.

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
applies to Web uploads.

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
