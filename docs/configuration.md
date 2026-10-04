# Configuration

This document owns configuration precedence and the settings reference: fields,
defaults, and examples. Runtime design belongs in [Architecture](architecture.md),
public payloads in [Interfaces](interfaces.md), security policy in
[Security](security.md), and procedures in [Operations](operations.md).

Root [`config.yaml`](../config.yaml) holds the checked-in non-secret settings.
Advanced fields are available through explicit YAML additions; nested
`DLIGHTRAG_*` variables are an override mechanism, not a second catalogue of
application settings.

```text
constructor args > environment variables > .env > config.yaml > code defaults
```

The process reads `config.yaml` and `.env` from its current working directory.
`config.yaml` is YAML 1.2: only `true` and `false` (also `True`, `TRUE`,
`False`, and `FALSE`) are implicit booleans, so plain `on`, `off`, `yes`, and
`no` stay strings. Duplicate keys and a `%YAML` directive naming another version
are rejected.

Precedence determines the effective value; it does not assign ownership. Within
one deployment, configure a setting in one place rather than relying on a
higher-precedence source to mask a duplicate lower-precedence value. Nested
environment variables follow the settings path with `__` separators:

```bash
DLIGHTRAG_MODELS__EMBEDDING__API_KEY=...
DLIGHTRAG_STORAGE__POSTGRES__HOST=postgres
```

DlightRAG has nine top-level sections: `deployment`, `storage`, `models`,
`corpus`, `answer`, `runtime`, `access`, `interfaces`, and `observability`.
Unknown keys are rejected, whether they come from YAML or from `DLIGHTRAG_*`
variables; the only `DLIGHTRAG_*` names that are not settings are the client
variables `DLIGHTRAG_API_URL`, `DLIGHTRAG_API_TOKEN`, and
`DLIGHTRAG_CLIENT_TIMEOUT`. The ownership decision is recorded in
[ADR 0006](adr/0006-configuration-ownership-and-deployment-bindings.md).

## Configuration Ownership

Use the first matching rule:

| Concern | Canonical owner | Examples |
|---|---|---|
| Credentials | `.env` locally; an orchestrator Secret in production | provider API keys, PostgreSQL password, JWT verification key |
| Deployment access | `.env` beside the checked-in `config.yaml` | auth mode, issuer, audience, access rules |
| Non-secret product behavior and integration choices | `config.yaml` | model selection, parser provider and endpoint, workspace identity, retrieval breadth, Answer policy, observability behavior |
| Facts created by deployment topology | Compose or another deployment manifest | Service DNS, container listener/transport binding, volumes, probes, resource limits, PostgreSQL server tuning |
| Stable low-level mechanics | Code defaults until measurement requires an explicit override | retry/backoff, cache bounds, parser polling, vector thresholds |

The checked-in `config.yaml` holds the development default
`access.auth_mode: none`. A deployment that runs that file sets its own access in
`.env` (see `.env.example`), so its issuer, audience, and people stay out of the
repository; a deployment that mounts its own `config.yaml` may keep access there.

Environment overrides suit Secrets, deployment access, topology bindings, and
short-lived operational exceptions. Do not copy ordinary model, retrieval, or
Answer policy from `config.yaml` into Compose or Kubernetes manifests.

For filesystem settings, prefer mounting storage at the configured or default
application path, and override the application path only when the deployment
cannot align its mount. For infrastructure endpoints such as PostgreSQL Service
DNS, the deployment manifest may own the typed setting; do not also place that
value in its mounted `config.yaml`.

Leave these at code defaults unless measurement proves otherwise:

- storage backend literals
- retry/backoff, HNSW, image-compression, and parser polling internals
- per-stage ingestion workers and queue sizes
- BM25 index signatures, RRF constants, and exact-vector thresholds
- thumbnail, highlight-cache, and URL-signing internals

### Service URLs carry no credentials

A configured service URL must not embed userinfo (`https://user:password@host`).
URLs are logged, and some are returned or published: the model catalogue lists
every entry's `base_url` to any authenticated caller, and OAuth metadata
carries the callback URL. Startup therefore refuses userinfo in every URL
setting, with an error that names the field and never repeats the value:

- `models.chat.default.base_url`, `models.chat.roles.*.base_url`,
  `models.catalogue[].base_url`, `models.embedding.base_url`, and
  `models.rerank.base_url`
- `corpus.sidecars.mineru.official_endpoint`, `.local_endpoint`, and
  `corpus.sidecars.docling.endpoint`
- `storage.lightrag.milvus_uri`
- `access.jwt_issuer`, `access.jwt_jwks_url`, `access.web_identity.issuer`, and
  `access.web_identity.jwks_url`
- `interfaces.mcp.resource_server_url` and `observability.langfuse_host`
- `answer.agent.connections.oauth_callback_url`
- `answer.agent.browser.endpoints[]` and `answer.agent.browser.egress_proxy`

Put the credential in the service's own secret setting instead (`api_key`,
`api_token`, `milvus_token`, and so on). A model catalogue entry published at
runtime is refused the same way.

## Container And Kubernetes Contract

The bundled image runs from `/app`, so Compose grants the canonical file through
its top-level `configs` resource at `/app/config.yaml`; this keeps configuration
distinct from data volumes. A Kubernetes ConfigMap should mount the same file at
the same path. Compose and Kubernetes mount the file but do not interpret its
fields or derive volume topology from them. Inject only credentials, deployment
access, and the minimal topology bindings each workload needs:

- mount corpus storage at `/app/dlightrag_storage`, matching the checked-in
  `deployment.working_dir: ./dlightrag_storage`;
- mount the shared Agent Workspace, when enabled, at
  `/home/app/.dlightrag/agent_workspaces`, matching the application default;
- inject the PostgreSQL Service DNS through
  `DLIGHTRAG_STORAGE__POSTGRES__HOST` unless the mounted YAML owns it;
- bind container listeners and select the MCP network transport in the workload
  manifest because these choices vary by process role;
- keep the development-only insecure-listener waiver
  (`DLIGHTRAG_ACCESS__ALLOW_INSECURE_NO_AUTH`) beside the manifest's
  loopback-only port publication; a deployment that publishes beyond loopback
  sets its access instead and drops the waiver;
- keep ports, Services, volumes, probes, resource requests/limits, and database
  server tuning entirely outside DlightRAG application configuration.

## Parser Sidecars

Configure one `mineru` or `docling` block; configuring both fails. With neither,
the code default is local MinerU at `http://127.0.0.1:8210`. DlightRAG derives
LightRAG's parser rule from the configured block. A parser change affects new
parses, not existing indexed data.

### MinerU (default)

The checked-in Docker configuration reaches a host service through
`host.docker.internal`:

```yaml
corpus:
  sidecars:
    mineru:
      api_mode: local
      local_endpoint: http://host.docker.internal:8210
      language: ch
      backend: hybrid-engine
```

| Field | Default | Notes |
|---|---|---|
| `api_mode` | `local` | `local` or `official` |
| `local_endpoint` | `http://127.0.0.1:8210` | Use the Docker host alias from containers |
| `official_endpoint` | `https://mineru.net` | Used only in official mode |
| `api_token` | unset | Prefer the nested environment variable |
| `language` | `ch` | OCR hint, separate from extraction output language |
| `backend` | `hybrid-engine` | `pipeline`, `vlm-engine`, or `hybrid-engine` |
| `poll_interval_seconds` | `5` | Parser polling interval |
| `max_polls` | `1440` | Two-hour default polling window |

DlightRAG leaves MinerU image analysis off; its own VLM sidecar describes
extracted figures instead. `make mineru-install` installs MinerU 3.4.5 or newer;
[Parser Services](operations.md#parser-services) covers running it.

### Docling

```yaml
corpus:
  sidecars:
    docling:
      endpoint: http://host.docker.internal:5001
      do_formula_enrichment: true
      force_ocr: true
      code_formula_preset: granite_docling
```

| Field | Default | Notes |
|---|---|---|
| `endpoint` | `http://127.0.0.1:5001` | External docling-serve endpoint |
| `do_formula_enrichment` | `true` | Transcribe detected formulas |
| `force_ocr` | `true` | Set `false` for reliable born-digital PDF text layers |
| `code_formula_preset` | `granite_docling` | Formula model; YAML `null` selects docling-serve's built-in model, which cannot run on Apple Silicon (MPS) |
| `poll_interval_seconds` | `5` | Parser polling interval |
| `max_polls` | `1440` | Two-hour default polling window |

DlightRAG always requests PDF heading hierarchy, which docling-serve 1.30.0+
(docling-jobkit 3.3.0+) honors; older services ignore it. The Docling service
owns its OCR engine and languages. The optional Compose CPU profile uses
`http://docling:5001` with `code_formula_preset: null`.

Both parser services need an HTTP keep-alive longer than the five-second poll
interval; DlightRAG's MinerU launcher sets 60 seconds.

### Figure VLM

```yaml
corpus:
  sidecars:
    vlm:
      enabled: true
      max_image_bytes: 5242880
      min_image_pixel: 80
      surrounding_leading_max_tokens: 256
      surrounding_trailing_max_tokens: 256
```

| Field | Default | Meaning |
|---|---|---|
| `corpus.sidecars.vlm.enabled` | `true` | Analyze parser-extracted figures |
| `corpus.sidecars.vlm.max_image_bytes` | `5242880` | Maximum source image bytes |
| `corpus.sidecars.vlm.min_image_pixel` | `80` | Minimum image side accepted |
| `corpus.sidecars.vlm.surrounding_leading_max_tokens` | `256` | Leading text supplied to figure analysis; `null` defers to LightRAG's own limit |
| `corpus.sidecars.vlm.surrounding_trailing_max_tokens` | `256` | Trailing text limit; `null` defers to LightRAG's own limit |

### Chunking And Extraction

| Field | Default | Meaning |
|---|---|---|
| `corpus.parser.chunk_options` | `{}` | Advanced LightRAG parser/chunk keyword arguments |
| `corpus.extraction.use_json` | `true` | Request structured extraction |
| `corpus.extraction.language` | `English` | Generated entity/relation and keyword language |
| `corpus.extraction.entity_type_prompt_file` | unset | `.yml`/`.yaml` filename under `prompts/entity_type/` |

Extraction language does not configure OCR or translate existing graph data.

## Embeddings

One embedding space is shared by ingestion and every retrieval leg. Never mix
models, dimensions, or vector spaces in one workspace; use a new workspace or a
complete offline rebuild.

### Providers

| `provider` | Typical model | Visual support | Dimension wire field |
|---|---|---|---|
| `openai` | `text-embedding-3-large` | Text only | `dimensions` for supported models |
| `openai_compatible` | Deployment-defined | Text only | Never sent; response is validated |
| `voyage` | `voyage-multimodal-3.5` | Native text+image fusion | `output_dimension` |
| `gemini` | `gemini-embedding-2` | Native content aggregation | `outputDimensionality` |
| `jina` | `jina-embeddings-v4` | Native text+image fusion | `dimensions` |
| `cohere` | `embed-v4.0` | Native mixed-input fusion | `output_dimension` |
| `azure_cohere` | `Cohere-embed-v4` | Native mixed-input fusion | `output_dimension` |

Unknown model names resolve conservatively to text-only operation.
`openai_compatible` does not invent vendor-specific image, dimension, or task
fields. Azure OpenAI v1 roots and Azure Cohere deployment scoring roots are
supported by their corresponding adapters.

### Fields

| Field | Default | Meaning |
|---|---|---|
| `provider` | `voyage` | Protocol adapter |
| `model` | `voyage-multimodal-3.5` | Exact model or deployment identifier |
| `api_key` | unset | Put secrets in `.env` |
| `base_url` | provider default | Protocol root or accepted complete endpoint; required for `openai_compatible` and `azure_cohere` |
| `dim` | `1024` | Vector/schema dimension; every response is validated |
| `max_token_size` | `8192` | LightRAG's embedding token limit; a longer chunk is split before embedding |
| `input_modality` | `auto` | `auto`, `text`, or `multimodal` |
| `startup_probe` | `true` | Verify configured visual paths |
| `timeout` | `120` | Request timeout in seconds |
| `max_concurrency` | `16` | Concurrent embedding calls per workspace runtime |
| `batch_size` | `64` | LightRAG embedding batch size |

`auto` enables native fused document vectors and image-query retrieval for
known multimodal models. `text` disables both image paths. `multimodal` requires
them and makes probe failure fatal. Fused output replaces the canonical chunk
vector; it never creates a second visual document vector.

Only a definitive probe outcome settles a workspace runtime's mode. A failure
the shared dependency classification treats as transient (a connection error, a
timeout, or a retryable status such as 429 or 503) settles nothing: the
workspace stays unavailable rather than embedding documents text-only beside the
corpus's fused vectors. While it is unavailable, requests that need it are
refused with HTTP 503 (`The model provider is temporarily unavailable`), Runs
that need it defer as a model provider outage, and for the default workspace
`GET /health` reports `providers` degraded. The first failure opens a 15-second
backoff that doubles with each consecutive failure up to 5 minutes; the first
request after it rebuilds the runtime and probes again. Corpus storage that is
briefly out while a workspace is built is reported the same way, as corpus
storage. Every other failure is definitive, including 5xx statuses outside the
retryable set (for example 501, 505, or 507): it leaves both image paths off
under `auto` and fails the runtime under `multimodal`.

Embedding batches split automatically at the provider's input-count and
inline-image byte limits while preserving order. Each embedding request is
retried at most twice, and only for a failure the durable Run classification
also treats as transient: a refused, reset, dropped, or timed-out connection, or
HTTP 408, 425, 429, 500, 502, 503, 504, 520-524 (an edge proxy reporting its
origin failed), or 529 (overloaded). HTTP 409 reports a conflict that resending
cannot resolve and is not retried; neither is a TLS certificate that fails
verification or a TLS protocol mismatch, such as an https URL for a plain-HTTP
service. A host name that does not resolve is retried, since a restarting
service stops resolving until it is back. `Retry-After` wins over exponential
backoff with jitter.

```yaml
models:
  embedding:
    provider: voyage
    model: voyage-multimodal-3.5
    base_url: https://api.voyageai.com/v1
    dim: 1024
    input_modality: auto
    startup_probe: true
```

For a host-side local endpoint used from Compose, replace `127.0.0.1` with
`host.docker.internal`. See
[Offline Vector Storage Rebuild](operations.md#offline-vector-storage-rebuild)
before changing an existing workspace's vector space.

## Chat Models

`provider` identifies the SDK/protocol, not the vendor:

| `provider` | Transport | Typical endpoints |
|---|---|---|
| `openai` | Chat Completions or Responses, selected per model | OpenAI, DeepSeek, OpenRouter, Azure OpenAI, vLLM, Ollama, other compatible APIs |
| `anthropic` | Anthropic native SDK | Claude |
| `gemini` | Google GenAI SDK, Interactions API ([Gemini](#gemini)) | Gemini |

Select OpenAI-compatible vendors with `base_url`; unknown provider names are
rejected. Model IDs are endpoint-specific: DeepSeek's flash model is
`deepseek-flash` on `api.deepseek.com` and `deepseek/deepseek-v4.1-flash` on
OpenRouter.

### Role Configuration

```yaml
models:
  chat:
    default:
      provider: openai
      model: openai/gpt-6.1-sol
      base_url: https://openrouter.ai/api/v1
    roles:
      extract:
        provider: openai
        model: deepseek-flash
        base_url: https://api.deepseek.com
```

A role override is complete, not a partial merge. Missing or incomplete roles
fall back to `models.chat.default` as a whole. The same fields apply to the
default and each `extract`, `keyword`, `query`, or `vlm` override:

| Field | Default | Meaning |
|---|---|---|
| `provider` | `openai` | `openai`, `anthropic`, or `gemini` protocol |
| `model` | required | Exact model/deployment ID |
| `api_key` | unset | Endpoint credential |
| `base_url` | provider default | Optional API root |
| `api_family` | `chat_completion`; `interactions` for `gemini` | `chat_completion` or `response`; `response` requires `provider: openai`, and a `gemini` model accepts only `interactions` |
| `structured_output` | `auto` | `auto`, `json_schema`, or `json_object` |
| `temperature` | unset | Nonnegative provider temperature; a `gemini` model refuses one |
| `timeout` | `240` | Request timeout seconds |
| `max_retries` | `3` | Provider SDK retries of a transient request failure; the retrieval planner adds none; Gemini's SDK retries at least once |
| `reasoning` | unset | Typed reasoning level |
| `agentic_reasoning` | inherits `reasoning` | Research-specific level; explicit `null` disables |
| `model_kwargs` | `{}` | Provider-specific ordinary options |
| `agentic_model_kwargs` | `{}` | Shallow Research overlay |

A `models.chat.default` that leaves fields out, as when only its API key comes
from the environment, takes them from the code's default endpoint
(`provider: openai`, `google/gemini-3.8-flash` on OpenRouter,
`temperature: 1.0`). When the default's `provider` is not `openai` it describes
another endpoint, Anthropic's or Gemini's, and inherits none of that endpoint's
`base_url`, `model`, or `temperature`.

### API Family

For `openai` and `anthropic`, omitting `api_family` selects **`chat_completion`**;
a `gemini` model's family is always `interactions` ([Gemini](#gemini)). The
shipped `config.yaml` selects **`response`** for the default model and the Query
role; Extract, Keyword, and VLM use Chat Completions, and reranking uses Voyage.
Its endpoint and transport settings, other model options omitted:

```yaml
models:
  chat:
    default:
      provider: openai
      api_family: response
      model: z-ai/glm-5.3-flash
      base_url: https://openrouter.ai/api/v1
    roles:
      query:
        provider: openai
        api_family: response
        model: deepseek-flash
        base_url: https://api.deepseek.com
        structured_output: json_object
```

Supply credentials through `DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY` and
`DLIGHTRAG_MODELS__CHAT__ROLES__QUERY__API_KEY`. The SDK appends `/responses` to
`base_url`, so direct DeepSeek is called at `https://api.deepseek.com/responses`
and OpenRouter at `https://openrouter.ai/api/v1/responses`; a custom compatible
root works the same way. An unsupported capability fails explicitly rather than
dropping a Tool, an image, or a requested reasoning level, and typed `reasoning`
owns the family-specific translation.

API Family selects the wire, not a provider or capacity profile; nothing probes
an endpoint or falls back to the other family. Response supports the same five
entrypoints, structured output, streaming, local function Tools, and user and
Tool-result images. `tool_choice=auto` lets the model answer without a call;
requested Tools are authorized and executed locally.

Every Response request sends the full local context with `store=false`,
`background=false`, and `truncation=disabled`, so no remote conversation,
background job, hosted tool, or provider-side compaction takes part
([Agent Session Recovery](run-runtime.md#agent-session-recovery) covers replay).
A request whose raw `model_kwargs` set a field the product owns fails: the input
or messages, model, instructions, tools or tool choice, streaming, storage,
truncation, output format or token limit, reasoning, temperature, metadata,
`include`, prompt templates or caching, or remote state.

`store=false` minimizes remote response state; **it does not establish Zero Data
Retention**. Provider logs, context caches, agreements, and OpenRouter routing
policies are deployment facts.

Live verification of Response covers only direct DeepSeek `deepseek-flash` and
OpenRouter `z-ai/glm-5.3-flash`. Official OpenAI
(`base_url: https://api.openai.com/v1`) is **experimental**: offline contract
tests cover it, but it is not live-qualified. Closing a stream does not prove the
provider stopped computing or billing.
[ADR 0027](adr/0027-api-family-selects-the-provider-wire.md) records the
decision.

### Gemini

`provider: gemini` calls Gemini through Google's Interactions API, statelessly.
Its API Family is always `interactions`, and no other value is accepted:

```yaml
models:
  chat:
    roles:
      query:
        provider: gemini
        model: gemini-3.8-flash
        reasoning: high
```

Every request carries the full local context with `store=false`, so the API
stores no interaction to resume or retrieve; DlightRAG never sends
`previous_interaction_id`, a background run, a webhook, an environment, or an
agent. As with Response, `store=false` is not Zero Data Retention.

The Interactions API's `GenerationConfig` has no `temperature` or `top_p`, so a
`temperature` on a Gemini model or a Gemini chat reranker fails configuration;
a Gemini default does not inherit the code default's
([Role Configuration](#role-configuration)).

Typed `reasoning` maps through the catalogue's `gemini` format to
`generation_config.thinking_level`, one of `minimal`, `low`, `medium`, or `high`
(`gemini-3.8-flash` takes `low`, `medium`, and `high`). An unsupported level
clamps to the nearest supported one, and `off` cannot be honored because Gemini
cannot turn thinking off. A configured level also asks for
`thinking_summaries: auto`, whose text becomes the turn's reasoning. Thoughts and
their signatures are stored with the Assistant Entry and sent back verbatim, in
place, to the same model; another model sees only the canonical text and calls.

`model_kwargs` accept `safety_settings`, in the Interactions shape
(`{type: dangerous_content, threshold: block_only_high}`), and `service_tier`.
Raw `thinking_level` and `thinking_summaries` are accepted only where no typed
level owns them, and any other key fails the request before it is sent.
Structured output is a JSON `response_format` (`type: text`,
`mime_type: application/json`, and the schema), and a Tool result's images ride
inside its own `function_result`. Reported output tokens include thinking, and
an overload Gemini reports inside a response or its stream is retried and
deferred like an HTTP 503.
[ADR 0030](adr/0030-gemini-uses-the-stateless-interactions-api.md) records the
decision.

### Model Catalogue And Reasoning

Model profiles are keyed by normalized `provider`, exact `model`, and normalized
`base_url`. An entry with a null `base_url` describes the provider's default
endpoint, so it also matches a model that writes that endpoint out
(`https://api.openai.com/v1`, `https://api.anthropic.com`, or
`https://generativelanguage.googleapis.com`). Resolution order is:

```text
PostgreSQL runtime overlay > models.catalogue > built-in catalogue > fallback
```

A profile defines context/input/output limits, image support, and optionally a
reasoning format with all seven typed levels: `off`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max`. Unsupported levels map to `null`; non-off requests
clamp to the nearest supported level. Uncatalogued endpoints use best-effort
protocol mapping and surface provider rejection.

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
      reasoning: max
```

Catalogue revisions are derived from canonical catalogue content and are never
configured manually. Startup catalogue changes require restart. Runtime overlay
operations and revision rules are in
[Interfaces](interfaces.md#model-catalogue-and-profile-memory).

`agentic_reasoning` inherits `reasoning`, and `agentic_model_kwargs` is a
shallow overlay for Research calls. When typed reasoning is configured, raw
provider reasoning keys in `model_kwargs` are rejected, so translation has one
owner. A role that sets reasoning through raw keys in `model_kwargs` or
`agentic_model_kwargs` instead ignores a caller-chosen agent effort, since it has
no typed level to replace. `agentic_reasoning` on the answering role is the
deployment's default, not a ceiling: one Answer request may choose its own
[`effort`](interfaces.md#request-fields).

### Structured Output

`structured_output` defaults to `auto`, which asks for a strict JSON schema. All
three chat protocols (`openai`, `anthropic`, `gemini`) serve one, so `auto` and
`json_schema` resolve identically; the decision that changes anything is the
`json_object` opt-out for an endpoint known to reject the `json_schema`
`response_format` type. Anthropic native rejects `json_object` outright.

A compatible endpoint that rejects the `json_schema` transport type is learned
once per process: the request retries with `json_object`, and later requests to
the same provider, model, endpoint, and API Family skip the rejected attempt
instead of paying the same 400 again, for both complete and streaming calls. Only
an explicit "type unavailable" rejection is remembered; a schema-validation
complaint retries once without becoming a permanent verdict. Restarting the
process probes the endpoint again. Chat writes the contract under
`response_format`; Response writes it under `text.format`; Gemini writes it as a
JSON text `response_format`. Changing API Family does not inherit the other
family's rejection verdict.

```yaml
models:
  chat:
    roles:
      extract:
        provider: openai
        model: deepseek-flash
        base_url: https://api.deepseek.com
        structured_output: json_object
```

## Reranking

Accepted strategy literals are:

| Strategy | Transport | Image policy |
|---|---|---|
| `chat_llm_reranker` | Configured chat endpoint | `auto` uses its vision probe |
| `jina_reranker` | Jina `/v1/rerank` | Explicit multimodal supported by suitable models |
| `aliyun_reranker` | Alibaba Model Studio | Explicit multimodal supported by Qwen VL rerank |
| `local_reranker` | Standard `{model,query,documents,top_n}` `/rerank` | Endpoint-defined |
| `voyage_reranker` | Voyage `/v1/rerank` | Text only |
| `cohere_reranker` | Cohere `/v2/rerank` | Text only |
| `azure_cohere` | Azure Cohere rerank | Text only |

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Enable final fused-candidate reranking |
| `strategy` | `chat_llm_reranker` | One literal above |
| `provider` | unset | Independent chat-rerank provider |
| `model`, `api_key`, `base_url` | unset | Rerank endpoint |
| `input_modality` | `auto` | `auto`, `text`, or `multimodal` |
| `score_threshold` | unset | Hard nonnegative post-rerank cutoff |
| `max_concurrency` | `8` | Concurrent `chat_llm_reranker` scoring requests |
| `batch_size` | `8` | Candidates per `chat_llm_reranker` request |
| `temperature` | unset | Chat reranker temperature |
| `model_kwargs` | `{}` | Provider-specific options |

An HTTP strategy sends one request per rerank, bounded only by
`models.max_concurrency`. Each HTTP strategy validates its required `api_key`
and/or `base_url`; invalid configuration fails startup rather than changing
strategy. Text-only strategies reject explicit multimodal mode.

## Remote Sources

`source_uri` is stable provenance. `download_uri` is the durable S3, Azure, or
queryless public HTTPS locator used when no local copy is retained.

```yaml
corpus:
  ingestion:
    retain_remote_source_files: false
    url_max_bytes: 104857600
    url_private_host_allowlist: []
  sources:
    blob_connection_string: null
    azure_sas_expiry: 3600
    s3_presign_expiry: 3600
    s3_region: null
```

Signed URLs with query or fragment tokens are not durable locators. Retain the
file or provide a separate queryless `download_uri`. Non-retained custom
`AsyncDataSource` connectors must provide the locator directly or through
`download_uri_for_key`; invalid contracts fail before parsing.

URL ingest rejects private hosts unless allowlisted, HTTPS-to-HTTP redirects,
and downloads over `url_max_bytes`. `blob_connection_string` is the Azure
credential; `azure_sas_expiry` and `s3_presign_expiry` bound projected URLs;
`s3_region` overrides SDK discovery. Prefer
`DLIGHTRAG_CORPUS__SOURCES__BLOB_CONNECTION_STRING` for the secret. S3 uses the
standard AWS credential chain. Deleting DlightRAG data never deletes provider
objects. See [Sources](interfaces.md#sources).

Built-in S3 ingestion stores the accepted region (request override, then this
configuration, then explicit SDK discovery) with each non-retained document, as
internal metadata separate from user fields and never a credential snapshot.
Failed-document retry and source-download signing reuse that region even if the
deployment default changes; a custom SDK source that declares no routing uses
the current default. A metadata-only update for the same locator keeps the
stored region unless a new routing choice is supplied. Credentials, signed-URL
expiry, URL size limits, and private-host policy always come from the current
deployment. Retained sources are replayed from their workspace-contained local
bytes, and an explicitly supplied mirror `download_uri` has its own routing.

## PostgreSQL And Process Role

DlightRAG requires PostgreSQL 18 for KV, graph, document status, BM25, metadata,
and Operational State. LightRAG's storage names stay at these defaults:

```yaml
storage:
  lightrag:
    vector_storage: PGVectorStorage
    graph_storage: PGTableGraphStorage
    kv_storage: PGKVStorage
    doc_status_storage: PGDocStatusStorage
  postgres:
    pool_min_size: 2
    pool_max_size: 16
    lightrag_pool_max_size: 16
```

`kv_storage`, `graph_storage`, and `doc_status_storage` accept only the values
above. The only vector alternative is LightRAG's `MilvusVectorDBStorage`; every
other storage class is rejected rather than substituted. Connection budgets and
pool sizing are in [PostgreSQL](postgresql.md#tuning-boundaries).

For a writer deployment using Milvus or a Milvus-compatible Zilliz endpoint,
install the client extra (`dlightrag[milvus]`) and select:

```yaml
storage:
  lightrag:
    vector_storage: MilvusVectorDBStorage
    milvus_uri: https://example-compatible-endpoint
    milvus_db_name: default
```

Keep `milvus_token` in `DLIGHTRAG_STORAGE__LIGHTRAG__MILVUS_TOKEN`. Resolved
DlightRAG values overwrite inherited `MILVUS_URI`, `MILVUS_TOKEN`, and
`MILVUS_DB_NAME`; an unset optional binding leaves the corresponding upstream
environment behavior untouched. DlightRAG never connects to Milvus for health
checks and never creates, copies, or drops Milvus infrastructure itself. Zilliz
uses `MilvusVectorDBStorage`, not a separate storage class.

DlightRAG's own chunk-vector operations exist for `PGVectorStorage` only, so on
Milvus a metadata scope filters vector results after retrieval rather than
inside the search, and a workspace whose embedding settles on fused visual
vectors refuses to start (set `models.embedding.input_modality: text`). Readers
and workspace promotion require `PGVectorStorage`; LightRAG offers no
non-mutating reader attach for an external vector adapter.

| Field | Default | Meaning |
|---|---|---|
| `deployment.service_role` | `writer` | `writer` or `reader`; see [Service roles](postgresql.md#service-roles-and-shared-artifacts) |
| `deployment.workspace` | `default` | Default workspace's display name. Its canonical id strips surrounding whitespace, replaces each character that is not an ASCII letter, digit, or `_` with `_`, lowercases the result, and prefixes a leading digit with `_`; the id must be 1-64 characters, or startup fails |
| `deployment.working_dir` | `./dlightrag_storage` | Corpus/input/artifact root; resolved absolute. Operators place local sources in `inputs/<workspace>`, which DlightRAG only reads; `corpus/` is DlightRAG's own |
| `storage.lightrag.vector_storage` | `PGVectorStorage` | `PGVectorStorage` or explicit `MilvusVectorDBStorage` |
| `storage.lightrag.milvus_uri` | unset | Optional `MILVUS_URI` bridge; may be a Milvus-compatible Zilliz URI |
| `storage.lightrag.milvus_token` | unset | Optional secret `MILVUS_TOKEN` bridge |
| `storage.lightrag.milvus_db_name` | unset | Optional `MILVUS_DB_NAME` bridge |
| `storage.postgres.host` | `localhost` | PostgreSQL host |
| `storage.postgres.port` | `5432` | PostgreSQL port |
| `storage.postgres.user` | `dlightrag` | Login role |
| `storage.postgres.password` | `dlightrag` | Password; override in `.env` |
| `storage.postgres.database` | `dlightrag` | Database |
| `storage.postgres.ssl_mode` | unset | `disable`, `allow`, `prefer`, `require`, `verify-ca`, or `verify-full` |
| `ssl_cert`, `ssl_key`, `ssl_root_cert`, `ssl_crl` | unset | TLS file paths |
| `pool_min_size`, `pool_max_size` | `2`, `16` | Domain pool bounds |
| `lightrag_pool_max_size` | `16` | Separate LightRAG pool maximum |
| `command_timeout`, `acquire_timeout` | `60`, `30` | SQL command/acquire seconds |
| `session_settings` | `{}` | asyncpg session parameters |
| `statement_cache_size` | driver default | Prepared-statement cache size |
| `connection_retries` | `10` | Attempts for each replay-safe domain-pool operation; also LightRAG's `POSTGRES_CONNECTION_RETRIES` |
| `connection_retry_backoff` / `_max` | `3` / `30` | Retry delay/cap seconds, for both pools |
| `pool_close_timeout` | `5` | Seconds LightRAG waits to close its pool before reconnecting after a transient error |

## Concurrency And Ingestion Limits

| Field | Default | Scope |
|---|---|---|
| `models.max_concurrency` | `16` | All provider requests in one process |
| `corpus.ingestion.pipeline.max_concurrency` | `16` | LLM requests one workspace's LightRAG pipeline makes at once, per role |
| `corpus.ingestion.chunk_token_size` | `2000` | LightRAG chunk size |
| `corpus.ingestion.image_margin` | `0.03` | White margin per side, as a fraction of each dimension (at most `0.5`), composited around an image source's parser copy so a full-bleed image keeps page context; the document keeps the bytes it was given, and `0` disables it |
| `corpus.ingestion.replace_default` | `false` | Default replacement policy |
| `corpus.ingestion.max_upload_bytes` | `104857600` | One ingest file |

Advanced stage defaults:

```yaml
corpus:
  ingestion:
    pipeline:
      max_parallel_insert: 3
      max_parallel_parse_native: 5
      max_parallel_parse_mineru: 2
      max_parallel_parse_docling: 2
      max_parallel_analyze: 5
      queue_size_parse: 20
      queue_size_analyze: 100
      queue_size_insert: 4
```

## Retrieval

```yaml
corpus:
  retrieval:
    top_k: 40
    chunk_top_k: 20
    direct_visual_top_k: 20
    timeout: 300
    bm25_enabled: true
```

`top_k` controls graph/entity breadth; `chunk_top_k` controls text/visual chunk
candidates. BM25 candidate breadth follows the chunk budget. `/answer` packs
evidence against the query model's remaining input capacity.

Advanced fields:

| Field | Default | Meaning |
|---|---|---|
| `timeout` | `300` | Claimed top-level Retrieval planning/search seconds; excludes queue residence and does not wrap Answer Runs |
| `bm25_k1` | `1.2` | BM25 term-frequency saturation |
| `bm25_b` | `0.75` | BM25 length normalization |
| `bm25_profiles` | built-in language set | pg_textsearch index signatures and language labels |
| `rrf_k` | `60` | Reciprocal-rank fusion constant |
| `metadata_filter_exact_vector_threshold` | `8192` | Exact vector scoring cutoff for a filtered candidate set |
| `max_entity_tokens` | `6000` | KG entity context ceiling |
| `max_relation_tokens` | `8000` | KG relation context ceiling |
| `max_total_tokens` | `40000` | Total LightRAG context ceiling |
| `kg_chunk_pick_method` | `VECTOR` | `VECTOR` or `WEIGHT` |
| `kg_entity_types` | `[]` | Empty uses LightRAG's general taxonomy; for stronger domain control set `corpus.extraction.entity_type_prompt_file` |

Enabling BM25 for existing data or changing profiles requires
[Workspace BM25 Rebuild](operations.md#workspace-bm25-rebuild). Algorithm and
budget semantics live in [Retrieval and Answer](retrieval-answer.md).

Workspace partition promotion is off unless `corpus.promotion.doc_threshold` or
`chunk_threshold` is set. Advanced worker defaults are `lease_seconds: 1800`,
`retry_backoff_seconds: 600`, and `claim_poll_seconds: 5.0`. Visual routes use
`corpus.visual_assets.thumb_max_px: 300` and `thumb_cache_size: 256`.

## RunRuntime Lanes And Retention

| Field | Default | Meaning |
|---|---|---|
| `runtime.query.worker_concurrency` | `16` | Query Lane Runs one process executes at once |
| `runtime.query.max_nonterminal_runs` | `30000` | Query Lane admission limit across the deployment |
| `runtime.corpus_mutation.worker_concurrency` | `2` | Corpus Mutation Runs one writer process executes at once |
| `runtime.corpus_mutation.max_nonterminal_runs` | `1000` | Corpus Mutation Lane admission limit across the deployment |
| `runtime.run_retention_days` | `365` | Days a terminal Answer Run is kept; superseded Profile Memory history is kept as long |

Every value is at least 1. What each lane carries and how its limit admits Runs
are in [RunRuntime](run-runtime.md#lanes); what retention removes, and
the fixed retention of other Runs, is in [RunRuntime](run-runtime.md#retention).

## Answer Generation And Attachments

```yaml
answer:
  generation:
    max_images: 12
    max_attachments: 6
    max_attachment_bytes: 104857600
    max_total_attachment_bytes: 134217728
    lineage_adoption: true
    image_max_bytes: 3000000
    image_max_total_bytes: 24000000
    image_max_px: 1536
    image_max_pixels: 40000000
    image_min_px: 1024
    image_quality: 89
    image_min_quality: 79
```

Attachments are run-scoped Resources whose full bytes never enter model context;
which uploads a Run admits is decided by type and content, not configuration
([Resource Reading](resource-reading.md#registration-and-acquisition)).
`query_images`, a separate retrieve-only path, takes at most three current
images. The final answer image count is clamped to the query model's discovered
capability. `lineage_adoption` lets a Run adopt a Resource an earlier Run on the
same Agent Session registered ([Earlier Runs](resource-reading.md#earlier-runs)).

## Research Agent

```yaml
answer:
  agent:
    execution_environment: trust   # disabled | trust
    workspace_root: null           # absolute path; null → ~/.dlightrag/agent_workspaces
    child_guidance_timeout_seconds: 300  # default ask_parent expiry; 1–86400
    session_notes:                 # durable Agent memory per Agent Session
      max_count: 64                # 1–1024 notes
      max_bytes: 262144            # 1024–16 MiB total
    skills_root: null              # absolute path; null → ~/.dlightrag/skills
    owner_skills_root: null        # absolute path; null → ~/.dlightrag/owner_skills
    disabled_builtin_skills: []    # packaged Skill names only
    fd_path: fd                    # PATH name or absolute path; fd 10.5.0+
    ripgrep_path: rg               # PATH name or absolute path; ripgrep 15.2.0+
    search_tool_cache_root: null   # absolute path; null → ~/.dlightrag/tools
    search_tool_auto_install: false  # fetch verified fd/ripgrep at runtime
    publication:
      max_artifacts: 20
      max_file_bytes: 31457280
      max_total_bytes: 104857600
      workspace_max_bytes: 1073741824  # at most 5 GiB
      preview_image_max_pixels: 16000000
      preview_image_max_edge: 4096
      original_image_max_pixels: 64000000
      original_image_max_edge: 8000
      active_html_max_bytes: 20971520
  conversations:
    active_html_preview_enabled: true
```

`session_notes` bounds the notes one Agent Session keeps, independently of
`publication.workspace_max_bytes`, which bounds one Run's workspace; a note that
does not fit is refused, never truncated or evicted
([Session Notes](retrieval-answer.md#session-notes)).

`trust` runs rooted tools as the service user. On a Linux kernel with Landlock
it confines every Agent process to its Agent Workspace; elsewhere processes run
unconfined, which `GET /health` and the Run trace report. What the boundary
admits is in [Security](security.md#answer-resources-and-execution). `disabled`
means no workspace root for any mode: no path tools, no artifacts, and no
Session notes. Reclamation follows a configured root rather than the mode, so a
deployment that turns execution off still deletes trees earlier Runs left.

An explicit workspace root must be absolute, must not overlap
`deployment.working_dir`, and must be the same shared RWX path on every worker.
Published artifacts fail whole when over budget; they are not truncated.
Interactive HTML is separately opt-in and isolated by the Web artifact boundary
([Security](security.md#answer-artifact-browser-boundary)).

Research reaches external tools only through its owner's Personal MCP
Connections and the deployment's [Agent Browser](#agent-browser);
`answer.agent.connections` holds the Connections' non-secret policy, whose
fields and limits are in
[Personal MCP Connections](personal-mcp-connections.md#streamable-http-security-and-limits).
No owner authorizes the Agent Browser: `read` uses it to render public pages, and the
`browser` tool drives one.

Research discovers Skills from packaged built-ins, an operator-global root, and
owner roots ([Architecture](architecture.md#agent-execution) gives the
precedence). `disabled_builtin_skills` hides named built-ins, not a same-named
global or owner Skill. The global root, `skills_root`, is operator-provisioned
and read-only for the answer agent. The bundled Compose stack keeps
`skills_root: null` and mounts the operator's
`${COMPOSE_GLOBAL_SKILLS_DIR:-$HOME/.dlightrag/skills}` read-only at that
default container path, refusing to create a missing host source as root; the
setup wizard prepares the directory, and manual operators create it before
`docker compose up`. Owners write their own Skills under `owner_skills_root`
only through the validated `publish_skill` and `delete_skill` tools, within a
20-skill / 20 MiB quota per owner. Every worker must see the same skill roots.

## Agent Browser

```yaml
answer:
  agent:
    browser:
      endpoints: []                   # DLIGHTRAG_ANSWER__AGENT__BROWSER__ENDPOINTS
      egress_proxy: null              # DLIGHTRAG_ANSWER__AGENT__BROWSER__EGRESS_PROXY
      chromium_sandbox: true
      lease_wait_seconds: 10          # 0–120
      connect_timeout_seconds: 15     # above 0, at most 120
      navigation_timeout_seconds: 30  # above 0, at most 300
      settle_timeout_seconds: 5       # 0–60
      action_timeout_seconds: 10      # above 0, at most 120
      snapshot_depth: 12              # 1–64
      idle_release_seconds: 30        # 0–600
      account_registration: true      # DLIGHTRAG_ANSWER__AGENT__BROWSER__ACCOUNT_REGISTRATION
```

The Agent Browser lets Research read a page as a browser renders it and drive one with
the `browser` tool, in a pool of Playwright containers the deployment runs
([ADR 0032](adr/0032-the-agent-browser.md); the pool's topology and boundary are in
[Security](security.md#agent-browser-boundary)). No endpoint means no Agent
Browser: `read` declares no `rendered` argument, the Extract chain has no browser step,
and Research has no `browser` tool.

- `endpoints` lists one Playwright run-server WebSocket URL (`ws://` or `wss://`) per
  pool container; each serves one Run at a time. They must be unique, hold no query,
  fragment, or userinfo, and be spelled identically in every process that runs Query
  workers, because the shared lease table is keyed by the URL.
- `egress_proxy` is the HTTP proxy every browser launch uses (`http://host:port`, no
  path). It is the pool's only way out, so `endpoints` require it: startup refuses
  endpoints without it, naming the field.
- Both are Compose Service names, so `docker-compose.yml` binds them
  ([ADR 0006](adr/0006-configuration-ownership-and-deployment-bindings.md)) and
  `config.yaml` leaves them unset. The bundled stack binds two members and the Squid
  proxy for `dlightrag-api`, `dlightrag-mcp`, and `dlightrag-reader`.
- `chromium_sandbox` is whether every browser launches inside Chromium's own process
  sandbox. Whether a pool host can start it is for the operator to state, so DlightRAG
  neither probes for it nor falls back: on a host that cannot, every launch fails and
  renders report the pool unreachable until the host is relaxed
  ([troubleshooting](operations.md#agent-browser-pool)) or this is `false`. `false` launches
  Chromium with `--no-sandbox`, leaving the container and its network as the only
  isolation ([Security](security.md#agent-browser-boundary)).
- `lease_wait_seconds` is how long a render, or the first `navigate` of an Agent Page,
  waits for a free browser before it reports the pool busy.
  `connect_timeout_seconds` bounds connecting to one browser.
- `navigation_timeout_seconds` is how long a page may take to load. It also bounds the
  `browser` tool's `wait` for text, `screenshot`, and `capture`, and the save of one
  downloaded file. `settle_timeout_seconds` is how long a loaded page may take to go quiet
  before it is read as it stands.
- `action_timeout_seconds` is how long one element action, one snapshot, or one `find` may
  take on a page the `browser` tool drives. `snapshot_depth` is how many levels of the
  page its accessibility snapshot shows; deeper elements keep their refs, and `find`
  locates them. A file a page downloads is bounded by `answer.generation.max_attachment_bytes`
  ([Answer Generation And Attachments](#answer-generation-and-attachments)).
- `idle_release_seconds`: a Run leases a browser at its first render or Agent Page and
  gives it back once it has gone this long with no Agent Page open and no render in
  flight, so a Run that rendered once does not hold a pool member for its whole duration.
  An Agent Page that stays open keeps the browser leased
  ([when](architecture.md#agent-browser)). The next render or page leases again, and `0`
  gives it back as soon as nothing is open. Settlement releases whatever is held either
  way.
- `account_registration` is the deployment's allowance for the Agent to register new
  accounts on third-party sites ([ADR 0034](adr/0034-agent-accounts-and-the-agent-mailbox.md);
  [what the actions do](retrieval-answer.md#agent-accounts-and-the-agent-mailbox)). `false`
  withdraws `register` alone: `login` and `inbox` still serve the accounts an owner has, and
  those stay in PostgreSQL and follow a key ring rotation. Each owner's own switch for new
  sign-ups in Settings can only turn registration off further than this. Without a key ring,
  register and login fail closed whatever this says.

The pool's size is the deployment's limit on Runs using a browser at the same moment
([sizing](operations.md#agent-browser-pool)). What `GET /health` says of the Agent
Browser is in [Interfaces](interfaces.md#health-and-errors).

## Agent Mailbox

```yaml
answer:
  agent:
    mailbox:
      endpoint: null        # DLIGHTRAG_ANSWER__AGENT__MAILBOX__ENDPOINT; null is AWS S3's own
      region: null          # DLIGHTRAG_ANSWER__AGENT__MAILBOX__REGION; null is the SDK's own resolution
      bucket: null          # DLIGHTRAG_ANSWER__AGENT__MAILBOX__BUCKET
      prefix: mail          # DLIGHTRAG_ANSWER__AGENT__MAILBOX__PREFIX
      alias_domain: null    # DLIGHTRAG_ANSWER__AGENT__MAILBOX__ALIAS_DOMAIN
```

The Agent Mailbox is optional ([ADR 0034](adr/0034-agent-accounts-and-the-agent-mailbox.md)). It
gives the Agent addresses of its own to register with and lets it read the mail that arrives
at them with `inbox`; without it the Agent may use a temporary-mail site through the
browser itself, and `register` records the address it typed. No bucket means no Agent
Mailbox, and an Agent Mailbox needs an [Agent Browser](#agent-browser), whose accounts it
serves.

- The two keys are secrets, so they belong in `.env` and nowhere in YAML
  ([ADR 0006](adr/0006-configuration-ownership-and-deployment-bindings.md)):
  `DLIGHTRAG_ANSWER__AGENT__MAILBOX__ACCESS_KEY_ID` and
  `DLIGHTRAG_ANSWER__AGENT__MAILBOX__SECRET_ACCESS_KEY`. They never render, and an Agent's
  own processes get no `DLIGHTRAG_*` variable. A deployment that runs the checked-in
  `config.yaml` keeps its endpoint, bucket, and alias domain in `.env` as well, as it keeps its
  access policy.
- Naming a `bucket` requires `alias_domain` and both keys, and without a bucket none of
  `endpoint`, `region`, `alias_domain`, or the keys may be set; startup refuses either mistake
  and names the setting. A blank variable in `.env` is an unset setting. `bucket` is a valid
  S3 bucket name, `prefix` is slash-separated segments of `A-Za-z0-9._-` or empty, and
  `alias_domain` is a lower-case domain with at least two labels.
- `region` is the bucket's region as its endpoint names it, such as `us-east-1` on AWS S3 or
  `auto` on Cloudflare R2. It has no default: unset, the SDK resolves the region as it does
  for any S3 client, from its usual environment and profile, and an unset `endpoint` is AWS
  S3's own.
- `alias_domain` is the domain the Agent's mailbox aliases are minted on, which the
  deployment's mail routing must deliver. A mailbox alias is the same for one owner and site
  in every Run.
- **The bucket's contract is its layout.** Each message is written whole, as one object, under
  `<prefix>/<envelope recipient, lower case>/`: the envelope recipient, because a `To:` header
  does not reliably name the address a message was delivered to, and lower case because
  DlightRAG mints lower-case mailbox aliases and a site may capitalize one. DlightRAG lists
  one mailbox alias's prefix and never scans the bucket, reads `LastModified` as when mail
  arrived, never an object's name or a header, and needs only list and get access. A listing
  reads at most the first 10,000 stored messages of a mailbox alias and says when it holds
  more, so a deployment keeps its retention short. How mail reaches the bucket, with an
  example, and how long it stays, are the deployment's
  ([Operations](operations.md#agent-mailbox)).

What `GET /health` says of it is in [Interfaces](interfaces.md#health-and-errors).

## Public Web Sources

```yaml
answer:
  web_sources:
    exa:
      api_key: null     # DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY
    tavily:
      api_key: null     # DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEY
    search_providers: null
    extract_providers: null
```

Exa and Tavily are optional provider adapters. A `null` provider order derives
an Exa-then-Tavily chain from configured keys; `[]` disables that operation.
Set each list explicitly to give Search and Extract independent failover order.
Every provider named in a list must have a key. The setup wizard can configure
Exa, Tavily, both with one shared order, or independent Search/Extract orders.

`extract_providers` may also name `browser`, the [Agent Browser](#agent-browser),
which holds no key. A configured Agent Browser always joins the automatic Extract
chain after the hosted providers, whether the order is derived or explicit, unless
the list names `browser` elsewhere, which only positions it (`[browser, exa]` renders
before it asks Exa). The wizard's explicit lists therefore need no `browser` entry,
and `[]` turns off the hosted providers while a configured browser still ends the
chain. Naming `browser` while `answer.agent.browser` has no endpoints is a startup
error, as naming a provider without its key is.

How Research uses the chains is in [Web Search](retrieval-answer.md#web-search):
`search_web` takes result count, domains, date range, and
`fast`/`balanced`/`deep` effort, and `read` uses the Extract chain only when a
direct anonymous fetch fails or yields no usable text; a browser step renders the
page ([Rendered reads](resource-reading.md#rendered-reads)).

## Citations And Highlights

Citation validation is always enabled. Semantic highlights default on for Web
Inspector Sources and off for other answer callers unless requested.

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

Set `enabled: false` to disable highlight extraction on every interface. Public
citation shapes are in [Interfaces](interfaces.md#citations).

## Access And Interfaces

Code defaults bind listeners to loopback with no auth; MCP defaults to `stdio`.
The checked-in Compose stack selects `streamable-http` on port 8101.

| Field | Default | Meaning |
|---|---|---|
| `interfaces.api.host`, `.port` | `127.0.0.1`, `8100` | REST/Web bind |
| `interfaces.mcp.transport` | `stdio` | `stdio` or `streamable-http` |
| `interfaces.mcp.host`, `.port` | `127.0.0.1`, `8101` | HTTP MCP bind |
| `interfaces.mcp.allowed_hosts` | local hosts | Host-header allowlist |
| `interfaces.mcp.allowed_origins` | local origins | Origin allowlist |
| `interfaces.mcp.resource_server_url` | unset | Public RFC 9728 resource URL |
| `interfaces.max_upload_size_mb` | `512` | One multi-file corpus upload request |
| `access.auth_mode` | `none` | `none`, `simple`, or `jwt` |
| `access.api_token` | unset | The deployment owner's bearer token for `simple` |
| `access.allow_insecure_no_auth` | `false` | Permit non-loopback no-auth bind |
| `access.jwt_verification_key` | unset | Static key: HMAC secret or public-key PEM |
| `access.jwt_issuer`, `.jwt_audience` | unset | Expected claims; the issuer's discovery names its keys |
| `access.jwt_jwks_url` | from discovery | Key set of an issuer without OpenID discovery |
| `access.jwt_algorithm` | the key's | Pins one algorithm; a static key is `HS256` |
| `access.cors_allow_origins` | `[]` | Cross-origin browser clients; the Web is same-origin |
| `access.web_identity` | disabled | Edge identity; its issuer and audience default to the API's, and its keys always come from published keys |
| `access.control.rules` | `[]` | Claim/workspace/action mappings; any rule puts every action under rules |

Do not expose listeners without auth and ingress protection. Security semantics
are in [Security](security.md); payload contracts are in
[Interfaces](interfaces.md).

The CLI reads `DLIGHTRAG_API_URL` (default `http://localhost:8100`), optional
`DLIGHTRAG_API_TOKEN`, and `DLIGHTRAG_CLIENT_TIMEOUT` (default 120 seconds); the
evaluation script reads the first two ([Evaluation](evaluation.md)).

## Observability

Tracing activates only when both Langfuse keys are set; keep them in `.env`.

| Field | Default | Meaning |
|---|---|---|
| `log_level` | `info` | Application logging level |
| `langfuse_public_key`, `langfuse_secret_key` | unset | Both required |
| `langfuse_host` | `https://cloud.langfuse.com` | Trace destination |
| `langfuse_trace_sensitive_data` | `true` | Suppress raw content — observation inputs, outputs, and error text — when false |
| `langfuse_export_external_spans` | `false` | Export third-party OTel spans |
| `langfuse_environment` | unset | Deployment label such as `local`, `staging`, or `production`, so deployments' traces stay apart: lowercase letters, digits, `_`, and `-`, at most 40 characters, starting with a letter or digit but not with `langfuse` |
| `langfuse_release` | running package version | Release label |
| `langfuse_sample_rate` | `1.0` | Export fraction |
| `langfuse_timeout` | SDK default | Export timeout |
| `langfuse_flush_at` | SDK default | Buffered event count |
| `langfuse_flush_interval` | SDK default | Flush cadence |

Memory traces include counts and character totals, never record bodies. Span
names, observation types, attribution, and redaction follow the
[observability contract](observability.md). Run the bundled stack with the
[Langfuse runbook](operations.md#local-langfuse-observability).

## Vector Index Fields

These LightRAG storage fields are supported but normally left at code defaults:

```yaml
storage:
  lightrag:
    vector_index_type: HNSW_HALFVEC   # HNSW, HNSW_HALFVEC, IVFFLAT, or VCHORDRQ
    hnsw_m: 32
    hnsw_ef_construction: 256
    hnsw_ef_search: 256
    vector_db_kwargs: {}
```

`vector_db_kwargs` holds at most 16 scalar adapter options in 4096 encoded
bytes; secrets and endpoint URIs are rejected. Milvus accepts only LightRAG's
documented index, metric, HNSW, SQ, and IVF keys and
`cosine_better_than_threshold`.
