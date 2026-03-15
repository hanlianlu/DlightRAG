# Provider Management Optimization Design

## Problem

The current provider architecture has grown unwieldy:

- **11 LLM providers** each with dedicated API key/base_url config fields (~20 fields in `config.py`)
- **828-line `llm.py`** with 11-way dispatch branches and ~400 lines of provider-specific vision code
  (`_build_openai_vision_func`, `_build_anthropic_vision_func`, `_build_google_vision_func`,
  `_convert_openai_to_anthropic_messages`)
- **Tight coupling** to LightRAG's native `*_complete_if_cache` functions — any new provider
  requires upstream support
- **Outdated model role split** — separate `chat_model` and `vision_model` no longer reflects
  modern multimodal LLMs where every model handles both text and images

## Industry Research

### OpenViking (Volcengine)

Splits providers into 3 categories: `volcengine` (first-party), `openai` (OpenAI-compatible
endpoints), `litellm` (everything else). Each model role (vlm, embedding, rerank) is a
self-contained config block with `provider/model/api_key/api_base`. JSON config file as primary
source.

### LiteLLM

Unified `completion(model="provider/model", messages=[...])` interface across 130+ providers.
Automatic message format translation, vision support, JSON mode, streaming. Provider prefix
convention (`anthropic/claude-3`, `azure/gpt-4o`, `ollama/qwen3`).

### Industry Consensus

Modern LLM APIs are **messages-first** — OpenAI, Anthropic, Google, LangChain ChatModel all use
`messages: list[dict]` as the primary interface. The `prompt: str` parameter is a legacy pattern
from the GPT-3 completion API era.

## Design

### Principle: Two-Track Provider, Messages-First Interface

- **openai** — Direct `AsyncOpenAI` SDK for truly OpenAI-compatible endpoints (OpenAI, Azure
  OpenAI, Qwen, MiniMax, and user-provided compatible endpoints)
- **litellm** — `litellm.acompletion()` / `litellm.aembedding()` for everything else (Anthropic,
  Gemini, Ollama, Xinference, OpenRouter, Voyage, and all future providers)
- Core callable interface is **messages-first**; LightRAG gets a thin adapter

### Architecture

```
DlightRAG (messages-first callables)
├─ provider=openai  → AsyncOpenAI SDK
│   ├─ chat.completions.create(messages=[...])
│   └─ embeddings.create(input=[...])
├─ provider=litellm → LiteLLM
│   ├─ litellm.acompletion(messages=[...])
│   └─ litellm.aembedding(input=[...])
└─ _adapt_for_lightrag(func)
    └─ (prompt, system_prompt=...) → messages → func(messages=...)

Rerank (independent system)
├─ backend=llm      → uses ingest model callable
├─ backend=cohere   → dedicated API client
├─ backend=jina     → dedicated API client
├─ backend=aliyun   → dedicated API client
└─ backend=azure_cohere → dedicated API client
```

### Component 1: Nested Configuration Schema

**Location**: `src/dlightrag/config.py`

Replace ~20 provider-specific fields with a reusable `ModelConfig` block.

```python
class ModelConfig(BaseModel):
    """Reusable model configuration block."""
    provider: Literal["openai", "litellm"] = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None   # None = provider default
    timeout: float = 120.0
    max_retries: int = 3
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

class EmbeddingConfig(ModelConfig):
    """Embedding-specific configuration."""
    model: str = "text-embedding-3-large"
    dim: int = 1024
    max_token_size: int = 8192

class RerankConfig(BaseModel):
    """Reranking configuration (independent provider system)."""
    enabled: bool = True
    backend: Literal["llm", "cohere", "jina", "aliyun", "azure_cohere"] = "llm"
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    score_threshold: float = 0.5

class DlightragConfig(BaseSettings):
    chat: ModelConfig = ModelConfig(model="gpt-4.1-mini", temperature=0.5)
    ingest: ModelConfig | None = None      # fallback → chat (inherits chat's temperature)
    embedding: EmbeddingConfig             # required, no default
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    # ... storage, workspace, etc. (unchanged)

    model_config = SettingsConfigDict(
        env_prefix="DLIGHTRAG_",
        env_nested_delimiter="__",
    )
```

**Config file format** (YAML):

```yaml
chat:
  provider: openai
  model: gpt-4.1-mini
  api_key: sk-xxx
  model_kwargs: {}

ingest:                            # optional, fallback to chat
  provider: litellm
  model: ollama/qwen3:8b
  base_url: http://localhost:11434
  model_kwargs: {}

embedding:                         # required
  provider: openai
  model: text-embedding-3-large
  api_key: sk-xxx
  dim: 1024

rerank:
  enabled: true
  backend: llm                     # uses ingest model (cheaper)
  score_threshold: 0.5
```

**Env var override**:

```bash
DLIGHTRAG_CHAT__PROVIDER=openai
DLIGHTRAG_CHAT__MODEL=gpt-4.1-mini
DLIGHTRAG_CHAT__API_KEY=sk-xxx
DLIGHTRAG_EMBEDDING__MODEL=text-embedding-3-large
DLIGHTRAG_EMBEDDING__API_KEY=sk-xxx
DLIGHTRAG_EMBEDDING__DIM=1024
```

**Fallback chains**:

- `ingest` not configured → use `chat`
- `ingest.api_key` not configured → use `chat.api_key`
- `rerank` backend=llm → use ingest model callable (cheaper)
- `embedding` — **no fallback**, must be explicitly configured

**Legacy field detection**:

```python
@model_validator(mode="before")
@classmethod
def _warn_legacy_fields(cls, values):
    legacy = {
        "openai_api_key", "azure_openai_api_key", "anthropic_api_key",
        "qwen_api_key", "minimax_api_key", "ollama_api_key",
        "xinference_api_key", "openrouter_api_key", "voyage_api_key",
        "llm_provider", "chat_model", "vision_model", "vision_provider",
        "embedding_provider",
    }
    found = [k for k in values if k in legacy]
    if found:
        raise ValueError(
            f"Legacy config fields detected: {found}. "
            "Please migrate to the new nested format. "
            "See docs/migration-guide.md"
        )
    return values
```

### Component 2: Messages-First Model Callables

**Location**: `src/dlightrag/models/llm.py`

Two completion functions with identical signatures, dispatched by `provider`:

```python
async def _openai_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str | None = None,
    stream: bool = False,
    response_format: Any = None,
    timeout: float = 120.0,
    max_retries: int = 3,
    _client: AsyncOpenAI | None = None,  # injected by factory for connection reuse
    **kwargs,
) -> str | AsyncIterator[str]:
    """OpenAI SDK path — for OpenAI, Azure, Qwen, MiniMax, etc."""
    client = _client or AsyncOpenAI(
        api_key=api_key, base_url=base_url,
        timeout=timeout, max_retries=max_retries,
    )
    call_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if response_format is not None:
        call_kwargs["response_format"] = response_format
    call_kwargs.update(kwargs)

    if stream:
        response = await client.chat.completions.create(**call_kwargs, stream=True)
        # return async iterator of content deltas
        ...
    else:
        response = await client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content


async def _litellm_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str | None = None,
    stream: bool = False,
    response_format: Any = None,
    timeout: float = 120.0,
    **kwargs,
) -> str | AsyncIterator[str]:
    """LiteLLM path — for Anthropic, Gemini, Ollama, OpenRouter, etc."""
    import litellm
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "api_key": api_key,
        "timeout": timeout,
    }
    if base_url:
        call_kwargs["api_base"] = base_url
    if response_format is not None:
        call_kwargs["response_format"] = response_format
    call_kwargs.update(kwargs)

    if stream:
        response = await litellm.acompletion(**call_kwargs, stream=True)
        # return async iterator of content deltas
        ...
    else:
        response = await litellm.acompletion(**call_kwargs)
        return response.choices[0].message.content
```

Two embedding functions:

```python
async def _openai_embedding(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    _client: AsyncOpenAI | None = None,  # injected by factory for connection reuse
    **kwargs,
) -> list[list[float]]:
    """OpenAI SDK embedding."""
    client = _client or AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.embeddings.create(model=model, input=texts, **kwargs)
    return [d.embedding for d in response.data]


async def _litellm_embedding(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    **kwargs,
) -> list[list[float]]:
    """LiteLLM embedding."""
    import litellm
    call_kwargs = {"model": model, "input": texts, "api_key": api_key}
    if base_url:
        call_kwargs["api_base"] = base_url
    call_kwargs.update(kwargs)
    response = await litellm.aembedding(**call_kwargs)
    return [d["embedding"] for d in response.data]
```

### Component 3: LightRAG Adapter

**Location**: `src/dlightrag/models/llm.py`

Thin adapter to bridge messages-first callable to LightRAG's expected
`(prompt, system_prompt=..., **kwargs)` signature:

```python
# LightRAG-internal kwargs that must not leak to API calls
_LIGHTRAG_STRIP_KWARGS = {
    "keyword_extraction", "token_tracker",
    "use_azure", "azure_deployment", "api_version",
}

def _adapt_for_lightrag(completion_func: Callable) -> Callable:
    """Wrap messages-first callable for LightRAG compatibility.

    - Converts (prompt, system_prompt=...) to messages array
    - Strips LightRAG-internal kwargs that would cause API errors
    - Handles ``hashing_kv`` for LightRAG's response caching: checks
      cache before calling, stores result after. This preserves
      LightRAG's ingestion caching behavior (avoids re-processing
      the same content during entity extraction, etc.)
    """
    async def wrapper(prompt: str, *, system_prompt: str | None = None, **kwargs):
        # Extract hashing_kv for cache handling
        hashing_kv = kwargs.pop("hashing_kv", None)
        # Strip remaining LightRAG-internal kwargs
        clean_kwargs = {k: v for k, v in kwargs.items()
                        if k not in _LIGHTRAG_STRIP_KWARGS}

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Check LightRAG cache
        if hashing_kv is not None:
            args_hash = compute_args_hash(
                model="", prompt=prompt, system_prompt=system_prompt, **clean_kwargs
            )
            cached = await hashing_kv.get_by_id(args_hash)
            if cached:
                return cached["return"]

        result = await completion_func(messages=messages, **clean_kwargs)

        # Store in LightRAG cache
        if hashing_kv is not None and isinstance(result, str):
            await hashing_kv.upsert(
                {args_hash: {"return": result, "model": "", "prompt": prompt}}
            )

        return result
    return wrapper
```

### Component 4: Factory Functions

**Location**: `src/dlightrag/models/llm.py`

Simplified factory functions — 2 branches instead of 11:

```python
def _make_completion_func(cfg: ModelConfig, fallback_api_key: str | None = None) -> Callable:
    """Build a messages-first completion callable from config.

    For openai provider, creates AsyncOpenAI client once (connection pooling).
    """
    api_key = cfg.api_key or fallback_api_key
    if cfg.provider == "openai":
        client = AsyncOpenAI(
            api_key=api_key, base_url=cfg.base_url,
            timeout=cfg.timeout, max_retries=cfg.max_retries,
        )
        extra = {**cfg.model_kwargs}
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        return partial(_openai_completion,
                       model=cfg.model, api_key=api_key,
                       base_url=cfg.base_url, _client=client,
                       **extra)
    else:  # litellm
        extra = {**cfg.model_kwargs, "num_retries": cfg.max_retries}
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        return partial(_litellm_completion,
                       model=cfg.model, api_key=api_key,
                       base_url=cfg.base_url, timeout=cfg.timeout,
                       **extra)


def get_chat_model_func(config: DlightragConfig) -> Callable:
    """Messages-first chat callable (for DlightRAG direct use)."""
    return _make_completion_func(config.chat)


def get_chat_model_func_for_lightrag(config: DlightragConfig) -> Callable:
    """LightRAG-compatible chat callable with adapter."""
    return _adapt_for_lightrag(get_chat_model_func(config))


def get_ingest_model_func(config: DlightragConfig) -> Callable:
    """Messages-first ingest callable, fallback to chat."""
    cfg = config.ingest or config.chat
    fallback_key = config.chat.api_key
    return _make_completion_func(cfg, fallback_api_key=fallback_key)


def get_ingest_model_func_for_lightrag(config: DlightragConfig) -> Callable:
    """LightRAG-compatible ingest callable with adapter."""
    return _adapt_for_lightrag(get_ingest_model_func(config))


def get_embedding_func(config: DlightragConfig) -> EmbeddingFunc:
    """Build LightRAG EmbeddingFunc from config."""
    cfg = config.embedding
    if cfg.provider == "openai":
        client = AsyncOpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url,
            timeout=cfg.timeout, max_retries=cfg.max_retries,
        )
        func = partial(_openai_embedding,
                       model=cfg.model, api_key=cfg.api_key,
                       base_url=cfg.base_url, _client=client)
    else:
        func = partial(_litellm_embedding,
                       model=cfg.model, api_key=cfg.api_key,
                       base_url=cfg.base_url)
    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=func,
        model_name=cfg.model,
    )


def get_rerank_func(config: DlightragConfig) -> Callable:
    """Build rerank callable. LLM backend uses ingest model."""
    # ... rerank logic mostly unchanged, but llm backend now calls
    # get_ingest_model_func(config) instead of get_llm_model_func()
```

### Component 5: Consumer Updates

**`src/dlightrag/core/service.py`** — Use new factory functions:

```python
# Before:
llm_func = get_llm_model_func(config)
vision_func = get_vision_model_func(config)

# After:
chat_func = get_chat_model_func(config)                      # messages-first (for AnswerEngine)
chat_func_lr = get_chat_model_func_for_lightrag(config)      # adapted (for LightRAG)
ingest_func_lr = get_ingest_model_func_for_lightrag(config)  # adapted (for LightRAG ingestion)
```

**`src/dlightrag/core/answer.py`** — AnswerEngine uses messages-first directly:

```python
# Before (prompt-based with separate vision path):
if has_images:
    messages = self._build_vlm_messages(system_prompt, user_prompt, contexts["chunks"])
    raw = await model_func(user_prompt, messages=messages, response_format=...)
else:
    raw = await model_func(user_prompt, system_prompt=system_prompt, response_format=...)

# After (messages-first, unified):
messages = self._build_messages(system_prompt, user_prompt, contexts)
raw = await self.model_func(messages=messages, response_format=...)
```

No separate VLM code path in AnswerEngine — multimodal content (images) is just part of
the messages array.

**`src/dlightrag/core/servicemanager.py`** — Uses `get_llm_model_func` and
`get_vision_model_func` for AnswerEngine and query rewriting:

```python
# Before:
llm_func = get_llm_model_func(config)
vision_func = get_vision_model_func(config)

# After:
chat_func = get_chat_model_func(config)  # messages-first for AnswerEngine
```

**Non-AnswerEngine vision consumers** — Several components call the vision callable
with the current signature `(prompt, *, image_data=..., messages=..., system_prompt=...)`:
`VlmOcrParser`, `UnifiedRepresentEngine`, `retriever.py`, `extractor.py`,
`multimodal_query.py`, RAGAnything itself.

These consumers already pass `messages=` in OpenAI format (content arrays with
`image_url` blocks). With the new messages-first callable, these callers simplify:
they build their messages array and call `func(messages=messages)` directly instead
of going through the intermediate `(prompt, image_data=...)` signature. The
`_adapt_for_lightrag` wrapper is only needed for LightRAG's `llm_model_func`, not
for these DlightRAG-native consumers.

**`supports_structured` handling** — The current code uses
`getattr(model_func, "supports_structured", False)` to decide between JSON and
freetext prompts. In the new architecture:

- **openai** provider: Always supports structured output (`response_format` works
  natively with AsyncOpenAI)
- **litellm** provider: LiteLLM normalizes `response_format` across providers
  (translates to provider-specific params automatically)

Therefore `supports_structured` is always `True` for both tracks. The attribute
is no longer needed — all callables support `response_format`. The `structured`
parameter in `get_answer_system_prompt(structured=...)` should default to `True`.
The streaming path continues to use `structured=False` as designed in the streaming
robustness work (streaming = freetext prompt, post-stream reference extraction).

**LLM-based rerank JSON mode** — The current `_json_kwargs_for_provider()` function
dispatches provider-specific JSON mode kwargs. In the new architecture, both tracks
support `response_format={"type": "json_object"}` natively (OpenAI SDK directly,
LiteLLM translates per provider). The per-provider dispatch function is deleted.

### Component 6: Config File Support

**Location**: `src/dlightrag/config.py`

Use pydantic-settings `settings_customise_sources` to support YAML config file +
env var override:

```python
@classmethod
def settings_customise_sources(cls, settings_cls, init_settings,
                                env_settings, dotenv_settings,
                                file_secret_settings):
    return (
        init_settings,          # highest priority: constructor args
        env_settings,           # env vars override config file
        dotenv_settings,        # .env file
        YamlConfigSettingsSource(settings_cls, yaml_file="dlightrag.yaml"),
    )
```

Priority: constructor args > env vars > .env > dlightrag.yaml

### Component 7: Migration Artifacts

**`.env.example`** — Rewrite to new nested format:

```bash
# === Chat Model (retrieval + answer generation) ===
DLIGHTRAG_CHAT__PROVIDER=openai          # openai | litellm
DLIGHTRAG_CHAT__MODEL=gpt-4.1-mini
DLIGHTRAG_CHAT__API_KEY=sk-xxx
# DLIGHTRAG_CHAT__BASE_URL=              # optional custom endpoint
# DLIGHTRAG_CHAT__MODEL_KWARGS={}        # extra params (JSON)

# === Ingest Model (optional, fallback to chat) ===
# DLIGHTRAG_INGEST__PROVIDER=litellm
# DLIGHTRAG_INGEST__MODEL=ollama/qwen3:8b
# DLIGHTRAG_INGEST__BASE_URL=http://localhost:11434

# === Embedding (required) ===
DLIGHTRAG_EMBEDDING__PROVIDER=openai
DLIGHTRAG_EMBEDDING__MODEL=text-embedding-3-large
DLIGHTRAG_EMBEDDING__API_KEY=sk-xxx
DLIGHTRAG_EMBEDDING__DIM=1024

# === Reranking ===
DLIGHTRAG_RERANK__ENABLED=true
DLIGHTRAG_RERANK__BACKEND=llm
DLIGHTRAG_RERANK__SCORE_THRESHOLD=0.5
```

**`.env`** — Migrate actual credentials to new format (already in .gitignore/.dockerignore).

## Files Changed

| File | Change |
|------|--------|
| `src/dlightrag/config.py` | Rewrite: `ModelConfig` + `EmbeddingConfig` + `RerankConfig` nested structure, delete ~20 provider-specific fields, add legacy field detection, add YAML config source |
| `src/dlightrag/models/llm.py` | Rewrite: `_openai_completion` + `_litellm_completion` + `_openai_embedding` + `_litellm_embedding` + `_adapt_for_lightrag` + simplified factories. 828→~200 lines |
| `src/dlightrag/models/__init__.py` | Update public API exports (remove `get_vision_model_func`, rename functions) |
| `src/dlightrag/core/service.py` | Adapt to new factory names, messages-first interface, `config.embedding_model` → `config.embedding.model`, `config.embedding_dim` → `config.embedding.dim` |
| `src/dlightrag/core/servicemanager.py` | Update `_get_answer_engine()`, `get_llm_func()`, and `_filter_compatible()` — all `config.embedding_model` → `config.embedding.model` |
| `src/dlightrag/core/answer.py` | AnswerEngine uses messages-first callable directly, remove separate VLM code path, `supports_structured` always True |
| `src/dlightrag/unifiedrepresent/embedder.py` | Adapt to new embedding interface |
| `src/dlightrag/web/routes.py` | Update `get_llm_model_func()` → `get_chat_model_func()` |
| `src/dlightrag/captionrag/vlm_parser.py` | Adapt VlmOcrParser to messages-first callable |
| `src/dlightrag/unifiedrepresent/engine.py` | Adapt UnifiedRepresentEngine to messages-first callable |
| `.env.example` | Rewrite to nested format |
| `.env` | Migrate credentials to nested format (not in git) |
| `tests/unit/test_config.py` | Rewrite for new nested config structure |
| `tests/unit/test_llm_providers.py` | Rewrite for 2-branch dispatch |
| `pyproject.toml` | Add `litellm` and `pyyaml` as required dependencies |

## Code Removed

- `_build_openai_vision_func()` (~120 lines)
- `_build_anthropic_vision_func()` (~110 lines)
- `_build_google_vision_func()` (~100 lines)
- `_convert_openai_to_anthropic_messages()` (~40 lines)
- `_json_kwargs_for_provider()` — replaced by universal `response_format` support
- `get_vision_model_func()` — no longer needed, chat/ingest callables are multimodal
- All `lightrag.llm.*` completion/embedding imports
- `_OPENAI_COMPATIBLE_PROVIDERS` set
- `LLMProvider` 11-value Literal type
- `supports_structured` function attribute pattern
- ~20 provider-specific `*_api_key` / `*_base_url` config fields

## Code Preserved

- Rerank `cohere/jina/aliyun/azure_cohere` backend logic (not LLM completion APIs)
- `MultimodalEmbedProvider` strategy pattern in `embedder.py` (Voyage etc.)
- Citation system, AnswerStream, and all non-provider code

## Non-Goals

- Changing rerank API backends (cohere/jina/aliyun) — these are specialized REST APIs,
  not covered by the openai/litellm split
- Modifying LightRAG upstream code
- Supporting config hot-reload (restart required for config changes)
- Provider-level load balancing or failover (LiteLLM handles this internally for
  litellm-routed providers)
