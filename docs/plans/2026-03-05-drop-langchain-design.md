# Drop LangChain: Use LightRAG Native LLM/Embedding Functions

**Date:** 2026-03-05
**Status:** Draft
**Scope:** `src/dlightrag/models/llm.py`, `src/dlightrag/service.py`, `pyproject.toml`

## Problem

DlightRAG uses LangChain as an intermediary layer between its configuration and
LightRAG/RAGAnything. This adds 4 heavy dependencies (`langchain-core`,
`langchain-openai`, `langchain-anthropic`, `langchain-google-genai`) that
duplicate functionality LightRAG already provides natively. Even with LangChain,
each provider still needs its own package.

## Decision

Replace LangChain with LightRAG's native LLM/embedding functions. Keep vision
(already uses native SDKs) and non-LLM rerank backends (cohere/jina/aliyun)
unchanged.

## Architecture Overview

### Current (LangChain)

```
DlightragConfig
  -> LangChain ChatOpenAI / ChatAnthropic / ChatGoogleGenerativeAI
    -> Underlying SDK (openai, anthropic, google-genai)
      -> LightRAG(llm_model_func=our_langchain_wrapper)
```

### New (LightRAG Native)

```
DlightragConfig
  -> LightRAG native functions (openai_complete, anthropic_complete, ...)
    -> Underlying SDK (already dependencies of lightrag-hku)
      -> LightRAG(llm_model_func=native_function)  # zero wrapper
```

## Verified Facts

All claims below were verified against LightRAG source code and provider SDK
documentation via context7.

### LightRAG's Internal Binding Mechanism

LightRAG handles model and kwargs binding internally (lightrag.py:664-674):

```python
self.llm_model_func = priority_limit_async_func_call(...)(
    partial(
        self.llm_model_func,
        hashing_kv=hashing_kv,
        **self.llm_model_kwargs,      # api_key, base_url, etc.
    )
)
```

Each provider's public function reads model from `hashing_kv.global_config`:

```python
# openai.py:629
model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
return await openai_complete_if_cache(model_name, prompt, ...)
```

**Consequence:** We pass the native function + `llm_model_name` +
`llm_model_kwargs` to LightRAG. No `partial()`, no wrapper needed for LLM.

### RAGAnything Transparency

RAGAnything passes `lightrag_kwargs` directly to `LightRAG(**lightrag_params)`
(raganything.py:318-338). Fields like `llm_model_name` and `llm_model_kwargs`
flow through untouched.

### Provider Function Mapping (Public Functions)

| DlightRAG provider | LightRAG function | Import path |
|---|---|---|
| `openai` | `openai_complete` | `lightrag.llm.openai` |
| `azure_openai` | `azure_openai_complete` | `lightrag.llm.azure_openai` |
| `anthropic` | `anthropic_complete` | `lightrag.llm.anthropic` |
| `google_gemini` | `gemini_model_complete` | `lightrag.llm.gemini` |
| `ollama` | `ollama_model_complete` | `lightrag.llm.ollama` |
| `qwen`, `minimax`, `openrouter`, `xinference` | `openai_complete` | `lightrag.llm.openai` (OpenAI-compatible) |

All public functions follow the same pattern: no `model` parameter, reads model
from `hashing_kv.global_config["llm_model_name"]`.

### Embedding Functions

Embedding functions are different — `EmbeddingFunc` has no `model_kwargs`
mechanism. The wrapped function is called as `func(texts)` directly. Therefore
we still need `partial()` to bind model/api_key/base_url.

| Function | Has `@wrap_embedding_func_with_attrs`? | How to use |
|---|---|---|
| `openai_embed` | Yes (returns EmbeddingFunc) | `partial(openai_embed.func, model=..., api_key=...)` |
| `gemini_embed` | Yes | `partial(gemini_embed.func, model=..., api_key=...)` |
| `ollama_embed` | Yes | `partial(ollama_embed.func, embed_model=..., host=...)` |
| `anthropic_embed` | **No** (raw function) | `partial(anthropic_embed, model=..., api_key=...)` |

All return `np.ndarray`. Wrap with `EmbeddingFunc(embedding_dim=..., func=bound_fn)`.

### Rerank Structured Output — Provider-Aware JSON Mode

LLM-based rerank needs structured JSON output. Provider JSON mode mechanisms
differ and were verified via context7 docs and LightRAG source kwargs analysis:

| Provider | JSON mode mechanism | How to pass | Verified source |
|---|---|---|---|
| OpenAI (+ azure, qwen, minimax, openrouter, xinference) | `response_format` | Via kwargs (transparent passthrough) | openai-python docs, LightRAG openai.py:329 |
| Ollama | `format` param | Via kwargs (transparent passthrough) | ollama docs, LightRAG ollama.py:162 |
| Gemini | `response_mime_type` in GenerateContentConfig | Via `generation_config` named param | python-genai docs, LightRAG gemini.py:134 |
| Anthropic | No native JSON mode | Prompt-only | anthropic-sdk-python docs, LightRAG anthropic.py:86 |

**Strategy: 3-layer defense**
1. Native JSON mode where available (via provider-specific kwargs)
2. Prompt always instructs JSON format (universal fallback)
3. Pydantic `model_validate_json()` as safety net

**kwargs passthrough verification:**
- OpenAI: `**kwargs` → `.parse()` / `.create()` — transparent
- Ollama: `**kwargs` → `ollama_client.chat()` — transparent
- Gemini: `**_: Any` — **discards all kwargs**, must use `generation_config=` param
- Anthropic: `**kwargs` → `messages.create()` — transparent but no JSON mode API

## Detailed Changes

### 1. `src/dlightrag/models/llm.py` — LLM Functions

**Delete:** `_build_chat_model`, `_build_llm_model_func`, `get_llm_model_func`,
`get_ingestion_llm_model_func`

**Add:** `_get_native_llm_func(provider)` — returns the LightRAG native function

```python
def _get_native_llm_func(provider: str) -> Callable:
    if provider in ("openai", "qwen", "minimax", "openrouter", "xinference"):
        from lightrag.llm.openai import openai_complete
        return openai_complete
    if provider == "azure_openai":
        from lightrag.llm.azure_openai import azure_openai_complete
        return azure_openai_complete
    if provider == "anthropic":
        from lightrag.llm.anthropic import anthropic_complete
        return anthropic_complete
    if provider == "google_gemini":
        from lightrag.llm.gemini import gemini_model_complete
        return gemini_model_complete
    if provider == "ollama":
        from lightrag.llm.ollama import ollama_model_complete
        return ollama_model_complete
    raise ValueError(f"Unsupported provider: {provider}")
```

**Add:** `_get_llm_model_kwargs(config, provider)` — returns kwargs dict for
the provider

```python
def _get_llm_model_kwargs(config: DlightragConfig, provider: str) -> dict:
    kwargs = {}
    api_key = config._get_provider_api_key(provider)
    if api_key:
        kwargs["api_key"] = api_key
    base_url = config._get_url(f"{provider}_base_url")
    if base_url:
        kwargs["base_url"] = base_url
    if provider == "azure_openai":
        kwargs["api_version"] = config.azure_api_version
    if provider == "ollama":
        kwargs["host"] = config.ollama_base_url.rstrip("/v1")
    return kwargs
```

### 2. `src/dlightrag/models/llm.py` — Embedding

**Replace** `get_embedding_func` body. Remove LangChain `OpenAIEmbeddings` /
`GoogleGenerativeAIEmbeddings`.

```python
def get_embedding_func(config) -> EmbeddingFunc:
    provider = config.effective_embedding_provider
    api_key = config._get_provider_api_key(provider)
    base_url = config._get_url(f"{provider}_base_url")

    if provider == "google_gemini":
        from lightrag.llm.gemini import gemini_embed
        raw_fn = partial(gemini_embed.func, model=config.embedding_model, api_key=api_key)
    elif provider == "anthropic":
        from lightrag.llm.anthropic import anthropic_embed
        raw_fn = partial(anthropic_embed, model=config.embedding_model, api_key=api_key)
    elif provider == "ollama":
        from lightrag.llm.ollama import ollama_embed
        host = config.ollama_base_url.rstrip("/v1")
        raw_fn = partial(ollama_embed.func, embed_model=config.embedding_model, host=host)
    else:
        from lightrag.llm.openai import openai_embed
        raw_fn = partial(openai_embed.func,
            model=config.embedding_model, api_key=api_key, base_url=base_url)

    return EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=8192,
        func=raw_fn,
        model_name=config.embedding_model,
    )
```

### 3. `src/dlightrag/models/llm.py` — Rerank (LLM-based)

**Replace** `_build_llm_rerank_func`. Remove LangChain
`with_structured_output`.

For the rerank LLM call, use `_if_cache` variant directly (rerank is not part
of LightRAG's pipeline, so no `hashing_kv` available):

```python
def _build_llm_rerank_func(config):
    provider = config.effective_rerank_llm_provider
    model = config.effective_rerank_model
    api_key = config._get_provider_api_key(provider)
    base_url = config._get_url(f"{provider}_base_url")
    json_kwargs = _json_kwargs_for_provider(provider)

    async def rerank_func(query, documents, **kw):
        prompt = _build_rerank_prompt(query, documents)
        result_str = await _call_rerank_llm(provider, model, prompt, SYSTEM, api_key, base_url, json_kwargs)
        parsed = RerankResult.model_validate_json(_extract_json(result_str))
        return [{"index": c.index, "relevance_score": c.relevance_score}
                for c in parsed.ranked_chunks]
    return rerank_func

def _json_kwargs_for_provider(provider):
    if provider in ("openai", "azure_openai", "qwen", "minimax", "openrouter", "xinference"):
        return {"response_format": {"type": "json_object"}}
    if provider == "ollama":
        return {"format": "json"}
    if provider == "google_gemini":
        return {"generation_config": {"response_mime_type": "application/json"}}
    return {}  # anthropic: prompt-only
```

### 4. `src/dlightrag/service.py` — Wire It Up

```python
# Before:
llm_func = get_llm_model_func(config)
ingestion_llm_func = get_ingestion_llm_model_func(config)

lightrag_kwargs = {
    # no llm_model_name, no llm_model_kwargs
    ...
}
rag = RAGAnything(None, ingestion_llm_func, vision_func, embedding_func, ...)

# After:
lightrag_kwargs = {
    "llm_model_func": _get_native_llm_func(config.llm_provider),
    "llm_model_name": config.chat_model_name,
    "llm_model_kwargs": _get_llm_model_kwargs(config, config.llm_provider),
    ...
}
rag = RAGAnything(None, None, vision_func, embedding_func, ...)
# llm_model_func now comes from lightrag_kwargs
```

Note: Ingestion and retrieval may use different models. Current config has
`ingestion_model_name` and `ingestion_temperature`. These can be passed via
separate RAGAnything instances or handled in lightrag_kwargs.

### 5. `src/dlightrag/models/llm.py` — Vision (NO CHANGE)

Vision functions (`_build_openai_vision_func`, `_build_anthropic_vision_func`,
`_build_google_vision_func`) already use native SDKs directly. No LangChain
involved. No changes needed.

### 6. `pyproject.toml` — Remove LangChain Dependencies

Remove:
- `langchain-core >= 1.2.16`
- `langchain-openai >= 1.1.10`
- `langchain-anthropic >= 1.3.0`
- `langchain-google-genai >= 4.2.0`

### 7. `src/dlightrag/models/llm.py` — Remove LangChain Imports

Remove all `from langchain_*` imports:
- `from langchain_core.language_models import BaseChatModel`
- `from langchain_core.messages import HumanMessage, SystemMessage`
- `from langchain_openai import ChatOpenAI`
- `from langchain_openai import OpenAIEmbeddings`
- `from langchain_anthropic import ChatAnthropic`
- `from langchain_google_genai import ChatGoogleGenerativeAI`
- `from langchain_google_genai import GoogleGenerativeAIEmbeddings`

## Open Questions

1. **Ingestion vs retrieval model split:** Currently DlightRAG supports
   separate models for ingestion (`ingestion_model_name`) and retrieval
   (`chat_model_name`). LightRAG has a single `llm_model_func`. Need to verify
   if RAGAnything supports overriding at the instance level, or if we need two
   LightRAG instances.

2. **Ollama base_url format:** DlightRAG config stores `ollama_base_url` as
   `http://localhost:11434/v1` (OpenAI-compatible). Native Ollama functions
   expect `http://localhost:11434` (no `/v1`). Need `.rstrip("/v1")` or store
   both formats.

## Test Strategy

- All 287 existing tests must pass
- Add unit tests for `_get_native_llm_func` dispatch
- Add unit tests for `_get_llm_model_kwargs` per provider
- Add unit tests for `_json_kwargs_for_provider`
- Verify embedding `partial` + `EmbeddingFunc` wrapping
- Verify rerank JSON parsing + fallback behavior
