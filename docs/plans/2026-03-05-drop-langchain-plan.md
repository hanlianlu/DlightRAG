# Drop LangChain Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LangChain LLM/embedding layer with LightRAG's native `_if_cache` functions + `partial()`, removing 4 heavy dependencies.

**Architecture:** Use `partial(lightrag_if_cache_fn, model=..., api_key=..., base_url=...)` for LLM functions. Use `partial(lightrag_embed.func, model=..., api_key=...)` for embedding. Replace `with_structured_output` with native provider JSON mode + Pydantic parsing for rerank.

**Tech Stack:** LightRAG native LLM functions (`lightrag.llm.*`), `functools.partial`, `EmbeddingFunc`, Pydantic `model_validate_json()`

**Design doc:** `docs/plans/2026-03-05-drop-langchain-design.md`

---

### Task 1: Remove LangChain from `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:15-16,40,42`

**Step 1: Remove LangChain dependencies**

Remove these 4 lines from `pyproject.toml` dependencies:
```
"langchain-openai>=1.1.10",
"langchain-core>=1.2.16",
"langchain-anthropic>=1.3.0",
"langchain-google-genai>=4.2.0",
```

Keep `openai`, `anthropic`, `google-genai` — they're direct SDK deps used by
LightRAG internally and by our vision functions.

**Step 2: Sync lockfile**

Run: `uv sync`
Expected: Resolves without langchain packages.

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: remove langchain dependencies from pyproject.toml"
```

---

### Task 2: Replace LLM functions in `llm.py`

**Files:**
- Modify: `src/dlightrag/models/llm.py:22-23,34-42,66-195`
- Test: `tests/unit/test_llm_providers.py`

**Step 1: Write the failing tests**

Replace `TestBuildChatModel` class in `tests/unit/test_llm_providers.py` with
tests for the new `get_llm_model_func` dispatch. Tests should verify that
`get_llm_model_func` returns a `functools.partial` wrapping the correct
LightRAG function for each provider.

```python
from functools import partial

from dlightrag.config import DlightragConfig
from dlightrag.models.llm import get_llm_model_func, get_ingestion_llm_model_func


class TestGetLlmModelFunc:
    """Test get_llm_model_func returns correct partial for each provider."""

    def _make_config(self, **overrides) -> DlightragConfig:
        defaults = {
            "openai_api_key": "test-key",
            "llm_provider": "openai",
        }
        defaults.update(overrides)
        return DlightragConfig(**defaults)

    def test_openai_returns_partial(self) -> None:
        config = self._make_config(llm_provider="openai")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"
        assert func.keywords["model"] == "gpt-4.1-mini"

    def test_qwen_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="qwen", qwen_api_key="qwen-key")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_minimax_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="minimax", minimax_api_key="mm-key")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_ollama_returns_ollama_partial(self) -> None:
        config = self._make_config(llm_provider="ollama")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.ollama"
        # Host should have /v1 stripped
        assert func.keywords.get("host", "").endswith("/v1") is False

    def test_openrouter_returns_openai_partial(self) -> None:
        config = self._make_config(
            llm_provider="openrouter", openrouter_api_key="sk-or-key"
        )
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_xinference_returns_openai_partial(self) -> None:
        config = self._make_config(llm_provider="xinference")
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.openai"

    def test_anthropic_returns_anthropic_partial(self) -> None:
        config = self._make_config(
            llm_provider="anthropic", anthropic_api_key="ant-key"
        )
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.anthropic"

    def test_google_gemini_returns_gemini_partial(self) -> None:
        config = self._make_config(
            llm_provider="google_gemini", google_gemini_api_key="google-key"
        )
        func = get_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.func.__module__ == "lightrag.llm.gemini"

    def test_unsupported_provider_raises(self) -> None:
        config = self._make_config()
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_llm_model_func(config, provider="unsupported")

    def test_ingestion_uses_ingestion_model(self) -> None:
        config = self._make_config(
            ingestion_model_name="gpt-4.1-nano",
        )
        func = get_ingestion_llm_model_func(config)
        assert isinstance(func, partial)
        assert func.keywords["model"] == "gpt-4.1-nano"

    def test_model_name_override(self) -> None:
        config = self._make_config()
        func = get_llm_model_func(config, model_name="gpt-4.1")
        assert func.keywords["model"] == "gpt-4.1"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_llm_providers.py::TestGetLlmModelFunc -v`
Expected: FAIL — old `get_llm_model_func` doesn't accept `provider`/`model_name` args.

**Step 3: Implement the new LLM functions**

In `src/dlightrag/models/llm.py`:

1. Remove top-level imports (lines 22-23):
   ```python
   from langchain_core.language_models import BaseChatModel
   from langchain_core.messages import HumanMessage, SystemMessage
   ```

2. Add `from functools import partial` to imports.

3. Delete `_OPENAI_COMPATIBLE_PROVIDERS` (lines 34-42).

4. Replace `_build_chat_model` (lines 71-135), `_build_llm_model_func` (lines
   138-174), `get_llm_model_func` (lines 177-182), and
   `get_ingestion_llm_model_func` (lines 185-195) with:

```python
def get_llm_model_func(
    config: DlightragConfig | None = None,
    model_name: str | None = None,
    provider: str | None = None,
) -> LLMFunc:
    """Build a LightRAG-compatible LLM function using native _if_cache functions.

    Returns a partial() with model/api_key/base_url bound. Works both when
    called directly (RAGAnything modal processing) and when called through
    LightRAG's pipeline (which injects hashing_kv).
    """
    from dlightrag.config import get_config

    cfg = config or get_config()
    prov = provider or cfg.llm_provider
    model = model_name or cfg.chat_model_name
    api_key = cfg._get_provider_api_key(prov)
    base_url = cfg._get_url(f"{prov}_base_url")

    if prov in ("openai", "qwen", "minimax", "openrouter", "xinference"):
        from lightrag.llm.openai import openai_complete_if_cache
        return partial(openai_complete_if_cache, model=model, api_key=api_key, base_url=base_url)

    if prov == "azure_openai":
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache
        return partial(
            azure_openai_complete_if_cache,
            model=model, api_key=api_key, base_url=base_url,
            api_version=cfg.azure_api_version,
        )

    if prov == "anthropic":
        from lightrag.llm.anthropic import anthropic_complete_if_cache
        return partial(anthropic_complete_if_cache, model=model, api_key=api_key)

    if prov == "google_gemini":
        from lightrag.llm.gemini import gemini_complete_if_cache
        return partial(gemini_complete_if_cache, model=model, api_key=api_key)

    if prov == "ollama":
        from lightrag.llm.ollama import ollama_model_if_cache
        host = (cfg.ollama_base_url or "http://localhost:11434").removesuffix("/v1")
        return partial(ollama_model_if_cache, model=model, host=host)

    raise ValueError(f"Unsupported LLM provider: {prov}")


def get_ingestion_llm_model_func(config: DlightragConfig | None = None) -> LLMFunc:
    """Dedicated ingestion LLM — uses ingestion_model_name."""
    from dlightrag.config import get_config

    cfg = config or get_config()
    return get_llm_model_func(cfg, model_name=cfg.ingestion_model_name)
```

Note: The exact function name for Ollama (`ollama_model_if_cache` vs
`_ollama_model_if_cache`) must be verified against the installed LightRAG
version at implementation time.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_llm_providers.py::TestGetLlmModelFunc -v`
Expected: PASS

**Step 5: Clean up old test class**

Remove `TestBuildChatModel` from `tests/unit/test_llm_providers.py` and the
`_build_chat_model` import (line 9).

**Step 6: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: PASS (any remaining references to `_build_chat_model` or LangChain
types should be caught here).

**Step 7: Commit**

```bash
git add src/dlightrag/models/llm.py tests/unit/test_llm_providers.py
git commit -m "refactor: replace LangChain LLM layer with LightRAG native functions"
```

---

### Task 3: Replace embedding function in `llm.py`

**Files:**
- Modify: `src/dlightrag/models/llm.py:557-602`
- Test: `tests/unit/test_llm_providers.py`

**Step 1: Write the failing tests**

Add/update `TestGetEmbeddingFunc` in `tests/unit/test_llm_providers.py`:

```python
class TestGetEmbeddingFunc:
    """Test embedding factory dispatching."""

    def _make_config(self, **overrides) -> DlightragConfig:
        defaults = {
            "openai_api_key": "test-key",
            "llm_provider": "openai",
        }
        defaults.update(overrides)
        return DlightragConfig(**defaults)

    def test_openai_embedding(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = self._make_config()
        func = get_embedding_func(config)
        assert func is not None
        assert func.embedding_dim == 1024
        # Internal func should be a partial wrapping lightrag's openai_embed
        assert isinstance(func.func, partial)

    def test_ollama_embedding(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = self._make_config(
            llm_provider="ollama",
            embedding_provider="ollama",
        )
        func = get_embedding_func(config)
        assert func is not None
        # Should use ollama_embed, not openai_embed
        assert "ollama" in func.func.func.__module__

    def test_google_embedding(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = self._make_config(
            llm_provider="google_gemini",
            google_gemini_api_key="google-key",
            embedding_provider="google_gemini",
        )
        func = get_embedding_func(config)
        assert func is not None
        assert "gemini" in func.func.func.__module__
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_llm_providers.py::TestGetEmbeddingFunc -v`
Expected: FAIL — old `get_embedding_func` uses LangChain `OpenAIEmbeddings`.

**Step 3: Implement new embedding function**

Replace `get_embedding_func` body (lines 557-602):

```python
def get_embedding_func(config: DlightragConfig | None = None) -> EmbeddingFunc:
    """Get embedding function using LightRAG native embed functions."""
    from dlightrag.config import get_config

    cfg = config or get_config()
    emb_provider = cfg.effective_embedding_provider
    api_key = cfg._get_provider_api_key(emb_provider)
    base_url = cfg._get_url(f"{emb_provider}_base_url")

    if emb_provider == "google_gemini":
        from lightrag.llm.gemini import gemini_embed
        raw_fn = partial(gemini_embed.func, model=cfg.embedding_model, api_key=api_key)
    elif emb_provider == "ollama":
        from lightrag.llm.ollama import ollama_embed
        host = (cfg.ollama_base_url or "http://localhost:11434").removesuffix("/v1")
        raw_fn = partial(ollama_embed.func, embed_model=cfg.embedding_model, host=host)
    else:
        # OpenAI-compatible: openai, qwen, minimax, xinference, openrouter, azure
        from lightrag.llm.openai import openai_embed
        raw_fn = partial(
            openai_embed.func, model=cfg.embedding_model,
            api_key=api_key, base_url=base_url,
        )

    return EmbeddingFunc(
        embedding_dim=cfg.embedding_dim,
        max_token_size=8192,
        func=raw_fn,
        model_name=cfg.embedding_model,
    )
```

Note: `openai_embed.func` unwraps the `@wrap_embedding_func_with_attrs`
decorator. If this `.func` attribute doesn't exist, use the function directly
and verify at implementation time.

**Step 4: Run tests**

Run: `pytest tests/unit/test_llm_providers.py::TestGetEmbeddingFunc -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dlightrag/models/llm.py tests/unit/test_llm_providers.py
git commit -m "refactor: replace LangChain embedding with LightRAG native embed functions"
```

---

### Task 4: Replace LLM rerank function in `llm.py`

**Files:**
- Modify: `src/dlightrag/models/llm.py:666-728`
- Test: `tests/unit/test_llm_providers.py`

**Step 1: Write the failing tests**

```python
class TestJsonKwargsForProvider:
    """Test _json_kwargs_for_provider returns correct JSON mode params."""

    def test_openai(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        result = _json_kwargs_for_provider("openai")
        assert result == {"response_format": {"type": "json_object"}}

    def test_ollama(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        result = _json_kwargs_for_provider("ollama")
        assert result == {"format": "json"}

    def test_google_gemini(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        result = _json_kwargs_for_provider("google_gemini")
        assert result == {"generation_config": {"response_mime_type": "application/json"}}

    def test_anthropic(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        result = _json_kwargs_for_provider("anthropic")
        assert result == {}

    def test_openai_compatible_providers(self) -> None:
        from dlightrag.models.llm import _json_kwargs_for_provider

        for provider in ("azure_openai", "qwen", "minimax", "openrouter", "xinference"):
            result = _json_kwargs_for_provider(provider)
            assert result == {"response_format": {"type": "json_object"}}, f"Failed for {provider}"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_llm_providers.py::TestJsonKwargsForProvider -v`
Expected: FAIL — `_json_kwargs_for_provider` doesn't exist yet.

**Step 3: Implement the rerank changes**

Replace `_build_llm_rerank_func` (lines 666-728) with:

```python
def _json_kwargs_for_provider(provider: str) -> dict[str, Any]:
    """Return provider-specific kwargs for JSON output mode."""
    if provider in ("openai", "azure_openai", "qwen", "minimax", "openrouter", "xinference"):
        return {"response_format": {"type": "json_object"}}
    if provider == "ollama":
        return {"format": "json"}
    if provider == "google_gemini":
        return {"generation_config": {"response_mime_type": "application/json"}}
    return {}  # anthropic: prompt-only


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response (handles markdown fences)."""
    import re
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    # Try to find raw JSON object
    start = text.find("{")
    if start != -1:
        return text[start:]
    return text


def _build_llm_rerank_func(config: DlightragConfig) -> Callable:
    """LLM-based listwise reranker using native LLM function + Pydantic parsing.

    3-layer defense:
    1. Native JSON mode where available (via provider-specific kwargs)
    2. Prompt always instructs JSON format (universal fallback)
    3. Pydantic model_validate_json() as safety net
    """
    from dlightrag.models.schemas import RerankResult

    provider = config.effective_rerank_llm_provider
    model = config.effective_rerank_model
    api_key = config._get_provider_api_key(provider)
    base_url = config._get_url(f"{provider}_base_url")
    json_kwargs = _json_kwargs_for_provider(provider)

    llm_func = get_llm_model_func(
        config, model_name=model, provider=provider
    )

    default_domain_knowledge = config.domain_knowledge_hints

    async def rerank_func(
        query: str,
        documents: list[str],
        domain_knowledge: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, float]]:
        if not documents:
            return []

        effective_domain_knowledge = domain_knowledge or default_domain_knowledge
        doc_lines = "\n".join([f"[{idx}] {doc}" for idx, doc in enumerate(documents)])

        system_parts = [
            "You are a reranker. Given a query and a list of chunks, rank them by relevance.",
            "\nRespond with JSON only: {\"ranked_chunks\": [{\"index\": 0, \"relevance_score\": 0.95}, ...]}",
        ]
        if effective_domain_knowledge:
            system_parts.append(f"\n{effective_domain_knowledge}")

        user_content = f"Query: {query}\n\nChunks:\n{doc_lines}"

        try:
            result_str = await llm_func(
                user_content,
                system_prompt="".join(system_parts),
                **json_kwargs,
            )

            parsed = RerankResult.model_validate_json(_extract_json(result_str))

            seen: set[int] = set()
            results = []
            for chunk in parsed.ranked_chunks:
                if 0 <= chunk.index < len(documents) and chunk.index not in seen:
                    seen.add(chunk.index)
                    results.append({"index": chunk.index, "relevance_score": chunk.relevance_score})

            if len(results) < len(documents):
                min_score = results[-1]["relevance_score"] if results else 0.5
                for idx in range(len(documents)):
                    if idx not in seen:
                        results.append(
                            {"index": idx, "relevance_score": max(0.0, min_score - 0.01)}
                        )

            return results or _fallback_ranking(len(documents))
        except Exception as exc:
            logger.warning("Rerank failed: %s", exc)
            return _fallback_ranking(len(documents))

    return rerank_func
```

**Step 4: Run tests**

Run: `pytest tests/unit/test_llm_providers.py::TestJsonKwargsForProvider -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/dlightrag/models/llm.py tests/unit/test_llm_providers.py
git commit -m "refactor: replace LangChain structured output with native JSON mode for rerank"
```

---

### Task 5: Clean up remaining LangChain references

**Files:**
- Modify: `src/dlightrag/models/llm.py` (top-level imports)
- Modify: `src/dlightrag/models/__init__.py` (verify exports)

**Step 1: Remove remaining LangChain imports**

In `src/dlightrag/models/llm.py`, remove:
- Line 22: `from langchain_core.language_models import BaseChatModel`
- Line 23: `from langchain_core.messages import HumanMessage, SystemMessage`
- Line 25: `from pydantic import SecretStr` (only used for LangChain API keys)

These should already be gone from Task 2, but verify no remaining references.
The `HumanMessage` / `SystemMessage` imports in the rerank function (lines
702-703) should now be gone too.

**Step 2: Verify `__init__.py` exports are unchanged**

Read `src/dlightrag/models/__init__.py` — it exports `get_llm_model_func`,
`get_ingestion_llm_model_func`, etc. These function names are unchanged, so no
modifications needed.

**Step 3: Run grep for any remaining langchain references**

Run: `grep -r "langchain" src/ tests/`
Expected: No matches.

**Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass (287+).

**Step 5: Commit**

```bash
git add src/dlightrag/models/llm.py
git commit -m "chore: clean up remaining LangChain imports"
```

---

### Task 6: Sync environment and run final verification

**Step 1: Clean install**

Run: `uv sync --reinstall`
Expected: No langchain packages in the resolved dependencies.

**Step 2: Verify langchain is not installed**

Run: `uv pip list | grep -i langchain`
Expected: No output (langchain packages not installed).

**Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests pass.

**Step 4: Run ruff**

Run: `ruff check src/ tests/`
Expected: No errors.

**Step 5: Commit any lock file changes**

```bash
git add uv.lock
git commit -m "chore: sync lockfile after dropping langchain dependencies"
```

---

## Summary of Changes

| File | Action |
|---|---|
| `pyproject.toml` | Remove 4 langchain deps |
| `src/dlightrag/models/llm.py` | Replace `_build_chat_model` + `_build_llm_model_func` with `get_llm_model_func` using `partial(_if_cache_fn, ...)` |
| `src/dlightrag/models/llm.py` | Replace `get_embedding_func` to use `partial(embed.func, ...)` |
| `src/dlightrag/models/llm.py` | Replace `_build_llm_rerank_func` to use native LLM + Pydantic |
| `src/dlightrag/models/llm.py` | Remove all `langchain_*` imports |
| `tests/unit/test_llm_providers.py` | Replace LangChain type checks with `partial` + module checks |
| `src/dlightrag/models/__init__.py` | No change (API unchanged) |
| `src/dlightrag/service.py` | No change (API unchanged) |

## Risk Mitigation

- **Anthropic streaming**: `anthropic_complete_if_cache` has `stream=True`
  hardcoded and returns `AsyncIterator[str]`. LightRAG handles this internally
  when used as `llm_model_func`. Verify RAGAnything modal processing also
  handles iterators (it should, since it passes the same function).

- **Gemini kwargs discarded**: `gemini_complete_if_cache` uses `**_: Any` and
  discards all kwargs. For rerank JSON mode, we pass
  `generation_config={"response_mime_type": "application/json"}` as a named
  parameter — verify at implementation time whether this is discarded or
  accepted (it's a named param, not a kwarg).

- **Ollama function name**: The internal function may be `_ollama_model_if_cache`
  (private) or `ollama_model_if_cache` (public). Check at implementation time.
