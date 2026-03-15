# Provider Management Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the provider architecture from 11 providers with dedicated config fields to a 2-track system (OpenAI SDK + LiteLLM) with messages-first callable interface.

**Architecture:** Two completion backends (`_openai_completion` via AsyncOpenAI, `_litellm_completion` via litellm) behind a messages-first interface. LightRAG gets a thin `_adapt_for_lightrag` wrapper that converts `(prompt, system_prompt=...)` to messages and handles `hashing_kv` caching. Config uses nested `ModelConfig` blocks per role (chat, ingest, embedding, rerank).

**Tech Stack:** OpenAI Python SDK (`openai`), LiteLLM (`litellm`), Pydantic Settings (`pydantic-settings`)

**Spec:** `docs/superpowers/specs/2026-03-15-provider-optimization-design.md`

---

## Chunk 1: Core Infrastructure

### Task 1: Config Schema — ModelConfig + EmbeddingConfig + RerankConfig

**Files:**
- Modify: `src/dlightrag/config.py`
- Test: `tests/unit/test_config.py`

**Context:** The current `config.py` has ~20 provider-specific fields (`openai_api_key`, `azure_openai_base_url`, etc.), an 11-value `LLMProvider` literal, and properties like `effective_embedding_provider`. All of this is replaced by 3 reusable config classes: `ModelConfig`, `EmbeddingConfig`, `RerankConfig`.

**Note:** This task will break existing tests that use old config fields (`tests/conftest.py`, `tests/unit/test_config.py`, `tests/unit/test_llm_providers.py`). This is expected — those tests and the shared `test_config` fixture are updated as part of this task. The project will be in a transitional state until Chunk 2 consumer updates are completed.

- [ ] **Step 1: Write tests for new config schema**

Create `tests/unit/test_config_v2.py` (new file, to avoid conflicts with existing tests during migration):

```python
"""Tests for the new nested provider config schema."""

import pytest
from pydantic import ValidationError

from dlightrag.config import DlightragConfig, EmbeddingConfig, ModelConfig, RerankConfig


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(model="gpt-4.1-mini")
        assert cfg.provider == "openai"
        assert cfg.model == "gpt-4.1-mini"
        assert cfg.api_key is None
        assert cfg.base_url is None
        assert cfg.temperature is None
        assert cfg.timeout == 120.0
        assert cfg.max_retries == 3
        assert cfg.model_kwargs == {}

    def test_litellm_provider(self):
        cfg = ModelConfig(provider="litellm", model="anthropic/claude-3")
        assert cfg.provider == "litellm"

    def test_invalid_provider(self):
        with pytest.raises(ValidationError):
            ModelConfig(provider="invalid", model="test")

    def test_model_kwargs(self):
        cfg = ModelConfig(model="gpt-4.1-mini", model_kwargs={"top_p": 0.9})
        assert cfg.model_kwargs == {"top_p": 0.9}


class TestEmbeddingConfig:
    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.model == "text-embedding-3-large"
        assert cfg.dim == 1024
        assert cfg.max_token_size == 8192
        assert cfg.provider == "openai"

    def test_custom(self):
        cfg = EmbeddingConfig(model="custom-embed", dim=768, max_token_size=4096)
        assert cfg.dim == 768
        assert cfg.max_token_size == 4096


class TestRerankConfig:
    def test_defaults(self):
        cfg = RerankConfig()
        assert cfg.enabled is True
        assert cfg.backend == "llm"
        assert cfg.model is None
        assert cfg.score_threshold == 0.5

    def test_cohere_backend(self):
        cfg = RerankConfig(backend="cohere", model="rerank-v4.0-pro", api_key="key")
        assert cfg.backend == "cohere"


class TestDlightragConfigNested:
    def test_minimal_config(self):
        """Only chat + embedding required."""
        cfg = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-test"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.chat.model == "gpt-4.1-mini"
        assert cfg.ingest is None
        assert cfg.embedding.model == "text-embedding-3-large"

    def test_chat_defaults(self):
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.chat.model == "gpt-4.1-mini"
        assert cfg.chat.temperature == 0.5

    def test_ingest_fallback(self):
        """When ingest is None, consumers should fall back to chat."""
        cfg = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-chat"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.ingest is None
        # Factory functions handle the fallback, not config itself

    def test_ingest_explicit(self):
        cfg = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-chat"),
            ingest=ModelConfig(provider="litellm", model="ollama/qwen3:8b"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        assert cfg.ingest.provider == "litellm"
        assert cfg.ingest.model == "ollama/qwen3:8b"
        # ingest.api_key is None — factory uses chat.api_key fallback

    def test_env_var_nested(self, monkeypatch):
        """Test env var override with __ delimiter."""
        monkeypatch.setenv("DLIGHTRAG_CHAT__MODEL", "gpt-4.1")
        monkeypatch.setenv("DLIGHTRAG_CHAT__API_KEY", "sk-env")
        monkeypatch.setenv("DLIGHTRAG_EMBEDDING__API_KEY", "sk-emb")
        cfg = DlightragConfig(
            embedding=EmbeddingConfig(api_key="sk-emb"),
        )
        assert cfg.chat.model == "gpt-4.1"
        assert cfg.chat.api_key == "sk-env"

    def test_legacy_field_detection(self):
        """Legacy field names should raise clear error."""
        with pytest.raises(ValidationError, match="Legacy config fields"):
            DlightragConfig(
                openai_api_key="sk-old",
                embedding=EmbeddingConfig(api_key="sk-test"),
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_config_v2.py -v`
Expected: FAIL (imports fail or classes don't exist yet)

- [ ] **Step 3: Implement new config schema**

In `src/dlightrag/config.py`:

1. Add new classes **before** the existing `DlightragConfig`:

```python
from pydantic import BaseModel, Field
from typing import Any, Literal

class ModelConfig(BaseModel):
    """Reusable model configuration block."""
    provider: Literal["openai", "litellm"] = "openai"
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float | None = None
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
```

2. In `DlightragConfig`, **replace** the old provider fields (lines 126-190, 202-227) with:

```python
    # --- New nested config ---
    chat: ModelConfig = Field(default_factory=lambda: ModelConfig(model="gpt-4.1-mini", temperature=0.5))
    ingest: ModelConfig | None = None
    embedding: EmbeddingConfig  # required, no default — RAG needs explicit embedding config
    rerank: RerankConfig = Field(default_factory=RerankConfig)
```

3. Delete old fields: `LLMProvider` type, `llm_provider`, `embedding_provider`, `vision_provider`, `chat_model`, `ingestion_model`, `vision_model`, all `*_api_key` / `*_base_url` fields (lines 164-186), `embedding_model`, `embedding_dim`, `chat_model_kwargs`, `vision_model_kwargs`, temperature fields (`llm_temperature`, `vision_temperature`, `ingestion_temperature`, `rerank_temperature`), `llm_request_timeout`, `llm_max_retries`.

4. Delete old properties: `effective_embedding_provider`, `effective_vision_provider`, `effective_rerank_llm_provider`, `effective_rerank_model`, `vision_model_name`, `chat_model_name`, `_get_provider_api_key`, `_get_url`.

5. Delete old validator `_validate_provider_fields`.

6. Add legacy field detection validator:

```python
    @model_validator(mode="before")
    @classmethod
    def _warn_legacy_fields(cls, values):
        if not isinstance(values, dict):
            return values
        legacy = {
            "openai_api_key", "azure_openai_api_key", "anthropic_api_key",
            "qwen_api_key", "minimax_api_key", "ollama_api_key",
            "xinference_api_key", "openrouter_api_key", "voyage_api_key",
            "llm_provider", "chat_model", "vision_model", "vision_provider",
            "embedding_provider", "embedding_model", "embedding_dim",
            "ingestion_model", "llm_temperature", "llm_request_timeout",
        }
        found = [k for k in values if k in legacy]
        if found:
            raise ValueError(
                f"Legacy config fields detected: {found}. "
                "Please migrate to the new nested format. "
                "See .env.example for the new configuration format."
            )
        return values
```

7. Update `SettingsConfigDict` to add nested delimiter:

```python
    model_config = SettingsConfigDict(
        env_prefix="DLIGHTRAG_",
        env_nested_delimiter="__",
    )
```

8. Keep all non-provider fields unchanged (storage backends, workspace, etc.).

- [ ] **Step 3b: Update `tests/conftest.py` fixture**

The shared `test_config` fixture uses old-style `openai_api_key=`. Update to new format:

```python
@pytest.fixture
def test_config(tmp_working_dir: Path) -> DlightragConfig:
    """Create a test config with temporary paths."""
    cfg = DlightragConfig(  # type: ignore[call-arg]
        working_dir=str(tmp_working_dir),
        chat=ModelConfig(model="gpt-4.1-mini", api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests")),
        embedding=EmbeddingConfig(api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests")),
        # Use JSON storage for unit tests (no PG dependency)
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
    )
    set_config(cfg)
    return cfg
```

Update the import at the top of `tests/conftest.py`:
```python
from dlightrag.config import DlightragConfig, EmbeddingConfig, ModelConfig, reset_config, set_config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_config_v2.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dlightrag/config.py tests/unit/test_config_v2.py tests/conftest.py
git commit -m "feat: add nested ModelConfig/EmbeddingConfig/RerankConfig schema"
```

---

### Task 2: Messages-First Model Callables

**Files:**
- Create: `src/dlightrag/models/completion.py` (new file — split from 828-line llm.py)
- Create: `src/dlightrag/models/embedding.py` (new file — split from llm.py)
- Test: `tests/unit/test_completion.py`
- Test: `tests/unit/test_embedding_func.py`

**Context:** The spec places completion and embedding functions in `llm.py`, but the plan splits them into focused modules for better maintainability: `completion.py` (~70 lines) and `embedding.py` (~40 lines). This deviates from the spec but follows the project's existing pattern of focused modules. The old 828-line `llm.py` will be rewritten to ~200 lines in Task 3 (factories + adapter only).

**Note:** `litellm` is added as a dependency in Task 7 (Chunk 3). These modules use lazy `import litellm` inside functions, so they won't fail at import time — only at call time if litellm is missing.

- [ ] **Step 1: Write tests for completion callables**

Create `tests/unit/test_completion.py`:

```python
"""Tests for messages-first completion callables."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAICompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
        )
        assert result == "Hello"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4.1-mini"
        assert call_kwargs.kwargs["messages"] == [{"role": "user", "content": "Hi"}]

    @pytest.mark.asyncio
    async def test_with_response_format(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"answer": "test"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
            response_format={"type": "json_object"},
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_streaming(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()

        async def mock_stream():
            for text in ["Hel", "lo"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        mock_client.chat.completions.create.return_value = mock_stream()

        result = await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
            stream=True,
        )
        chunks = [c async for c in result]
        assert chunks == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_extra_kwargs_forwarded(self):
        from dlightrag.models.completion import _openai_completion

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await _openai_completion(
            messages=[{"role": "user", "content": "Hi"}],
            model="gpt-4.1-mini",
            api_key="sk-test",
            _client=mock_client,
            temperature=0.5,
        )
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.5


class TestLiteLLMCompletion:
    @pytest.mark.asyncio
    async def test_non_streaming(self):
        from dlightrag.models.completion import _litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello"))]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = await _litellm_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="anthropic/claude-3",
                api_key="sk-test",
            )
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_base_url_mapped_to_api_base(self):
        from dlightrag.models.completion import _litellm_completion

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response) as mock:
            await _litellm_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="ollama/qwen3:8b",
                api_key="ollama",
                base_url="http://localhost:11434",
            )
        call_kwargs = mock.call_args
        assert call_kwargs.kwargs["api_base"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_streaming(self):
        from dlightrag.models.completion import _litellm_completion

        async def mock_stream():
            for text in ["He", "llo"]:
                chunk = MagicMock()
                chunk.choices = [MagicMock(delta=MagicMock(content=text))]
                yield chunk

        with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_stream()):
            result = await _litellm_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="anthropic/claude-3",
                api_key="sk-test",
                stream=True,
            )
        chunks = [c async for c in result]
        assert chunks == ["He", "llo"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_completion.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement completion callables**

Create `src/dlightrag/models/completion.py`:

```python
"""Messages-first LLM completion callables.

Two tracks:
- _openai_completion: AsyncOpenAI SDK (OpenAI, Azure, Qwen, MiniMax, etc.)
- _litellm_completion: LiteLLM (Anthropic, Gemini, Ollama, OpenRouter, etc.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI


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
    _client: AsyncOpenAI | None = None,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """OpenAI SDK completion."""
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

        async def _stream() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield delta.content

        return _stream()
    else:
        response = await client.chat.completions.create(**call_kwargs)
        return response.choices[0].message.content or ""


async def _litellm_completion(
    *,
    messages: list[dict[str, Any]],
    model: str,
    api_key: str,
    base_url: str | None = None,
    stream: bool = False,
    response_format: Any = None,
    timeout: float = 120.0,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """LiteLLM completion."""
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

        async def _stream() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    yield delta.content

        return _stream()
    else:
        response = await litellm.acompletion(**call_kwargs)
        return response.choices[0].message.content or ""


__all__ = ["_openai_completion", "_litellm_completion"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_completion.py -v`
Expected: PASS

- [ ] **Step 5: Write tests for embedding callables**

Create `tests/unit/test_embedding_func.py`:

```python
"""Tests for embedding callables."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestOpenAIEmbedding:
    @pytest.mark.asyncio
    async def test_embed_texts(self):
        from dlightrag.models.embedding import _openai_embedding

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_client.embeddings.create.return_value = mock_response

        result = await _openai_embedding(
            ["hello", "world"],
            model="text-embedding-3-large",
            api_key="sk-test",
            _client=mock_client,
        )
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


class TestLiteLLMEmbedding:
    @pytest.mark.asyncio
    async def test_embed_texts(self):
        from dlightrag.models.embedding import _litellm_embedding

        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2]},
            {"embedding": [0.3, 0.4]},
        ]

        with patch("litellm.aembedding", new_callable=AsyncMock, return_value=mock_response):
            result = await _litellm_embedding(
                ["hello", "world"],
                model="text-embedding-3-large",
                api_key="sk-test",
            )
        assert result == [[0.1, 0.2], [0.3, 0.4]]
```

- [ ] **Step 6: Implement embedding callables**

Create `src/dlightrag/models/embedding.py`:

```python
"""Embedding callables.

Two tracks matching the completion module:
- _openai_embedding: AsyncOpenAI SDK
- _litellm_embedding: LiteLLM
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI


async def _openai_embedding(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    _client: AsyncOpenAI | None = None,
    **kwargs: Any,
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
    **kwargs: Any,
) -> list[list[float]]:
    """LiteLLM embedding."""
    import litellm

    call_kwargs: dict[str, Any] = {"model": model, "input": texts, "api_key": api_key}
    if base_url:
        call_kwargs["api_base"] = base_url
    call_kwargs.update(kwargs)
    response = await litellm.aembedding(**call_kwargs)
    return [d["embedding"] for d in response.data]


__all__ = ["_openai_embedding", "_litellm_embedding"]
```

- [ ] **Step 7: Run all new tests**

Run: `uv run pytest tests/unit/test_completion.py tests/unit/test_embedding_func.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/dlightrag/models/completion.py src/dlightrag/models/embedding.py \
        tests/unit/test_completion.py tests/unit/test_embedding_func.py
git commit -m "feat: add messages-first completion and embedding callables"
```

---

### Task 3: LightRAG Adapter + Factory Functions

**Files:**
- Modify: `src/dlightrag/models/llm.py` (rewrite — replace 828 lines with ~200)
- Test: `tests/unit/test_adapter.py` (new)
- Test: `tests/unit/test_factories.py` (new)

**Context:** The adapter bridges messages-first callables to LightRAG's `(prompt, system_prompt=...)` signature, handling `hashing_kv` caching. Factory functions build callables from config with 2 branches instead of 11. The old `llm.py` content is replaced entirely.

- [ ] **Step 1: Write tests for LightRAG adapter**

Create `tests/unit/test_adapter.py`:

```python
"""Tests for _adapt_for_lightrag wrapper."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from dlightrag.models.llm import _adapt_for_lightrag, _LIGHTRAG_STRIP_KWARGS


class TestAdaptForLightrag:
    @pytest.mark.asyncio
    async def test_prompt_to_messages(self):
        """Converts (prompt, system_prompt) to messages array."""
        inner = AsyncMock(return_value="response")
        adapted = _adapt_for_lightrag(inner)

        result = await adapted("What is AI?", system_prompt="You are helpful")
        assert result == "response"

        call_args = inner.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "What is AI?"}

    @pytest.mark.asyncio
    async def test_no_system_prompt(self):
        inner = AsyncMock(return_value="response")
        adapted = _adapt_for_lightrag(inner)

        await adapted("Hello")
        messages = inner.call_args.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_strips_lightrag_kwargs(self):
        """LightRAG-internal kwargs are stripped before forwarding."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        await adapted(
            "test",
            keyword_extraction="foo",
            token_tracker=MagicMock(),
            use_azure=False,
            temperature=0.5,  # should be preserved
        )
        call_kwargs = inner.call_args.kwargs
        assert "keyword_extraction" not in call_kwargs
        assert "token_tracker" not in call_kwargs
        assert "use_azure" not in call_kwargs
        assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_hashing_kv_cache_miss(self):
        """hashing_kv: cache miss → call inner → store result."""
        inner = AsyncMock(return_value="computed result")
        adapted = _adapt_for_lightrag(inner)

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = None  # cache miss

        result = await adapted("test", hashing_kv=mock_kv)
        assert result == "computed result"
        inner.assert_called_once()
        mock_kv.upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_hashing_kv_cache_hit(self):
        """hashing_kv: cache hit → return cached, don't call inner."""
        inner = AsyncMock()
        adapted = _adapt_for_lightrag(inner)

        mock_kv = AsyncMock()
        mock_kv.get_by_id.return_value = {"return": "cached result"}

        result = await adapted("test", hashing_kv=mock_kv)
        assert result == "cached result"
        inner.assert_not_called()

    @pytest.mark.asyncio
    async def test_forwards_stream_and_response_format(self):
        """Non-LightRAG kwargs like stream and response_format pass through."""
        inner = AsyncMock(return_value="ok")
        adapted = _adapt_for_lightrag(inner)

        await adapted("test", stream=True, response_format={"type": "json_object"})
        call_kwargs = inner.call_args.kwargs
        assert call_kwargs["stream"] is True
        assert call_kwargs["response_format"] == {"type": "json_object"}
```

- [ ] **Step 2: Write tests for factory functions**

Create `tests/unit/test_factories.py`:

```python
"""Tests for model factory functions."""

from unittest.mock import AsyncMock, patch

import pytest

from dlightrag.config import DlightragConfig, EmbeddingConfig, ModelConfig


class TestMakeCompletionFunc:
    def test_openai_provider(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(model="gpt-4.1-mini", api_key="sk-test", temperature=0.5)
        func = _make_completion_func(cfg)
        # partial should have model and api_key bound
        assert func.keywords["model"] == "gpt-4.1-mini"
        assert func.keywords["api_key"] == "sk-test"
        assert func.keywords["temperature"] == 0.5
        assert "_client" in func.keywords  # client created

    def test_litellm_provider(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(provider="litellm", model="anthropic/claude-3", api_key="sk-ant")
        func = _make_completion_func(cfg)
        assert func.keywords["model"] == "anthropic/claude-3"
        assert func.keywords["num_retries"] == 3

    def test_fallback_api_key(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(model="gpt-4.1-mini")  # no api_key
        func = _make_completion_func(cfg, fallback_api_key="sk-fallback")
        assert func.keywords["api_key"] == "sk-fallback"


class TestGetChatModelFunc:
    def test_returns_callable(self):
        from dlightrag.models.llm import get_chat_model_func

        config = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-test"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        func = get_chat_model_func(config)
        assert callable(func)


class TestGetIngestModelFunc:
    def test_fallback_to_chat(self):
        from dlightrag.models.llm import get_ingest_model_func

        config = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-chat"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        func = get_ingest_model_func(config)
        assert func.keywords["model"] == "gpt-4.1-mini"

    def test_explicit_ingest(self):
        from dlightrag.models.llm import get_ingest_model_func

        config = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-chat"),
            ingest=ModelConfig(provider="litellm", model="ollama/qwen3:8b"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        func = get_ingest_model_func(config)
        assert func.keywords["model"] == "ollama/qwen3:8b"
        # api_key falls back to chat's
        assert func.keywords["api_key"] == "sk-chat"


class TestGetEmbeddingFunc:
    def test_returns_embedding_func(self):
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(
            chat=ModelConfig(model="gpt-4.1-mini", api_key="sk-test"),
            embedding=EmbeddingConfig(api_key="sk-test", dim=1024),
        )
        emb = get_embedding_func(config)
        assert emb.embedding_dim == 1024
        assert emb.max_token_size == 8192
        assert emb.model_name == "text-embedding-3-large"
```

- [ ] **Step 3: Rewrite `src/dlightrag/models/llm.py`**

Replace the entire file content with:

```python
"""Model factory functions.

Builds messages-first callables from config with 2-track dispatch:
- provider=openai  → AsyncOpenAI SDK
- provider=litellm → LiteLLM

Provides _adapt_for_lightrag() to bridge to LightRAG's (prompt, system_prompt) signature.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import Any

from openai import AsyncOpenAI

from dlightrag.config import DlightragConfig, ModelConfig
from dlightrag.models.completion import _litellm_completion, _openai_completion
from dlightrag.models.embedding import _litellm_embedding, _openai_embedding

logger = logging.getLogger(__name__)

# LightRAG-internal kwargs that must not leak to API calls
_LIGHTRAG_STRIP_KWARGS = {
    "keyword_extraction", "token_tracker",
    "use_azure", "azure_deployment", "api_version",
}


def _adapt_for_lightrag(completion_func: Callable) -> Callable:
    """Wrap messages-first callable for LightRAG compatibility.

    - Converts (prompt, system_prompt=...) to messages array
    - Strips LightRAG-internal kwargs that would cause API errors
    - Handles ``hashing_kv`` for LightRAG's response caching
    """
    from lightrag.utils import compute_args_hash

    async def wrapper(prompt: str, *, system_prompt: str | None = None, **kwargs: Any) -> Any:
        hashing_kv = kwargs.pop("hashing_kv", None)
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in _LIGHTRAG_STRIP_KWARGS}

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Check LightRAG cache
        if hashing_kv is not None:
            args_hash = compute_args_hash(
                "", prompt, system_prompt or "", str(clean_kwargs)
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


# ------------------------------------------------------------------ #
# Factory helpers                                                      #
# ------------------------------------------------------------------ #


def _make_completion_func(cfg: ModelConfig, fallback_api_key: str | None = None) -> partial:
    """Build a messages-first completion callable from config."""
    api_key = cfg.api_key or fallback_api_key
    if cfg.provider == "openai":
        client = AsyncOpenAI(
            api_key=api_key, base_url=cfg.base_url,
            timeout=cfg.timeout, max_retries=cfg.max_retries,
        )
        extra: dict[str, Any] = {**cfg.model_kwargs}
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        return partial(
            _openai_completion,
            model=cfg.model, api_key=api_key,
            base_url=cfg.base_url, _client=client,
            **extra,
        )
    else:  # litellm
        extra = {**cfg.model_kwargs, "num_retries": cfg.max_retries}
        if cfg.temperature is not None:
            extra["temperature"] = cfg.temperature
        return partial(
            _litellm_completion,
            model=cfg.model, api_key=api_key,
            base_url=cfg.base_url, timeout=cfg.timeout,
            **extra,
        )


# ------------------------------------------------------------------ #
# Public factory functions                                             #
# ------------------------------------------------------------------ #


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


def get_embedding_func(config: DlightragConfig) -> Any:
    """Build LightRAG EmbeddingFunc from config."""
    from lightrag.utils import EmbeddingFunc

    cfg = config.embedding
    if cfg.provider == "openai":
        client = AsyncOpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url,
            timeout=cfg.timeout, max_retries=cfg.max_retries,
        )
        func = partial(
            _openai_embedding,
            model=cfg.model, api_key=cfg.api_key,
            base_url=cfg.base_url, _client=client,
        )
    else:
        func = partial(
            _litellm_embedding,
            model=cfg.model, api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
    return EmbeddingFunc(
        embedding_dim=cfg.dim,
        max_token_size=cfg.max_token_size,
        func=func,
        model_name=cfg.model,
    )


def get_rerank_func(config: DlightragConfig) -> Callable | None:
    """Build rerank callable.

    LLM backend uses ingest model (cheaper). API-based backends
    (cohere, jina, aliyun, azure_cohere) use their dedicated clients
    unchanged from the current implementation.
    """
    rc = config.rerank
    if not rc.enabled:
        return None

    if rc.backend == "llm":
        # LLM-based reranking uses the ingest model
        ingest_func = get_ingest_model_func(config)
        return _build_llm_rerank_func(ingest_func, config)

    # API-based rerankers — keep existing implementation
    # (cohere_rerank, jina_rerank, ali_rerank, azure_cohere)
    return _build_api_rerank_func(rc)


def _build_llm_rerank_func(ingest_func: Callable, config: DlightragConfig) -> Callable:
    """Build LLM-based rerank function using the ingest model.

    NOTE: This function's internal implementation should be ported from
    the current _build_llm_rerank_func in llm.py, replacing the provider-
    specific _json_kwargs_for_provider() with universal
    response_format={"type": "json_object"}.
    """
    # Port from current llm.py lines 697-769, replacing:
    # - _json_kwargs_for_provider(provider) → response_format={"type": "json_object"}
    # - get_llm_model_func() → ingest_func parameter
    raise NotImplementedError("Port from current llm.py")


def _build_api_rerank_func(rc: Any) -> Callable:
    """Build API-based rerank function (cohere, jina, aliyun, azure_cohere).

    NOTE: Port unchanged from current llm.py lines 650-676, 777-817.
    """
    raise NotImplementedError("Port from current llm.py")


__all__ = [
    "get_chat_model_func",
    "get_chat_model_func_for_lightrag",
    "get_embedding_func",
    "get_ingest_model_func",
    "get_ingest_model_func_for_lightrag",
    "get_rerank_func",
]
```

**Important:** The rerank functions (`_build_llm_rerank_func`, `_build_api_rerank_func`) are `NotImplementedError` stubs in this task. The implementer must port them from the current `llm.py` (lines 650-817) during implementation:
- LLM rerank: replace `_json_kwargs_for_provider(provider)` with `response_format={"type": "json_object"}`, use `ingest_func` directly, port `_fallback_ranking` helper.
- API rerank (cohere/jina/aliyun/azure_cohere): port unchanged — only imports and provider resolution change.

- [ ] **Step 4: Update `src/dlightrag/models/__init__.py`**

Replace exports to match new public API:

```python
from dlightrag.models.llm import (
    get_chat_model_func,
    get_chat_model_func_for_lightrag,
    get_embedding_func,
    get_ingest_model_func,
    get_ingest_model_func_for_lightrag,
    get_rerank_func,
)
```

- [ ] **Step 5: Run all tests**

Run: `uv run pytest tests/unit/test_adapter.py tests/unit/test_factories.py tests/unit/test_completion.py tests/unit/test_embedding_func.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/dlightrag/models/llm.py src/dlightrag/models/__init__.py \
        tests/unit/test_adapter.py tests/unit/test_factories.py
git commit -m "feat: rewrite llm.py with 2-track factories and LightRAG adapter"
```

---

## Chunk 2: Consumer Updates

### Task 4: AnswerEngine — Messages-First Interface

**Files:**
- Modify: `src/dlightrag/core/answer.py`
- Create: `tests/unit/test_answer_engine.py`

**Context:** AnswerEngine currently has separate code paths for text LLM and VLM (vision), uses `supports_structured` attribute check, and calls model functions with the old signature. No existing test file — tests are created from scratch. Change to:
1. Single constructor arg `model_func` (messages-first, replaces `llm_model_func` + `vision_model_func`)
2. Always `structured=True` for non-streaming (both tracks support `response_format`)
3. Build messages array with images inline (no separate VLM path)
4. Streaming: unchanged (still `structured=False`, freetext prompt)

- [ ] **Step 1: Update AnswerEngine**

In `src/dlightrag/core/answer.py`:

1. Change constructor to take single `model_func` (messages-first callable):

```python
def __init__(self, *, model_func: Callable[..., Any] | None = None) -> None:
    self.model_func = model_func
```

2. In `generate()`: Remove `supports_structured` check, always use `structured=True`. Remove `has_images` branching. Build unified messages:

```python
async def generate(self, query, contexts):
    if self.model_func is None:
        return RetrievalResult(answer=None, contexts=contexts)

    system_prompt = get_answer_system_prompt(structured=True)
    user_prompt = self._build_user_prompt(query, contexts)
    messages = self._build_messages(system_prompt, user_prompt, contexts)

    raw = await self.model_func(
        messages=messages,
        response_format=StructuredAnswer,
    )
    result = self._parse_response(raw, structured=True, query=query)
    return RetrievalResult(answer=result.answer, contexts=contexts, references=result.references)
```

3. In `generate_stream()`: Keep `structured=False`. Build messages similarly:

```python
async def generate_stream(self, query, contexts):
    if self.model_func is None:
        return contexts, None

    system_prompt = get_answer_system_prompt(structured=False)
    user_prompt = self._build_user_prompt(query, contexts)
    messages = self._build_messages(system_prompt, user_prompt, contexts)

    token_iterator = await self.model_func(messages=messages, stream=True)
    if hasattr(token_iterator, "__aiter__"):
        token_iterator = AnswerStream(token_iterator)
    return contexts, token_iterator
```

4. Replace `_build_vlm_messages` + `_has_images` with unified `_build_messages`:

```python
@staticmethod
def _build_messages(
    system_prompt: str,
    user_prompt: str,
    contexts: RetrievalContexts,
) -> list[dict[str, Any]]:
    """Build messages array with inline images if present."""
    content: list[dict[str, Any]] = []

    # Add images from chunks
    for chunk in contexts.get("chunks", []):
        img_data = chunk.get("image_data")
        if img_data:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"},
            })

    content.append({"type": "text", "text": user_prompt})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
```

5. Delete: `_has_images()`, `_select_model_func()`, `_build_vlm_messages()`.

6. Remove `vision_model_func` from constructor and all references.

7. Preserve existing logging patterns (replace `has_images`/`structured` branching logs with equivalent messages like `image_count=N` and `structured=True` hardcoded).

- [ ] **Step 2: Create AnswerEngine tests**

Create `tests/unit/test_answer_engine.py` (no existing tests for AnswerEngine). Write tests for:
- `generate()` with mock `model_func` returning JSON — verify `StructuredAnswer` parsing
- `generate()` with `model_func=None` — returns `RetrievalResult(answer=None)`
- `generate_stream()` with mock async iterator — verify `AnswerStream` wrapping
- `_build_messages()` with image chunks — verify images appear in content array
- `_build_messages()` without images — verify text-only content

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/unit/test_answer_engine.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/core/answer.py tests/unit/test_answer_engine.py
git commit -m "refactor: AnswerEngine uses messages-first interface, remove VLM split"
```

---

### Task 5: Service Layer Updates

**Files:**
- Modify: `src/dlightrag/core/service.py`
- Modify: `src/dlightrag/core/servicemanager.py`

**Context:** `service.py` creates model functions during initialization. `servicemanager.py` creates AnswerEngine and provides `get_llm_func()` for query rewriting. Both must switch to the new factory functions and config field paths.

**Key decision:** In the new architecture, there is no separate `vision_model_func`. Modern LLMs are multimodal — the chat model handles both text and images via messages content arrays. So `vision_func` is replaced by `chat_func` (messages-first callable) everywhere.

- [ ] **Step 1: Update `service.py`**

1. Replace imports (line 112-116):
```python
# Before:
from dlightrag.models import get_embedding_func, get_llm_model_func, get_rerank_func, get_vision_model_func
# After:
from dlightrag.models.llm import (
    get_chat_model_func,
    get_chat_model_func_for_lightrag,
    get_embedding_func,
    get_ingest_model_func,
    get_ingest_model_func_for_lightrag,
    get_rerank_func,
)
```

2. In `_do_initialize()` (caption mode, lines 365-368):
```python
# Before:
llm_func = get_llm_model_func(config)
vision_func = get_vision_model_func(config) if self.enable_vlm else None
# After:
chat_func_lr = get_chat_model_func_for_lightrag(config)
chat_func = get_chat_model_func(config)
```

3. In `_do_initialize_unified()` (unified mode, lines 475-476): Same pattern.

4. Replace all `config.embedding_model` → `config.embedding.model`, `config.embedding_dim` → `config.embedding.dim` (including `_record_workspace_meta()` line 945, and lines 502, 519).

5. Update call sites — which callable goes where:
```python
# LightRAG constructor (needs adapted signature):
llm_model_func=chat_func_lr

# RAGAnything (needs adapted signature for its LightRAG internals):
# vision_func param → chat_func_lr (RAGAnything uses LightRAG-style calls internally)
self.rag = RAGAnything(None, chat_func_lr, chat_func_lr, embedding_func, ...)

# AnswerEngine (messages-first):
model_func=chat_func

# VlmOcrParser (messages-first):
vision_model_func=chat_func

# UnifiedRepresentEngine (messages-first):
vision_model_func=chat_func
```

- [ ] **Step 2: Update `servicemanager.py`**

1. In `_get_answer_engine()` (lines 210-219):
```python
# Before:
from dlightrag.models.llm import get_llm_model_func, get_vision_model_func
llm_model_func=get_llm_model_func(self._config)
vision_model_func=get_vision_model_func(self._config)
# After:
from dlightrag.models.llm import get_chat_model_func
model_func=get_chat_model_func(self._config)
```

2. In `get_llm_func()` (lines 221-225):
```python
# Before:
from dlightrag.models.llm import get_llm_model_func
return get_llm_model_func(self._config)
# After:
from dlightrag.models.llm import get_chat_model_func
return get_chat_model_func(self._config)
```

3. In `_filter_compatible()` (line 367): `config.embedding_model` → `config.embedding.model`.

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest tests/unit/ -v -k "service" --no-header`
Expected: PASS (or identify remaining failures)

- [ ] **Step 4: Commit**

```bash
git add src/dlightrag/core/service.py src/dlightrag/core/servicemanager.py
git commit -m "refactor: service layer uses new factory functions and nested config"
```

---

### Task 6: Remaining Consumer Updates

**Files:**
- Modify: `src/dlightrag/web/routes.py`
- Modify: `src/dlightrag/citations/highlight.py`
- Modify: `src/dlightrag/captionrag/vlm_parser.py`
- Modify: `src/dlightrag/unifiedrepresent/engine.py`
- Modify: `src/dlightrag/unifiedrepresent/retriever.py`
- Modify: `src/dlightrag/unifiedrepresent/extractor.py`
- Modify: `src/dlightrag/unifiedrepresent/multimodal_query.py`

**Context:** These files import old factory functions or call model functions with the old `(prompt, system_prompt=..., image_data=...)` signature. Changes are mechanical: update call sites to messages-first. `models/__init__.py` was already updated in Task 3.

- [ ] **Step 1: Update `web/routes.py`**

1. Update import (line 206-208):
```python
# Before:
from dlightrag.models.llm import get_llm_model_func
llm_func = get_llm_model_func(get_config())
# After:
from dlightrag.models.llm import get_chat_model_func
chat_func = get_chat_model_func(get_config())
```

2. Update `_rewrite_query` (line 69) — convert old signature to messages-first:
```python
# Before:
return await llm_func(user_prompt, system_prompt=system)

# After:
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user_prompt},
]
return await llm_func(messages=messages)
```

3. Update `extract_highlights_for_sources` call — it receives `llm_func` which is now messages-first. The highlight extractor itself (`src/dlightrag/citations/highlight.py`) must also be updated to call `llm_func(messages=[...])` instead of `llm_func(prompt, system_prompt=...)`.

- [ ] **Step 2: Update `citations/highlight.py`**

Convert the LLM call inside `HighlightExtractor._extract_one()` from `(prompt, system_prompt=...)` to messages-first:
```python
# Before:
raw = await self.llm_func(user_prompt, system_prompt=system_prompt)

# After:
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
raw = await self.llm_func(messages=messages)
```

- [ ] **Step 3: Update `vlm_parser.py`**

Lines 178-182: VlmOcrParser calls vision model with `(prompt, image_data=..., system_prompt=...)`. Change to messages-first:

```python
# Before:
raw = await self.vision_model_func(
    OCR_USER_PROMPT,
    image_data=image_bytes,
    system_prompt=OCR_SYSTEM_PROMPT,
)

# After:
import base64
b64 = base64.b64encode(image_bytes).decode()
messages = [
    {"role": "system", "content": OCR_SYSTEM_PROMPT},
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        {"type": "text", "text": OCR_USER_PROMPT},
    ]},
]
raw = await self.vision_model_func(messages=messages)
```

- [ ] **Step 4: Update `engine.py`**

In `UnifiedRepresentEngine.__init__`:
- Replace `config.effective_embedding_provider` → `config.embedding.provider`
- Replace `config._get_url(f"{emb_provider}_base_url")` → `config.embedding.base_url`
- Replace `config._get_provider_api_key(emb_provider)` → `config.embedding.api_key`
- Replace `config.embedding_dim` → `config.embedding.dim`
- Replace rerank config fields:
  - `config.enable_rerank` → `config.rerank.enabled`
  - `config.effective_rerank_model` → `config.rerank.model`
  - `config.rerank_base_url` → `config.rerank.base_url`
  - `config.effective_rerank_api_key` → `config.rerank.api_key`
  - `config.rerank_backend` → `config.rerank.backend`

- [ ] **Step 5: Update `retriever.py`, `extractor.py`, `multimodal_query.py`**

All three consume `vision_model_func` with old signature. Apply same messages-first pattern:

1. **`extractor.py` line 131:** Same pattern as vlm_parser — `(OCR_USER_PROMPT, image_data=..., system_prompt=...)` → messages array with inline base64 image.

2. **`retriever.py` lines 396-413:** Uses `vision_model_func(prompt, image_data=img_bytes)` and checks `getattr(self.vision_model_func, "supports_structured", False)`. Remove `supports_structured` check (always True now). Convert all calls to messages-first with inline images.

3. **`multimodal_query.py` line 51:** `vision_model_func("Describe this image...", image_data=img_bytes)` → messages-first with inline base64 image.

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/unit/ -v --no-header`
Expected: PASS (identify and fix any remaining failures)

- [ ] **Step 7: Commit**

```bash
git add src/dlightrag/web/routes.py src/dlightrag/citations/highlight.py \
        src/dlightrag/captionrag/vlm_parser.py \
        src/dlightrag/unifiedrepresent/engine.py \
        src/dlightrag/unifiedrepresent/retriever.py \
        src/dlightrag/unifiedrepresent/extractor.py \
        src/dlightrag/unifiedrepresent/multimodal_query.py
git commit -m "refactor: update all consumers to messages-first interface"
```

---

## Chunk 3: Migration and Cleanup

### Task 7: Dependencies + Migration Artifacts

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `.env`
- Delete: `tests/unit/test_config.py` (old tests)
- Delete: `tests/unit/test_llm_providers.py` (old tests)
- Rename: `tests/unit/test_config_v2.py` → `tests/unit/test_config.py`

- [ ] **Step 1: Update `pyproject.toml`**

Add `litellm` to dependencies (YAML config support is deferred — `pyyaml` will be added when Component 6 is implemented):

```toml
dependencies = [
    # ... existing deps ...
    "litellm>=1.60.0",
]
```

Remove provider-specific SDKs that are now handled by LiteLLM:
```toml
# DELETE these lines:
"anthropic>=0.84.0",
"google-genai>=1.65.0",
```

Keep `openai>=2.24.0` (used directly by the openai track).

- [ ] **Step 2: Install new dependencies**

Run: `uv sync`

- [ ] **Step 3: Rewrite `.env.example`**

Replace the provider-related sections with the new nested format (spec Component 7). Specifically:
- **Replace:** "Provider API Keys" section, "Models" section (chat_model, vision_model, etc.), "Reranking" provider fields
- **Keep unchanged:** RAG Mode, Ingestion Performance, Storage, Server, Data Sources, and all other non-provider sections
- The new format uses `DLIGHTRAG_CHAT__*`, `DLIGHTRAG_INGEST__*`, `DLIGHTRAG_EMBEDDING__*`, `DLIGHTRAG_RERANK__*` prefix patterns

- [ ] **Step 4: Migrate `.env`**

Convert the actual `.env` credentials to the new nested format. This file is not in git.

- [ ] **Step 5: Clean up old test files**

```bash
# Delete old tests that test the 11-provider config and dispatch
rm -f tests/unit/test_config.py tests/unit/test_llm_providers.py
# Rename new config tests (safe if already renamed)
[ -f tests/unit/test_config_v2.py ] && mv tests/unit/test_config_v2.py tests/unit/test_config.py
```

- [ ] **Step 6: Run full test suite**

Run: `uv run pytest tests/unit/ -v --no-header`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .env.example tests/unit/ src/dlightrag/
git commit -m "chore: add litellm dependency, migrate config, clean up old tests"
```

---

### Task 8: Final Verification + Cleanup

**Files:**
- All files from previous tasks
- Any remaining references to old API

- [ ] **Step 1: Search for remaining old references**

```bash
# Search for any remaining old provider references
grep -rn "get_llm_model_func\|get_vision_model_func\|get_ingestion_llm" src/
grep -rn "LLMProvider\|_OPENAI_COMPATIBLE_PROVIDERS\|supports_structured" src/
grep -rn "openai_api_key\|azure_openai_api_key\|anthropic_api_key" src/
grep -rn "effective_embedding_provider\|effective_vision_provider" src/
# Search for old flat config field patterns (not nested config.embedding.*)
grep -rn "config\.embedding_model\|config\.embedding_dim\|self\.embedding_model\|self\.embedding_dim" src/
```

Fix any remaining references found.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest tests/ -v --no-header`
Expected: ALL PASS

- [ ] **Step 3: Verify import structure**

```bash
uv run python -c "from dlightrag.models.llm import get_chat_model_func, get_chat_model_func_for_lightrag, get_ingest_model_func, get_ingest_model_func_for_lightrag, get_embedding_func, get_rerank_func; print('OK')"
uv run python -c "from dlightrag.config import ModelConfig, EmbeddingConfig, RerankConfig; print('OK')"
```

- [ ] **Step 4: Final commit**

```bash
git add -u  # only tracked files — avoids staging .env or stray files
git commit -m "refactor: complete provider optimization — 11 providers → 2-track (openai + litellm)"
```
