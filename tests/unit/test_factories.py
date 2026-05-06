"""Tests for model factory functions."""

from unittest.mock import AsyncMock

import pytest

from dlightrag.config import DlightragConfig, EmbeddingConfig, ModelConfig


class TestMakeCompletionFunc:
    def test_openai_provider(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(
            provider="openai", model="gpt-4.1-mini", api_key="sk-test", temperature=0.5
        )
        func = _make_completion_func(cfg)
        # partial wraps completion_wrapper; calling it should invoke provider.complete()
        assert callable(func)
        assert func.keywords == {}  # no preset args on the partial itself

    def test_anthropic_provider(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(provider="anthropic", model="claude-3-5-sonnet", api_key="sk-ant")
        func = _make_completion_func(cfg)
        assert callable(func)

    def test_fallback_api_key(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(provider="openai", model="gpt-4.1-mini")  # no api_key
        func = _make_completion_func(cfg, fallback_api_key="sk-fallback")
        assert callable(func)


class TestGetChatModelFunc:
    def test_returns_callable(self):
        from dlightrag.models.llm import get_chat_model_func

        config = DlightragConfig(
            chat=ModelConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-test"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        func = get_chat_model_func(config)
        assert callable(func)


class TestGetIngestModelFunc:
    def test_fallback_to_chat(self):
        from dlightrag.models.llm import get_ingest_model_func

        config = DlightragConfig(
            chat=ModelConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-chat"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        func = get_ingest_model_func(config)
        assert callable(func)

    def test_explicit_ingest(self):
        from dlightrag.models.llm import get_ingest_model_func

        config = DlightragConfig(
            chat=ModelConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-chat"),
            ingest=ModelConfig(provider="anthropic", model="claude-3-5-sonnet"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )
        func = get_ingest_model_func(config)
        assert callable(func)


class TestGetRerankFunc:
    def test_chat_llm_reranker_uses_chat_fallback_without_override(self, monkeypatch):
        from dlightrag.models import llm

        seen_models: list[str] = []
        seen_ingest_func = None

        def fake_make_completion_func(cfg, fallback_api_key=None):
            seen_models.append(cfg.model)
            return f"completion:{cfg.model}"

        def fake_build_rerank_func(rc, ingest_func=None):
            nonlocal seen_ingest_func
            seen_ingest_func = ingest_func
            return "rerank-func"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        monkeypatch.setattr("dlightrag.models.rerank.build_rerank_func", fake_build_rerank_func)

        config = DlightragConfig(
            chat=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            vlm=ModelConfig(provider="openai", model="vlm-model"),
            embedding=EmbeddingConfig(api_key="sk-test"),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert seen_ingest_func == "completion:chat-model"
        assert seen_models == ["chat-model"]


class TestGetEmbeddingFunc:
    def test_returns_embedding_func(self):
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(
            chat=ModelConfig(provider="openai", model="gpt-4.1-mini", api_key="sk-test"),
            embedding=EmbeddingConfig(api_key="sk-test", dim=1024),
        )
        emb = get_embedding_func(config)
        assert emb.embedding_dim == 1024
        assert emb.max_token_size == 8192


class TestAdaptForLightrag:
    @pytest.mark.asyncio
    async def test_adapt_wraps_messages_first(self):
        from dlightrag.models.llm import _adapt_for_lightrag

        mock_complete = AsyncMock(return_value="Hello world")
        wrapped = _adapt_for_lightrag(mock_complete)

        result = await wrapped("Tell me", system_prompt="You are helpful")
        mock_complete.assert_called_once()
        call_kwargs = mock_complete.call_args.kwargs
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Tell me"},
        ]
        assert result == "Hello world"
