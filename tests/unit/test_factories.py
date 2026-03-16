"""Tests for model factory functions."""

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
