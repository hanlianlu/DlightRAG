"""Tests for model factory functions."""

from unittest.mock import AsyncMock

import pytest

from dlightrag.config import (
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
    RerankConfig,
)


def _embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="sk-test",
        startup_probe=False,
    )


class TestMakeCompletionFunc:
    def test_openai_provider(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(
            provider="openai", model="gpt-5.4-mini", api_key="sk-test", temperature=0.5
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

    def test_default_api_key(self):
        from dlightrag.models.llm import _make_completion_func

        cfg = ModelConfig(provider="openai", model="gpt-5.4-mini")  # no api_key
        func = _make_completion_func(cfg, default_api_key="sk-default")
        assert callable(func)

    def test_query_model_func_is_queue_managed_by_default(self):
        from dlightrag.models import llm

        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="sk-test")),
            embedding=_embedding_config(),
        )

        func = llm.get_query_model_func(cfg)
        raw_func = llm.get_query_model_func(cfg, bounded=False)

        assert callable(func)
        assert callable(getattr(func, "shutdown", None))
        assert not hasattr(raw_func, "shutdown")

    def test_vlm_model_func_is_queue_managed_by_default(self):
        from dlightrag.models import llm

        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="sk-test")),
            embedding=_embedding_config(),
        )

        func = llm.get_vlm_model_func(cfg)

        assert callable(func)
        assert callable(getattr(func, "shutdown", None))

    @pytest.mark.asyncio
    async def test_structured_output_uses_openai_json_schema(self, monkeypatch):
        from pydantic import BaseModel, ConfigDict

        from dlightrag.models import llm
        from dlightrag.models.structured import StructuredOutput

        class DemoPlan(BaseModel):
            model_config = ConfigDict(extra="forbid")

            answer: str

        seen: dict[str, object] = {}

        class FakeProvider:
            async def complete(self, **kwargs):
                seen.update(kwargs)
                return '{"answer": "ok"}'

            def stream(self, **kwargs):  # pragma: no cover - not used
                raise AssertionError("stream should not be called")

        monkeypatch.setattr(llm, "get_provider", lambda *args, **kwargs: FakeProvider())
        func = llm._make_completion_func(
            ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-test")
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=StructuredOutput(name="demo_plan", schema=DemoPlan),
        )

        response_format = seen["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "demo_plan"
        assert response_format["json_schema"]["strict"] is True
        assert "extra_body" not in seen

    @pytest.mark.asyncio
    async def test_structured_output_falls_back_to_json_object_for_non_openai(self, monkeypatch):
        from pydantic import BaseModel, ConfigDict

        from dlightrag.models import llm
        from dlightrag.models.structured import StructuredOutput

        class DemoPlan(BaseModel):
            model_config = ConfigDict(extra="forbid")

            answer: str

        seen: dict[str, object] = {}

        class FakeProvider:
            async def complete(self, **kwargs):
                seen.update(kwargs)
                return '{"answer": "ok"}'

            def stream(self, **kwargs):  # pragma: no cover - not used
                raise AssertionError("stream should not be called")

        monkeypatch.setattr(llm, "get_provider", lambda *args, **kwargs: FakeProvider())
        func = llm._make_completion_func(
            ModelConfig(provider="anthropic", model="claude-sonnet-4", api_key="sk-test")
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=StructuredOutput(name="demo_plan", schema=DemoPlan),
        )

        assert seen["response_format"] == {"type": "json_object"}
        assert "extra_body" not in seen

    @pytest.mark.asyncio
    async def test_openai_structured_output_retries_json_object_when_strict_fails(
        self, monkeypatch
    ):
        from pydantic import BaseModel, ConfigDict

        from dlightrag.models import llm
        from dlightrag.models.structured import StructuredOutput

        class DemoPlan(BaseModel):
            model_config = ConfigDict(extra="forbid")

            answer: str

        seen: list[dict[str, object]] = []

        class FakeProvider:
            async def complete(self, **kwargs):
                seen.append(kwargs)
                if kwargs["response_format"]["type"] == "json_schema":
                    raise RuntimeError("strict schemas unsupported")
                return '{"answer": "ok"}'

            def stream(self, **kwargs):  # pragma: no cover - not used
                raise AssertionError("stream should not be called")

        monkeypatch.setattr(llm, "get_provider", lambda *args, **kwargs: FakeProvider())
        func = llm._make_completion_func(
            ModelConfig(provider="openai", model="local-openai-compatible", api_key="sk-test")
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=StructuredOutput(name="demo_plan", schema=DemoPlan),
        )

        assert [call["response_format"]["type"] for call in seen] == [
            "json_schema",
            "json_object",
        ]


class TestModelFactoryExports:
    def test_removed_chat_model_factory_is_not_exported(self):
        import dlightrag.models as models
        from dlightrag.models import llm

        assert not hasattr(llm, "get_chat_model_func")
        assert not hasattr(llm, "get_chat_model_func_for_lightrag")
        assert "get_chat_model_func" not in models.__all__
        assert "get_chat_model_func_for_lightrag" not in models.__all__

    def test_lightrag_default_adapter_is_explicitly_exported(self):
        import dlightrag.models as models
        from dlightrag.models import llm

        assert hasattr(llm, "get_default_model_func_for_lightrag")
        assert "get_default_model_func_for_lightrag" in models.__all__


class TestGetDefaultModelFunc:
    def test_returns_callable(self):
        from dlightrag.models.llm import get_default_model_func

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-test")
            ),
            embedding=_embedding_config(),
        )
        func = get_default_model_func(config)
        assert callable(func)


class TestGetExtractModelFunc:
    def test_uses_default_llm_when_extract_role_is_unset(self):
        from dlightrag.models.llm import get_extract_model_func

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-chat")
            ),
            embedding=_embedding_config(),
        )
        func = get_extract_model_func(config)
        assert callable(func)

    def test_explicit_extract_role(self):
        from dlightrag.models.llm import get_extract_model_func

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-chat"),
                roles=LLMRolesConfig(
                    extract=ModelConfig(provider="anthropic", model="claude-3-5-sonnet")
                ),
            ),
            embedding=_embedding_config(),
        )
        func = get_extract_model_func(config)
        assert callable(func)


class TestGetKeywordModelFunc:
    def test_keyword_model_factory_is_exported(self):
        import dlightrag.models as models
        from dlightrag.models import llm

        assert hasattr(llm, "get_keyword_model_func")
        assert "get_keyword_model_func" in models.__all__

    def test_explicit_keyword_role(self, monkeypatch):
        from dlightrag.models import llm

        seen_models: list[str] = []

        def fake_make_completion_func(cfg, default_api_key=None):
            seen_models.append(cfg.model)
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-chat"),
                roles=LLMRolesConfig(
                    keyword=ModelConfig(provider="openai", model="deepseek-v4-flash")
                ),
            ),
            embedding=_embedding_config(),
        )

        func = llm.get_keyword_model_func(config)

        assert func == "completion:deepseek-v4-flash"
        assert seen_models == ["deepseek-v4-flash"]


class TestGetRerankFunc:
    @staticmethod
    def _capture_scoring_model(monkeypatch):
        from dlightrag.models import llm

        seen_models: list[str] = []
        captured: dict[str, object] = {}

        def fake_make_completion_func(cfg, default_api_key=None):
            seen_models.append(cfg.model)
            return f"completion:{cfg.model}"

        def fake_build_rerank_func(rc, ingest_func=None):
            captured["ingest_func"] = ingest_func
            return "rerank-func"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        monkeypatch.setattr("dlightrag.models.rerank.build_rerank_func", fake_build_rerank_func)
        return llm, seen_models, captured

    def test_chat_llm_reranker_prefers_vlm_role_without_override(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
                roles=LLMRolesConfig(vlm=ModelConfig(provider="openai", model="vlm-model")),
            ),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:vlm-model"
        assert seen_models == ["vlm-model"]

    def test_chat_llm_reranker_uses_query_role_when_vlm_role_is_unset(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
                roles=LLMRolesConfig(query=ModelConfig(provider="openai", model="query-model")),
            ),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:query-model"
        assert seen_models == ["query-model"]

    def test_chat_llm_reranker_uses_default_when_no_role_override_exists(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            ),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:chat-model"
        assert seen_models == ["chat-model"]

    def test_chat_llm_reranker_explicit_config_overrides_roles(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
                roles=LLMRolesConfig(vlm=ModelConfig(provider="openai", model="vlm-model")),
            ),
            rerank=RerankConfig(provider="openai", model="rerank-model"),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:rerank-model"
        assert seen_models == ["rerank-model"]


class TestGetEmbeddingFunc:
    def test_returns_embedding_func(self):
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-test")
            ),
            embedding=_embedding_config(),
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
