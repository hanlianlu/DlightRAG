"""Tests for model factory functions."""

import asyncio
import base64
import io
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image
from pydantic import BaseModel, ConfigDict

from dlightrag.config import (
    AnswerConfig,
    DlightragConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMRolesConfig,
    ModelConfig,
    ParserSidecarsConfig,
    QueryImagesConfig,
    RerankConfig,
    VLMSidecarConfig,
)
from dlightrag.models.structured import StructuredOutput


class DemoPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str


DEMO_STRUCTURED_OUTPUT = StructuredOutput(name="demo_plan", schema=DemoPlan)


class CapturingProvider:
    supports_native_json_schema: bool = False

    def __init__(self, seen: dict[str, Any], *, supports_native_json_schema: bool = False) -> None:
        self.seen = seen
        self.supports_native_json_schema = supports_native_json_schema

    async def complete(self, **kwargs: Any) -> str:
        self.seen.update(kwargs)
        return '{"answer": "ok"}'

    def stream(self, **kwargs: Any):  # pragma: no cover - not used
        raise AssertionError("stream should not be called")


def _capture_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    supports_native_json_schema: bool = False,
) -> tuple[Any, dict[str, Any]]:
    from dlightrag.models import llm

    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        llm,
        "get_provider",
        lambda *args, **kwargs: CapturingProvider(
            seen, supports_native_json_schema=supports_native_json_schema
        ),
    )
    return llm, seen


def _embedding_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="voyage",
        model="voyage-multimodal-3.5",
        api_key="sk-test",
        startup_probe=False,
    )


class TestMakeCompletionFunc:
    def test_role_with_explicit_null_key_keeps_complete_local_config(self):
        from dlightrag.models.llm_roles import model_for_role

        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(
                    provider="openai",
                    model="default-model",
                    api_key="default-key",
                    base_url="https://default.example/v1",
                ),
                roles=LLMRolesConfig(
                    query=ModelConfig(
                        provider="openai",
                        model="local-query",
                        api_key=None,
                        base_url="http://host.docker.internal:8888/v1",
                    )
                ),
            ),
            embedding=_embedding_config(),
        )

        resolved = model_for_role(cfg, "query")

        assert resolved is cfg.llm.roles.query
        assert resolved is not None
        assert resolved.api_key is None
        assert resolved.base_url == "http://host.docker.internal:8888/v1"

    def test_role_without_own_key_falls_back_to_complete_default_config(self):
        from dlightrag.models.llm_roles import model_for_role

        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(
                    provider="openai",
                    model="default-model",
                    api_key="default-key",
                    base_url="https://default.example/v1",
                ),
                roles=LLMRolesConfig(
                    query=ModelConfig(
                        provider="openai",
                        model="local-query",
                        base_url="http://host.docker.internal:8888/v1",
                    )
                ),
            ),
            embedding=_embedding_config(),
        )

        resolved = model_for_role(cfg, "query")

        assert resolved is cfg.llm.default
        assert resolved.model == "default-model"
        assert resolved.api_key == "default-key"
        assert resolved.base_url == "https://default.example/v1"

    @pytest.mark.parametrize("api_key", ["", "   "])
    def test_role_with_blank_key_falls_back_to_complete_default_config(self, api_key: str):
        from dlightrag.models.llm_roles import model_for_role

        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(model="default-model", api_key="default-key"),
                roles=LLMRolesConfig(
                    query=ModelConfig(
                        model="incomplete-query",
                        api_key=api_key,
                        base_url="http://host.docker.internal:8888/v1",
                    )
                ),
            ),
            embedding=_embedding_config(),
        )

        assert model_for_role(cfg, "query") is cfg.llm.default

    def test_planner_model_func_prefers_keyword_role_direct(self, monkeypatch):
        from dlightrag.models import llm

        captured: dict[str, Any] = {}

        def fake_make_completion_func(cfg, *, root=False):
            captured["model"] = cfg.model
            captured["api_key"] = cfg.api_key
            captured["root"] = root
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-chat"),
                roles=LLMRolesConfig(
                    keyword=ModelConfig(
                        provider="openai",
                        model="deepseek-v4-flash",
                        api_key="sk-keyword",
                    )
                ),
            ),
            embedding=_embedding_config(),
            max_async=7,
        )

        func = llm.get_planner_model_func(cfg)

        # DlightRAG-owned planner: direct completion (no queue), nests (root=False)
        assert func == "completion:deepseek-v4-flash"
        assert captured == {
            "model": "deepseek-v4-flash",
            "api_key": "sk-keyword",
            "root": False,
        }

    def test_planner_model_func_uses_default_when_keyword_role_is_unset(self, monkeypatch):
        from dlightrag.models import llm

        captured: dict[str, Any] = {}

        def fake_make_completion_func(cfg, *, root=False):
            captured["model"] = cfg.model
            captured["api_key"] = cfg.api_key
            captured["root"] = root
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="default-model", api_key="sk-chat"),
            ),
            embedding=_embedding_config(),
        )

        func = llm.get_planner_model_func(cfg)

        assert func == "completion:default-model"
        assert captured == {
            "model": "default-model",
            "api_key": "sk-chat",
            "root": False,
        }

    def test_lightrag_facing_funcs_use_root(self, monkeypatch):
        from dlightrag.models import llm

        roots: list[bool] = []

        def fake_make_completion_func(cfg, *, root=False):
            roots.append(root)
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk"),
                roles=LLMRolesConfig(keyword=ModelConfig(provider="openai", model="kw")),
            ),
            embedding=_embedding_config(),
        )

        llm.get_default_model_func(cfg)  # handed to LightRAG → root
        llm.build_role_llm_configs(cfg)  # handed to LightRAG → root
        assert roots and all(roots)

    def test_lightrag_role_overrides_ignore_blank_keys(self, monkeypatch):
        from dlightrag.models import llm

        built_models: list[str] = []

        def fake_make_completion_func(cfg, *, root=False):
            built_models.append(cfg.model)
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(model="default-model", api_key="default-key"),
                roles=LLMRolesConfig(keyword=ModelConfig(model="incomplete-keyword", api_key="")),
            ),
            embedding=_embedding_config(),
        )

        assert llm.build_role_llm_configs(cfg) is None
        assert built_models == []

    def test_owned_funcs_are_not_root(self, monkeypatch):
        from dlightrag.models import llm

        roots: list[bool] = []

        def fake_make_completion_func(cfg, *, root=False):
            roots.append(root)
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(provider="openai", model="m", api_key="sk")),
            embedding=_embedding_config(),
        )

        llm.get_query_model_func(cfg)  # answer → nests
        llm.get_vlm_model_func(cfg)  # vlm → nests
        llm.get_keyword_model_func(cfg)  # highlights → nests
        assert roots == [False, False, False]

    def test_embedding_func_is_root(self, monkeypatch):
        from types import SimpleNamespace

        from dlightrag.models import llm

        captured: dict[str, Any] = {}

        def fake_wrap_embedding_func(fn, *, name="embedding", root=False):
            captured["root"] = root
            return fn

        monkeypatch.setattr("dlightrag.observability.wrap_embedding_func", fake_wrap_embedding_func)
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(provider="openai", model="m", api_key="sk")),
            embedding=_embedding_config(),
        )

        llm.get_embedding_func(cfg, embedder=SimpleNamespace(supports_asymmetric=False))
        assert captured["root"] is True

    @pytest.mark.asyncio
    async def test_structured_output_uses_openai_json_schema(self, monkeypatch):
        llm, seen = _capture_provider(monkeypatch)
        func = llm._make_completion_func(
            ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-test")
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=DEMO_STRUCTURED_OUTPUT,
        )

        response_format = seen["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "demo_plan"
        assert response_format["json_schema"]["strict"] is True
        assert "extra_body" not in seen

    @pytest.mark.asyncio
    async def test_structured_output_auto_uses_json_object_for_openai_compatible_base_url(
        self, monkeypatch
    ):
        llm, seen = _capture_provider(monkeypatch)
        func = llm._make_completion_func(
            ModelConfig(
                provider="openai",
                model="deepseek-v4-flash",
                api_key="sk-test",
                base_url="https://api.deepseek.com",
            )
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=DEMO_STRUCTURED_OUTPUT,
        )

        assert seen["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_structured_output_explicit_json_schema_overrides_custom_base_url(
        self, monkeypatch
    ):
        llm, seen = _capture_provider(monkeypatch)
        func = llm._make_completion_func(
            ModelConfig(
                provider="openai",
                model="openai-compatible-with-schema",
                api_key="sk-test",
                base_url="https://llm.example.test/v1",
                structured_output="json_schema",
            )
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=DEMO_STRUCTURED_OUTPUT,
        )

        response_format = seen["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "demo_plan"

    @pytest.mark.parametrize(
        ("provider", "model"),
        [
            ("anthropic", "claude-sonnet-4"),
            ("gemini", "gemini-2.5-flash"),
        ],
    )
    @pytest.mark.asyncio
    async def test_structured_output_auto_uses_json_schema_for_native_providers(
        self, monkeypatch, provider, model
    ):
        llm, seen = _capture_provider(monkeypatch, supports_native_json_schema=True)
        func = llm._make_completion_func(
            ModelConfig(provider=provider, model=model, api_key="sk-test")
        )

        await func(
            messages=[{"role": "user", "content": "hi"}],
            structured_output=DEMO_STRUCTURED_OUTPUT,
        )

        response_format = seen["response_format"]
        assert isinstance(response_format, dict)
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["name"] == "demo_plan"

    @pytest.mark.asyncio
    async def test_openai_structured_output_retries_json_object_when_strict_fails(
        self, monkeypatch
    ):
        from dlightrag.models import llm

        seen: list[dict[str, Any]] = []

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
            structured_output=DEMO_STRUCTURED_OUTPUT,
        )

        assert [call["response_format"]["type"] for call in seen] == [
            "json_schema",
            "json_object",
        ]


class TestModelFactoryExports:
    def test_lightrag_default_adapter_is_explicitly_exported(self):
        import dlightrag.models as models
        from dlightrag.models import llm

        assert hasattr(llm, "get_default_model_func_for_lightrag")
        assert "get_default_model_func_for_lightrag" in models.__all__


class TestGetKeywordModelFunc:
    def test_keyword_model_factory_is_exported(self):
        import dlightrag.models as models
        from dlightrag.models import llm

        assert hasattr(llm, "get_keyword_model_func")
        assert "get_keyword_model_func" in models.__all__

    def test_explicit_keyword_role(self, monkeypatch):
        from dlightrag.models import llm

        seen_models: list[str] = []

        def fake_make_completion_func(cfg):
            seen_models.append(cfg.model)
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-chat"),
                roles=LLMRolesConfig(
                    keyword=ModelConfig(
                        provider="openai",
                        model="deepseek-v4-flash",
                        api_key="sk-keyword",
                    )
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
        captured: dict[str, Any] = {}

        def fake_make_completion_func(cfg):
            seen_models.append(cfg.model)
            return f"completion:{cfg.model}"

        def fake_build_rerank_func(rc, ingest_func=None):
            captured["rerank_config"] = rc
            captured["ingest_func"] = ingest_func
            return "rerank-func"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        monkeypatch.setattr("dlightrag.models.rerank.build_rerank_func", fake_build_rerank_func)
        return llm, seen_models, captured

    @pytest.mark.parametrize(
        "roles",
        [
            LLMRolesConfig(vlm=ModelConfig(provider="openai", model="vlm-model")),
            LLMRolesConfig(query=ModelConfig(provider="openai", model="query-model")),
        ],
        ids=["vlm-role", "query-role"],
    )
    def test_chat_llm_reranker_uses_default_even_when_roles_exist(
        self, monkeypatch, roles: LLMRolesConfig
    ):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
                roles=roles,
            ),
            rerank=RerankConfig(strategy="chat_llm_reranker"),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:chat-model"
        assert seen_models == ["chat-model"]

    def test_chat_llm_reranker_uses_default_when_no_role_override_exists(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            ),
            rerank=RerankConfig(strategy="chat_llm_reranker"),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:chat-model"
        assert seen_models == ["chat-model"]

    def test_chat_llm_reranker_auto_reuses_positive_vision_probe(self, monkeypatch):
        llm, _, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            ),
            rerank=RerankConfig(strategy="chat_llm_reranker"),
            embedding=_embedding_config(),
        )

        llm.get_rerank_func(config, supports_vision=True)

        assert captured["rerank_config"].input_modality == "multimodal"

    def test_chat_llm_reranker_auto_reuses_negative_vision_probe(self, monkeypatch):
        llm, _, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            ),
            rerank=RerankConfig(strategy="chat_llm_reranker"),
            embedding=_embedding_config(),
        )

        llm.get_rerank_func(config, supports_vision=False)

        assert captured["rerank_config"].input_modality == "text"

    def test_chat_llm_reranker_forced_multimodal_rejects_negative_probe(self, monkeypatch):
        llm, _, _ = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            ),
            rerank=RerankConfig(strategy="chat_llm_reranker", input_modality="multimodal"),
            embedding=_embedding_config(),
        )

        with pytest.raises(ValueError, match="does not support image input"):
            llm.get_rerank_func(config, supports_vision=False)

    def test_chat_llm_reranker_explicit_config_overrides_roles(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
                roles=LLMRolesConfig(vlm=ModelConfig(provider="openai", model="vlm-model")),
            ),
            rerank=RerankConfig(
                strategy="chat_llm_reranker",
                provider="openai",
                model="rerank-model",
                api_key="sk-rerank",
            ),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:rerank-model"
        assert seen_models == ["rerank-model"]

    def test_chat_llm_reranker_blank_key_falls_back_to_default(self, monkeypatch):
        llm, seen_models, captured = self._capture_scoring_model(monkeypatch)

        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(model="chat-model", api_key="default-key"),
            ),
            rerank=RerankConfig(
                strategy="chat_llm_reranker",
                provider="openai",
                model="incomplete-reranker",
                api_key="",
            ),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:chat-model"
        assert seen_models == ["chat-model"]

    def test_provider_reranker_missing_key_fails_fast_without_chat_fallback(self, monkeypatch):
        from dlightrag.models import llm

        make_completion = MagicMock()
        monkeypatch.setattr(llm, "_make_completion_func", make_completion)
        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="chat-model", api_key="sk-chat"),
            ),
            rerank=RerankConfig(strategy="voyage_reranker"),
            embedding=_embedding_config(),
        )

        with pytest.raises(ValueError, match="voyage_reranker requires api_key"):
            llm.get_rerank_func(config)

        make_completion.assert_not_called()


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

    def test_enables_asymmetric_by_default_for_capable_provider(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(embedding=_embedding_config())

        embedding_func = get_embedding_func(config)

        assert embedding_func.supports_asymmetric is True

    def test_uses_symmetric_fallback_for_unsupported_auto(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="openai_compatible",
                model="qwen3-vl-embedding-2b",
                api_key="sk-test",
                dim=2048,
                input_modality="multimodal",
                startup_probe=False,
            )
        )

        embedding_func = get_embedding_func(config)

        assert embedding_func.supports_asymmetric is False

    @pytest.mark.asyncio
    async def test_can_reuse_service_embedder(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="ollama",
                model="nomic-embed-text",
                api_key="",
                dim=3,
                startup_probe=False,
            )
        )
        embedder = MagicMock()
        embedder.supports_asymmetric = False
        embedder.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        embedding_func = get_embedding_func(config, embedder=embedder)
        result = await embedding_func.func(["hello"], context="query")

        assert result.tolist() == [[0.1, 0.2, 0.3]]
        embedder.embed_texts.assert_awaited_once_with(["hello"], context="query")

    def test_rejects_required_asymmetric_for_unsupported_provider(self) -> None:
        from dlightrag.models.llm import get_embedding_func

        config = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="openai_compatible",
                model="qwen3-vl-embedding-2b",
                api_key="sk-test",
                dim=2048,
                input_modality="multimodal",
                asymmetric="require",
                startup_probe=False,
            )
        )

        with pytest.raises(ValueError, match="does not support asymmetric"):
            get_embedding_func(config)


class TestGetMultimodalEmbedder:
    def test_factory_does_not_pass_batch_size(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from dlightrag.models import llm

        captured: dict[str, Any] = {}

        class FakeEmbedder:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "dlightrag.models.multimodal_embedding.MultimodalEmbedder", FakeEmbedder
        )
        config = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-test")
            ),
            embedding=_embedding_config(),
        )

        llm.get_multimodal_embedder(config)

        assert "batch_size" not in captured

    @pytest.mark.asyncio
    async def test_factory_applies_input_modality(self) -> None:
        from dlightrag.models.llm import get_multimodal_embedder

        config = DlightragConfig(
            embedding=EmbeddingConfig(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key="sk-test",
                dim=1024,
                input_modality="text",
                startup_probe=False,
            )
        )

        embedder = get_multimodal_embedder(config)
        try:
            assert embedder.supports_images is False
        finally:
            await embedder.aclose()


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


class TestComposerAnalysisAdapter:
    @pytest.mark.asyncio
    async def test_shared_vlm_adapter_preserves_messages_first_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm, seen = _capture_provider(monkeypatch)
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="vision-model", api_key="sk-test")),
            embedding=_embedding_config(),
        )
        adapter, _identity, _close = llm.create_composer_analysis_adapter(cfg, role="vlm")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ]

        result = await adapter(messages=messages)

        assert result == '{"answer": "ok"}'
        assert seen["messages"] is messages
        assert seen["model_kwargs"] == {}

    @pytest.mark.asyncio
    async def test_consumes_lightrag_controls_and_converts_image_inputs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm, seen = _capture_provider(monkeypatch)
        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(
                    provider="openai",
                    model="default-model",
                    api_key="sk-secret",
                ),
                roles=LLMRolesConfig(
                    vlm=ModelConfig(
                        provider="openai",
                        model="vision-model",
                        api_key="sk-role-secret",
                        base_url="https://user:pass@example.test/v1?token=secret#fragment",
                    )
                ),
            ),
            embedding=_embedding_config(),
        )

        adapter, identity, _close = llm.create_composer_analysis_adapter(cfg, role="vlm")
        response_format = {"type": "json_object"}
        result = await adapter(
            "Describe the drawing",
            hashing_kv=object(),
            _priority=7,
            token_tracker=object(),
            keyword_extraction=True,
            pipeline_status={"cancellation_requested": False},
            pipeline_status_lock=object(),
            image_inputs=[
                {
                    "base64": (
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
                        "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
                    ),
                    "mime_type": "image/png",
                }
            ],
            response_format=response_format,
            stream=False,
        )

        assert result == '{"answer": "ok"}'
        from dlightrag.models.composer import normalized_endpoint_fingerprint

        assert identity == {
            "provider": "openai",
            "model": "vision-model",
            "endpoint_fingerprint": normalized_endpoint_fingerprint(
                "https://user:pass@example.test/v1?token=secret#fragment"
            ),
        }
        content = seen["messages"][-1]["content"]
        assert content[-1] == {"type": "text", "text": "Describe the drawing"}
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/")
        assert seen["response_format"] is response_format
        assert seen["model_kwargs"] == {}

    @pytest.mark.parametrize(
        ("control", "value"),
        [
            ("entity_extraction", True),
            ("future_cache_metadata", {"cache_scope": "workspace"}),
        ],
    )
    @pytest.mark.asyncio
    async def test_rejects_unknown_lightrag_controls_without_calling_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
        control: str,
        value: object,
    ) -> None:
        llm, seen = _capture_provider(monkeypatch)
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="vision-model", api_key="sk-test")),
            embedding=_embedding_config(),
        )
        adapter, _identity, _close = llm.create_composer_analysis_adapter(cfg, role="vlm")

        with pytest.raises(TypeError, match=control):
            await adapter(
                "Describe the drawing",
                **{control: value},
            )

        assert seen == {}

    @pytest.mark.asyncio
    async def test_bounds_images_and_preserves_stream_semantics(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.models import llm

        seen: dict[str, Any] = {}
        stream_result = object()

        class StreamingProvider(CapturingProvider):
            def stream(self, **kwargs: Any) -> object:
                self.seen.update(kwargs)
                return stream_result

        monkeypatch.setattr(
            llm,
            "get_provider",
            lambda *args, **kwargs: StreamingProvider(seen),
        )
        source = Image.effect_noise((256, 192), 180).convert("RGB")
        buffer = io.BytesIO()
        source.save(buffer, format="PNG")
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="vision", api_key="sk-test")),
            embedding=_embedding_config(),
            answer=AnswerConfig(
                image_max_bytes=5_000,
                image_max_total_bytes=5_000,
                image_max_px=96,
                image_min_px=32,
            ),
        )
        adapter, _identity, _close = llm.create_composer_analysis_adapter(cfg, role="vlm")
        response_format = {"type": "json_object"}

        result = await adapter(
            "Describe",
            image_inputs=[
                {
                    "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    "mime_type": "image/png",
                }
            ],
            response_format=response_format,
            stream=True,
        )

        assert result is stream_result
        assert seen["response_format"] is response_format
        assert seen["model_kwargs"] == {}
        uri = seen["messages"][-1]["content"][0]["image_url"]["url"]
        raw = base64.b64decode(uri.split(",", 1)[1])
        assert len(raw) <= 5_000
        with Image.open(io.BytesIO(raw)) as bounded:
            assert max(bounded.size) <= 96

    @pytest.mark.asyncio
    async def test_uses_effective_lightrag_vlm_limits_for_image_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.utils import images

        llm, seen = _capture_provider(monkeypatch)
        bounded_kwargs: dict[str, Any] = {}

        def bounded_image_data_uri(value: str, **kwargs: Any) -> tuple[str, int]:
            bounded_kwargs.update(kwargs)
            return value, 100

        monkeypatch.setattr(images, "bounded_image_data_uri", bounded_image_data_uri)
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="vision", api_key="sk-test")),
            embedding=_embedding_config(),
            answer=AnswerConfig(
                image_max_bytes=1_000_000,
                image_max_total_bytes=2_000_000,
                image_min_px=32,
            ),
            parser_sidecars=ParserSidecarsConfig(
                vlm=VLMSidecarConfig(
                    max_image_bytes=300_000,
                    min_image_pixel=128,
                )
            ),
        )
        adapter, _identity, _close = llm.create_composer_analysis_adapter(cfg, role="vlm")

        await adapter(
            "Describe",
            image_inputs=[{"base64": "aGVsbG8=", "mime_type": "image/png"}],
        )

        assert bounded_kwargs["max_bytes"] == 300_000
        assert bounded_kwargs["min_px"] == 128
        assert seen["messages"][-1]["content"][0]["type"] == "image_url"

    @pytest.mark.parametrize(
        "answer",
        [
            AnswerConfig(max_images=0),
            AnswerConfig(
                max_images=1,
                image_max_bytes=1,
                image_max_total_bytes=1,
                image_max_px=16,
                image_min_px=1,
            ),
        ],
        ids=["no-image-slots", "compression-rejected"],
    )
    @pytest.mark.asyncio
    async def test_rejected_composer_images_never_fall_back_to_text_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        answer: AnswerConfig,
    ) -> None:
        llm, seen = _capture_provider(monkeypatch)
        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="vision-model", api_key="sk-test")),
            embedding=_embedding_config(),
            answer=answer,
            query_images=QueryImagesConfig(max_current_images=answer.max_images),
        )
        adapter, _identity, _close = llm.create_composer_analysis_adapter(cfg, role="vlm")

        with pytest.raises(RuntimeError, match="Composer image payload") as exc_info:
            await adapter(
                "Describe the drawing",
                image_inputs=[
                    {
                        "base64": (
                            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/"
                            "x8AAwMCAO+/p9sAAAAASUVORK5CYII="
                        ),
                        "mime_type": "image/png",
                    }
                ],
            )

        assert type(exc_info.value).__name__ == "ComposerImagePayloadError"
        assert seen == {}

    @pytest.mark.asyncio
    async def test_composer_bundle_owns_and_closes_each_role_once(self) -> None:
        from dlightrag.models.composer import ComposerModelBundle

        vlm = AsyncMock()
        extract = AsyncMock()
        close_vlm = AsyncMock()
        close_extract = AsyncMock()
        created_roles: list[str] = []

        def create_adapter(config: DlightragConfig, *, role: str):
            created_roles.append(role)
            if role == "vlm":
                return vlm, {"provider": "openai", "model": "vision"}, close_vlm
            return extract, {"provider": "openai", "model": "extract"}, close_extract

        cfg = DlightragConfig(embedding=_embedding_config())

        bundle = await ComposerModelBundle.acreate(
            cfg, bind=lambda func: func, adapter_factory=create_adapter
        )

        assert bundle.vlm_func is vlm
        assert bundle.extract_func is extract
        assert bundle.vlm_identity == {"provider": "openai", "model": "vision"}
        assert bundle.extract_identity == {"provider": "openai", "model": "extract"}
        assert created_roles == ["vlm", "extract"]

        await bundle.aclose()
        await bundle.aclose()

        close_vlm.assert_awaited_once()
        close_extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_composer_bundle_rolls_back_vlm_when_extract_creation_fails(self) -> None:
        from dlightrag.models.composer import ComposerModelBundle

        close_vlm = AsyncMock()

        def create_adapter(config: DlightragConfig, *, role: str):
            if role == "extract":
                raise RuntimeError("extract construction failed")
            return AsyncMock(), {"provider": "openai", "model": "vision"}, close_vlm

        cfg = DlightragConfig(embedding=_embedding_config())

        with pytest.raises(RuntimeError, match="extract construction failed"):
            await ComposerModelBundle.acreate(
                cfg, bind=lambda func: func, adapter_factory=create_adapter
            )

        close_vlm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_composer_bundle_rolls_back_both_roles_when_extract_binding_fails(
        self,
    ) -> None:
        from dlightrag.models.composer import ComposerModelBundle

        vlm = AsyncMock()
        extract = AsyncMock()
        close_vlm = AsyncMock()
        close_extract = AsyncMock()

        def create_adapter(config: DlightragConfig, *, role: str):
            if role == "vlm":
                return vlm, {}, close_vlm
            return extract, {}, close_extract

        def bind(func: Any) -> Any:
            if func is extract:
                raise RuntimeError("extract binding failed")
            return func

        cfg = DlightragConfig(embedding=_embedding_config())

        with pytest.raises(RuntimeError, match="extract binding failed"):
            await ComposerModelBundle.acreate(cfg, bind=bind, adapter_factory=create_adapter)

        close_vlm.assert_awaited_once()
        close_extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_composer_bundle_construction_cancellation_rolls_back_vlm(self) -> None:
        from dlightrag.models.composer import ComposerModelBundle

        close_vlm = AsyncMock()
        task = asyncio.current_task()
        assert task is not None

        def create_adapter(config: DlightragConfig, *, role: str):
            assert role == "vlm"
            task.cancel("construction cancelled")
            return AsyncMock(), {}, close_vlm

        cfg = DlightragConfig(embedding=_embedding_config())

        with pytest.raises(asyncio.CancelledError, match="construction cancelled"):
            await ComposerModelBundle.acreate(
                cfg, bind=lambda func: func, adapter_factory=create_adapter
            )

        close_vlm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_composer_bundle_close_continues_after_cancellation_and_retries(
        self,
    ) -> None:
        from dlightrag.models.composer import ComposerModelBundle

        calls: list[str] = []
        first_attempt = True

        async def close_vlm() -> None:
            nonlocal first_attempt
            calls.append("vlm")
            if first_attempt:
                first_attempt = False
                raise asyncio.CancelledError("shutdown")

        async def close_extract() -> None:
            calls.append("extract")

        bundle = ComposerModelBundle(
            vlm_func=AsyncMock(),
            extract_func=AsyncMock(),
            vlm_identity={},
            extract_identity={},
            _closers=(close_vlm, close_extract),
        )

        with pytest.raises(asyncio.CancelledError, match="shutdown"):
            await bundle.aclose()

        assert calls == ["vlm", "extract"]
        assert bundle._closed is False

        await bundle.aclose()
        await bundle.aclose()

        assert calls == ["vlm", "extract", "vlm"]
        assert bundle._closed is True

    @pytest.mark.asyncio
    async def test_composer_bundle_close_retries_exception_without_double_closing(
        self,
    ) -> None:
        from dlightrag.models.composer import ComposerModelBundle

        close_vlm = AsyncMock(side_effect=[RuntimeError("temporary"), None])
        close_extract = AsyncMock()
        bundle = ComposerModelBundle(
            vlm_func=AsyncMock(),
            extract_func=AsyncMock(),
            vlm_identity={},
            extract_identity={},
            _closers=(close_vlm, close_extract),
        )

        await bundle.aclose()
        assert bundle._closed is False
        await bundle.aclose()
        await bundle.aclose()

        assert close_vlm.await_count == 2
        close_extract.assert_awaited_once()
        assert bundle._closed is True
