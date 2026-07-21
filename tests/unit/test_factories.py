"""Tests for model factory functions."""

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
    RerankConfig,
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

    def test_query_model_func_is_direct_not_queue_managed(self):
        from dlightrag.models import llm

        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="sk-test")),
            embedding=_embedding_config(),
        )

        func = llm.get_query_model_func(cfg)

        # DlightRAG-owned: a plain completion callable, not a queue-managed wrapper
        assert callable(func)
        assert not hasattr(func, "shutdown")

    def test_planner_model_func_prefers_keyword_role_direct(self, monkeypatch):
        from dlightrag.models import llm

        captured: dict[str, Any] = {}

        def fake_make_completion_func(cfg, default_api_key=None, *, root=False):
            captured["model"] = cfg.model
            captured["default_api_key"] = default_api_key
            captured["root"] = root
            return f"completion:{cfg.model}"

        monkeypatch.setattr(llm, "_make_completion_func", fake_make_completion_func)
        cfg = DlightragConfig(
            llm=LLMConfig(
                default=ModelConfig(provider="openai", model="gpt-5.4-mini", api_key="sk-chat"),
                roles=LLMRolesConfig(
                    keyword=ModelConfig(provider="openai", model="deepseek-v4-flash")
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
            "default_api_key": "sk-chat",
            "root": False,
        }

    def test_planner_model_func_uses_default_when_keyword_role_is_unset(self, monkeypatch):
        from dlightrag.models import llm

        captured: dict[str, Any] = {}

        def fake_make_completion_func(cfg, default_api_key=None, *, root=False):
            captured["model"] = cfg.model
            captured["default_api_key"] = default_api_key
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
            "default_api_key": "sk-chat",
            "root": False,
        }

    def test_vlm_model_func_is_direct_not_queue_managed(self):
        from dlightrag.models import llm

        cfg = DlightragConfig(
            llm=LLMConfig(default=ModelConfig(model="gpt-5.4-mini", api_key="sk-test")),
            embedding=_embedding_config(),
        )

        func = llm.get_vlm_model_func(cfg)

        # DlightRAG-owned: a plain completion callable, not a queue-managed wrapper
        assert callable(func)
        assert not hasattr(func, "shutdown")

    def test_lightrag_facing_funcs_use_root(self, monkeypatch):
        from dlightrag.models import llm

        roots: list[bool] = []

        def fake_make_completion_func(cfg, default_api_key=None, *, root=False):
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

    def test_owned_funcs_are_not_root(self, monkeypatch):
        from dlightrag.models import llm

        roots: list[bool] = []

        def fake_make_completion_func(cfg, default_api_key=None, *, root=False):
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
        captured: dict[str, Any] = {}

        def fake_make_completion_func(cfg, default_api_key=None):
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
                strategy="chat_llm_reranker", provider="openai", model="rerank-model"
            ),
            embedding=_embedding_config(),
        )

        result = llm.get_rerank_func(config)

        assert result == "rerank-func"
        assert captured["ingest_func"] == "completion:rerank-model"
        assert seen_models == ["rerank-model"]

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
        assert identity == {
            "provider": "openai",
            "model": "vision-model",
            "base_url": "https://example.test/v1",
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
    async def test_ignores_unknown_lightrag_controls_without_provider_leakage(
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

        result = await adapter(
            "Describe the drawing",
            **{control: value},
        )

        assert result == '{"answer": "ok"}'
        assert seen["model_kwargs"] == {}

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
    async def test_composer_bundle_owns_and_closes_each_role_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from dlightrag.models import llm
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

        monkeypatch.setattr(llm, "create_composer_analysis_adapter", create_adapter)
        cfg = DlightragConfig(embedding=_embedding_config())

        bundle = ComposerModelBundle.create(cfg, bind=lambda func: func)

        assert bundle.vlm_func is vlm
        assert bundle.extract_func is extract
        assert bundle.vlm_identity == {"provider": "openai", "model": "vision"}
        assert bundle.extract_identity == {"provider": "openai", "model": "extract"}
        assert created_roles == ["vlm", "extract"]

        await bundle.aclose()
        await bundle.aclose()

        close_vlm.assert_awaited_once()
        close_extract.assert_awaited_once()
