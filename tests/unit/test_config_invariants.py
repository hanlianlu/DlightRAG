# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral invariants retained by the canonical 3.0 configuration."""

from __future__ import annotations

import dataclasses
from typing import Any, cast, get_args

import pytest
from pydantic import SecretStr, ValidationError

from dlightrag.application.access.authentication import (
    AuthenticationSettings,
)
from dlightrag.application.config.loading import load_config
from dlightrag.application.config.sections import (
    AccessControlConfig,
    AccessControlRuleConfig,
    AccessSectionSettings,
    AgentBrowserConfig,
    AgentExecutionConfig,
    AgentMailboxConfig,
    AnswerConfig,
    AnswerSectionSettings,
    ApiInterfaceSettings,
    DeploymentSettings,
    DlightragConfig,
    InterfacesSettings,
    LightRAGStorageSettings,
    McpInterfaceSettings,
    ObservabilitySettings,
    PostgresSettings,
    StorageSettings,
    WebSourceProviderConfig,
    WebSourcesConfig,
)
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelSettings,
    ModelsSettings,
    RerankSettings,
)
from dlightrag.engine.ai.telemetry import is_secret_key
from dlightrag.engine.answer.model_runtime import WebSourceRuntimeSettings
from dlightrag.engine.rag.workspace.settings import (
    RERANK_DEADLINE_SECONDS,
    BM25ProfileSettings,
    CorpusSettings,
    DoclingSidecarSettings,
    ExtractionSettings,
    MinerUSidecarSettings,
    ParserSidecarsSettings,
    RetrievalSettings,
    SourceSettings,
    VisualAssetSettings,
)
from tests.support.settings_models import settings_models


@pytest.fixture(autouse=True)
def _clean_config_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root conftest hides DLIGHTRAG_*; these three names are read by name elsewhere."""
    for key in ("LIGHTRAG_PARSER", "POSTGRES_SERVER_SETTINGS", "POSTGRES_WORKSPACE"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("api://dlightrag", "api://dlightrag"),
        (["api://dlightrag", "proxy-id"], ("api://dlightrag", "proxy-id")),
        ('["a", "b"]', ("a", "b")),
        ("   ", None),
    ],
)
def test_jwt_audience_normalization(raw: Any, expected: Any) -> None:
    assert AccessSectionSettings(jwt_audience=raw).jwt_audience == expected


def test_model_provider_is_case_folded() -> None:
    settings = ModelSettings(provider=cast(Any, " GEMINI "), model="gemini-model")

    assert settings.provider == "gemini"


@pytest.mark.parametrize(
    "values",
    [
        {"model": "x", "temperature": -0.1},
        {"model": "x", "timeout": 0},
        {"model": "x", "max_retries": -1},
        {"model": "x", "structured_output": "json_yaml"},
        {"provider": "anthropic", "model": "x", "structured_output": "json_object"},
        {"provider": "anthropic", "model": "x", "api_family": "response"},
        {"model": "x", "api_family": "responses"},
        {"provider": "invalid", "model": "x"},
    ],
)
def test_invalid_model_settings_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelSettings(**values)  # type: ignore[arg-type]


def test_startup_catalogue_requires_complete_profile_facts() -> None:
    with pytest.raises(ValidationError, match="max_input_tokens"):
        ModelsSettings.model_validate(
            {
                "catalogue": [
                    {
                        "provider": "openai",
                        "model": "new-model",
                        "base_url": None,
                        "profile": {
                            "context_window_tokens": 100_000,
                            "max_output_tokens": 10_000,
                            "supports_images": False,
                            "reasoning": None,
                        },
                    }
                ]
            }
        )


@pytest.mark.parametrize("values", [{"dim": 0}, {"max_token_size": 0}, {"batch_size": 0}])
def test_invalid_embedding_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingSettings(**values)


@pytest.mark.parametrize(
    "provider",
    [
        "azure_cohere",
        "cohere",
        "gemini",
        "jina",
        "openai",
        "openai_compatible",
        "voyage",
    ],
)
def test_embedding_wire_protocols_are_accepted(provider: str) -> None:
    assert EmbeddingSettings(provider=provider).provider == provider  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "values",
    [
        {"max_concurrency": 0},
        {"batch_size": 0},
        {"score_threshold": -0.1},
        {"temperature": -0.1},
    ],
)
def test_invalid_rerank_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RerankSettings(**values)


def test_rerank_provider_is_case_folded() -> None:
    assert RerankSettings(provider=cast(Any, " OpenAI ")).provider == "openai"


@pytest.mark.parametrize(
    "values",
    [
        {"max_images": -1},
        {"image_max_total_bytes": 0},
        {"image_min_quality": 96},
        {"max_attachments": -1},
        {"max_attachment_bytes": 0},
    ],
)
def test_invalid_answer_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        AnswerConfig(**values)


@pytest.mark.parametrize("values", [{"thumb_max_px": 0}, {"thumb_cache_size": 0}])
def test_invalid_visual_asset_bounds_are_rejected(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        VisualAssetSettings(**values)


def test_only_an_explicit_milvus_vector_alternative_is_accepted_for_storage() -> None:
    assert (
        LightRAGStorageSettings(vector_storage="MilvusVectorDBStorage").vector_storage
        == "MilvusVectorDBStorage"
    )
    for values in (
        {"vector_storage": "QdrantStorage"},
        {"graph_storage": "PGGraphStorage"},
        {"graph_storage": "AGEStorage"},
        {"kv_storage": "RedisKVStorage"},
        {"doc_status_storage": "JsonDocStatusStorage"},
    ):
        with pytest.raises(ValidationError):
            LightRAGStorageSettings(**values)  # type: ignore[arg-type]


def test_milvus_reader_and_pg_only_options_are_rejected_without_secret_echo() -> None:
    secret = "never-echo-this-milvus-token"
    storage = StorageSettings(
        lightrag=LightRAGStorageSettings(
            vector_storage="MilvusVectorDBStorage",
            milvus_uri="https://milvus.example",
            milvus_token=secret,
            milvus_db_name="default",
        )
    )
    with pytest.raises(ValidationError) as raised:
        DlightragConfig(
            deployment=DeploymentSettings(service_role="reader"),
            storage=storage,
        )
    assert secret not in str(raised.value)

    with pytest.raises(ValidationError, match="require"):
        LightRAGStorageSettings(milvus_token=secret)


def test_milvus_vector_kwargs_are_bounded_and_exclude_credentials() -> None:
    with pytest.raises(ValidationError, match="unsupported keys"):
        LightRAGStorageSettings(
            vector_storage="MilvusVectorDBStorage",
            vector_db_kwargs={"unknown_index_knob": 1},
        )
    with pytest.raises(ValidationError, match="must not contain credentials"):
        LightRAGStorageSettings(vector_db_kwargs={"token": "secret"})
    with pytest.raises(ValidationError, match="at most 16"):
        LightRAGStorageSettings(vector_db_kwargs={f"key_{index}": index for index in range(17)})


def test_milvus_rejects_postgres_only_promotion_configuration() -> None:
    with pytest.raises(ValidationError, match="promotion thresholds require PGVectorStorage"):
        DlightragConfig(
            storage=StorageSettings(
                lightrag=LightRAGStorageSettings(
                    vector_storage="MilvusVectorDBStorage",
                    milvus_uri="https://milvus.example",
                    milvus_db_name="default",
                )
            ),
            corpus=CorpusSettings(
                promotion={"chunk_threshold": 100},  # type: ignore[arg-type]
            ),
        )


@pytest.mark.parametrize("timeout", [1, RERANK_DEADLINE_SECONDS - 1, RERANK_DEADLINE_SECONDS])
def test_a_retrieval_timeout_the_rerank_deadline_cannot_fit_inside_is_refused(
    timeout: int,
) -> None:
    with pytest.raises(ValidationError, match="corpus.retrieval.timeout must exceed"):
        DlightragConfig(corpus=CorpusSettings(retrieval=RetrievalSettings(timeout=timeout)))


def test_a_retrieval_timeout_just_past_the_rerank_deadline_is_accepted() -> None:
    timeout = RERANK_DEADLINE_SECONDS + 1
    config = DlightragConfig(corpus=CorpusSettings(retrieval=RetrievalSettings(timeout=timeout)))
    assert config.corpus.retrieval.timeout == timeout


def test_the_default_bm25_profiles_name_exactly_one_fallback() -> None:
    assert sum(profile.fallback for profile in RetrievalSettings().bm25_profiles) == 1


@pytest.mark.parametrize(
    "profile",
    [
        {"name": "bad-name", "text_config": "english", "languages": ("en",)},
        {"name": "x", "text_config": "unsafe;drop", "languages": ("en",)},
        {"name": "x", "text_config": "english", "languages": ("en", "de")},
        {"name": "x", "text_config": "simple", "languages": ("en",), "fallback": True},
    ],
)
def test_invalid_bm25_profiles_are_rejected(profile: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        BM25ProfileSettings(**profile)


def test_a_second_parser_block_is_refused() -> None:
    # The documented contract is exactly one parser block: a second one is refused
    # rather than ignored, so a Docling configuration cannot silently run MinerU.
    with pytest.raises(ValidationError, match="not both"):
        ParserSidecarsSettings(
            mineru=MinerUSidecarSettings(),
            docling=DoclingSidecarSettings(),
        )


def test_entity_type_prompt_file_is_one_yaml_filename() -> None:
    assert ExtractionSettings(entity_type_prompt_file="finance.yaml").entity_type_prompt_file == (
        "finance.yaml"
    )
    for value in ("../finance.yaml", "/tmp/finance.yaml", "finance.txt"):
        with pytest.raises(ValidationError):
            ExtractionSettings(entity_type_prompt_file=value)


def test_public_listener_without_auth_is_refused_and_override_is_explicit() -> None:
    public = InterfacesSettings(api=ApiInterfaceSettings(host="0.0.0.0"))
    with pytest.raises(ValidationError, match="non-loopback"):
        DlightragConfig(interfaces=public)
    config = DlightragConfig(
        interfaces=public,
        access=AccessSectionSettings(allow_insecure_no_auth=True),
    )
    assert config.interfaces.api.host == "0.0.0.0"


def test_jwt_jwks_and_mcp_oauth_validation() -> None:
    access = AccessSectionSettings(
        auth_mode="jwt",
        jwt_jwks_url="https://issuer.example/jwks.json",
        jwt_issuer="https://issuer.example",
        jwt_audience="api://dlightrag",
        jwt_algorithm="RS256",
    )
    interfaces = InterfacesSettings(
        mcp=McpInterfaceSettings(
            transport="streamable-http",
            resource_server_url="https://rag.example/mcp",
        )
    )
    assert DlightragConfig(access=access, interfaces=interfaces).access.jwt_algorithm == "RS256"
    # Published keys (named or discovered) verify any token their issuer signs.
    for published in (
        AccessSectionSettings(auth_mode="jwt", jwt_jwks_url="https://x/jwks"),
        AccessSectionSettings(auth_mode="jwt", jwt_issuer="https://x"),
    ):
        with pytest.raises(ValidationError, match="requires jwt_issuer and jwt_audience"):
            DlightragConfig(access=published)
    with pytest.raises(ValidationError, match="requires jwt_issuer or jwt_verification_key"):
        DlightragConfig(access=AccessSectionSettings(auth_mode="jwt"))
    rule = AccessControlRuleConfig(claim="email", value="a@example.com", actions=["admin"])
    with pytest.raises(ValidationError, match="rules require auth_mode='jwt'"):
        DlightragConfig(
            access=AccessSectionSettings(
                auth_mode="simple", api_token="token", control=AccessControlConfig(rules=[rule])
            )
        )


def test_explicit_env_file_and_error_redaction(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=super-secret-value\n",
        encoding="utf-8",
    )
    config = load_config(env_file)
    assert config.models.chat.default.api_key == "super-secret-value"
    assert "super-secret-value" not in repr(config)

    bad = tmp_path / "bad.env"
    bad.write_text(
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__API_KEY=never-echo-me\n"
        "DLIGHTRAG_MODELS__CHAT__DEFAULT__TIMEOUT=0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as caught:
        load_config(bad)
    assert "never-echo-me" not in str(caught.value)


def test_incomplete_secret_only_role_error_never_echoes_key(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_MODELS__CHAT__ROLES__EXTRACT__API_KEY=never-echo-role-secret\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="models.chat.roles.extract.model") as caught:
        load_config(env_file)
    assert "never-echo-role-secret" not in str(caught.value)


def test_web_source_orders_derive_from_available_credentials() -> None:
    config = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
    )

    assert config.search_order() == ("exa", "tavily")
    assert config.extract_order() == ("exa", "tavily")


def test_web_source_orders_put_deepseek_last_whatever_order_the_keys_were_given_in() -> None:
    config = WebSourcesConfig(
        deepseek=WebSourceProviderConfig(api_key="deepseek-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
        exa=WebSourceProviderConfig(api_key="exa-key"),
    )

    assert config.configured_providers() == ("exa", "tavily", "deepseek")
    assert config.search_order() == ("exa", "tavily", "deepseek")
    assert config.extract_order() == ("exa", "tavily", "deepseek")
    assert WebSourcesConfig(
        deepseek=WebSourceProviderConfig(api_key="deepseek-key")
    ).search_order() == ("deepseek",)


def test_web_source_orders_may_name_deepseek_anywhere_and_the_browser_beside_it() -> None:
    config = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        deepseek=WebSourceProviderConfig(api_key="deepseek-key"),
        search_providers=("deepseek", "exa"),
        extract_providers=("deepseek", "browser", "exa"),
    )

    assert config.search_order() == ("deepseek", "exa")
    assert config.extract_order() == ("deepseek", "browser", "exa")


@pytest.mark.parametrize("field", ["search_providers", "extract_providers"])
def test_web_source_order_naming_deepseek_without_its_key_is_refused(field: str) -> None:
    with pytest.raises(ValidationError, match="lack api_key: deepseek"):
        WebSourcesConfig(
            exa=WebSourceProviderConfig(api_key="exa-key"),
            **{field: ("exa", "deepseek")},  # pyright: ignore[reportArgumentType]
        )


def test_the_extract_chain_puts_a_configured_browser_after_deepseek() -> None:
    web_sources = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
        deepseek=WebSourceProviderConfig(api_key="deepseek-key"),
    )
    browser = AgentBrowserConfig(
        endpoints=("ws://agent-browser-1:3000/",),
        egress_proxy="http://agent-browser-egress:3128",
    )

    with_browser = AnswerSectionSettings(
        web_sources=web_sources, agent=AgentExecutionConfig(browser=browser)
    )
    without_browser = AnswerSectionSettings(web_sources=web_sources)

    assert with_browser.extract_chain() == ("exa", "tavily", "deepseek", "browser")
    assert without_browser.extract_chain() == ("exa", "tavily", "deepseek")


def test_the_deepseek_web_key_comes_from_its_environment_variable(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_ANSWER__WEB_SOURCES__DEEPSEEK__API_KEY=deepseek-key\n", encoding="utf-8"
    )

    config = load_config(env_file)

    assert config.answer.web_sources.deepseek.api_key == "deepseek-key"
    assert config.answer.web_sources.search_order() == ("deepseek",)
    assert "deepseek-key" not in repr(config)


def test_web_source_orders_are_independent_and_empty_disables_operation() -> None:
    config = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-key"),
        tavily=WebSourceProviderConfig(api_key="tavily-key"),
        search_providers=("tavily", "exa"),
        extract_providers=(),
    )

    assert config.search_order() == ("tavily", "exa")
    assert config.extract_order() == ()


def test_web_source_order_rejects_provider_without_credential() -> None:
    with pytest.raises(ValidationError, match="lack api_key"):
        WebSourcesConfig(search_providers=("tavily",))


def test_web_source_order_rejects_duplicate_provider() -> None:
    with pytest.raises(ValidationError, match="contain duplicates"):
        WebSourcesConfig(
            exa=WebSourceProviderConfig(api_key="exa-key"),
            search_providers=("exa", "exa"),
        )


def test_a_providers_extra_keys_join_one_rotation_pool() -> None:
    """One key is the historical single-key setup; extra keys multiply it, never replace it."""
    single = WebSourcesConfig(exa=WebSourceProviderConfig(api_key="exa-one"))
    pair = WebSourcesConfig(
        exa=WebSourceProviderConfig(api_key="exa-one", api_keys=("exa-two", "exa-one"))
    )
    extras_only = WebSourcesConfig(exa=WebSourceProviderConfig(api_keys=("exa-one", "exa-two")))

    assert single.exa.all_api_keys() == ("exa-one",)
    assert pair.exa.all_api_keys() == ("exa-one", "exa-two"), "duplicates collapse, order holds"
    assert extras_only.exa.all_api_keys() == ("exa-one", "exa-two")
    # A key may arrive only in the plural setting and still derive the same order.
    assert extras_only.search_order() == ("exa",)
    assert extras_only.extract_order() == ("exa",)


def test_an_empty_key_in_the_plural_setting_is_an_unset_one() -> None:
    assert WebSourceProviderConfig(api_keys=("  ", "\n")).api_keys == ()
    assert WebSourceProviderConfig(api_keys="").api_keys == ()  # pyright: ignore[reportArgumentType]
    assert WebSourceProviderConfig(api_keys=None).api_keys == ()  # pyright: ignore[reportArgumentType]
    assert WebSourceProviderConfig(api_keys=(" exa-one ", "exa-two", "exa-one")).all_api_keys() == (
        "exa-one",
        "exa-two",
    )


@pytest.mark.parametrize(
    "value",
    [
        pytest.param('["unterminated', id="malformed-json"),
        pytest.param(1, id="not-a-list"),
        pytest.param(("key", 1), id="non-string-entry"),
    ],
)
def test_the_plural_key_setting_refuses_a_shape_it_cannot_read(value: object) -> None:
    with pytest.raises(ValidationError):
        WebSourceProviderConfig(api_keys=value)  # pyright: ignore[reportArgumentType]


def test_the_web_key_pool_comes_from_the_environment_as_a_list_or_a_run(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEY=exa-one\n"
        'DLIGHTRAG_ANSWER__WEB_SOURCES__EXA__API_KEYS=\'["exa-two", "exa-three"]\'\n'
        "DLIGHTRAG_ANSWER__WEB_SOURCES__TAVILY__API_KEYS=tavily-one,tavily-two\n",
        encoding="utf-8",
    )

    config = load_config(env_file)

    assert config.answer.web_sources.exa.all_api_keys() == ("exa-one", "exa-two", "exa-three")
    assert config.answer.web_sources.tavily.all_api_keys() == ("tavily-one", "tavily-two")
    assert config.answer.web_sources.search_order() == ("exa", "tavily")
    assert "exa-two" not in repr(config) and "tavily-one" not in repr(config)


def test_config_composes_canonical_models_without_snapshot_copy() -> None:
    from dlightrag.application.settings.projections import rag_settings

    models = ModelsSettings(embedding=EmbeddingSettings(startup_probe=False))
    answer = AnswerSectionSettings()
    config = DlightragConfig(models=models, answer=answer)
    runtime = rag_settings(config)
    assert config.models is models
    assert config.answer is answer
    assert runtime.models is config.models
    assert runtime.corpus is config.corpus


_SECRET = "zq7X-never-echo-nested-K9wP"


def _holds_text(annotation: Any) -> bool:
    return annotation in (str, SecretStr) or any(_holds_text(arg) for arg in get_args(annotation))


def _secret_named(name: str) -> bool:
    return is_secret_key(name)


def test_every_secret_named_setting_is_hidden_from_repr() -> None:
    """A new secret field that forgets repr=False fails here, before it can reach a log."""
    exposed = [
        f"{model.__name__}.{name}"
        for model in settings_models()
        for name, field in model.model_fields.items()
        if _secret_named(name) and _holds_text(field.annotation) and field.repr
    ]
    exposed += [
        f"{projection.__name__}.{item.name}"
        for projection in (AuthenticationSettings, WebSourceRuntimeSettings)
        for item in dataclasses.fields(projection)
        if _secret_named(item.name) and item.repr
    ]

    assert exposed == []


def test_every_settings_model_hides_rejected_input_in_errors() -> None:
    """A validation error must not echo a secret that sits in the rejected input."""
    exposed = sorted(
        model.__name__
        for model in settings_models()
        if model.model_config.get("hide_input_in_errors") is not True
    )

    assert exposed == []


def test_nested_secret_settings_never_render_in_repr() -> None:
    """The root redacts its own repr; nested settings and runtime projections hide secrets too."""
    rendered = [
        ModelSettings(model="m", api_key=_SECRET),
        EmbeddingSettings(model="e", api_key=_SECRET),
        RerankSettings(api_key=_SECRET),
        PostgresSettings(password=_SECRET),
        LightRAGStorageSettings(
            vector_storage="MilvusVectorDBStorage", milvus_uri=_SECRET, milvus_token=_SECRET
        ),
        AccessSectionSettings(api_token=_SECRET, jwt_verification_key=_SECRET),
        ObservabilitySettings(otlp_headers={"authorization": _SECRET}, browser_api_key=_SECRET),
        WebSourceProviderConfig(api_key=_SECRET),
        WebSourceProviderConfig(api_keys=(_SECRET,)),
        WebSourcesConfig(exa=WebSourceProviderConfig(api_key=_SECRET)),
        WebSourcesConfig(tavily=WebSourceProviderConfig(api_keys=(_SECRET,))),
        AgentMailboxConfig(
            bucket="agent-mail",
            alias_domain="mail.example",
            access_key_id=_SECRET,
            secret_access_key=_SECRET,
        ),
        MinerUSidecarSettings(api_token=_SECRET),
        SourceSettings(blob_connection_string=_SECRET),
        WebSourceRuntimeSettings(
            exa_api_keys=(_SECRET,),
            tavily_api_keys=(_SECRET,),
            deepseek_api_keys=(_SECRET,),
        ),
        WebSourcesConfig(deepseek=WebSourceProviderConfig(api_key=_SECRET)),
        AuthenticationSettings(api_token=_SECRET, jwt_verification_key=_SECRET),
    ]
    for item in rendered:
        assert _SECRET not in repr(item), type(item).__name__
        assert _SECRET not in str(item), type(item).__name__


def test_root_config_renders_no_part_of_a_secret() -> None:
    """str() and f-strings go through pydantic field reprs unless the root routes them."""
    # Gateway credentials need not match any secret name pattern.
    headers = {
        "extra_headers": {"Helicone-Auth": _SECRET, "Cookie": _SECRET},
        "authorization": {"bearer": _SECRET},
    }
    config = DlightragConfig(
        storage=StorageSettings(postgres=PostgresSettings(password=_SECRET)),
        models=ModelsSettings(
            chat={"default": {"model": "m", "model_kwargs": headers}},  # pyright: ignore[reportArgumentType]
        ),
    )

    for text in (repr(config), str(config), f"{config}", config.model_dump_json()):
        assert _SECRET[:4] not in text
        assert _SECRET[-4:] not in text
    assert _SECRET not in repr(config.models.chat.default)


def test_dotenv_may_carry_client_and_test_names(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "DLIGHTRAG_API_TOKEN=client-token\n"
        "dlightrag_client_timeout=30\n"
        "RUN_LOAD_RUNTIME=1\n"
        "E2E_ARTIFACT_DIR=/tmp/e2e\n",
        encoding="utf-8",
    )

    config = load_config(env_file)

    assert config.access.api_token is None


def test_client_and_test_names_are_not_settings_elsewhere() -> None:
    """Only a shared .env may carry them; constructor values stay strict."""
    with pytest.raises(ValidationError, match="api_token"):
        DlightragConfig(api_token="client-token")  # type: ignore[call-arg]


def test_nested_settings_hide_rejected_input_in_errors() -> None:
    with pytest.raises(ValidationError) as caught:
        PostgresSettings.model_validate({"passwrod": _SECRET})
    assert _SECRET not in str(caught.value)

    with pytest.raises(ValidationError) as caught:
        WebSourceProviderConfig.model_validate({"api_key": _SECRET, "unexpected": _SECRET})
    assert _SECRET not in str(caught.value)
