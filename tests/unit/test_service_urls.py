# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Configured service URLs never carry credentials, and refusing one never echoes it."""

from typing import Any

import pytest

from dlightrag.application.config import load_config
from dlightrag.engine.ai.settings import EmbeddingSettings, reject_url_credentials

_SECRET = "never-echo-url-secret"
_URL = f"https://operator:{_SECRET}@service.example/v1"


def _nested(path: str, value: object) -> dict[str, Any]:
    head, _, rest = path.partition(".")
    return {head: _nested(rest, value) if rest else value}


_CATALOGUE_ENTRY = {
    "provider": "openai",
    "model": "m",
    "base_url": _URL,
    "profile": {
        "context_window_tokens": 1000,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "supports_images": False,
        "reasoning": None,
    },
}

#: Every URL setting DlightRAG calls, verifies against, or publishes, with the
#: location the startup error names. The default chat model is merged over the
#: shipped default before validation, so its errors name ``models.chat``.
_SERVICE_URLS = [
    ("models.chat.base_url", _nested("models.chat.default.base_url", _URL)),
    (
        "models.chat.roles.query.base_url",
        _nested("models.chat.roles.query", {"model": "m", "base_url": _URL}),
    ),
    ("models.catalogue.0.base_url", {"models": {"catalogue": [_CATALOGUE_ENTRY]}}),
    ("models.embedding.base_url", _nested("models.embedding.base_url", _URL)),
    ("models.rerank.base_url", _nested("models.rerank.base_url", _URL)),
    *(
        (path, _nested(path, _URL))
        for path in (
            "corpus.sidecars.mineru.official_endpoint",
            "corpus.sidecars.mineru.local_endpoint",
            "corpus.sidecars.docling.endpoint",
            "storage.lightrag.milvus_uri",
            "access.jwt_jwks_url",
            "access.jwt_issuer",
            "access.web_identity.issuer",
            "access.web_identity.jwks_url",
            "interfaces.mcp.resource_server_url",
            "observability.langfuse_host",
            "answer.agent.connections.oauth_callback_url",
            "answer.agent.browser.egress_proxy",
        )
    ),
    (
        "answer.agent.browser.endpoints.0",
        _nested("answer.agent.browser.endpoints", [_URL.replace("https", "ws")]),
    ),
]


@pytest.mark.parametrize(
    ("location", "overrides"), [pytest.param(*case, id=case[0]) for case in _SERVICE_URLS]
)
def test_a_service_url_with_credentials_fails_naming_only_the_field(
    location: str, overrides: dict[str, Any]
) -> None:
    # A list item's location ends in its index, but the error names the setting.
    field = next(part for part in reversed(location.split(".")) if not part.isdigit())

    with pytest.raises(ValueError) as caught:
        load_config(**overrides)

    message = str(caught.value)
    assert f"{location}: Value error, {field} must not include credentials" in message
    assert _SECRET not in message


@pytest.mark.parametrize(
    "url",
    [
        "https://user@service.example",
        "https://:password@service.example",
        "https://user:password@service.example:8443/v1",
        "//user:password@service.example",
    ],
)
def test_every_userinfo_form_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="base_url must not include credentials"):
        reject_url_credentials(url, "base_url")


@pytest.mark.parametrize(
    "url",
    [
        "https://service.example/v1",
        "https://service.example/v1/@team?owner=a@b#c@d",
        "http://127.0.0.1:19530",
        "./milvus_lite.db",
    ],
)
def test_an_at_sign_outside_the_authority_is_not_a_credential(url: str) -> None:
    assert reject_url_credentials(url, "base_url") == url


def test_an_unparsable_url_is_refused_without_echoing_it() -> None:
    with pytest.raises(ValueError, match="^base_url must be a valid URL$"):
        reject_url_credentials("http://[::1", "base_url")


def test_the_settings_type_keeps_a_credential_free_url() -> None:
    url = "https://gateway.example/v1/@team"

    assert EmbeddingSettings(base_url=url).base_url == url
