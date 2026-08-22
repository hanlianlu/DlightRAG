# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for SDK HTTP configuration."""

from dlightrag.sdk.http import auth_token, client_timeout


def test_client_timeout_reads_only_dlightrag_client_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DLIGHTRAG_CLIENT_TIMEOUT", "17")
    monkeypatch.setenv("DLIGHTRAG_CORPUS__RETRIEVAL__TIMEOUT", "999")

    assert client_timeout() == 17.0


def test_auth_token_reads_client_then_nested_simple_auth_environment(monkeypatch) -> None:
    monkeypatch.setenv("DLIGHTRAG_ACCESS__API_TOKEN", "deployment-token")
    assert auth_token() == "deployment-token"

    monkeypatch.setenv("DLIGHTRAG_API_TOKEN", "client-token")
    assert auth_token() == "client-token"


def test_client_timeout_ignores_server_retrieval_timeout(monkeypatch) -> None:
    monkeypatch.delenv("DLIGHTRAG_CLIENT_TIMEOUT", raising=False)
    monkeypatch.setenv("DLIGHTRAG_CORPUS__RETRIEVAL__TIMEOUT", "999")

    assert client_timeout() == 120.0
