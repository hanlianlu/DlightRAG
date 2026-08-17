# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for SDK HTTP configuration."""

from dlightrag.sdk.http import client_timeout


def test_client_timeout_reads_only_dlightrag_client_timeout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DLIGHTRAG_CLIENT_TIMEOUT", "17")
    monkeypatch.setenv("DLIGHTRAG_RETRIEVAL_TIMEOUT", "999")

    assert client_timeout() == 17.0


def test_client_timeout_ignores_server_retrieval_timeout(monkeypatch) -> None:
    monkeypatch.delenv("DLIGHTRAG_CLIENT_TIMEOUT", raising=False)
    monkeypatch.setenv("DLIGHTRAG_RETRIEVAL_TIMEOUT", "999")

    assert client_timeout() == 120.0
