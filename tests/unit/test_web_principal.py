# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for server-owned Web conversation principal identities."""

from dlightrag.api.auth import UserContext
from dlightrag.web.principal import principal_id_from_user


def test_jwt_principal_uses_trust_domain_and_subject() -> None:
    alice = UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )
    bob = UserContext(
        user_id="bob",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )

    assert principal_id_from_user(alice) == principal_id_from_user(alice)
    assert principal_id_from_user(alice) != principal_id_from_user(bob)


def test_simple_and_none_are_deployment_scoped() -> None:
    first = UserContext(user_id="header-a", auth_mode="simple")
    second = UserContext(user_id="header-b", auth_mode="simple")

    assert principal_id_from_user(first) == principal_id_from_user(second)
    assert principal_id_from_user(None) != principal_id_from_user(first)
