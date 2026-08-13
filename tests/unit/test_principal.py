# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the transport-neutral owner identity every durable run is scoped to."""

from dlightrag.api.auth import UserContext
from dlightrag.api.principal import owner_id_from_user
from dlightrag.core.principal import DEPLOYMENT_OWNER_ID


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

    assert owner_id_from_user(alice) == owner_id_from_user(alice)
    assert owner_id_from_user(alice) != owner_id_from_user(bob)


def test_simple_and_none_are_deployment_scoped() -> None:
    first = UserContext(user_id="header-a", auth_mode="simple")
    second = UserContext(user_id="header-b", auth_mode="simple")

    assert owner_id_from_user(first) == owner_id_from_user(second)
    assert owner_id_from_user(None) != owner_id_from_user(first)


def test_direct_in_process_calls_share_the_deployment_owner() -> None:
    assert owner_id_from_user(None) == DEPLOYMENT_OWNER_ID
    assert owner_id_from_user(UserContext(user_id="anonymous", auth_mode="none")) == (
        DEPLOYMENT_OWNER_ID
    )
