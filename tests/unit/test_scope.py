# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for request/session scoping."""

from __future__ import annotations

from dlightrag.core.scope import RequestScope


def test_request_scope_namespaces_session_by_user_and_workspace() -> None:
    base = RequestScope(user_id="alice", auth_mode="jwt")

    alice_reports = base.for_workspaces(["Reports"])
    alice_finance = base.for_workspaces(["Finance"])
    bob_reports = RequestScope(user_id="bob", auth_mode="jwt").for_workspaces(["Reports"])

    assert alice_reports.session_key("s1") != alice_finance.session_key("s1")
    assert alice_reports.session_key("s1") != bob_reports.session_key("s1")
    assert alice_reports.session_key("s1") == alice_reports.session_key("s1")


def test_empty_session_id_stays_empty() -> None:
    scope = RequestScope(user_id="alice", auth_mode="simple").for_workspaces(["default"])

    assert scope.session_key(None) is None
    assert scope.session_key("") is None
