# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""`/.well-known/security.txt`: reachable without a session, and never stale."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

from starlette.routing import Route

from dlightrag.adapters.http.well_known import SECURITY_CONTACT, router, security_txt


def test_the_document_states_a_contact_an_expiry_and_the_languages() -> None:
    body = security_txt()
    assert f"Contact: {SECURITY_CONTACT}" in body
    # RFC 9116: a Contact is a mailto: or an https: URI.
    assert SECURITY_CONTACT.startswith(("mailto:", "https://"))
    assert "Preferred-Languages: en, zh" in body
    expires = datetime.strptime(body.split("Expires: ")[1].split("\n")[0], "%Y-%m-%dT%H:%M:%SZ")
    assert expires.replace(tzinfo=UTC) > datetime.now(UTC) + timedelta(days=30)


def test_a_report_always_finds_a_future_expiry() -> None:
    """An expired file tells a reporter nobody is listening, so it is computed."""
    ahead = datetime(2030, 1, 1, tzinfo=UTC)
    body = security_txt(now=ahead)
    assert "Expires: 2030-06-30T00:00:00Z" in body


def test_the_route_is_the_well_known_path_and_carries_no_prefix() -> None:
    """It sits outside `/web`, which is the only prefix the session middleware guards."""
    assert [cast(Route, route).path for route in router.routes] == ["/.well-known/security.txt"]
