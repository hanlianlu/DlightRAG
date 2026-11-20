# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""`/.well-known/security.txt`: where to report a vulnerability (RFC 9116).

A hosted service is expected to answer this path without a session, so the route
lives outside the Web cookie middleware's reach (`/web*` only) and states a
mailbox rather than a form. `Expires` is computed per request instead of frozen
in a file: a stale security.txt is worse than none, because a reporter who reads
an expired one concludes nobody is listening.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

router = APIRouter()

#: Where a report goes, and the languages it may be written in.
SECURITY_CONTACT = "https://github.com/hanlianlu/DlightRAG/security/advisories/new"
_PREFERRED_LANGUAGES = "en, zh"
#: RFC 9116 wants a future date and a short one: a deployment is reminded to
#: review the file when this rolls over, rather than trusting a file forever.
_VALIDITY_DAYS = 180


def security_txt(now: datetime | None = None) -> str:
    """The document's body: a contact, an expiry and the languages we read."""
    expires = (now or datetime.now(UTC)) + timedelta(days=_VALIDITY_DAYS)
    return (
        f"Contact: {SECURITY_CONTACT}\n"
        f"Expires: {expires.strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"Preferred-Languages: {_PREFERRED_LANGUAGES}\n"
    )


@router.get("/.well-known/security.txt", include_in_schema=False)
async def read_security_txt() -> PlainTextResponse:
    """Answer the well-known path in the type the standard names."""
    return PlainTextResponse(security_txt(), media_type="text/plain; charset=utf-8")


__all__ = ["SECURITY_CONTACT", "read_security_txt", "router", "security_txt"]
