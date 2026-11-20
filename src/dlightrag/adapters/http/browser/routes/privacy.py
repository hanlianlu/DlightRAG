# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin owner-only Settings routes for Conversation Contribution (ADR 0041).

The setting is the owner's own state, like their Memory and their Agent Accounts, so no access
action guards it, and Settings is its only surface: there is no REST or MCP counterpart. Every
answer is the setting as it is kept.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, StrictBool

from dlightrag.adapters.http.browser.deps import get_application
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.answer_runs.child_roster import stamp
from dlightrag.application.privacy import PrivacySettings

router = APIRouter(prefix="/privacy")


class PrivacySettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contribute: StrictBool


class PrivacySettingsView(BaseModel):
    contribute: bool
    since: str | None
    """When the permission began, UTC ISO 8601 with a ``Z``; null while it is off."""


def _owner(request: Request) -> str:
    return owner_id_from_user(request.state.user_context)


def _view(settings: PrivacySettings) -> PrivacySettingsView:
    return PrivacySettingsView(contribute=settings.contribute, since=stamp(settings.since))


@router.get("/settings", response_model=PrivacySettingsView)
async def read_privacy_settings(request: Request) -> PrivacySettingsView:
    return _view(await get_application(request).privacy.settings(owner_id=_owner(request)))


@router.put("/settings", response_model=PrivacySettingsView)
async def update_privacy_settings(
    request: Request, body: PrivacySettingsInput
) -> PrivacySettingsView:
    return _view(
        await get_application(request).privacy.set_contribute(
            owner_id=_owner(request), contribute=body.contribute
        )
    )


__all__ = ["router"]
