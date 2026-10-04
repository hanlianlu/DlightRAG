# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Same-origin owner-only Settings routes for Agent Accounts; a password is never returned.

Settings is the only surface for an owner's Agent Accounts, so these are Web routes and have no
REST counterpart. Every answer is the owner's whole view, which the page shows as it is.
"""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, StrictBool

from dlightrag.adapters.http.browser.deps import get_application
from dlightrag.application.access import owner_id_from_user

router = APIRouter(prefix="/agent-accounts")


class SettingsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    registration_enabled: StrictBool


def _owner(request: Request) -> str:
    return owner_id_from_user(request.state.user_context)


@router.get("")
async def read_agent_accounts(request: Request) -> dict[str, Any]:
    return asdict(await get_application(request).agent_accounts.view(owner_id=_owner(request)))


@router.put("/settings")
async def update_agent_account_settings(request: Request, body: SettingsInput) -> dict[str, Any]:
    view = await get_application(request).agent_accounts.set_registration(
        owner_id=_owner(request), enabled=body.registration_enabled
    )
    return asdict(view)


@router.delete("/{site}")
async def remove_agent_account(site: str, request: Request) -> dict[str, Any]:
    return asdict(
        await get_application(request).agent_accounts.remove(owner_id=_owner(request), site=site)
    )


__all__ = ["router"]
