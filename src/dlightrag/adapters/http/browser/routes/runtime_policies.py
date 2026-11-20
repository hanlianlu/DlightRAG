# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Browser-session adapter for the Runtime Policies an administrator edits."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Request

from dlightrag.adapters.http.browser.deps import enforce_web_access, get_application
from dlightrag.application.access.control import AccessAction
from dlightrag.application.answer_runs.child_roster import stamp
from dlightrag.application.policies.service import PolicyState

router = APIRouter()


def _actor(request: Request) -> str:
    user = getattr(request.state, "user_context", None)
    return str(getattr(user, "user_id", None) or "anonymous")


def _view(state: PolicyState) -> dict[str, Any]:
    return {
        "key": state.key,
        "policy": state.document.model_dump(mode="json"),
        "updated_at": stamp(state.updated_at),
        "updated_by": state.updated_by,
    }


@router.get("/runtime-policies/{key}")
async def read_runtime_policy(key: str, request: Request) -> dict[str, Any]:
    """A policy as it stands. It lists the people it targets, so only a manager reads it."""
    await enforce_web_access(request, AccessAction.RUNTIME_POLICY_MANAGE, None)
    return _view(await get_application(request).policies.read(key))


@router.put("/runtime-policies/{key}")
async def write_runtime_policy(
    key: str, document: Annotated[dict[str, Any], Body()], request: Request
) -> dict[str, Any]:
    """Replace a policy's document whole. It holds from the next request on."""
    await enforce_web_access(request, AccessAction.RUNTIME_POLICY_MANAGE, None)
    return _view(await get_application(request).policies.write(key, document, _actor(request)))


__all__ = ["router"]
