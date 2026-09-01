# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes exposing the discovered Agent Skill catalog.

Names and descriptions only: skill documents themselves stay reachable through
the answer agent's ``load_skill`` tool inside an authorized run. The catalog
merges operator-global skills with the current user's own published skills.
"""

from typing import Any

from fastapi import APIRouter, Depends, Request

from dlightrag.adapters.http.browser.deps import (
    enforce_web_access,
    get_application,
    get_workspace,
)
from dlightrag.application.access import AccessAction, owner_id_from_user
from dlightrag.application.skills import discover_skill_catalog

router = APIRouter()


def require_known_skill(application: Any, owner_id: str, name: str) -> str:
    """Validate one requested skill against the viewer's merged catalog."""
    catalog = discover_skill_catalog(application.config, owner_id=owner_id)
    if name not in {skill.name for skill in catalog.metadata}:
        raise ValueError(f"Unknown Agent Skill: {name}")
    return name


@router.get("/skills")
async def list_skills(
    request: Request,
    workspace: str = Depends(get_workspace),
) -> dict[str, Any]:
    """List discovered Agent Skills (metadata only) for slash autocomplete."""
    await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace)
    application = get_application(request)
    user = getattr(request.state, "user_context", None)
    catalog = discover_skill_catalog(application.config, owner_id=owner_id_from_user(user))
    return {
        "skills": [
            {
                "name": skill.name,
                "description": skill.description,
                "source": skill.source,
            }
            for skill in catalog.metadata
        ]
    }
