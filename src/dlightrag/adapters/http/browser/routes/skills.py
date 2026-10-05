# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web routes exposing the discovered Agent Skill catalog and an owner's own Skills.

The catalog lists names and descriptions only: skill documents themselves stay reachable
through the answer agent's ``load_skill`` tool inside an authorized run. It merges
packaged built-ins, operator-global skills, and the current user's own published skills,
less the ones the user turned off.

Settings manages that user's own Skills and nothing else: it lists them on or off, turns
one off or on, reads its document, and deletes it. Another tier's Skills and another
owner's are never named, so a name that is not the viewer's own is simply not found.
"""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import StrictBool

from dlightrag.adapters.http.browser.deps import (
    enforce_web_access,
    get_application,
    get_workspace,
)
from dlightrag.application.access import AccessAction, owner_id_from_user
from dlightrag.application.skills import owner_skills_directory, skills_bundle_factory
from dlightrag.engine.agent.skills import (
    OWNER_MAX_SKILLS,
    OwnerSkill,
    delete_owner_skill,
    list_owner_skills,
    read_owner_skill,
    set_owner_skill_enabled,
)
from dlightrag.engine.answer.client_contracts import ClientContractModel

router = APIRouter()


class OwnSkillEnabledInput(ClientContractModel):
    enabled: StrictBool


def require_known_skill(application: Any, owner_id: str, name: str) -> str:
    """Validate one requested skill against the viewer's three-tier catalog.

    A Skill the owner turned off, and that no lower tier serves, is refused as disabled
    rather than unknown, since Settings is where it comes back.
    """
    catalog = skills_bundle_factory(application.config)(owner_id).catalog()
    if catalog is not None:
        if name in {skill.name for skill in catalog.metadata}:
            return name
        if name in catalog.disabled:
            raise ValueError(f"Agent Skill {name} is disabled. Turn it on in Settings to use it.")
    raise ValueError(f"Unknown Agent Skill: {name}")


@router.get("/skills")
async def list_skills(
    request: Request,
    workspace: str = Depends(get_workspace),
) -> dict[str, Any]:
    """List discovered Agent Skills (metadata only) for slash autocomplete."""
    await enforce_web_access(request, AccessAction.WORKSPACE_QUERY, workspace)
    application = get_application(request)
    user = getattr(request.state, "user_context", None)
    owner_id = owner_id_from_user(user)
    catalog = skills_bundle_factory(application.config)(owner_id).catalog()
    return {
        "skills": [
            {
                "name": skill.name,
                "description": skill.description,
                "source": skill.source,
            }
            for skill in (() if catalog is None else catalog.metadata)
        ]
    }


def _own_skills(request: Request) -> Path:
    """The viewer's own Skill directory, the only one these routes ever act on."""
    owner_id = owner_id_from_user(getattr(request.state, "user_context", None))
    return owner_skills_directory(get_application(request).config, owner_id=owner_id)


def _row(skill: OwnerSkill) -> dict[str, Any]:
    return {"name": skill.name, "description": skill.description, "enabled": skill.enabled}


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent Skill not found")


@router.get("/skills/mine")
async def list_own_skills(request: Request) -> dict[str, Any]:
    """The viewer's own Skills, on or off, and how many they may keep."""
    return {
        "skills": [_row(skill) for skill in list_owner_skills(_own_skills(request))],
        "limit": OWNER_MAX_SKILLS,
    }


@router.put("/skills/mine/{name}/enabled")
async def set_own_skill_enabled(
    name: str, body: OwnSkillEnabledInput, request: Request
) -> dict[str, Any]:
    skill = set_owner_skill_enabled(_own_skills(request), name, body.enabled)
    if skill is None:
        raise _not_found()
    return _row(skill)


@router.get("/skills/mine/{name}/document")
async def read_own_skill_document(name: str, request: Request) -> Response:
    """One of the viewer's Skills as the text it is, for them to read and never to run."""
    document = read_owner_skill(_own_skills(request), name)
    if document is None:
        raise _not_found()
    return Response(
        content=document,
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.delete("/skills/mine/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_own_skill(name: str, request: Request) -> None:
    if not delete_owner_skill(_own_skills(request), name):
        raise _not_found()
