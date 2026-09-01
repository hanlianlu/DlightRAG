# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-level Agent Skill catalog queries for transports."""

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import agent_skills_root, owner_skills_root
from dlightrag.engine.agent.skills import SkillCatalog, owner_skill_root


def discover_skill_catalog(
    config: DlightragConfig,
    *,
    owner_id: str | None = None,
) -> SkillCatalog:
    """Discover the merged Agent Skill catalog for one viewer.

    The global tier is operator-provisioned; the owner tier carries the
    viewer's own published skills and takes precedence by name.
    """
    owner_root = (
        owner_skill_root(owner_skills_root(config), owner_id) if owner_id is not None else None
    )
    return SkillCatalog.discover(
        global_root=agent_skills_root(config),
        owner_root=owner_root,
    )


__all__ = ["discover_skill_catalog"]
