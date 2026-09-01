# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-level Agent Skill catalog queries for transports."""

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import agent_skills_root
from dlightrag.engine.agent.skills import SkillCatalog


def discover_global_skill_catalog(config: DlightragConfig) -> SkillCatalog:
    """Discover the global Agent Skill catalog for one application config.

    Workspace skills are per-run and cannot exist before a run, so transports
    list the global root only.
    """
    return SkillCatalog.discover(
        global_root=agent_skills_root(config),
        workspace_root=None,
    )


__all__ = ["discover_global_skill_catalog"]
