# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-level Agent Skill slice for transports and composition."""

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import agent_skills_root, owner_skills_root
from dlightrag.engine.agent.skills import SkillsBundle, SkillsBundleFactory, owner_skill_root


def skills_bundle_factory(
    config: DlightragConfig,
    *,
    ensure_dirs: bool = False,
) -> SkillsBundleFactory:
    """Build the per-run skills slice for one application config.

    Resolves both roots once; composition passes ``ensure_dirs=True`` to create
    them eagerly, request paths leave it False.
    """
    global_root = agent_skills_root(config)
    owner_root = owner_skills_root(config)
    if ensure_dirs:
        global_root.mkdir(parents=True, exist_ok=True)
        owner_root.mkdir(parents=True, exist_ok=True)

    def build(owner_id: str, requested_skill: str | None = None) -> SkillsBundle:
        return SkillsBundle(
            global_root=global_root,
            owner_root=owner_skill_root(owner_root, owner_id),
            requested_skill=requested_skill,
        )

    return build


__all__ = ["skills_bundle_factory"]
