# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-level Agent Skill slice for transports and composition."""

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import (
    agent_skills_root,
    disabled_builtin_skills,
    owner_skills_root,
)
from dlightrag.engine.agent.environment.confinement import DeclaredLayer
from dlightrag.engine.agent.skills import (
    SkillsBundle,
    SkillsBundleFactory,
    builtin_skills_root,
    owner_skill_root,
)


def skill_read_layers(config: DlightragConfig) -> tuple[DeclaredLayer, ...]:
    """Return the Skill roots an Agent's processes read, as confinement layers.

    A Skill may point at an executable asset, so the roots have to be readable from
    inside the confinement; the declaration lives here, beside the capability that
    needs it, rather than in a list the execution environment owns (ADR 0024). The
    packaged built-ins are absent on purpose: they already sit inside the
    interpreter prefix the runtime grants, and declaring them would ask the policy to
    accept a path inside the project tree on a checkout that runs from one.
    """
    return (
        DeclaredLayer(path=agent_skills_root(config), capability="skills (operator-global)"),
        DeclaredLayer(path=owner_skills_root(config), capability="skills (owner)"),
    )


def skills_bundle_factory(
    config: DlightragConfig,
    *,
    ensure_dirs: bool = False,
) -> SkillsBundleFactory:
    """Build the per-run skills slice for one application config.

    Resolves both roots once; composition passes ``ensure_dirs=True`` to create
    them eagerly, request paths leave it False.
    """
    builtin_root = builtin_skills_root()
    global_root = agent_skills_root(config)
    owner_root = owner_skills_root(config)
    disabled_builtins = disabled_builtin_skills(config)
    if ensure_dirs:
        global_root.mkdir(parents=True, exist_ok=True)
        owner_root.mkdir(parents=True, exist_ok=True)

    def build(owner_id: str, requested_skill: str | None = None) -> SkillsBundle:
        return SkillsBundle(
            builtin_root=builtin_root,
            global_root=global_root,
            owner_root=owner_skill_root(owner_root, owner_id),
            disabled_builtin_skills=disabled_builtins,
            requested_skill=requested_skill,
        )

    return build


__all__ = ["skill_read_layers", "skills_bundle_factory"]
