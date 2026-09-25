# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-level Agent Skill slice for transports and composition."""

from pathlib import Path

from dlightrag.application.config import DlightragConfig
from dlightrag.application.settings import (
    agent_skills_root,
    disabled_builtin_skills,
    owner_skills_root,
)
from dlightrag.engine.agent.environment.confinement import DeclaredLayer
from dlightrag.engine.agent.skills import (
    SkillsBundleFactory,
    builtin_skills_root,
    owner_skill_root,
)


def skill_read_layers(config: DlightragConfig, *, owner_id: str) -> tuple[DeclaredLayer, ...]:
    """Return the Skill roots this owner's Agent processes read.

    A Skill may point at an executable asset, so the roots have to be readable from
    inside the confinement; the declaration lives here, beside the capability that
    needs it, rather than in a list the execution environment owns (ADR 0024). The
    owner's own shard is the unit, never the shared parent: one owner's skills are that
    owner's, and an Agent must not be able to read a sibling owner's.

    The packaged built-ins are absent on purpose: the serving process loads them
    through its own ``load_skill`` tool, and their assets are reachable where the
    install puts them under the runtime prefix.
    """
    return (
        DeclaredLayer(path=agent_skills_root(config), capability="skills (operator-global)"),
        DeclaredLayer(
            path=owner_skill_root(owner_skills_root(config), owner_id),
            capability="skills (this owner)",
        ),
    )


def skill_roots(config: DlightragConfig) -> tuple[tuple[Path, str], ...]:
    """Return every root Skill lookup may serve, for composition's refusal to check.

    Composition refuses each of these against the trees an Agent may never see, so a
    deployment that points a Skill root at its own project fails at startup rather
    than the first Run that happens to announce that owner.
    """
    return (
        (agent_skills_root(config), "skills (operator-global)"),
        (owner_skills_root(config), "skills (owner shards)"),
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

    return SkillsBundleFactory(
        builtin_root=builtin_root,
        global_root=global_root,
        owner_root=owner_root,
        disabled_builtin_skills=disabled_builtins,
    )


__all__ = ["skill_read_layers", "skill_roots", "skills_bundle_factory"]
