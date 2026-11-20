# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The process-wide holder of the reader content gate, over a real Skill root."""

import json
from pathlib import Path

import pytest

from dlightrag.application.skills import SkillProtection
from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER
from dlightrag.engine.agent.skills import SkillsBundleFactory

PARAGRAPH = (
    "A driver moves the value only through a mechanism: the volume falls, the fixed cost "
    "spreads over fewer units, and the margin compresses before the price can answer it."
)


def _factory(global_root: Path) -> SkillsBundleFactory:
    return SkillsBundleFactory(builtin_root=None, global_root=global_root, owner_root=None)


ROUTING_DESCRIPTION = (
    "Determine whether a listed company is worth buying, and state the operating mechanism "
    "behind every conclusion the analysis reaches, with the evidence it rests on."
)
"""A long description: the routing surface an agent may legitimately repeat."""


def _write_skill(root: Path, text: str = PARAGRAPH) -> None:
    references = root / "sample" / "references"
    references.mkdir(parents=True)
    (root / "sample" / "SKILL.md").write_text(
        f"---\nname: sample\ndescription: {ROUTING_DESCRIPTION}\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (references / "method.md").write_text(text, encoding="utf-8")


def test_masks_a_global_skills_document(tmp_path: Path) -> None:
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path))
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER


def test_the_routing_surface_is_not_masked_and_the_body_is(tmp_path: Path) -> None:
    """The description the catalog publishes is public; the document's own words are not.

    Where the frontmatter ends is the catalog's rule, and this asserts the index honours it:
    a second copy of that delimiter rule is how a public description would eventually be
    replaced in an answer that legitimately repeats it.
    """
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path))
    assert protection.mask(ROUTING_DESCRIPTION) == ROUTING_DESCRIPTION
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER


def test_a_root_with_no_skills_is_the_identity(tmp_path: Path) -> None:
    protection = SkillProtection(_factory(tmp_path))
    assert not protection.index
    assert protection.mask(PARAGRAPH) == PARAGRAPH


def test_a_revalidation_picks_up_an_edited_document(tmp_path: Path) -> None:
    """A document changed on disk is protected as it now reads, not as it read at boot."""
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path), revalidate_seconds=0.0)
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER
    replacement = (
        "A different procedure replaces the first one: it states the observation the "
        "conclusion would fail on, and names the period that observation belongs to."
    )
    (tmp_path / "sample" / "references" / "method.md").write_text(replacement, encoding="utf-8")
    assert protection.mask(replacement) == MASK_PLACEHOLDER
    assert protection.mask(PARAGRAPH) == PARAGRAPH


def test_one_index_serves_repeated_calls(tmp_path: Path) -> None:
    """The built index is a process resource: the second call does not build it again."""
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path))
    first = protection.index
    assert protection.index is first


def test_an_unreadable_root_keeps_the_index_it_had(tmp_path: Path) -> None:
    """A directory that cannot be listed is not evidence that nothing is protected."""
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path), revalidate_seconds=0.0)
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER

    def unreadable() -> None:
        return None

    protection._root_signature = unreadable  # type: ignore[method-assign]
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER


def test_a_failed_rebuild_does_not_publish_a_root_it_did_not_read(tmp_path: Path) -> None:
    """A rebuild that failed must leave the signature alone so the next window retries."""
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path), revalidate_seconds=0.0)
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER

    failing = protection._build

    def failed(*, builtin_only: bool = False) -> None:
        return None

    protection._build = failed  # type: ignore[method-assign]
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER
    protection._build = failing  # type: ignore[method-assign]

    added = tmp_path / "added" / "references"
    added.mkdir(parents=True)
    (tmp_path / "added" / "SKILL.md").write_text("---\nname: added\ndescription: x.\n---\n")
    (added / "other.md").write_text(PARAGRAPH, encoding="utf-8")
    assert protection.mask("A passage nobody indexed yet: " + PARAGRAPH) == (
        "A passage nobody indexed yet: " + MASK_PLACEHOLDER
    )


def test_a_failed_first_build_does_not_cache_an_unmasked_answer(tmp_path: Path) -> None:
    """A gate that could not see its documents must not teach the cache that nothing matches."""
    _write_skill(tmp_path)
    protection = SkillProtection(_factory(tmp_path), revalidate_seconds=0.0)
    real_build = protection._build

    def failed(*, builtin_only: bool = False) -> None:
        return None

    protection._build = failed  # type: ignore[method-assign]
    assert protection.mask(PARAGRAPH) == PARAGRAPH
    protection._build = real_build  # type: ignore[method-assign]
    assert protection.mask(PARAGRAPH) == MASK_PLACEHOLDER


SCRIPT_TEXT = (
    "a MODELLED tail, not a measured one: it is the simulated distribution's own tail, "
    "not a frequency measured on this issuer, and model error is not estimated from this "
    "sample. Set against it the checks that can be run: the implied discount rate against "
    "the market-implied set, a second method on a different basis, and the pre-committed "
    "observations that score the miss."
)
"""Engine code the Skill writes into its case; not prose any reference repeats."""


def _write_skill_with_kit(root: Path) -> None:
    _write_skill(root)
    skill = root / "sample"
    (skill / "SKILL.md").write_text(
        "---\nname: sample\nkit: scripts\ndescription: x.\n---\n\nBody.\n",
        encoding="utf-8",
    )
    (skill / "scripts").mkdir()
    (skill / "scripts" / "engine.py").write_text(SCRIPT_TEXT, encoding="utf-8")


def test_reader_masks_script_prose_while_publication_text_excludes_the_declared_kit(
    tmp_path: Path,
) -> None:
    _write_skill_with_kit(tmp_path)
    protection = SkillProtection(_factory(tmp_path))
    assert protection.mask(SCRIPT_TEXT) == MASK_PLACEHOLDER
    assert protection.publication_index.scan(SCRIPT_TEXT) == ()


def test_artifact_text_gate_accepts_engine_case_text_but_reader_index_would_refuse(
    tmp_path: Path,
) -> None:
    from dlightrag.engine.answer.publication import (
        ArtifactValidationError,
        prepare_artifact_attachment,
    )

    _write_skill_with_kit(tmp_path)
    protection = SkillProtection(_factory(tmp_path))
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "case.json").write_text(json.dumps({"label": SCRIPT_TEXT}), encoding="utf-8")

    attachment = prepare_artifact_attachment(
        root,
        path="case.json",
        protected_index=lambda: protection.publication_index,
    )
    assert attachment.relative_path == "case.json"
    with pytest.raises(ArtifactValidationError):
        prepare_artifact_attachment(
            root,
            path="case.json",
            protected_index=lambda: protection.index,
        )
