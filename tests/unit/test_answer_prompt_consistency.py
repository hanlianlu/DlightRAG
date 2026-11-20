# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research and Fast answers share one citation contract and one set of link rules.

The prompts are provider prefix-cache input, so what these tests hold is structure: which section
sits in which prompt, what a capability adds, and that nothing else moves. The wording of a section
is prose and is not pinned here.
"""

import pytest

from dlightrag.engine.answer.goal import BLOCKED_AFTER_ROUNDS, DEFAULT_MAX_ROUNDS
from dlightrag.engine.answer.prompts.agent import (
    _ARTIFACT_PUBLICATION_GUIDANCE,
    _CONNECTION_GUIDANCE,
    _GOAL_GUIDANCE,
    _MCP_RESOURCE_GUIDANCE,
    _PROFILE_MEMORY_GUIDANCE,
    _RUN_NOTE_GUIDANCE,
    agent_control_prompt,
)
from dlightrag.engine.answer.prompts.answer import (
    CITATION_GUIDANCE,
    EVIDENCE_USE_GUIDANCE,
    PRESENTATION_GUIDANCE,
    fast_answer_prompt,
)
from dlightrag.engine.answer.prompts.identity import core_self_conception
from dlightrag.engine.answer.tools.goals import goal_tool_declarations

_CAPABILITY_SECTIONS = {
    "connection_tools": _CONNECTION_GUIDANCE,
    "mcp_resources": _MCP_RESOURCE_GUIDANCE,
    "artifact_publication": _ARTIFACT_PUBLICATION_GUIDANCE,
    "run_notes": _RUN_NOTE_GUIDANCE,
    "profile_memory_write": _PROFILE_MEMORY_GUIDANCE,
    "goal_tools": _GOAL_GUIDANCE,
}


def test_both_paths_share_the_citation_evidence_and_link_rules() -> None:
    fast = fast_answer_prompt()
    research = agent_control_prompt()

    for shared in (CITATION_GUIDANCE, EVIDENCE_USE_GUIDANCE, PRESENTATION_GUIDANCE):
        assert shared in fast
        assert shared in research


def test_both_paths_open_with_one_self_conception_block() -> None:
    """Identity and the philosophical stance are one block, composed once for both paths."""
    assert fast_answer_prompt().startswith(f"{core_self_conception(environment_clock=False)}\n\n")
    assert agent_control_prompt().startswith(f"{core_self_conception(environment_clock=True)}\n\n")


@pytest.mark.parametrize("capability", _CAPABILITY_SECTIONS)
def test_a_capability_adds_exactly_its_guidance_section_and_changes_no_other_byte(
    capability: str,
) -> None:
    """A Run without the capability keeps the exact bytes its provider prefix cache holds."""
    section = _CAPABILITY_SECTIONS[capability]
    without = agent_control_prompt()
    with_capability = agent_control_prompt(**{capability: True})

    assert section not in without
    assert with_capability.replace("\n\n" + section, "", 1) == without


def test_a_run_that_cannot_publish_an_artifact_is_not_told_about_them() -> None:
    assert "Artifact" not in agent_control_prompt()
    assert "Artifact" not in fast_answer_prompt()
    assert "attach_artifact" in _ARTIFACT_PUBLICATION_GUIDANCE


def test_the_goal_guidance_names_each_goal_tool_and_the_live_round_limits() -> None:
    for declaration in goal_tool_declarations():
        assert declaration.name in _GOAL_GUIDANCE
    assert str(BLOCKED_AFTER_ROUNDS) in _GOAL_GUIDANCE
    assert str(DEFAULT_MAX_ROUNDS) in _GOAL_GUIDANCE
