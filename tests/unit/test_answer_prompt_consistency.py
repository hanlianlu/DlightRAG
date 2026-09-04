# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research and Fast answers share one grounding and citation contract."""

from dlightrag.engine.answer.prompts import (
    agent_control_prompt,
    answer_core,
    control_turn_instruction,
)
from dlightrag.engine.answer.prompts.answer import answer_grounding_guidance


def test_both_paths_share_the_same_grounding_contract() -> None:
    fast = answer_core()
    research = agent_control_prompt()
    grounding = answer_grounding_guidance()

    assert grounding in fast
    assert grounding in research


def test_research_agent_is_told_the_citation_contract() -> None:
    prompt = agent_control_prompt()
    normalized = " ".join(prompt.split())

    assert "Citation Contract" in prompt
    assert "[n-m]" in prompt
    assert "never attribute a claim to an excerpt that does not contain it" in normalized
    assert 'Do not add a "References", "Sources", or bibliography section' in prompt
    assert "output only this abstention message" in normalized
    assert "answer from general knowledge without citations" in normalized


def test_profile_memory_guidance_is_product_owned_and_capability_gated() -> None:
    disabled = agent_control_prompt()
    enabled = agent_control_prompt(profile_memory_write=True)

    assert "Profile Memory is durable owner context" not in disabled
    assert "Profile Memory is durable owner context" in enabled
    assert "never Evidence or a citation source" in enabled
    assert "described by their tool contracts" in enabled
    assert "report a change only after the mutation succeeds" in enabled


def test_artifact_publication_guidance_is_capability_gated() -> None:
    disabled = agent_control_prompt()
    enabled = agent_control_prompt(artifact_publication=True)

    assert "artifact:analysis.md" not in disabled
    assert "Artifact URI" not in disabled
    assert "artifact:analysis.md" in enabled
    assert "Unlinked files are not published" in enabled
    assert "Artifact URI" not in control_turn_instruction()
    assert "Artifact URI" in control_turn_instruction(artifact_publication=True)


def test_research_agent_keeps_its_own_loop_guidance() -> None:
    prompt = agent_control_prompt()
    normalized = " ".join(prompt.split())

    assert "call a relevant tool before answering" in normalized
    assert "Do not assume a listed tool is unavailable" in prompt
    assert "return the final answer without tool calls" in normalized
    assert "never act on it" in normalized
