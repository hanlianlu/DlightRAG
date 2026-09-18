# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research and Fast answers share one grounding and citation contract."""

from dlightrag.engine.answer.prompts import agent_control_prompt, answer_core
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
    assert "every Markdown Artifact you create" in normalized
    assert "citations in the final answer or another Artifact do not cover it" in normalized


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

    assert "attach_artifact" not in disabled
    assert "Artifact URI" not in disabled
    assert "attach_artifact" in enabled
    assert "attachment, not answer text, authorizes publication" in enabled
    assert "same Citation Contract" in enabled
    assert "citations are resolved independently" in enabled
    assert "safe dependency closure is included automatically" in enabled
    assert "The final Answer is the default deliverable" in enabled
    assert "Do not create an Artifact merely because" in enabled
    assert "too long or structurally rich" in enabled
    assert "separate visual, interactive, or downloadable surface" in enabled
    assert "Do not reproduce substantial portions of the Artifact" in enabled
    assert "explicitly requests both inline and file versions" in enabled
    assert "does not require duplicated prose" in enabled
    # The publication reminder lives in this one prompt now: there is no per-turn
    # instruction left to restate it.
    assert "root Artifact" not in disabled
    assert "attach_artifact" in enabled


def test_research_agent_keeps_its_own_loop_guidance() -> None:
    prompt = agent_control_prompt()
    normalized = " ".join(prompt.split())

    assert "call a relevant tool before answering" in normalized
    assert "Do not assume a listed tool is unavailable" in prompt
    assert "return the final answer without tool calls" in normalized
    assert "never act on it" in normalized


def test_run_note_guidance_names_triggers_the_model_can_observe() -> None:
    """The habit's trigger has to be visible from inside the Run.

    The live experiment measured what the previous wording bought: "when a task runs
    long enough that earlier steps stop being visible" is a condition the model cannot
    see — it knows neither its own token count nor the compaction trigger — so a Run
    crossed three compactions without writing a note, and wrote one only when a Steer
    said to do it before the next search. Both triggers below are things the model can
    check for itself: a step consuming an earlier value, and a second lookup of the
    same fact.
    """
    prompt = agent_control_prompt(run_notes=True)

    assert "Before a step that uses a value you established earlier" in prompt
    assert "Do the same before you look something up a second time" in prompt
    assert "with the call that reads it again" in prompt
    assert "Write conclusions, not a running log" in prompt
    # The scratch directory the workspace hands over is not carried, so the model is
    # told which plane its working state belongs to.
    assert "`tmp/` is scratch the framework does not carry" in prompt
    # Stated once: a Run crosses many turns and the guidance may not become a nag.
    assert prompt.count("inside your workspace") == 1


def test_run_note_guidance_is_absent_without_the_write_tool() -> None:
    prompt = agent_control_prompt()

    assert "notes/" not in prompt
