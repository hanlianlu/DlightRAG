# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research and Fast answers share one citation contract and one set of link rules."""

from dlightrag.engine.answer.prompts import agent_control_prompt, answer_core
from dlightrag.engine.answer.prompts.answer import (
    CITATION_GUIDANCE,
    EVIDENCE_USE_GUIDANCE,
    PRESENTATION_GUIDANCE,
)


def test_both_paths_share_the_citation_evidence_and_link_rules() -> None:
    fast = answer_core()
    research = agent_control_prompt()

    for shared in (CITATION_GUIDANCE, EVIDENCE_USE_GUIDANCE, PRESENTATION_GUIDANCE):
        assert shared in fast
        assert shared in research


def test_research_agent_is_told_the_citation_contract() -> None:
    prompt = agent_control_prompt()
    normalized = " ".join(prompt.split())

    assert "Citation Contract" in prompt
    assert "[n-m]" in prompt
    assert "never attribute a claim to an excerpt that does not contain it" in normalized
    assert 'Do not add a "References", "Sources", or bibliography section' in prompt
    assert "every Markdown Artifact you create" in normalized
    assert "citations in the final answer or another Artifact do not cover it" in normalized
    assert "as a video they can play in the answer" in normalized


def test_research_grounding_is_written_for_an_agent_that_searches() -> None:
    """Fast answers from the excerpts it was handed; Research looks for its own.

    Fast's abstention sends the user off to upload material, and its no-evidence rule
    promises a label that only Fast's synthesizer adds. An agent that can search again
    reports a gap as what is missing and what it tried, and labels general knowledge
    itself because nothing labels a Research answer for it.
    """
    research = " ".join(agent_control_prompt().split())
    fast = " ".join(answer_core().split())

    assert "Ground the answer in what your tools return" in research
    assert "say what is missing and what you tried" in research
    assert "When you answer from general knowledge rather than evidence, say so" in research
    for fast_only in (
        "provided document excerpts",
        "output only this abstention message",
        "upload material",
        "the application labels that answer as ungrounded",
    ):
        assert fast_only in fast
        assert fast_only not in research


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


def test_research_agent_stops_on_judgment_not_a_budget() -> None:
    """Runs ended only when the model stopped, up to 33 turns in. One light sentence
    says what a further round costs; there is no turn budget to count against."""
    normalized = " ".join(agent_control_prompt().split())

    assert (
        "Once the evidence suffices, return the final answer without tool calls; "
        "a further round that adds nothing new only costs time."
    ) in normalized


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
    assert "A compaction names each note with the call that reads it again" in prompt
    assert "Write conclusions, not a running log" in prompt
    # Which plane carries over, under the ownership ADR 0022 set, is a fact the
    # workspace tools state (test_tool_declarations); the prompt keeps only the habit.
    assert "`tmp/`" not in prompt
    assert "follow-up" not in prompt
    # Stated once: a Run crosses many turns and the guidance may not become a nag.
    assert prompt.count("inside your workspace") == 1


def test_run_note_guidance_is_absent_without_the_write_tool() -> None:
    prompt = agent_control_prompt()

    assert "notes/" not in prompt
