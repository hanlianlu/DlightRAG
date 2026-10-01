# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Assemble one research request from the run's memory under one capacity."""

import asyncio
from collections.abc import Sequence
from typing import Any

from dlightrag.engine.agent.context import ContextContribution, ContextProjector
from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.continuation_handles import session_notes_message
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.memory import standing_memory_message
from dlightrag.engine.answer.mode import resource_role
from dlightrag.engine.answer.prompts import agent_control_prompt
from dlightrag.engine.answer.resources.converters import conversion_format
from dlightrag.engine.answer.resources.models import ResourceManifestEntry
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename
from dlightrag.engine.runtime.workspace import SessionNoteRecord


class ContextAssembler:
    """Build each turn of one request as an append-only transcript.

    Every request is the previous request plus new material, and the first request of
    a follow-up Run is the previous Run's last one plus new material. The Session fold
    only appends, evidence text is frozen into the Tool result that admitted it, and
    whatever one Run composes rides after the transcript. That shape is what a
    provider prefix cache can reuse, and no request states a clock.

    The former shape re-packed the whole evidence ledger after the growing fold on
    every turn, which put the pack past every matched cache prefix: on this
    deployment one Run was billed at 0% cache hits on all nine turns, and the turns
    that did hit reused only the fold (7.7k-11k of a 76k-106k prompt) while the
    65k-90k pack was charged at the full input rate again. The Run's question block
    then did the same across Runs from the other end: it sat in front of the Session
    fold, so each follow-up Run's first request reused only the system prompt and the
    Tool definitions (0.8k-7.8k of 70k-244k tokens), and it repeated a question the
    fold already held.

    The terminal in-loop assistant text is the Research answer; citation and source
    finalization remain deterministic outside model generation.
    """

    def __init__(
        self,
        *,
        model_profile: ModelProfile,
        context_policy: ContextPolicy = CONTEXT_POLICY,
        query: str,
        history: PriorTurns,
        query_images: list[dict[str, Any]] | None,
        resource_manifest: tuple[ResourceManifestEntry, ...],
        memory_text: str = "",
        contributions: tuple[ContextContribution, ...] = (),
        tool_guidance: tuple[str, ...] = (),
        profile_memory_write: bool = False,
        artifact_publication: bool = False,
        run_notes: bool = False,
        connection_tools: bool = False,
        session_notes: Sequence[SessionNoteRecord] = (),
        instructions: str = "",
    ) -> None:
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._input_limit = context_policy.hard_input_limit(model_profile)
        self._history = history
        #: The question as the Run's own User Entry states it.
        self._query = query
        #: Host instructions that hold for every request of this Session, stated with
        #: the system prompt: a Child's role and scratch directory, for one.
        self._instructions = instructions
        manifest = _resource_manifest_context(resource_manifest)
        #: What this Run adds to its question: the Resources it registered and the
        #: pixels the caller attached. The Session records the question's text, and
        #: none of this.
        self._question_material: tuple[dict[str, Any], ...] = (
            *(({"type": "text", "text": manifest},) if manifest else ()),
            *(query_images or ()),
        )
        self._memory_text = memory_text
        self._contributions = contributions
        self._tool_guidance = tool_guidance
        self._profile_memory_write = profile_memory_write
        self._artifact_publication = artifact_publication
        self._run_notes = run_notes
        self._connection_tools = connection_tools
        self._session_notes = tuple(session_notes)
        #: Provider-anchored estimator correction; see ``observe_provider_input``.
        self._estimated_bias_tokens = 0
        self._last_measured_tokens: int | None = None
        #: Whether the last measured request carried pixel blocks. The estimator
        #: charges nothing for them by design, so their tokens would otherwise be
        #: mistaken for the text undercount the anchor exists to correct.
        self._last_measured_had_pixels = False

    async def control_turn(
        self,
        *,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
    ) -> list[dict[str, Any]]:
        """Compose one request's messages. Tool-schema input is the caller's to add."""
        return await asyncio.to_thread(self._compose_control_turn, evidence, working)

    def measure_control_input(
        self,
        *,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
    ) -> int:
        """Measure the exact control-turn messages without enforcing the limit."""
        return estimate_messages_tokens(self._compose_control_turn(evidence, working))

    def accounted_input_tokens(
        self,
        *,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
    ) -> int:
        """Measure the request about to be sent, and remember it for the next anchor.

        Only request assembly calls this. The remembered measure is the one the
        provider is about to answer, so only this path may record it: a measurement
        of a request that is never sent (a compaction's accounted-before, for
        instance) would make the next anchor compare a billed count against
        something the provider never saw.
        """
        messages = self._compose_control_turn(evidence, working)
        measured = estimate_messages_tokens(messages)
        self._last_measured_tokens = measured
        self._last_measured_had_pixels = carries_pixels(messages)
        return measured + self._estimated_bias_tokens

    def corrected_input_tokens(
        self,
        *,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
    ) -> int:
        """Measure with the carried correction, without moving the next anchor."""
        measured = self.measure_control_input(evidence=evidence, working=working)
        return measured + self._estimated_bias_tokens

    def observe_provider_input(
        self,
        prompt_tokens: int | None,
        *,
        tool_schema_tokens: int = 0,
    ) -> None:
        """Anchor the estimator on the provider's own count for the last request.

        The character heuristic undercounts recorded Session content by a median of
        8% and up to 46%, and the reservation policy deliberately carries no
        safety margin for it, so the error has to be absorbed somewhere. The
        provider states the exact input it billed; the difference against what this
        assembler measured for that same request is the correction carried forward.

        A request that carried pixels is skipped: the estimator charges no tokens
        for image blocks by design, so its gap to the provider would measure the
        provider's image accounting rather than the text undercount. The correction
        never exceeds the raw estimate it corrects, so one bad anchor cannot more
        than double the accounted input. The pixel fact is read from the messages
        this assembler composed for that request, not from the provider's counters.
        """
        if (
            prompt_tokens is None
            or prompt_tokens <= 0
            or self._last_measured_tokens is None
            or self._last_measured_had_pixels
        ):
            return
        billed_text = prompt_tokens - tool_schema_tokens
        self._estimated_bias_tokens = min(
            max(0, billed_text - self._last_measured_tokens),
            self._last_measured_tokens,
        )

    def output_allowance(
        self,
        messages: list[dict[str, Any]],
        *,
        additional_input_tokens: int = 0,
    ) -> int | None:
        """Preflight one exact model request and return its provider output cap."""
        if additional_input_tokens < 0:
            raise ValueError("additional_input_tokens cannot be negative")
        input_tokens = (
            estimate_messages_tokens(messages)
            + additional_input_tokens
            + self._estimated_bias_tokens
        )
        self._check_input_tokens(input_tokens)
        return self._context_policy.output_allowance(
            self._model_profile,
            input_tokens=input_tokens,
        )

    def _compose_control_turn(
        self,
        evidence: EvidenceLedger,
        working: WorkingContextProjection,
    ) -> list[dict[str, Any]]:
        system = agent_control_prompt(
            profile_memory_write=self._profile_memory_write,
            artifact_publication=self._artifact_publication,
            run_notes=self._run_notes,
            connection_tools=self._connection_tools,
        )
        contributions = [
            ContextContribution(
                source="answer.system",
                authority="system",
                messages=(
                    {
                        "role": "system",
                        "content": f"{system}\n\n{self._instructions}"
                        if self._instructions
                        else system,
                    },
                ),
            )
        ]
        if self._history.episodic_summary:
            contributions.append(
                ContextContribution(
                    source="conversation.episodic",
                    authority="conversation",
                    messages=(
                        {
                            "role": "user",
                            "content": self._history.episodic_summary,
                        },
                    ),
                )
            )
        if self._history.messages:
            contributions.append(
                ContextContribution(
                    source="conversation.tail",
                    authority="conversation",
                    messages=tuple(self._history.messages),
                )
            )
        contributions.append(
            ContextContribution(
                source="agent.session",
                authority="working",
                messages=tuple(_stating_the_question(working, self._query)),
            )
        )
        notes = session_notes_message(self._session_notes) if self._run_notes else ""
        if notes:
            contributions.append(
                ContextContribution(
                    source="answer.session_notes",
                    authority="workspace",
                    messages=({"role": "user", "content": notes},),
                )
            )
        material = _material_message(self._question_material)
        if material is not None:
            contributions.append(
                ContextContribution(
                    source="answer.question",
                    authority="user",
                    messages=(material,),
                )
            )
        memory_message = standing_memory_message(self._memory_text)
        if memory_message is not None:
            contributions.append(
                ContextContribution(
                    source="profile.memory",
                    authority="profile",
                    messages=(memory_message,),
                )
            )
        visual_blocks = evidence.visual_blocks()
        if visual_blocks:
            contributions.append(
                ContextContribution(
                    source="answer.evidence_images",
                    authority="visual",
                    messages=({"role": "user", "content": visual_blocks},),
                )
            )
        if self._tool_guidance:
            contributions.append(
                ContextContribution(
                    source="answer.tools",
                    authority="reference",
                    messages=(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Tool usage guidance:\n"
                                    + "\n".join(self._tool_guidance),
                                }
                            ],
                        },
                    ),
                )
            )
        contributions.extend(self._contributions)
        # Authority order is the request order. The system prompt and the transcript
        # lead, and the transcript states the question once, in its place, so a later
        # request, of this Run or of the next one, extends an earlier one and a later
        # steer still follows the question. Everything this Run composes comes after
        # the transcript: the notes it holds and its
        # question's material, which differ from one Run to the next; memory, tool
        # guidance and skill context, byte-stable for the Run; and last the visual
        # lane, which re-renders per request (ADR 0015). Nothing is composed per turn:
        # the loop-termination guidance lives in the system prompt, and both reference
        # harnesses send no nudge either.
        return list(ContextProjector().project(contributions).messages)

    def _check_input_tokens(self, input_tokens: int) -> None:
        if input_tokens > self._input_limit:
            raise AnswerInputOverflowError(
                "Research input exceeds the resolved model input limit: "
                f"{input_tokens} > {self._input_limit} estimated input tokens"
            )


def carries_pixels(messages: list[dict[str, Any]]) -> bool:
    """Return whether any request message states pixels rather than only text."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"image_url", "input_image"}:
                return True
    return False


def _stating_the_question(
    working: WorkingContextProjection,
    query: str,
) -> list[dict[str, Any]]:
    """Return the transcript with the Run's question in its place, stated once.

    Acceptance records the question as the Run's own User Entry, so the transcript
    already states it, before the Tool work it asked for and before any steer or
    follow-up that came later. Where the transcript does not hold it — before that
    Entry exists, as in the acceptance measure, or once a compaction has covered it —
    it is restated where it stood: right after the summary that replaced the covered
    prefix, so a later message still has the last word.
    """
    transcript = working.messages()
    if any(
        message.get("role") == "user" and message.get("content") == query for message in transcript
    ):
        return transcript
    lead = 0 if working.summary is None else 1
    return [*transcript[:lead], {"role": "user", "content": query}, *transcript[lead:]]


def _material_message(material: tuple[dict[str, Any], ...]) -> dict[str, Any] | None:
    """Return what this Run adds to its question, or nothing when it adds nothing.

    The Resource manifest and the attached pixels belong to this Run alone and differ
    from one Run to the next, so they follow the transcript rather than sit in it.
    """
    if not material:
        return None
    if len(material) == 1 and material[0].get("type") == "text":
        return {"role": "user", "content": material[0]["text"]}
    return {"role": "user", "content": list(material)}


def _resource_manifest_context(manifest: tuple[ResourceManifestEntry, ...]) -> str:
    if not manifest:
        return ""
    lines = ["## Registered run-scoped Resources"]
    for entry in manifest:
        filename = safe_source_filename(entry.filename or "resource")
        mime = (entry.declared_mime or "").split(";", 1)[0].lower()
        format_name = conversion_format(filename, entry.declared_mime)
        if resource_role(filename=filename, mime_type=entry.declared_mime) == "image":
            kind = "image; view"
        elif format_name == "pdf":
            kind = "PDF; read text or view physical pages"
        elif format_name in {"docx", "pptx", "xlsx"}:
            kind = f"{format_name.upper()}; read extracted text and embedded-image inventory"
        else:
            kind = mime or "resource; type verified on acquisition"
        lines.append(f"- [resource: {entry.resource_id}] {filename} ({kind})")
    lines.append(
        "Use these opaque resource ids with read for text or view for pixels. A cursor "
        "printed by an earlier turn is historical; call read or view on the resource again "
        "for a current one. A resource id printed by an earlier turn may still resolve "
        "here; only if read or view refuses it must that document be attached again."
    )
    return "\n".join(lines)


__all__ = ["ContextAssembler", "carries_pixels"]
