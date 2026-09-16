# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Assemble one research request from the run's memory under one capacity."""

import asyncio
from typing import Any

from dlightrag.engine.agent.context import ContextContribution, ContextProjector
from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.errors import AnswerInputOverflowError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.memory import standing_memory_message
from dlightrag.engine.answer.mode import resource_role
from dlightrag.engine.answer.prompts import agent_control_prompt, control_turn_instruction
from dlightrag.engine.answer.resources.converters import conversion_format
from dlightrag.engine.answer.resources.models import ResourceManifestEntry
from dlightrag.engine.rag.corpus.sources.source_contract import safe_source_filename


class ContextAssembler:
    """Build each turn of one request as an append-only transcript.

    Every request is the previous request plus new material. The Session fold only
    appends, evidence text is frozen into the Tool result that admitted it, and the
    control instruction rides after the transcript. That shape is what a provider
    prefix cache can reuse, and no request states a clock.

    The former shape re-packed the whole evidence ledger after the growing fold on
    every turn, which put the pack past every matched cache prefix: on this
    deployment one Run was billed at 0% cache hits on all nine turns, and the turns
    that did hit reused only the fold (7.7k-11k of a 76k-106k prompt) while the
    65k-90k pack was charged at the full input rate again.

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
    ) -> None:
        self._model_profile = model_profile
        self._context_policy = context_policy
        self._input_limit = context_policy.hard_input_limit(model_profile)
        self._history = history
        self._question = _question_message(query, query_images, resource_manifest)
        self._memory_text = memory_text
        self._contributions = contributions
        self._tool_guidance = tool_guidance
        self._profile_memory_write = profile_memory_write
        self._artifact_publication = artifact_publication
        self._control_instruction = control_turn_instruction(
            artifact_publication=artifact_publication
        )
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
        self._last_measured_had_pixels = _carries_pixels(messages)
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
        system = {
            "role": "system",
            "content": agent_control_prompt(
                profile_memory_write=self._profile_memory_write,
                artifact_publication=self._artifact_publication,
            ),
        }
        head = self._head(system, working.messages())
        memory_message = standing_memory_message(self._memory_text)
        tail: list[ContextContribution] = []
        if memory_message is not None:
            tail.append(
                ContextContribution(
                    source="profile.memory",
                    authority="profile",
                    messages=(memory_message,),
                )
            )
        visual_blocks = evidence.visual_blocks()
        if visual_blocks:
            tail.append(
                ContextContribution(
                    source="answer.evidence_images",
                    authority="evidence",
                    messages=({"role": "user", "content": visual_blocks},),
                    citable=True,
                )
            )
        if self._tool_guidance:
            tail.append(
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
        tail.extend(self._contributions)
        # The instruction is the only per-turn prose in the request and its last
        # message, so nothing after it can move and the last thing the model reads is
        # what to do next. It stays derived rather than durable: the reusable prefix
        # ends before it, which costs those few tokens per turn and nothing else.
        tail.append(
            ContextContribution(
                source="answer.control",
                authority="reference",
                messages=({"role": "user", "content": self._control_instruction},),
            )
        )
        return [*head, *ContextProjector().project(tail).messages]

    def _head(
        self,
        system: dict[str, Any],
        carried: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        contributions = [
            ContextContribution(
                source="answer.system",
                authority="system",
                messages=(system,),
                compressible=False,
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
        contributions.extend(
            (
                ContextContribution(
                    source="answer.question",
                    authority="user",
                    messages=(self._question,),
                    compressible=False,
                ),
                ContextContribution(
                    source="agent.session",
                    authority="working",
                    messages=tuple(carried),
                ),
            )
        )
        return list(ContextProjector().project(contributions).messages)

    def _check_input_tokens(self, input_tokens: int) -> None:
        if input_tokens > self._input_limit:
            raise AnswerInputOverflowError(
                "Research input exceeds the resolved model input limit: "
                f"{input_tokens} > {self._input_limit} estimated input tokens"
            )


def _carries_pixels(messages: list[dict[str, Any]]) -> bool:
    """Return whether any request message states pixels rather than only text."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"image_url", "input_image"}:
                return True
    return False


def _question_message(
    query: str,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
) -> dict[str, Any]:
    manifest = _resource_manifest_context(resource_manifest)
    if not query_images and not manifest:
        return {"role": "user", "content": query}
    content: list[dict[str, Any]] = [{"type": "text", "text": query}]
    if manifest:
        content.append({"type": "text", "text": manifest})
    content.extend(query_images or [])
    return {"role": "user", "content": content}


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
        "Use these opaque resource ids with read for text or view for pixels. They belong to "
        "this run: a handle or cursor printed by an earlier turn is historical, and that "
        "document must be attached again before it can be read or viewed here."
    )
    return "\n".join(lines)


__all__ = ["ContextAssembler"]
