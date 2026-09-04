# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research acceptance measurement shared without importing the orchestrator."""

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.agent.tools import AgentTool
from dlightrag.engine.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.images import AnswerImageBudget
from dlightrag.engine.answer.research.context import ContextAssembler
from dlightrag.engine.answer.resources.models import ResourceManifestEntry


def research_history_input_measure(
    *,
    model_profile: ModelProfile,
    context_policy: ContextPolicy,
    query: str,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
    image_budget: AnswerImageBudget | None,
    tools: list[AgentTool],
    retained_tail_tokens: int,
    memory_text: str = "",
    episodic_summary: str = "",
) -> Callable[..., int]:
    """Return the exact zero-evidence Research seed serializer used at acceptance."""
    tool_schema_tokens = estimate_tokens(
        json.dumps(
            [asdict(tool.definition) for tool in tools],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    tool_guidance = tuple(f"- {tool.guidance}" for tool in tools if tool.guidance)

    def measure(
        history: list[dict[str, Any]],
        projected_summary: str = "",
    ) -> int:
        context = ContextAssembler(
            model_profile=model_profile,
            context_policy=context_policy,
            query=query,
            history=PriorTurns(
                history,
                episodic_summary="\n\n".join(
                    part for part in (episodic_summary, projected_summary) if part.strip()
                ),
            ),
            query_images=query_images,
            resource_manifest=resource_manifest,
            memory_text=memory_text,
            tool_guidance=tool_guidance,
            profile_memory_write=any(tool.name == "remember" for tool in tools),
            artifact_publication=any(tool.name == "write" for tool in tools),
        )
        return (
            context.measure_control_input(
                evidence=EvidenceLedger(image_budget=image_budget),
                working=WorkingContextProjection(retained_tail_tokens=retained_tail_tokens),
            )
            + tool_schema_tokens
        )

    return measure


__all__ = ["research_history_input_measure"]
