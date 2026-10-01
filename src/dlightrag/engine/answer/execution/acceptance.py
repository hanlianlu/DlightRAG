# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The history budget one Answer Run is accepted and executed under.

Every model call an accepted Run can reach is one :class:`HistoryProjectionTarget`:
the model profile it runs on, the exact zero-evidence serializer of its request,
and whether it compacts proactively. Acceptance projects the caller's history
across the calls of every mode the Run may still resolve to; a Fast Run measures
its durable Session history against the same Fast calls before it compacts. Both
build the calls here, so which calls exist and how each one is measured is stated
once rather than remembered by each side.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.agent.tools import ToolDeclaration
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ContextPolicy, ModelProfile
from dlightrag.engine.ai.settings import ChatModelSelector
from dlightrag.engine.ai.tokens import estimate_tokens
from dlightrag.engine.answer.errors import AnswerInputOverflowError, UnsupportedAnswerModeError
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution.connection_binding import is_connection_tool
from dlightrag.engine.answer.execution.input import AttachmentReference
from dlightrag.engine.answer.history import (
    HistoryInputMeasure,
    HistoryProjectionOverflowError,
    HistoryProjectionTarget,
    project_history,
)
from dlightrag.engine.answer.images import AnswerImageBudget, AnswerImagePolicy
from dlightrag.engine.answer.memory import standing_memory_for_acceptance
from dlightrag.engine.answer.mode import AnswerMode, ModeResource, ResolvedMode, resource_role
from dlightrag.engine.answer.research.context import ContextAssembler
from dlightrag.engine.answer.resources.models import ResourceManifestEntry
from dlightrag.engine.answer.router import AnswerModeRouter
from dlightrag.engine.answer.synthesizer import AnswerSynthesizer

if TYPE_CHECKING:
    from dlightrag.engine.rag.retrieval.planner import RetrievalPlanner

logger = logging.getLogger(__name__)

#: The model calls a Run can reach, by the name a projection overflow and a Fast
#: compaction trace report them under.
FAST_PLANNER = "fast_planner"
FAST_GENERATION = "fast_generation"
RESEARCH_PLANNER = "research_planner"
RESEARCH_SEED = "research_seed"
ROUTER = "router"


class RetrievalPlanning(Protocol):
    """The retrieval planner and metadata schema a Run's planning call uses."""

    def planner_for(self, model_profile: ModelProfile | None = None) -> RetrievalPlanner: ...

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]: ...


def reserved_memory_text(*, auth_mode: str, enabled: bool) -> str:
    """The standing memory block a Run reserves room for before recall runs.

    Execution injects at most this block, so acceptance and execution measure with
    it: reserving room recall does not use is safe, under-reserving spends the
    difference on evidence the request can no longer hold.
    """
    return standing_memory_for_acceptance(auth_mode) if enabled else ""


def routing_resources(
    attachments: Sequence[AttachmentReference],
    history_attachments: Sequence[AttachmentReference],
) -> tuple[ModeResource, ...]:
    """The uploads the ``auto`` router is told about: this turn's, then earlier ones."""
    return tuple(
        ModeResource(role=resource_role(filename=item.filename, mime_type=item.mime_type))
        for item in (*attachments, *history_attachments)
    )


def research_history_input_measure(
    *,
    model_profile: ModelProfile,
    context_policy: ContextPolicy,
    query: str,
    query_images: list[dict[str, Any]] | None,
    resource_manifest: tuple[ResourceManifestEntry, ...],
    image_budget: AnswerImageBudget | None,
    tools: Sequence[ToolDeclaration],
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
            artifact_publication=any(tool.name == "attach_artifact" for tool in tools),
            # The measurement must carry the same habit the run's own request does.
            run_notes=any(tool.name == "write" for tool in tools),
            connection_tools=any(is_connection_tool(tool.name) for tool in tools),
        )
        return (
            context.measure_control_input(
                evidence=EvidenceLedger(image_budget=image_budget),
                working=WorkingContextProjection(),
            )
            + tool_schema_tokens
        )

    return measure


@dataclass(frozen=True, slots=True)
class ResearchSeed:
    """What the Research seed request carries beyond the shared budget facts."""

    tools: Sequence[ToolDeclaration]
    query_images: list[dict[str, Any]] | None
    resource_manifest: tuple[ResourceManifestEntry, ...]
    image_budget: AnswerImageBudget | None


@dataclass(frozen=True, slots=True)
class AnswerHistoryBudget:
    """The facts every reachable model call of one Answer Run is measured with.

    ``profiles`` are the Run's resolved role profiles, so each call is measured
    against the model that serves it: planning against ``extract``, generation and
    the Research seed against ``query``, and routing against ``keyword``, the model
    the executor routes ``auto`` with.
    """

    query: str
    profiles: Mapping[ChatModelSelector, ModelProfile]
    planner_for: Callable[[ModelProfile], RetrievalPlanner]
    schema: dict[str, Any] | None
    answer_image_policy: Callable[[ModelProfile], AnswerImagePolicy]
    image_descriptions: Sequence[str] = ()
    current_images: Sequence[dict[str, Any]] = ()
    memory_text: str = ""
    #: The caller's own continuation summary. A durable Session history carries its
    #: summary with its messages instead, so execution leaves this empty.
    episodic_summary: str = ""

    def fast_targets(self) -> tuple[HistoryProjectionTarget, ...]:
        """Fast's planning and generation calls, each keeping its full dynamic reserve."""
        synthesizer = AnswerSynthesizer(
            image_policy=self.answer_image_policy(self.profiles["query"]),
            model_profile=self.profiles["query"],
            context_policy=CONTEXT_POLICY,
        )
        return (
            HistoryProjectionTarget(
                FAST_PLANNER,
                self.profiles["extract"],
                self._planner_measure(preserve_query=None),
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
            HistoryProjectionTarget(
                FAST_GENERATION,
                self.profiles["query"],
                synthesizer.history_input_measure(
                    self.query,
                    memory_text=self.memory_text,
                    episodic_summary=self.episodic_summary,
                    current_images=list(self.current_images) or None,
                ),
                proactive_compaction=True,
                require_full_dynamic_reserve=True,
            ),
        )

    def research_targets(self, seed: ResearchSeed) -> tuple[HistoryProjectionTarget, ...]:
        """Research's planning call and its first Agent request."""
        return (
            HistoryProjectionTarget(
                RESEARCH_PLANNER,
                self.profiles["extract"],
                self._planner_measure(preserve_query=True),
            ),
            HistoryProjectionTarget(
                RESEARCH_SEED,
                self.profiles["query"],
                research_history_input_measure(
                    model_profile=self.profiles["query"],
                    context_policy=CONTEXT_POLICY,
                    query=self.query,
                    query_images=seed.query_images,
                    resource_manifest=seed.resource_manifest,
                    image_budget=seed.image_budget,
                    tools=seed.tools,
                    memory_text=self.memory_text,
                    episodic_summary=self.episodic_summary,
                ),
                proactive_compaction=True,
            ),
        )

    def router_target(
        self,
        *,
        resources: Sequence[ModeResource],
        valid_modes: Sequence[str],
        web_search: bool,
    ) -> HistoryProjectionTarget:
        """The ``auto`` routing call, measured as the executor will send it."""

        async def _unused_model(**_kwargs: Any) -> str:
            raise RuntimeError("a routing measurement never calls the model")

        return HistoryProjectionTarget(
            ROUTER,
            self.profiles["keyword"],
            AnswerModeRouter(_unused_model).history_input_measure(
                self.query,
                resources=resources,
                valid_modes=valid_modes,
                web_search=web_search,
            ),
        )

    def _planner_measure(self, *, preserve_query: bool | None) -> HistoryInputMeasure:
        planner = self.planner_for(self.profiles["extract"])
        return planner.history_input_measure(
            self.query,
            schema=self.schema,
            current_image_descriptions=list(self.image_descriptions) or None,
            preserve_query=preserve_query,
        )


@dataclass(frozen=True, slots=True)
class AcceptedHistory:
    """The caller history an accepted Run keeps, and the modes it may resolve to."""

    history: tuple[dict[str, Any], ...]
    episodic_summary: str
    valid_modes: frozenset[ResolvedMode]


def accept_history(
    budget: AnswerHistoryBudget,
    *,
    history: Sequence[Mapping[str, Any]],
    requested_mode: AnswerMode,
    allowed_modes: frozenset[ResolvedMode],
    research: ResearchSeed | None,
    mode_resources: Sequence[ModeResource],
    web_search: bool,
) -> AcceptedHistory:
    """Project the caller's history across every call the accepted Run can reach.

    Fast must hold its full dynamic reserve with no history at all. An explicit
    Fast request that cannot is refused as an overflow; ``auto`` drops Fast and
    keeps whatever else is valid. The newest complete history pairs every
    remaining call accepts are kept, older pairs become a bounded extractive
    summary, and a fixed request that overflows refuses the Run. A routing call
    that cannot fit resolves ``auto`` to Research, which needs no routing.
    """
    effective_modes = allowed_modes
    fast_targets: tuple[HistoryProjectionTarget, ...] = ()
    if "fast" in effective_modes:
        fast_targets = budget.fast_targets()
        try:
            project_history([], targets=fast_targets)
        except HistoryProjectionOverflowError as exc:
            if requested_mode == "fast":
                raise AnswerInputOverflowError(str(exc)) from exc
            # Observability for ADR 0020's residual: a reserved standing memory
            # block can be what makes Fast unviable, and only a line like this
            # says whether that ever happens.
            logger.info(
                "Fast is not viable for this request; resolving without it",
                extra={
                    "target": exc.target,
                    "fixed_input_tokens": exc.fixed_input_tokens,
                    "acceptance_limit_tokens": exc.acceptance_limit_tokens,
                    "memory_chars": len(budget.memory_text),
                    "requested_mode": requested_mode,
                },
            )
            effective_modes = cast(
                frozenset[ResolvedMode],
                frozenset(mode for mode in effective_modes if mode != "fast"),
            )
            if not effective_modes:
                raise UnsupportedAnswerModeError(requested_mode) from exc

    targets: list[HistoryProjectionTarget] = []
    if "research" in effective_modes:
        if research is None:
            raise ValueError("a Research-capable acceptance requires its seed request")
        targets.extend(budget.research_targets(research))
    if "fast" in effective_modes:
        targets.extend(fast_targets)
    if requested_mode == "auto" and effective_modes >= {"fast", "research"}:
        targets.append(
            budget.router_target(
                resources=mode_resources,
                valid_modes=tuple(sorted(effective_modes)),
                web_search=web_search,
            )
        )
    try:
        projected = project_history([dict(message) for message in history], targets=targets)
    except HistoryProjectionOverflowError as exc:
        if exc.target != ROUTER or research is None:
            raise AnswerInputOverflowError(str(exc)) from exc
        # The keyword model cannot take even the routing request, so nothing can
        # choose between the modes: resolve ``auto`` to Research, the mode that
        # handles any request, and fit the history to its calls alone.
        logger.info(
            "The auto routing call cannot fit; resolving to Research",
            extra={
                "fixed_input_tokens": exc.fixed_input_tokens,
                "acceptance_limit_tokens": exc.acceptance_limit_tokens,
            },
        )
        effective_modes = frozenset[ResolvedMode]({"research"})
        try:
            projected = project_history(
                [dict(message) for message in history],
                targets=list(budget.research_targets(research)),
            )
        except HistoryProjectionOverflowError as overflow:
            raise AnswerInputOverflowError(str(overflow)) from overflow
    summaries = (
        part.strip()
        for part in (budget.episodic_summary, projected.episodic_summary)
        if part.strip()
    )
    return AcceptedHistory(
        history=tuple(dict(message) for message in projected.messages),
        episodic_summary="\n\n".join(dict.fromkeys(summaries)),
        valid_modes=effective_modes,
    )


__all__ = [
    "FAST_GENERATION",
    "FAST_PLANNER",
    "RESEARCH_PLANNER",
    "RESEARCH_SEED",
    "ROUTER",
    "AcceptedHistory",
    "AnswerHistoryBudget",
    "ResearchSeed",
    "RetrievalPlanning",
    "accept_history",
    "research_history_input_measure",
    "reserved_memory_text",
    "routing_resources",
]
