# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The harness's goal tools: the one surface through which an Agent talks about a goal (ADR 0060).

The tools are harness code and name no Skill. They bind to the engine's
:class:`AnswerGoalService`, whose store is the only port an adapter implements; a Run composes them
only when the deployment supplies that service. The host carries the Run's identity and whether this
turn is a goal round, because the execution Tool sees neither. Which requests may create, edit, pause
or resume is enforced by the service, close to the durable record; the tools only carry the context
across.

The docstrings are what the model reads. They state the opt-in rule, the round cap, that the platform
continues an active goal automatically — a process restart included — and the authority split: the
model completes or blocks its own goal, while a human pauses, resumes and edits.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolDeclaration,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.goal import (
    BLOCKED_AFTER_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    GOAL_COMPLETION_UNATTESTED,
    Goal,
    GoalAction,
    GoalError,
)
from dlightrag.engine.answer.goal_service import AnswerGoalService

#: The names this module composes. Child Sessions never hold them (ADR 0025): a Child spends a
#: Run's capability, and goals are the Run's own authority.
GOAL_TOOL_NAMES = ("create_goal", "get_goal", "update_goal")


@dataclass(slots=True)
class GoalHost:
    """One Run's identity and its turn's provenance, bound before the tools compose.

    ``goal_round`` is the number of the goal round this turn is, or ``None`` for a direct human
    turn. The executor sets it from the durable goal and the Run's continuation kind, so the same
    three tools answer in both kinds of turn without being rebuilt.
    """

    service: AnswerGoalService
    owner_id: str
    run_id: str
    session_id: str
    goal_round: int | None = None
    #: None means no delivery gate is bound; a reason string makes complete a refusal while the
    #: current Run's sealed Skill delivery is unattested.
    completion_refusal: Callable[[], str | None] | None = None


class CreateGoalArgs(BaseModel):
    """Arguments for `create_goal`."""

    objective: str = Field(
        description="One sentence naming what must be achieved. The message and the session keep "
        "the plan; this field is the contract, not a description of the work."
    )
    max_goal_rounds: int = Field(
        default=DEFAULT_MAX_ROUNDS,
        ge=1,
        description=f"Continue at most this many times. The default is {DEFAULT_MAX_ROUNDS} and "
        "there is no product ceiling, so name a larger cap when the objective needs more rounds.",
    )


class GetGoalArgs(BaseModel):
    """Arguments for `get_goal`; it takes none."""


class UpdateGoalArgs(BaseModel):
    """Arguments for `update_goal`."""

    goal_id: str = Field(description="The goal's id, as `create_goal` or `get_goal` returned it.")
    revision: int = Field(
        ge=1,
        description="The goal's revision from `get_goal` in this turn; a stale revision is "
        "refused, never merged.",
    )
    action: GoalAction = Field(
        description="edit | pause | resume | complete | blocked. complete and blocked are the "
        "Agent's calls; edit, pause and resume act on a direct human turn. pause stops the goal "
        "immediately and cancels the running goal round, and a paused goal is the human's to resume."
    )
    objective: str | None = Field(default=None, description="For action=edit: the new objective.")
    max_goal_rounds: int | None = Field(
        default=None,
        ge=1,
        description="For action=edit: the new round cap. Any positive integer is accepted.",
    )
    blocked_reason: str | None = Field(
        default=None,
        description="For action=blocked: the concrete condition that blocks the objective, which "
        "a goal round may report only after the threshold of started rounds.",
    )


def _goal_line(goal: Goal) -> str:
    """One stable rendering of a goal, for both the tool text and its details."""
    line = (
        f"goal {goal.id} revision {goal.revision}: {goal.phase}, "
        f"rounds {goal.rounds_started}/{goal.max_rounds}; objective: {goal.objective}"
    )
    if goal.blocked_reason is not None:
        line = f"{line}; blocked: {goal.blocked_reason}"
    return line


#: What a terminal goal round tells the model to write. The closing message is the reader's
#: answer, so it reports an outcome and its evidence rather than a status, and it names what the
#: reader must do next when continuation stopped.
_CLOSING_COMPLETE = (
    "Write the closing message to the user now, and do not call any more tools in this Run. The "
    "closing message reports the outcome: what the objective asked for and what has been achieved, "
    "the evidence and verification that support it, and the concrete results the user can open or "
    "use — files, published artifacts, links. Report only what this Session establishes; name a gap "
    "instead of inventing a result. Say what the user should review or do next, and address the "
    "user directly."
)
_CLOSING_BLOCKED = (
    "Write the closing message to the user now, and do not call any more tools in this Run. The "
    "closing message says what has been completed, what remains, the concrete blocking condition, "
    "what was tried, and exactly what the user must do to continue — raise the round cap, resume "
    "the goal, or change the objective. Report only what this Session establishes, and address the "
    "user directly."
)


def _result(goal: Goal, *, action: str) -> ToolResult:
    text = _goal_line(goal)
    if action == "create":
        text = (
            f"{text}\n\nWhile this goal is active and unfinished the platform continues it "
            "automatically after this Run; if the process restarts, recovery takes the durable "
            "goal up again and starts the next round. Mark it complete when the objective is "
            "achieved."
        )
    elif action == "complete":
        text = f"{text}\n\nThe goal is now complete. {_CLOSING_COMPLETE}"
    elif action == "blocked":
        text = f"{text}\n\nThe goal is now blocked. {_CLOSING_BLOCKED}"
    return ToolResult.text(
        text,
        subject=f"goal {goal.id}",
        details={
            "goal_id": goal.id,
            "goal_revision": goal.revision,
            "goal_phase": str(goal.phase),
            "goal_rounds_started": goal.rounds_started,
            "goal_max_rounds": goal.max_rounds,
            "goal_action": action,
        },
    )


def _refused(error: GoalError) -> ToolResult:
    return ToolResult.text(
        f"goal refused ({error.code}): {error}",
        details={"goal_error": error.code},
        is_error=True,
    )


_CREATE_DESCRIPTION = (
    "Create a goal before starting a direct human request whose work will not fit one Run — a "
    "deliverable needing several passes to gather, build and verify, a change across several "
    "files, a long iterative job — even if the human did not say 'goal'. While the goal is active "
    "and unfinished the platform starts the next round of this conversation automatically after "
    "this Run ends; a process restart does not stop it, because the service that serves this "
    "conversation recovers active goals from their durable record and starts the next round, a "
    "goal with no Run to continue from included. Continuation needs no human action, so mark it "
    "complete in the Run that finishes the objective. Creating it is the Agent's call, never "
    "forced; a single lookup, chat reply or one-pass edit needs none. One conversation carries one "
    "goal: a completed goal may be replaced by a new one, while an unfinished goal must be paused, "
    "blocked or cleared before another takes its place. The objective is yours to state, but it "
    "must be the outcome the human asked for, not a weaker alternative invented to finish early — "
    "do not append an escape hatch such as 'or deliver an unsealed memo'. A loaded Skill's "
    "documents may inform how you scope it; the sealed delivery gate still refuses complete while "
    "the Skill's required reads or verifier are not attested."
)

_GET_DESCRIPTION = (
    "Read the goal this conversation is carrying, if any: its objective, phase, rounds and "
    "revision. Read it before `update_goal` or before claiming completion, so the goal id and "
    "revision you send are current."
)

_UPDATE_DESCRIPTION = (
    "Update the goal this conversation carries; read it with `get_goal` first, because `goal_id` "
    "and `revision` must be current and a stale revision is refused, never merged. complete and "
    "blocked are yours to call: complete only once the objective is actually achieved and the Run's "
    "loaded sealed Skill delivery is attested — a skipped required read, an unpassed verifier or a "
    "refused seal makes complete a refusal that leaves the goal active; blocked only with the "
    "concrete condition, and from a goal round only "
    f"after the threshold of {BLOCKED_AFTER_ROUNDS} started rounds. edit, pause and resume act on "
    "a direct human turn: edit changes the objective or the cap, pause stops the goal immediately "
    "and cancels the goal round already running, starts no next round, and a paused goal is resumed "
    "by the human, not by you. The platform keeps starting rounds while the goal is active and "
    "rounds remain; difficulty or remaining work is not blocked."
)


def create_goal_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="create_goal",
        description=_CREATE_DESCRIPTION,
        input_model=CreateGoalArgs,
        replay_policy="never",
        contract_version=1,
    )


def get_goal_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="get_goal",
        description=_GET_DESCRIPTION,
        input_model=GetGoalArgs,
        replay_policy="replayable",
        read_only=True,
        contract_version=1,
    )


def update_goal_declaration() -> ToolDeclaration:
    return ToolDeclaration(
        name="update_goal",
        description=_UPDATE_DESCRIPTION,
        input_model=UpdateGoalArgs,
        replay_policy="never",
        contract_version=1,
    )


def goal_tool_declarations() -> tuple[ToolDeclaration, ...]:
    """The three declarations, for a Run's accepted catalog and its execution composition."""
    return (create_goal_declaration(), get_goal_declaration(), update_goal_declaration())


def goal_tools(*, host: GoalHost) -> list[AgentTool]:
    """Bind the three goal tools to one Run's identity and its turn's round context."""

    async def create(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = CreateGoalArgs.model_validate(raw)
        del runtime  # the identity is the host's, not the call's
        try:
            goal = await host.service.create(
                owner_id=host.owner_id,
                session_id=host.session_id,
                run_id=host.run_id,
                objective=args.objective,
                max_rounds=args.max_goal_rounds,
                goal_round=host.goal_round,
            )
        except GoalError as error:
            return _refused(error)
        return _result(goal, action="create")

    async def read(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        del raw
        await runtime.in_source_order()
        goal = await host.service.get(owner_id=host.owner_id, session_id=host.session_id)
        if goal is None:
            return ToolResult.text(
                "this conversation carries no goal",
                details={"goal_id": "", "goal_action": "read"},
            )
        return _result(goal, action="read")

    async def update(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = UpdateGoalArgs.model_validate(raw)
        del runtime
        if args.action == "complete" and host.completion_refusal is not None:
            refusal = host.completion_refusal()
            if refusal:
                return ToolResult.text(
                    "goal completion refused "
                    f"({GOAL_COMPLETION_UNATTESTED}): {refusal}\n\nThe goal is still active. "
                    "Fix the sealed delivery and complete in this Run, or call update_goal "
                    "action=blocked with the concrete condition; otherwise the platform will "
                    "continue the goal after this Run.",
                    details={
                        "goal_error": GOAL_COMPLETION_UNATTESTED,
                        "goal_id": args.goal_id,
                        "goal_action": "complete",
                    },
                    is_error=True,
                )
        try:
            goal = await host.service.update(
                owner_id=host.owner_id,
                session_id=host.session_id,
                run_id=host.run_id,
                goal_id=args.goal_id,
                revision=args.revision,
                action=args.action,
                objective=args.objective,
                max_rounds=args.max_goal_rounds,
                blocked_reason=args.blocked_reason,
                goal_round=host.goal_round,
            )
        except GoalError as error:
            return _refused(error)
        return _result(goal, action=args.action)

    return [
        create_goal_declaration().bind(create),
        get_goal_declaration().bind(read),
        update_goal_declaration().bind(update),
    ]


__all__ = [
    "BLOCKED_AFTER_ROUNDS",
    "GOAL_TOOL_NAMES",
    "CreateGoalArgs",
    "GetGoalArgs",
    "GoalAction",
    "GoalHost",
    "UpdateGoalArgs",
    "create_goal_declaration",
    "get_goal_declaration",
    "goal_tool_declarations",
    "goal_tools",
    "update_goal_declaration",
]
