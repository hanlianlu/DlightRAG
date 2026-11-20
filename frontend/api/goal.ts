// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The conversation's Goal: what the bar reads, and the command the reader may send. */

import * as v from 'valibot';
import {csrfHeaders} from './csrf.ts';
import {
  apiErrorFromBody,
  parseWire,
  refusalBody,
  refusalEnvelope,
} from './wire.ts';

const goalState = v.pipe(
  v.object({
    conversation_id: v.string(),
    goal_id: v.string(),
    revision: v.number(),
    objective: v.string(),
    phase: v.string(),
    max_rounds: v.number(),
    rounds_started: v.number(),
    blocked_reason: v.nullable(v.string()),
  }),
  v.transform((wire) => ({
    conversationId: wire.conversation_id,
    goalId: wire.goal_id,
    revision: wire.revision,
    objective: wire.objective,
    phase: wire.phase as GoalPhase,
    maxRounds: wire.max_rounds,
    roundsStarted: wire.rounds_started,
    roundsLeft: Math.max(0, wire.max_rounds - wire.rounds_started),
    blockedReason: wire.blocked_reason,
  })),
);
const nullableGoal = v.nullable(goalState);

type GoalPhase = 'active' | 'paused' | 'blocked' | 'complete';
export type ConversationGoal = v.InferOutput<typeof goalState>;
export type GoalAction = 'create' | 'edit' | 'pause' | 'resume' | 'clear';

export interface GoalCommand {
  readonly action: GoalAction;
  readonly objective?: string;
  readonly maxGoalRounds?: number;
  /** The composer's current workspace selection: round 1 continues it when it creates a goal. */
  readonly workspaces?: readonly string[];
}

/** A refused goal command, carrying the domain reason the server named.
 *
 * The goal route answers a refusal with `{detail: {kind, message}}` rather than the general
 * envelope: `kind` is the domain's stable code and `message` its sentence, so the goal bar can
 * show the specific reason instead of a generic failure line. */
export class GoalCommandError extends Error {
  readonly kind: string | null;

  constructor(kind: string | null, message: string | null) {
    super(message ?? 'The goal command failed.');
    this.name = 'GoalCommandError';
    this.kind = kind;
  }
}

/** Read a goal refusal through its module's own envelope, then the general one. */
async function goalRefusal(response: Response): Promise<Error> {
  const body = await refusalBody(response);
  const {detail} = refusalEnvelope(body);
  if (detail !== null && typeof detail === 'object' && !Array.isArray(detail)) {
    const fields = detail as Record<string, unknown>;
    const kind = typeof fields.kind === 'string' && fields.kind ? fields.kind : null;
    const message = typeof fields.message === 'string' && fields.message.trim()
      ? fields.message
      : null;
    return new GoalCommandError(kind, message);
  }
  return apiErrorFromBody(response.status, body);
}

const location = (conversationId: string): string =>
  `/web/api/conversations/${encodeURIComponent(conversationId)}/goal`;

/** Read the conversation's current goal, or null when it carries none. */
export async function readConversationGoal(
  conversationId: string,
  signal?: AbortSignal,
): Promise<ConversationGoal | null> {
  const response = await fetch(location(conversationId), {signal});
  return parseWire(response, nullableGoal);
}

/** Send one human goal command; null is the answer to a clear. */
export async function sendGoalCommand(
  conversationId: string,
  command: GoalCommand,
  signal?: AbortSignal,
): Promise<ConversationGoal | null> {
  const response = await fetch(location(conversationId), {
    method: 'POST',
    headers: csrfHeaders('application/json'),
    body: JSON.stringify({
      action: command.action,
      ...(command.objective === undefined ? {} : {objective: command.objective}),
      ...(command.maxGoalRounds === undefined
        ? {}
        : {max_goal_rounds: command.maxGoalRounds}),
      ...(command.workspaces === undefined ? {} : {workspaces: [...command.workspaces]}),
    }),
    signal,
  });
  return parseWire(response, nullableGoal, goalRefusal);
}
