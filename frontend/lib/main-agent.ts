// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Run's main agent as the roster states it: the same status row a child has, drawn from the turn the
 *  Run answers, so one row and one page show every agent. Pure, so it is tested without a page. */

import type {AgentChildStatus, AgentRunStatus} from '../api/conversations.ts';
import {MAIN_AGENT} from './agent-id.ts';
import {isTerminalTurnState, type ChatTurnView} from './chat-views.ts';
import {localizedText} from './localized-text.ts';

/** The status row of the main agent. Its objective is the question, its result the answer or, when the Run
 *  failed, why, and its Evidence the sources the answer cites — all from the turn it answers. Where it
 *  stands and how long it has run come from the Run's own row, which the roster page states beside every
 *  child's, so this row is drawn from the same facts those are. */
export function mainAgentStatus(
  turn: ChatTurnView | undefined,
  run: AgentRunStatus | null = null,
): AgentChildStatus & {childSessionId: string} {
  const total = (turn?.usage.usage_details as Record<string, unknown> | undefined)?.total_tokens;
  return {
    childSessionId: MAIN_AGENT,
    status: run?.status
      ?? (turn === undefined || !isTerminalTurnState(turn.state) ? 'running' : turn.state),
    objective: turn?.userText,
    modelRole: undefined,
    usage: typeof total === 'number' && Number.isFinite(total) ? {total_tokens: total} : null,
    operationId: null,
    operationSequence: null,
    operationStatus: null,
    cancellationOrigin: null,
    summary: turn?.presentation?.answerText.trim()
      || (turn?.error ? localizedText(turn.error) : null),
    resultHandles: turn?.presentation?.sources.map((source) => `[${source.id}] ${source.title}`) ?? [],
    startedAt: run?.startedAt ?? null,
    finishedAt: run?.finishedAt ?? null,
    pendingQuestions: 0,
    // The Run's answer is judged under the chat's turn, never from this row.
    resultSequence: null,
    feedback: null,
  };
}
