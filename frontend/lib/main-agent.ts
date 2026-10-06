// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The Run's main agent as the roster states it: the same status row a child has, drawn from the turn the
 *  Run answers, so one row and one page show every agent. Pure, so it is tested without a page. */

import type {AgentChildStatus} from '../api/conversations.ts';
import {isTerminalTurnState, type ChatTurnView} from './chat-views.ts';

/** What names the main agent where a child's id would. */
export const MAIN_AGENT = 'main';

/** The status row of the main agent. Its objective is the question, its result the answer or, when the Run
 *  failed, why, and its Evidence the sources the answer cites. */
export function mainAgentStatus(turn: ChatTurnView | undefined): AgentChildStatus & {childSessionId: string} {
  const total = (turn?.usage.usage_details as Record<string, unknown> | undefined)?.total_tokens;
  return {
    childSessionId: MAIN_AGENT,
    status: turn === undefined || !isTerminalTurnState(turn.state) ? 'running' : turn.state,
    objective: turn?.userText,
    modelRole: undefined,
    usage: typeof total === 'number' && Number.isFinite(total) ? {total_tokens: total} : null,
    operationId: null,
    operationSequence: null,
    operationStatus: null,
    cancellationOrigin: null,
    summary: turn?.presentation?.answerText.trim() || turn?.error || null,
    resultHandles: turn?.presentation?.sources.map((source) => `[${source.id}] ${source.title}`) ?? [],
    startedAt: null,
    finishedAt: null,
    pendingQuestions: 0,
  };
}
