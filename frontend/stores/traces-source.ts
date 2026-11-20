// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What the Inspector's Run traces dock reads for one Run.
 *
 * The dock draws a Run's agents and reads their pages; it knows nothing of the chat that started the
 * Run. Almost all of what it needs is a call to the server, which this builds straight on the api. One
 * thing only the chat can say is handed in: the turn the Run answers, which the main agent's row and
 * the Run's answer are drawn from.
 */

import type {
  ActivityPage,
  AgentChildRosterPage,
  AgentChildStatus,
  AgentRunStatus,
  AnswerPresentation,
  RunOverview,
  StepSearchPage,
  ChildObservation,
  TraceSearch,
} from '../api/conversations.ts';
import {
  getAnswerActivityPage,
  getAnswerActivitySince,
  getAnswerRunChild,
  getAnswerRunChildrenPage,
  getAnswerRunOverview,
  getAnswerStepSearch,
  getAnswerTraceSearch,
} from '../api/conversations.ts';
import type {ChatTurnView} from '../lib/chat-views.ts';
import {mainAgentStatus} from '../lib/main-agent.ts';

/** What the dock reads for one Run's agents. */
export interface TracesSource {
  readonly runId: string;
  /** The Run's main agent as a status row: its words from the turn it answers, and
   *  its own row — status and clock — from the roster page, or null before it arrives. */
  mainAgent(run: AgentRunStatus | null): AgentChildStatus & {childSessionId: string};
  /** The Run's answer, once it has one: what any agent's Evidence line can open. */
  presentation(): AnswerPresentation | null;
  page(cursor: string | null, signal: AbortSignal): Promise<AgentChildRosterPage>;
  observe(childSessionId: string, signal: AbortSignal): Promise<ChildObservation>;
  /** Read a child ahead of the click: a source that keeps pages reads it when its row is warmed. */
  warm?(childSessionId: string): void;
  /** The messages of one agent's transcript holding the reader's words, when the source can search. */
  search?(text: string): Promise<TraceSearch>;
  /** The Run's whole-Run overview: totals, agents, and the capped findings list. It is an aggregate
   *  read, so the dock can follow a Run without reading its steps. */
  overview(signal: AbortSignal): Promise<RunOverview>;
  /** One page of the Run's steps whose messages hold `text`, newest page first; `before` is the
   *  cursor the previous page stated. */
  searchSteps(text: string, before: string | null, signal: AbortSignal): Promise<StepSearchPage>;
  /** One page of an agent's transcript: the Run's main agent when `agent` is null. */
  activity(agent: string | null, cursor: string | null, signal: AbortSignal): Promise<ActivityPage>;
  /** The messages written after a sequence the reader holds; absent from a source that cannot say. */
  since?(agent: string | null, sequence: number, signal: AbortSignal): Promise<ActivityPage>;
}

/** What only the chat can say about the Run it shows. */
interface RunTracesDeps {
  /** The turn on show that the Run answers, if the chat holds it now. */
  turn(): ChatTurnView | undefined;
}

/** The source for one Run. */
export function runTracesSource(runId: string, deps: RunTracesDeps): TracesSource {
  return {
    runId,
    mainAgent: (run) => mainAgentStatus(deps.turn(), run),
    presentation: () => deps.turn()?.presentation ?? null,
    page: (cursor, signal) => getAnswerRunChildrenPage(runId, cursor, signal),
    observe: (childSessionId, signal) => getAnswerRunChild(runId, childSessionId, signal),
    overview: (signal) => getAnswerRunOverview(runId, signal),
    searchSteps: (text, before, signal) => getAnswerStepSearch(runId, text, before, signal),
    activity: (agent, cursor, signal) => getAnswerActivityPage(runId, agent, cursor, signal),
    since: (agent, sequence, signal) => getAnswerActivitySince(runId, agent, sequence, signal),
    search: (text) => getAnswerTraceSearch(runId, text),
  };
}
