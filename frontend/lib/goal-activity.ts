// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Whether a run batch wrote the conversation's Goal, so the Goal bar must read it again. */

import type {AnswerRunEvent} from './run-controller.ts';
import type {ToolEventPayload} from './tool-events.ts';

/** The Tools whose settled call changes the conversation's Goal: `create_goal` writes it, and
 *  `update_goal` moves its phase, objective, round cap or blocked reason. `get_goal` only reads
 *  it, so a reader running that Tool sees no reason for a second read. */
const GOAL_WRITING_TOOLS: ReadonlySet<string> = new Set(['create_goal', 'update_goal']);

/** Whether this batch carries a settled Goal write.
 *
 *  A Goal the Agent creates mid-Run is durable before its tool's end frame is committed, so a
 *  read taken when that frame arrives sees it: the bar appears while the Run is still going,
 *  without waiting for the turn to settle and without a page reload. A write that failed changed
 *  nothing, so it asks for no read. */
export function goalWasWritten(events: readonly AnswerRunEvent[]): boolean {
  return events.some((event) => {
    if (event.kind !== 'tool' || event.eventType !== 'tool_end') return false;
    const payload = event.payload;
    if (typeof payload !== 'object' || payload === null) return false;
    const settled = payload as ToolEventPayload;
    return settled.outcome === 'succeeded'
      && typeof settled.tool_name === 'string'
      && GOAL_WRITING_TOOLS.has(settled.tool_name);
  });
}
