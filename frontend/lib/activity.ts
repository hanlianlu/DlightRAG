// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What an agent did, as steps a reader can follow: what it said, each tool it called beside what came
 *  back, and what it was told. The server sends the transcript a page at a time, newest page first, so a
 *  step never depends on an earlier page being there.
 *
 *  Pure, so the timeline's rules are tested without a page. */

import type {ActivityMessage} from '../api/conversations.ts';
import {toolDisplay, toolVerbText} from './tool-display.ts';

/** The first line of a result stands for it on a timeline row, at most this many characters. */
const EXCERPT_LIMIT = 120;

export type ActivityStep =
  | {
      kind: 'say';
      key: string;
      text: string;
      /** The reasoning the model wrote before it said this, trimmed; left off when the wire states
       *  none. The pane folds it behind a "Thinking" line, since it is how the step was reached
       *  rather than the step. */
      reasoning?: string;
    }
  | {
      /** Reasoning an assistant turn wrote with no words of its own — a turn that only chose a Tool.
       *  It stands on its own before the calls it explains, since there is no step to fold it into. */
      kind: 'thinking';
      key: string;
      text: string;
    }
  | {
      kind: 'tool';
      key: string;
      verb: string;
      state: 'running' | 'done' | 'failed';
      /** The bounded, tool-reported subject of the call; empty when the wire carried none. */
      subject: string;
      /** The result's first non-empty line, shortened; empty while the call runs or when it left no result. */
      excerpt: string;
      /** The whole result. */
      full: string;
      /** How long the call took, or has been running so far; null when the wire gave no clock. */
      durationMs: number | null;
    }
  | {
      /** What the agent concluded: the words its page folds at the end. */
      kind: 'brief';
      key: string;
      text: string;
      /** The reasoning that produced the words, trimmed, carried off the say step they fold. */
      reasoning?: string;
    }
  | {
      kind: 'instruction';
      key: string;
      /** Who wrote it, as a control record names its origin; empty when nobody is known. */
      sender: string;
      text: string;
    }
  | {
      /** Internal evidence stated to the model before delivery; not a human instruction. */
      kind: 'delivery';
      key: string;
      text: string;
    };

/** What a control record says of a steer the transcript holds. */
export interface ActivityControl {
  origin: string;
  content: string;
}

interface ActivityContext {
  /** The words a search narrowed this pane to, left out when the pane shows a Session as it is. */
  terms?: readonly string[];

  /** What this agent concluded. The projection owns it: it closes the list, exactly once. */
  brief?: string;

  /** The agent's objective: the user message that records it is its task, not an instruction it was given. */
  objective: string;
  /** Whether the agent is working: a call without a result is then still in flight, else it was lost. */
  running: boolean;
  /** The control records of the agent, which name who sent each steer. */
  controls: readonly ActivityControl[];
  /** The clock this render holds: how far a call still in flight has run by now. Absent in a caller
   *  that has no clock to give, and then a running call states no time. */
  now?: number;
}

/** When a message's entry was written, in milliseconds, or null when the server did not say. */
function instantOf(message: ActivityMessage): number | null {
  const parsed = message.at === null ? Number.NaN : Date.parse(message.at);
  return Number.isNaN(parsed) ? null : parsed;
}

/** How long a Tool call took: from the entry that made it to the clock that ended it — the result
 *  that settled it, or now while it is still in flight. Null when either end is unknown, so a step
 *  never states a time nobody measured. */
function timeBetween(calledAt: number | null, endedAt: number | null | undefined): number | null {
  if (calledAt === null || endedAt === null || endedAt === undefined) return null;
  return Math.max(0, endedAt - calledAt);
}

function excerptOf(result: string): string {
  const line = result.split('\n').map((part) => part.replace(/\s+/g, ' ').trim()).find(Boolean) ?? '';
  // Characters, not UTF-16 units: a cut must not split a pair.
  const characters = Array.from(line);
  return characters.length > EXCERPT_LIMIT
    ? `${characters.slice(0, EXCERPT_LIMIT - 1).join('').trimEnd()}…`
    : line;
}

/** A steer is written into the transcript as "<Origin> steer: <content>", so a control record claims
 *  exactly that text. */
function told(text: string, context: ActivityContext): {sender: string; text: string} {
  const control = context.controls.find((record) => {
    const origin = record.origin || 'unknown';
    return text === `${origin.charAt(0).toUpperCase()}${origin.slice(1).toLowerCase()} steer: ${record.content}`;
  });
  return control
    ? {sender: control.origin, text: control.content}
    : {sender: '', text};
}

/** A message's reasoning, trimmed to what a reader would see, or '' when the wire states none: an
 *  empty fold would give a reader a label to open with nothing under it. */
function trimmedReasoning(message: ActivityMessage): string {
  return message.reasoning.trim();
}

export function projectActivity(
  messages: readonly ActivityMessage[],
  context: ActivityContext,
): ActivityStep[] {
  const results = new Map<string, ActivityMessage>();
  for (const message of messages) {
    if (message.role === 'tool' && message.toolCallId && !results.has(message.toolCallId)) {
      results.set(message.toolCallId, message);
    }
  }
  const task = context.objective.trim();
  const steps: ActivityStep[] = [];
  // A pane narrowed to a search shows what was found: a hit is a message the words are in, and
  // every one of them is shown, whether it was said, asked, answered by a tool, or told. The
  // count a reader sees beside an agent is the number of these, so the two agree. A hit carries
  // no reasoning (the search projection states none), so these steps fold only what they say.
  if ((context.terms?.length ?? 0) > 0) {
    for (const message of messages) {
      if (message.content.trim()) {
        steps.push({kind: 'say', key: `say:${message.sequence}`, text: message.content});
      }
    }
    return concluded(steps, context);
  }
  for (const message of messages) {
    if (message.role === 'assistant') {
      const reasoning = trimmedReasoning(message);
      if (message.content.trim()) {
        steps.push({
          kind: 'say', key: `say:${message.sequence}`, text: message.content,
          ...(reasoning === '' ? {} : {reasoning}),
        });
      } else if (reasoning !== '') {
        // The turn said nothing around its reasoning because it was choosing a Tool: the thinking
        // stands on its own, before the calls it explains.
        steps.push({kind: 'thinking', key: `thinking:${message.sequence}`, text: reasoning});
      }
      const calledAt = instantOf(message);
      message.toolCalls.forEach((call, order) => {
        const result = call.id ? results.get(call.id) : undefined;
        const display = toolDisplay(call.name || result?.name || '');
        const verb = toolVerbText(display.verb, display.verbId);
        const key = call.id ? `tool:${call.id}` : `tool:${message.sequence}.${order}`;
        // A call still in flight is timed to the clock this render holds, so a reader watching one
        // sees it climb; a call whose result is here is timed to the entry that settled it.
        const durationMs = result === undefined
          ? (context.running ? timeBetween(calledAt, context.now) : null)
          : timeBetween(calledAt, instantOf(result));
        if (result === undefined) {
          steps.push({
            kind: 'tool', key, verb, state: context.running ? 'running' : 'failed',
            subject: call.subject, excerpt: '', full: '', durationMs,
          });
        } else {
          steps.push({
            kind: 'tool', key, verb, state: result.isError ? 'failed' : 'done',
            subject: call.subject, excerpt: excerptOf(result.content), full: result.content, durationMs,
          });
        }
      });
    } else if (message.role === 'user') {
      const text = message.content.trim();
      if (text && !(task && text.includes(task))) {
        steps.push({kind: 'instruction', key: `told:${message.sequence}`, ...told(text, context)});
      }
    } else if (message.role === 'delivery') {
      const text = message.content.trim();
      if (text) steps.push({kind: 'delivery', key: `delivery:${message.sequence}`, text});
    }
  }
  return concluded(steps, context);
}

/** Join a freshly read newest page to the messages already shown. A page that reaches back to what is
 *  shown replaces its own range and keeps the older messages; a page that does not (the agent moved on by
 *  more than a page, or sequences have a hole there) leaves a possible gap, so it is null and the caller
 *  starts over from the page. */
export function joinActivity(
  shown: readonly ActivityMessage[],
  newest: readonly ActivityMessage[],
): ActivityMessage[] | null {
  const first = newest[0];
  const last = shown.at(-1);
  if (first === undefined) return [...shown];
  if (last === undefined) return [...newest];
  if (first.sequence > last.sequence) return null;
  return [...shown.filter((message) => message.sequence < first.sequence), ...newest];
}

/** The list a reader ends on: what the agent concluded, said once, after the work it did. The
 *  conclusion takes over the say step it folds, and the reasoning that produced its words with it. */
function concluded(steps: ActivityStep[], context: ActivityContext): ActivityStep[] {
  const said = context.brief?.trim() ?? '';
  if (said === '') return steps;
  const folded = steps.find((step) => step.kind === 'say' && step.text.trim() === said);
  const brief: Extract<ActivityStep, {kind: 'brief'}> = {kind: 'brief', key: 'brief', text: said};
  if (folded?.kind === 'say' && folded.reasoning !== undefined) brief.reasoning = folded.reasoning;
  return [...steps.filter((step) => step.kind !== 'say' || step.text.trim() !== said), brief];
}
