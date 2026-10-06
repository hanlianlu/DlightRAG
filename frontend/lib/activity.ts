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
  | {kind: 'say'; key: string; text: string}
  | {
      kind: 'tool';
      key: string;
      verb: string;
      state: 'running' | 'done' | 'failed';
      /** The result's first non-empty line, shortened; empty while the call runs or when it left no result. */
      excerpt: string;
      /** The whole result. */
      full: string;
    }
  | {
      kind: 'instruction';
      key: string;
      /** Who wrote it, as a control record names its origin; empty when nobody is known. */
      sender: string;
      text: string;
    };

/** What a control record says of a steer the transcript holds. */
export interface ActivityControl {
  origin: string;
  content: string;
}

export interface ActivityContext {
  /** The agent's objective: the user message that records it is its task, not an instruction it was given. */
  objective: string;
  /** Whether the agent is working: a call without a result is then still in flight, else it was lost. */
  running: boolean;
  /** The control records of the agent, which name who sent each steer. */
  controls: readonly ActivityControl[];
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
  for (const message of messages) {
    if (message.role === 'assistant') {
      if (message.content.trim()) steps.push({kind: 'say', key: `say:${message.sequence}`, text: message.content});
      message.toolCalls.forEach((call, order) => {
        const result = call.id ? results.get(call.id) : undefined;
        const display = toolDisplay(call.name || result?.name || '');
        const verb = toolVerbText(display.verb, display.verbId);
        const key = call.id ? `tool:${call.id}` : `tool:${message.sequence}.${order}`;
        if (result === undefined) {
          steps.push({kind: 'tool', key, verb, state: context.running ? 'running' : 'failed', excerpt: '', full: ''});
        } else {
          steps.push({
            kind: 'tool', key, verb, state: result.isError ? 'failed' : 'done',
            excerpt: excerptOf(result.content), full: result.content,
          });
        }
      });
    } else if (message.role === 'user') {
      const text = message.content.trim();
      if (text && !(task && text.includes(task))) {
        steps.push({kind: 'instruction', key: `told:${message.sequence}`, ...told(text, context)});
      }
    }
  }
  return steps;
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
