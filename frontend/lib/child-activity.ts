// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** What a child did, as steps a reader can follow: what it said, each tool it called beside what came
 *  back, and what it was told. The server sends the transcript as a bounded tail (the latest
 *  messages), so a step never depends on an earlier message being there.
 *
 *  Pure, so the timeline's rules are tested without a page. */

import type {ChildTranscriptMessage} from '../api/conversations.ts';
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
  | {kind: 'instruction'; key: string; text: string};

export interface ActivityContext {
  /** The child's objective: the User Entry that records it is its task, not an instruction it was given. */
  objective: string;
  /** Whether the child is running: a call without a result is then still in flight, else it was lost. */
  childRunning: boolean;
}

function excerptOf(result: string): string {
  const line = result.split('\n').map((part) => part.replace(/\s+/g, ' ').trim()).find(Boolean) ?? '';
  // Characters, not UTF-16 units: a cut must not split a pair.
  const characters = Array.from(line);
  return characters.length > EXCERPT_LIMIT
    ? `${characters.slice(0, EXCERPT_LIMIT - 1).join('').trimEnd()}…`
    : line;
}

export function projectActivity(
  transcript: readonly ChildTranscriptMessage[],
  {objective, childRunning}: ActivityContext,
): ActivityStep[] {
  const results = new Map<string, ChildTranscriptMessage>();
  for (const message of transcript) {
    if (message.role === 'tool' && message.toolCallId && !results.has(message.toolCallId)) {
      results.set(message.toolCallId, message);
    }
  }
  const task = objective.trim();
  const steps: ActivityStep[] = [];
  transcript.forEach((message, index) => {
    if (message.role === 'assistant') {
      if (message.content.trim()) steps.push({kind: 'say', key: `say:${index}`, text: message.content});
      message.toolCalls.forEach((call, order) => {
        const result = call.id ? results.get(call.id) : undefined;
        const display = toolDisplay(call.name || result?.name || '');
        const verb = toolVerbText(display.verb, display.verbId);
        const key = call.id ? `tool:${call.id}` : `tool:${index}.${order}`;
        if (result === undefined) {
          steps.push({kind: 'tool', key, verb, state: childRunning ? 'running' : 'failed', excerpt: '', full: ''});
        } else {
          steps.push({
            kind: 'tool', key, verb, state: result.isError ? 'failed' : 'done',
            excerpt: excerptOf(result.content), full: result.content,
          });
        }
      });
    } else if (message.role === 'user') {
      const text = message.content.trim();
      if (text && !(task && text.includes(task))) steps.push({kind: 'instruction', key: `told:${index}`, text});
    }
  });
  return steps;
}
