// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The turn view contract shared by projection, components, and tests. It is a
 *  data shape, not a component concern, so it lives in lib where the pure
 *  projection logic can depend on it without reaching into ui/. */

import type {
  AnswerPresentation,
  ConversationAttachmentReference,
  ConversationCommand,
} from '../api/conversations.ts';
import type {AnswerJudgement} from '../api/feedback.ts';
import {commandResultText, type CommandResultText} from './web-commands.ts';
import type {LocalizedText} from './localized-text.ts';
import type {ToolRow} from './tool-events.ts';

/** The rows a live turn's process block shows at once: reasoning folds and Tool rows share one
 *  bounded viewport, so the newest steps stay in view while the page keeps its place. The rest of
 *  the trace is read by scrolling inside the block; nothing a Run thought or called grows the page. */
export const PROCESS_TRACE_VISIBLE_ROWS = 5;

/** The most entries a live turn's process trace keeps: reasoning folds and Tool rows oldest first.
 *  Enough that the block's own scroll holds a long Run's work; a Run longer than this keeps its
 *  count in the line and its whole record in the Inspector, which is where a settled Run is read. */
export const MAX_PROCESS_ENTRIES = 200;

/** One entry of a live turn's process trace, in the order it happened. Both arms are process, not
 *  answer: the model's reasoning fold, and one Tool row the trace holds. */
export type ProcessEntry =
  | {
      readonly kind: 'thinking';
      /** The reasoning the fold holds, as it has streamed so far. */
      readonly text: string;
      /** Whether this is the block the model is still writing: the one fold left open. */
      readonly open: boolean;
    }
  | {
      readonly kind: 'tool';
      /** The row the Tool trace's own state machine holds, the same object `toolRows` carries. */
      readonly row: ToolRow;
    };

export interface ChatTurnView {
  id: string;
  userText: string;
  userAttachments: readonly ConversationAttachmentReference[];
  runId: string;
  state: 'pending' | 'streaming' | 'succeeded' | 'failed' | 'cancelled' | 'retryable';
  streamText: string;
  /** The turn's process as one trace: reasoning folds and Tool rows interleaved oldest first and
   *  kept up to `MAX_PROCESS_ENTRIES`. The component draws it in a bounded, scrollable block that
   *  shows the newest `PROCESS_TRACE_VISIBLE_ROWS` steps at rest. It lasts only while the turn runs —
   *  the terminal view clears it, and the Inspector is where the thinking and the calls stay readable. */
  processEntries: readonly ProcessEntry[];
  presentation: AnswerPresentation | null;
  usage: Record<string, unknown>;
  /** Why the turn failed, as the source a component applies words to. */
  error: LocalizedText | null;
  /** What a live turn is doing, before a component applies words to it. The view
   *  holds the source and not the resolved line, so a language change redraws it. */
  activity: TurnActivity | null;
  cancelRequested: boolean;
  steeringMessages: readonly string[];
  toolRows: readonly ToolRow[];
  /** The turn's place in the conversation's one ordering space, which commands share: the stored
   *  `turn_number`. A turn the reader has just sent has none until the server answers with its
   *  number; a reader orders it after everything already known. */
  sequence?: number;
  /** The `sequence` of the Run's final answer message, which a judgement names. The server states it with
   *  the stored turn, so a turn that has just finished has none until history is read again. */
  answerSequence: number | null;
  /** The reader's judgement of that message, null when they have given none. */
  feedback: AnswerJudgement | null;
}

/** One command the server executed and kept for this conversation, as a node of its own beside the
 *  turns. It is the conversation's control surface, not an answer: it has no Run behind it, no
 *  judgement, and nothing to fork. */
export interface ChatCommandView {
  readonly kind: 'command';
  readonly id: string;
  /** Where the command sits in the conversation's one ordering space: the same counter as a turn's
   *  `sequence`, so a list can interleave the two. */
  readonly sequence: number;
  readonly name: string;
  readonly input: string;
  /** The result to draw, resolved when the node draws it: the client's own words in the language
   *  showing then, or the server's canonical English when the payload states none of them. */
  readonly result: CommandResultText;
  readonly tone: 'success' | 'error';
}

/** The command as the chat draws it: one persisted command event, shaped for the view. The result
 *  travels as the catalog renderer's resolved-later text, never as words settled here. */
export function chatCommandView(command: ConversationCommand): ChatCommandView {
  return {
    kind: 'command',
    id: command.commandId,
    sequence: command.sequence,
    name: command.name,
    input: command.input,
    result: commandResultText(command),
    tone: command.tone,
  };
}

/** What a live turn's status line shows. Every arm is a machine fact, so a
 *  component applies the words and a language change redraws the line: the server
 *  phase that named it, the newest tool row, or copy the client owns. */
type TurnActivity =
  | {readonly kind: 'phase'; readonly phase: string}
  | {readonly kind: 'tool'}
  | ({readonly kind: 'copy'} & LocalizedText);

/** A turn whose Run has settled: it streams and changes no more. */
export function isTerminalTurnState(state: ChatTurnView['state']): boolean {
  return state === 'succeeded' || state === 'failed' || state === 'cancelled';
}
