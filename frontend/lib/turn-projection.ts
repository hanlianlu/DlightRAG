// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Pure projection of one answer-run event stream onto one turn view. The
 *  feature component reduces a batch through this module; nothing here
 *  touches the DOM or dispatches events. */

import {parseAnswerPresentation} from '../api/conversations.ts';
import type {AnswerRunEvent} from './run-controller.ts';
import {runErrorText, SERVICE_ERROR_TEXT} from './run-errors.ts';
import {MAX_PROCESS_ENTRIES, type ChatTurnView, type ProcessEntry} from './chat-views.ts';
import {applyToolEvent, type ToolEventPayload, type ToolRow} from './tool-events.ts';

type AnswerPhase = 'routing' | 'planning' | 'searching' | 'researching' | 'generating';

export const ANSWER_PHASE_LABELS = {
  routing: 'Routing answer...',
  planning: 'Planning answer...',
  searching: 'Searching knowledge base...',
  researching: 'Researching sources...',
  generating: 'Generating answer...',
} as const satisfies Record<AnswerPhase, string>;

export function answerPhaseLabel(phase: string): string | null {
  if (!Object.hasOwn(ANSWER_PHASE_LABELS, phase)) return null;
  return ANSWER_PHASE_LABELS[phase as AnswerPhase];
}

interface DonePayload {
  status: 'succeeded' | 'cancelled';
  presentation: unknown;
  usage?: Record<string, unknown>;
}

function isDonePayload(payload: unknown): payload is DonePayload {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) return false;
  const candidate = payload as Record<string, unknown>;
  return candidate.status === 'succeeded' || candidate.status === 'cancelled';
}

/** The key one Tool row is held by in the merged trace: the call id the wire stated, or else the
 *  ordinal the trace's own state machine gave it (a row a caller built by hand carries neither). */
function toolEntryKey(row: ToolRow): string {
  return row.callId === '' ? `ordinal:${row.ordinal ?? ''}` : `call:${row.callId}`;
}

/** Keep only the newest entries the live trace's history holds: the block scrolls what remains, and
 *  a run longer than that keeps its record in the Inspector rather than in the page's DOM. */
function cappedProcess(entries: ProcessEntry[]): ProcessEntry[] {
  return entries.length > MAX_PROCESS_ENTRIES ? entries.slice(-MAX_PROCESS_ENTRIES) : entries;
}

/** Fold one reasoning delta onto the merged trace: the fold still being written grows, and a delta
 *  that starts a new block opens one after whatever the model did before it. */
function withReasoning(entries: readonly ProcessEntry[], text: string): ProcessEntry[] {
  const last = entries.at(-1);
  if (last?.kind === 'thinking' && last.open) {
    return [...entries.slice(0, -1), {kind: 'thinking', text: last.text + text, open: true}];
  }
  return cappedProcess([...entries, {kind: 'thinking', text, open: true}]);
}

/** Fold the Tool rows the state machine now holds into the merged trace. A call shuts the reasoning
 *  fold that was still open; an entry already in the trace takes the row's newest state; and a call
 *  the trace has not seen yet appends after the entries that came before it. Entries whose row the
 *  Tool window dropped go with it, so both arms stay one list. */
function withToolRows(entries: readonly ProcessEntry[], toolRows: readonly ToolRow[]): ProcessEntry[] {
  const unplaced = new Map(toolRows.map((row) => [toolEntryKey(row), row]));
  const merged: ProcessEntry[] = [];
  for (const entry of entries) {
    if (entry.kind === 'thinking') {
      merged.push(entry.open ? {...entry, open: false} : entry);
      continue;
    }
    const key = toolEntryKey(entry.row);
    const row = unplaced.get(key);
    if (row === undefined) continue;
    unplaced.delete(key);
    merged.push({kind: 'tool', row});
  }
  for (const row of toolRows) {
    if (unplaced.delete(toolEntryKey(row))) merged.push({kind: 'tool', row});
  }
  return cappedProcess(merged);
}

/** A turn that is no longer writing draws none of its live text: the streamed answer and the process
 *  trace it showed are cleared, and the patch states how the turn ended. One helper, so every arm
 *  that settles a turn silences the same reading surface. */
function settled(
  turn: ChatTurnView,
  patch: Partial<ChatTurnView> & Pick<ChatTurnView, 'state'>,
): ChatTurnView {
  return {...turn, ...patch, processEntries: []};
}

/**
 * Fold one event onto the current turn view. Returns the same reference when
 * the event does not change the view (memory events, unknown phases), so the
 * caller can cheaply detect a no-op batch.
 */
export function applyAnswerEvent(
  turn: ChatTurnView,
  event: AnswerRunEvent,
  now: number,
): ChatTurnView {
  switch (event.kind) {
    case 'memory':
      return turn;
    case 'token':
      return {
        ...turn,
        state: 'streaming',
        streamText: turn.streamText + event.text,
        activity: turn.streamText ? turn.activity : null,
        error: null,
      };
    case 'reasoning':
      return {
        ...turn,
        state: turn.state === 'pending' ? 'streaming' : turn.state,
        processEntries: withReasoning(turn.processEntries, event.text),
      };
    case 'reset':
      return settled(turn, {state: 'pending', streamText: '', activity: null, error: null});
    case 'progress': {
      const phase = String((event.payload as {phase?: string})?.phase || '');
      if (answerPhaseLabel(phase) === null) return turn;
      return {...turn, activity: {kind: 'phase', phase}, error: null};
    }
    case 'tool': {
      const info = event.payload as ToolEventPayload;
      if (!info || typeof info.tool_name !== 'string') return turn;
      const toolRows = applyToolEvent(turn.toolRows, event.eventType, info, now);
      // A Tool call ends the reasoning that chose it: its fold shuts, and the call's own entry joins
      // the same rolling trace.
      return {
        ...turn,
        toolRows,
        processEntries: withToolRows(turn.processEntries, toolRows),
        activity: {kind: 'tool'},
        error: null,
      };
    }
    case 'error':
      return settled(turn, {
        state: 'failed', error: runErrorText(event.payload), activity: null,
      });
    case 'done': {
      const payload = event.payload;
      if (!isDonePayload(payload)) {
        return settled(turn, {state: 'failed', error: SERVICE_ERROR_TEXT, activity: null});
      }
      if (payload.status === 'cancelled') {
        return settled(turn, {state: 'cancelled', activity: null});
      }
      // The frame's presentation is the one the history serves, so the schema that
      // owns the type reads it; a missing or malformed one is a service error.
      const presentation = parseAnswerPresentation(payload.presentation);
      if (presentation === null) {
        return settled(turn, {state: 'failed', error: SERVICE_ERROR_TEXT, activity: null});
      }
      return settled(turn, {
        state: 'succeeded',
        presentation,
        streamText: presentation.answerText,
        usage: payload.usage ?? {},
        activity: null,
      });
    }
  }
}
