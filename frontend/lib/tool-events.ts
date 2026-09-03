// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Pure state machine for the compact per-run tool trace shown while a turn
 *  runs and re-inspectable after it settles. */

import {toolDisplay, toolVerbText} from './tool-display.ts';

export interface ToolRow {
  readonly callId: string;
  readonly name: string;
  readonly verb: string;
  readonly verbId: string | null;
  readonly object: string;
  readonly state: 'running' | 'done' | 'failed';
  readonly durationMs: number | null;
}

export interface ToolEventPayload {
  tool_name?: string;
  call_id?: string;
  object_label?: string;
  outcome?: string;
  duration_ms?: number;
}

export const MAX_TOOL_ROWS = 5;

export function applyToolEvent(
  rows: readonly ToolRow[],
  eventType: string,
  payload: ToolEventPayload,
): readonly ToolRow[] {
  const name = typeof payload.tool_name === 'string' ? payload.tool_name : '';
  const callId = typeof payload.call_id === 'string' ? payload.call_id : '';
  if (eventType === 'tool_start') {
    if (!name) return rows;
    const display = toolDisplay(name);
    const row: ToolRow = {
      callId,
      name,
      verb: display.verb,
      verbId: display.verbId,
      object: '',
      state: 'running',
      durationMs: null,
    };
    return [...rows, row].slice(-MAX_TOOL_ROWS);
  }
  const index = rows.findIndex((row) => row.callId !== '' && row.callId === callId);
  if (index < 0) return rows;
  if (eventType === 'tool_progress') {
    const object = typeof payload.object_label === 'string' ? payload.object_label : '';
    if (!object) return rows;
    return rows.map((row, i) => (i === index ? {...row, object} : row));
  }
  if (eventType === 'tool_end') {
    const failed = payload.outcome === 'failed';
    const durationMs = typeof payload.duration_ms === 'number' ? payload.duration_ms : null;
    return rows.map((row, i) => (
      i === index ? {...row, state: failed ? 'failed' : 'done', durationMs} : row
    ));
  }
  return rows;
}

/** One display line for a tool row (localized verb + optional object name). */
export function toolRowText(row: ToolRow): string {
  const verb = toolVerbText(row.verb, row.verbId);
  return row.object ? `${verb} — ${row.object}` : verb;
}

/** The live status line for the most recent row. */
export function toolStatusText(rows: readonly ToolRow[]): string {
  const row = rows[rows.length - 1];
  return row ? toolRowText(row) : '';
}
