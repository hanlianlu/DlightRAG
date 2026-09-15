// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Pure state machine for the compact per-run tool trace shown while a turn
 *  runs and re-inspectable after it settles.
 *
 *  The clock is a parameter, never a global read: a row is stamped when the
 *  viewer observes its start, so the fold stays deterministic and testable
 *  while the view owns the ticking. */

import {toolDisplay, toolVerbText} from './tool-display.ts';

export interface ToolRow {
  readonly callId: string;
  /** Server-resolved display label; null means this viewer must name the tool. */
  readonly label: string | null;
  readonly name: string;
  readonly verb: string;
  readonly verbId: string | null;
  readonly object: string;
  readonly state: 'running' | 'done' | 'failed';
  /** Monotonic milliseconds when this viewer saw the call start, while running. */
  readonly startedAt: number | null;
  /** Server-measured wall time, present only once the call settled. */
  readonly durationMs: number | null;
}

export interface ToolEventPayload {
  tool_name?: string;
  call_id?: string;
  object_label?: string;
  tool_label?: string;
  outcome?: string;
  duration_ms?: number;
}

export const MAX_TOOL_ROWS = 5;

export function applyToolEvent(
  rows: readonly ToolRow[],
  eventType: string,
  payload: ToolEventPayload,
  now: number,
): readonly ToolRow[] {
  const name = typeof payload.tool_name === 'string' ? payload.tool_name : '';
  const callId = typeof payload.call_id === 'string' ? payload.call_id : '';
  const label = typeof payload.tool_label === 'string' && payload.tool_label
    ? payload.tool_label
    : null;
  if (eventType === 'tool_start') {
    if (!name) return rows;
    const display = toolDisplay(name);
    const row: ToolRow = {
      callId,
      label,
      name,
      verb: display.verb,
      verbId: display.verbId,
      object: '',
      state: 'running',
      startedAt: now,
      durationMs: null,
    };
    return [...rows, row].slice(-MAX_TOOL_ROWS);
  }
  const index = rows.findIndex((row) => row.callId !== '' && row.callId === callId);
  if (index < 0) return rows;
  if (eventType === 'tool_progress') {
    const object = typeof payload.object_label === 'string' ? payload.object_label : '';
    return rows.map((row, i) => (
      i === index ? {...row, label: label ?? row.label, object: object || row.object} : row
    ));
  }
  if (eventType === 'tool_end') {
    const durationMs = typeof payload.duration_ms === 'number' && payload.duration_ms >= 0
      ? payload.duration_ms
      : null;
    return rows.map((row, i) => (
      i === index
        ? {
            ...row,
            label: label ?? row.label,
            state: payload.outcome === 'succeeded' ? 'done' : 'failed',
            startedAt: null,
            durationMs,
          }
        : row
    ));
  }
  return rows;
}

/** The duration this row shows now: server truth once settled, viewer time while
 *  running. A settlement that published no duration shows none, so a number a
 *  reader sees is never this viewer's guess about a finished call. */
export function rowDurationMs(row: ToolRow, now: number): number | null {
  if (row.state !== 'running') return row.durationMs;
  if (row.startedAt === null) return null;
  return Math.max(0, now - row.startedAt);
}

/** One display line for a tool row (server label, else localized verb, plus an
 *  optional object name). */
export function toolRowText(row: ToolRow): string {
  const verb = row.label ?? toolVerbText(row.verb, row.verbId);
  return row.object ? `${verb} — ${row.object}` : verb;
}

/** The live status line for the most recent row. */
export function toolStatusText(rows: readonly ToolRow[]): string {
  const row = rows[rows.length - 1];
  return row ? toolRowText(row) : '';
}
