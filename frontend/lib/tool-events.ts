// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** Pure state machine for the compact per-run tool trace shown while a turn
 *  runs and re-inspectable afterwards in the tab that watched it settle. Rows are
 *  a live affordance: they are never stored with the turn, so a reload or another
 *  browser reads the answer without them. While the turn works they are one
 *  process group: a line that names what the Run is doing, how many calls it has
 *  made, and how long it has been at it, which opens on the rows.
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
  /** This call's 1-based place in the turn's trace, so the process group states how many calls
   *  the turn made even after older rows have fallen off the capped list. Absent only on a row
   *  a caller built by hand. */
  readonly ordinal?: number;
  /** Monotonic milliseconds when this viewer saw the trace's first call start; every row of one
   *  turn carries it, so a capped list still times the whole group. */
  readonly traceStartedAt?: number | null;
}

export interface ToolEventPayload {
  tool_name?: string;
  call_id?: string;
  object_label?: string;
  tool_label?: string;
  outcome?: string;
  duration_ms?: number;
  /** Server-measured running time of a replayed start, when it arrived late. */
  elapsed_ms?: number;
}

/** The Tool rows a live turn keeps before the oldest fall out. Enough that the process block's own
 *  scroll holds the turn's calls, not only the newest few; the count and the clock outlive the rows
 *  a run longer than this drops, and the Inspector keeps the whole record. */
export const MAX_TOOL_ROWS = 200;

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
    // A replayed start tells us how long the call has already run, so a reloaded
    // trace keeps counting instead of pretending the call just began.
    const runningFor = typeof payload.elapsed_ms === 'number' && payload.elapsed_ms > 0
      ? payload.elapsed_ms
      : 0;
    const row: ToolRow = {
      callId,
      label,
      name,
      verb: display.verb,
      verbId: display.verbId,
      object: '',
      state: 'running',
      startedAt: now - runningFor,
      durationMs: null,
      // The count and the group's clock outlive the rows the capped list drops, so each row
      // carries them: the newest row states how many calls the trace has seen, and when the
      // first of them began.
      ordinal: (rows[rows.length - 1]?.ordinal ?? rows.length) + 1,
      traceStartedAt: rows[0]?.traceStartedAt ?? now - runningFor,
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

/** What a live turn's process group states in one line, before a component applies the words.
 *
 *  The newest call names what the turn is doing ("Searching the web"), the count is every call
 *  the viewer has seen rather than the rows the capped list still holds, and the clock runs from
 *  the first call the viewer saw, so both outlive the rows that fell off the list. Before the
 *  first call there is only the server's phase to name, and neither a count nor a clock. Null
 *  when there is nothing to say at all.
 *
 *  Wordless: `verb` carries its catalog `verbId` (or is a server label, which needs none), so a
 *  component resolves it and a language change redraws the line. */
export interface ProcessGroupSummary {
  /** The newest call's own words: a server label, or the tool verb the client named it with. */
  readonly verb: string;
  /** The catalog id for `verb`, null when `verb` is a server label or a prettified name. */
  readonly verbId: string | null;
  /** The server phase, present only while no call has landed yet. */
  readonly phase: string | null;
  /** How many calls this viewer has seen this turn; 0 while only a phase names the work. */
  readonly count: number;
  /** Viewer-clock milliseconds since the first call; null before one, or when unmeasured. */
  readonly durationMs: number | null;
}

export function processGroupSummary(
  rows: readonly ToolRow[],
  phase: string | null,
  now: number,
): ProcessGroupSummary | null {
  const newest = rows[rows.length - 1];
  if (newest === undefined) {
    return phase === null ? null : {verb: '', verbId: null, phase, count: 0, durationMs: null};
  }
  const startedAt = typeof newest.traceStartedAt === 'number' ? newest.traceStartedAt : null;
  return {
    verb: newest.label ?? newest.verb,
    verbId: newest.label === null ? newest.verbId : null,
    phase: null,
    count: newest.ordinal ?? rows.length,
    durationMs: startedAt === null ? null : Math.max(0, now - startedAt),
  };
}
