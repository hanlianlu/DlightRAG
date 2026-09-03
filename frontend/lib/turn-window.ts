// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Visible index window for conversation turns. Streaming turns stay mounted. */

export const TURN_WINDOW_PAD = 20;
export const TURN_PLACEHOLDER_MIN_PX = 96;

export interface TurnWindowInput {
  readonly count: number;
  readonly heights: readonly number[];
  readonly scrollTop: number;
  readonly viewportHeight: number;
  readonly alwaysOn: ReadonlySet<number>;
}

export interface TurnWindow {
  readonly start: number;
  readonly end: number;
}

function heightAt(heights: readonly number[], index: number): number {
  const value = heights[index];
  return value > 0 ? value : TURN_PLACEHOLDER_MIN_PX;
}

export function visibleTurnWindow(input: TurnWindowInput): TurnWindow {
  const {count, heights, scrollTop, viewportHeight, alwaysOn} = input;
  if (count <= 0) return {start: 0, end: -1};
  if (count <= TURN_WINDOW_PAD * 2 + alwaysOn.size) {
    return {start: 0, end: count - 1};
  }
  const viewTop = Math.max(0, scrollTop);
  const viewBottom = viewTop + Math.max(1, viewportHeight);
  let acc = 0;
  let first = 0;
  let last = count - 1;
  let sawFirst = false;
  for (let index = 0; index < count; index += 1) {
    const top = acc;
    acc += heightAt(heights, index);
    if (!sawFirst && acc > viewTop) {
      first = index;
      sawFirst = true;
    }
    if (top < viewBottom) last = index;
  }
  let start = Math.max(0, first - TURN_WINDOW_PAD);
  let end = Math.min(count - 1, last + TURN_WINDOW_PAD);
  for (const index of alwaysOn) {
    if (index < 0 || index >= count) continue;
    start = Math.min(start, index);
    end = Math.max(end, index);
  }
  return {start, end};
}

export function turnIsLive(state: string): boolean {
  return state === 'pending' || state === 'streaming' || state === 'retryable';
}
