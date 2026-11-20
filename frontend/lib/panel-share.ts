// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The Shell's proportional panel defaults, in one place (ADR 0068).
 *
 * Every docked panel opens at a share of the split it lives in — not at a fixed pixel token — so the
 * policy has one source: the table below and `shareWidth`. The split adapter in `ui/split-panel.ts`
 * owns the pixel state (drag, persistence, clamping to the room the chat keeps); it asks this module
 * only what width a grain opens at. A width the reader stated is pixels and never passes through here. */

/** The read grain a docked panel shows. Files, Sources and the live browser are one grain. */
export type PanelGrain = 'conversation' | 'inspector' | 'traces' | 'canvas';

/** One grain's default: a share of its split's axis, held between a minimum and a maximum. */
export interface PanelShare {
  readonly fraction: number;
  readonly min: number;
  readonly max: number;
}

/** Owner-approved defaults (ADR 0068): see the table there for why each share is what it is. */
export const PANEL_SHARES: Readonly<Record<PanelGrain, PanelShare>> = {
  conversation: {fraction: 0.18, min: 240, max: 360},
  // Files, Sources and the live browser are the same read grain, and open at the same share.
  inspector: {fraction: 0.38, min: 320, max: 560},
  // The traces grain is the wider read: half the split, with its own minimum and maximum.
  traces: {fraction: 0.5, min: 420, max: 960},
  canvas: {fraction: 0.42, min: 320, max: 720},
};

/** The width one grain opens at on a split whose axis is `axis` pixels: its share, rounded, held
 *  between the grain's own minimum and maximum. The caller still clamps it to the room the split
 *  leaves beside the chat and any other docked panel. */
export function shareWidth(grain: PanelGrain, axis: number): number {
  const share = PANEL_SHARES[grain];
  return Math.min(Math.max(Math.round(share.fraction * axis), share.min), share.max);
}
