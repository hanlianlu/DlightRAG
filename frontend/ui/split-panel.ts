// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Product adapter for split-layout state, breakpoints, and persistence. */

import {DlSplitLayout} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA} from '../lib/breakpoints.ts';

const MIN_WIDTH = 320;
const CHAT_RESERVE = 520;
const DIVIDER_SIZE = 1;

type WidthVar = '--panel-width' | '--artifact-canvas-width';

interface SplitState {
  split: DlSplitLayout;
  panel: HTMLElement;
  widthVar: WidthVar;
  storageKey: string;
  preferred: number;
}

let states: SplitState[] = [];
let widthFrame: number | null = null;
let initialized = false;

function cssDefault(widthVar: WidthVar): number {
  const value = Number.parseInt(
    getComputedStyle(document.documentElement).getPropertyValue(widthVar),
    10,
  );
  return Number.isFinite(value) && value >= MIN_WIDTH ? value : 420;
}

function loadPreferred(storageKey: string, widthVar: WidthVar): number {
  try {
    const stored = localStorage.getItem(storageKey);
    const value = stored === null ? NaN : Number.parseInt(stored, 10);
    if (Number.isFinite(value) && value >= MIN_WIDTH) return value;
  } catch {
    // Storage is an enhancement; the CSS default remains authoritative.
  }
  return cssDefault(widthVar);
}

function savePreferred(state: SplitState): void {
  try {
    localStorage.setItem(state.storageKey, String(state.preferred));
  } catch {
    // Resizing remains available when storage is blocked.
  }
}

function syncRenderedWidthsNow(): void {
  for (const state of states) {
    if (!state.panel.classList.contains('open')) continue;
    const width = Math.round(state.panel.getBoundingClientRect().width || state.split.size);
    document.documentElement.style.setProperty(state.widthVar, `${width}px`);
  }
}

function scheduleRenderedWidthSync(): void {
  if (widthFrame !== null) cancelAnimationFrame(widthFrame);
  widthFrame = requestAnimationFrame(() => {
    widthFrame = null;
    syncRenderedWidthsNow();
  });
}

function conversationReserve(): number {
  if (!document.body.classList.contains('conversation-sidebar-open')) return CHAT_RESERVE;
  const sidebar = document.getElementById('chat-sidebar');
  return CHAT_RESERVE + (sidebar?.getBoundingClientRect().width ?? 0);
}

function updateMaximums(): void {
  const drawer = window.matchMedia(COMPACT_SHELL_MEDIA).matches;
  const reserve = conversationReserve();
  const artifact = states.find((state) => state.widthVar === '--artifact-canvas-width');
  for (const state of states) {
    if (drawer) {
      state.split.max = Math.max(MIN_WIDTH, state.split.clientWidth);
      continue;
    }
    let otherWidth = 0;
    if (state.widthVar === '--panel-width'
      && document.body.classList.contains('artifact-canvas-open')
      && !document.body.classList.contains('artifact-canvas-overlay')) {
      otherWidth = artifact?.split.size ?? 0;
    }
    const dividerReserve = DIVIDER_SIZE * (otherWidth > 0 ? 2 : 1);
    state.split.max = Math.max(
      MIN_WIDTH,
      state.split.clientWidth - reserve - otherWidth - dividerReserve,
    );
  }
}

export function syncPanelSplitState(): void {
  const drawer = window.matchMedia(COMPACT_SHELL_MEDIA).matches;
  for (const state of states) {
    const open = state.panel.classList.contains('open');
    state.split.disabled = drawer || !open;
    state.split.toggleAttribute('data-open', open);
    state.split.size = open ? state.preferred : 0;
  }
  // Bounds depend on the other split's rendered size, so clamp only after both
  // open/closed positions have been projected in the first pass.
  updateMaximums();
  for (const state of states) {
    if (state.panel.classList.contains('open')) {
      state.split.size = Math.min(state.preferred, state.split.max);
    }
  }
  scheduleRenderedWidthSync();
}

function setResizing(active: boolean): void {
  document.body.toggleAttribute('data-resizing', active);
  document.body.style.userSelect = active ? 'none' : '';
  document.body.style.cursor = active ? 'col-resize' : '';
}

function createState(
  splitId: string,
  panelId: string,
  widthVar: WidthVar,
  storageKey: string,
): SplitState | null {
  const split = document.getElementById(splitId);
  const panel = document.getElementById(panelId);
  if (!(split instanceof DlSplitLayout) || !panel) return null;
  return {
    split,
    panel,
    widthVar,
    storageKey,
    preferred: loadPreferred(storageKey, widthVar),
  };
}

function bindState(state: SplitState): void {
  state.split.divider.setAttribute(
    'aria-label',
    state.widthVar === '--panel-width' ? 'Resize Files or Sources' : 'Resize Artifact Canvas',
  );
  state.split.divider.addEventListener('pointerdown', () => {
    if (!state.split.disabled) setResizing(true);
  });
  state.split.addEventListener('dl-split-input', (event) => {
    if (event.target !== state.split) return;
    updateMaximums();
    document.documentElement.style.setProperty(state.widthVar, `${state.split.size}px`);
    scheduleRenderedWidthSync();
  });
  state.split.addEventListener('dl-split-change', (event) => {
    if (event.target !== state.split || !state.panel.classList.contains('open')) return;
    updateMaximums();
    state.preferred = Math.round(state.split.size);
    state.split.size = state.preferred;
    savePreferred(state);
    setResizing(false);
    scheduleRenderedWidthSync();
  });
}

/** Bind the app-unique split controls for the Vite document lifetime. */
export function setupPanelSplits(): void {
  if (initialized) return;
  const nextStates = [
    createState('panel-split', 'inspector', '--panel-width', 'dlightrag-panel-width'),
    createState(
      'artifact-canvas-split',
      'artifact-canvas',
      '--artifact-canvas-width',
      'dlightrag-artifact-canvas-width',
    ),
  ].filter((state): state is SplitState => state !== null);
  if (nextStates.length === 0) return;
  initialized = true;
  states = nextStates;
  for (const state of states) bindState(state);
  window.addEventListener('pointerup', () => setResizing(false));
  window.addEventListener('pointercancel', () => setResizing(false));
  window.addEventListener('blur', () => setResizing(false));
  window.addEventListener('resize', syncPanelSplitState);
  syncPanelSplitState();
}
