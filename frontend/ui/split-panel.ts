// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Product adapter for split-layout state, breakpoints, and persistence. */

import {DlSplitLayout} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA} from '../lib/breakpoints.ts';

const INSPECTOR_MIN_WIDTH = 320;
const CONVERSATION_MIN_WIDTH = 240;
const CONVERSATION_MAX_WIDTH = 360;
const CONVERSATION_DEFAULT_WIDTH = 260;
const INSPECTOR_DEFAULT_WIDTH = 420;
const CHAT_RESERVE = 520;
const DIVIDER_SIZE = 1;

type WidthVar = '--panel-width' | '--artifact-canvas-width' | '--layout-chat-sidebar-width';

interface SplitState {
  split: DlSplitLayout;
  panel: HTMLElement;
  widthVar: WidthVar;
  storageKey: string;
  preferred: number;
  minWidth: number;
  maxWidth: number;
}

let states: SplitState[] = [];
let widthFrame: number | null = null;
let initialized = false;

function isConversation(state: SplitState): boolean {
  return state.widthVar === '--layout-chat-sidebar-width';
}

function cssDefault(state: Pick<SplitState, 'widthVar' | 'minWidth'>): number {
  const value = Number.parseInt(
    getComputedStyle(document.documentElement).getPropertyValue(state.widthVar),
    10,
  );
  if (Number.isFinite(value) && value >= state.minWidth) return value;
  return state.widthVar === '--layout-chat-sidebar-width'
    ? CONVERSATION_DEFAULT_WIDTH
    : INSPECTOR_DEFAULT_WIDTH;
}

function loadPreferred(state: Omit<SplitState, 'split' | 'panel' | 'preferred'>): number {
  try {
    const stored = localStorage.getItem(state.storageKey);
    const value = stored === null ? NaN : Number.parseInt(stored, 10);
    if (Number.isFinite(value) && value >= state.minWidth) {
      return Math.min(state.maxWidth, value);
    }
  } catch {
    // Storage is an enhancement; the CSS default remains authoritative.
  }
  return Math.min(state.maxWidth, cssDefault(state));
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
    if (isConversation(state)) {
      const width = state.split.size > 0 ? state.split.size : state.preferred;
      document.documentElement.style.setProperty(state.widthVar, `${Math.round(width)}px`);
      continue;
    }
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

function updateMaximums(): void {
  const drawer = window.matchMedia(COMPACT_SHELL_MEDIA).matches;
  const artifact = states.find((state) => state.widthVar === '--artifact-canvas-width');
  for (const state of states) {
    if (isConversation(state)) {
      state.split.min = state.minWidth;
      const axis = state.split.clientWidth;
      state.split.max = drawer || axis <= 0
        ? (drawer ? state.minWidth : state.maxWidth)
        : Math.min(
          state.maxWidth,
          Math.max(state.minWidth, axis - CHAT_RESERVE - DIVIDER_SIZE),
        );
      continue;
    }
    if (drawer) {
      state.split.max = Math.max(state.minWidth, state.split.clientWidth);
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
      state.minWidth,
      state.split.clientWidth - CHAT_RESERVE - otherWidth - dividerReserve,
    );
  }
}

function desktopOpen(state: SplitState): boolean {
  if (isConversation(state)) {
    return state.panel.classList.contains('open')
      || document.body.classList.contains('conversation-sidebar-open');
  }
  return state.panel.classList.contains('open');
}

export function syncPanelSplitState(): void {
  const drawer = window.matchMedia(COMPACT_SHELL_MEDIA).matches;
  for (const state of states) {
    const open = desktopOpen(state);
    const splitOpen = isConversation(state) ? open && !drawer : open;
    state.split.disabled = drawer || !splitOpen;
    state.split.toggleAttribute('data-open', splitOpen);
    state.split.size = splitOpen ? state.preferred : 0;
  }
  // Bounds depend on the other split's rendered size, so clamp only after both
  // open/closed positions have been projected in the first pass.
  updateMaximums();
  for (const state of states) {
    const open = desktopOpen(state);
    const splitOpen = isConversation(state) ? open && !drawer : open;
    if (splitOpen) {
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
  minWidth: number,
  maxWidth: number,
): SplitState | null {
  const split = document.getElementById(splitId);
  const panel = document.getElementById(panelId);
  if (!(split instanceof DlSplitLayout) || !panel) return null;
  const partial = {widthVar, storageKey, minWidth, maxWidth};
  return {
    split,
    panel,
    ...partial,
    preferred: loadPreferred(partial),
  };
}

function resizeLabel(state: SplitState): string {
  if (state.widthVar === '--panel-width') return 'Resize Files or Sources';
  if (state.widthVar === '--artifact-canvas-width') return 'Resize Artifact Canvas';
  return 'Resize conversations';
}

function bindState(state: SplitState): void {
  state.split.min = state.minWidth;
  state.split.divider.setAttribute('aria-label', resizeLabel(state));
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
    if (event.target !== state.split || !desktopOpen(state)) return;
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
    createState(
      'conversation-split',
      'conversation-sidebar',
      '--layout-chat-sidebar-width',
      'dlightrag-conversation-sidebar-width',
      CONVERSATION_MIN_WIDTH,
      CONVERSATION_MAX_WIDTH,
    ),
    createState(
      'panel-split',
      'inspector',
      '--panel-width',
      'dlightrag-panel-width',
      INSPECTOR_MIN_WIDTH,
      Number.POSITIVE_INFINITY,
    ),
    createState(
      'artifact-canvas-split',
      'artifact-canvas',
      '--artifact-canvas-width',
      'dlightrag-artifact-canvas-width',
      INSPECTOR_MIN_WIDTH,
      Number.POSITIVE_INFINITY,
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
