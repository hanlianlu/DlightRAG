// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Product adapter for split-layout state, breakpoints, and persistence. */

import {DlSplitLayout} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA, MOBILE_MEDIA} from '../lib/breakpoints.ts';
import {PANEL_SHARES, shareWidth, type PanelGrain} from '../lib/panel-share.ts';
import {readStored, writeStored} from '../lib/storage.ts';
import type {InspectorKind} from './inspector.ts';

const INSPECTOR_MIN_WIDTH = PANEL_SHARES.inspector.min;
const CONVERSATION_MIN_WIDTH = PANEL_SHARES.conversation.min;
const CONVERSATION_MAX_WIDTH = PANEL_SHARES.conversation.max;
const CHAT_RESERVE = 520;
const DIVIDER_SIZE = 1;
/** The width a panel opens at as a compact drawer, where a phone gives it the whole column. */
const DRAWER_WIDTH = 420;

type WidthVar = '--panel-width' | '--artifact-canvas-width' | '--layout-chat-sidebar-width';

interface SplitState {
  split: DlSplitLayout;
  panel: HTMLElement;
  widthVar: WidthVar;
  storageKey: string;
  /** The width the stylesheet declares, kept only for a split with no rendered axis to measure. */
  fallbackWidth: number;
  /** The width a reader stated in pixels; a panel that is not explicit follows its share instead. */
  preferred: number;
  /** Whether `preferred` is the reader's own width, which a default never replaces. */
  explicit: boolean;
  /** What the panel last showed. An opening or a kind change asks for a default; every other sync
   *  (a resize, the other panels opening) only clamps what is drawn. */
  kind: PanelGrain;
  /** Whether the panel was open at the previous sync, whatever the Shell's mode was: crossing the
   *  compact breakpoint must not read as a fresh opening. */
  wasOpen: boolean;
  /** The panel opened, or changed grain, as a compact drawer, so its share is still owed: it is asked
   *  for when the panel docks. */
  awaitsDock: boolean;
  minWidth: number;
  maxWidth: number;
}

/** What the Shell knows of its panels that the splits need. The Shell passes it with each sync, so the
 *  splits read the Shell's state directly instead of the classes it keeps on <body> for the stylesheets. */
export interface ShellSplitState {
  /** The conversation list is showing, as a column or as a drawer. */
  conversationOpen: boolean;
  /** The Artifact Canvas covers the screen. */
  canvasOverlay: boolean;
  /** The Artifact Canvas takes the whole column beside the chat. */
  canvasWide: boolean;
  /** The Inspector's read grain, which is only "traces" for the wider share; null when it is closed. */
  inspectorKind: InspectorKind | null;
}

/** The Shell with nothing open, which is what the splits assume until a Shell says otherwise. */
export const IDLE_SHELL_SPLIT_STATE: ShellSplitState = {
  conversationOpen: false,
  canvasOverlay: false,
  canvasWide: false,
  inspectorKind: null,
};

let shell: ShellSplitState = IDLE_SHELL_SPLIT_STATE;
let states: SplitState[] = [];
let widthFrame: number | null = null;
let initialized = false;

function isConversation(state: SplitState): boolean {
  return state.widthVar === '--layout-chat-sidebar-width';
}

/** The grain a panel shows, which decides the share of the split it opens at. */
function panelKind(state: Pick<SplitState, 'widthVar'>): PanelGrain {
  if (state.widthVar === '--layout-chat-sidebar-width') return 'conversation';
  if (state.widthVar === '--artifact-canvas-width') return 'canvas';
  return shell.inspectorKind === 'traces' ? 'traces' : 'inspector';
}

/**
 * The width a split opens at before it has an axis to measure: the grain's own minimum. A layout
 * that has not rendered cannot know its share; the first measurable sync derives the share, and a
 * reader's stored pixels win before either.
 */
function fallbackWidth(widthVar: WidthVar): number {
  if (widthVar === '--layout-chat-sidebar-width') return PANEL_SHARES.conversation.min;
  if (widthVar === '--artifact-canvas-width') return PANEL_SHARES.canvas.min;
  return PANEL_SHARES.inspector.min;
}

/** The split's own axis in pixels: the conversation split spans the Shell, the panel split is the
 *  column beside it, and the Canvas split is what that column has left beside the chat. */
function splitAxis(state: SplitState): number {
  return state.split.clientWidth;
}

/**
 * The most a docked panel can take: its split's axis less the chat's reserve, the panel docked
 * opposite it, and the dividers between them — and never less than the panel's own minimum.
 */
function dockedMaximum(state: SplitState, axis: number): number {
  const artifact = states.find((candidate) => candidate.widthVar === '--artifact-canvas-width');
  const otherWidth = state.widthVar === '--panel-width'
    && artifact !== undefined
    && desktopOpen(artifact)
    && !shell.canvasOverlay
    ? artifact.split.size
    : 0;
  const dividerReserve = DIVIDER_SIZE * (otherWidth > 0 ? 2 : 1);
  const available = Math.max(
    state.minWidth,
    axis - CHAT_RESERVE - otherWidth - dividerReserve,
  );
  return isConversation(state) ? Math.min(state.maxWidth, available) : available;
}

/**
 * The width a panel opens at: its share of the split's own axis, rounded, held between the share's
 * minimum and maximum, then held again to the room the split leaves beside the chat and the other
 * docked panel. A split with no rendered width yet falls back to the stylesheet's width.
 */
function proportionalDefault(state: SplitState): number {
  const axis = splitAxis(state);
  if (!(axis > 0)) return state.fallbackWidth;
  return Math.min(shareWidth(panelKind(state), axis), dockedMaximum(state, axis));
}

/** What a compact drawer draws when the reader has stated no width: the whole column on a phone,
 *  otherwise DRAWER_WIDTH, held to the room the column has. */
function drawerWidth(state: SplitState): number {
  return window.matchMedia(MOBILE_MEDIA).matches
    ? state.split.max
    : Math.min(DRAWER_WIDTH, state.split.max);
}

function savePreferred(state: SplitState): void {
  writeStored(state.storageKey, String(state.preferred));
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
  const expandedCanvas = shell.canvasWide || shell.canvasOverlay;
  for (const state of states) {
    if (isConversation(state)) {
      state.split.min = state.minWidth;
      const axis = state.split.clientWidth;
      state.split.max = drawer
        ? state.minWidth
        : (axis <= 0 ? state.maxWidth : dockedMaximum(state, axis));
      continue;
    }
    if (drawer) {
      state.split.max = Math.max(state.minWidth, state.split.clientWidth);
      continue;
    }
    if (state.widthVar === '--artifact-canvas-width' && expandedCanvas) {
      // The divider is hidden while expanded, so the canvas may take the full column.
      state.split.max = Math.max(state.minWidth, state.split.clientWidth);
      continue;
    }
    state.split.max = dockedMaximum(state, state.split.clientWidth);
  }
}

function desktopOpen(state: SplitState): boolean {
  if (isConversation(state)) {
    return state.panel.classList.contains('open') || shell.conversationOpen;
  }
  return state.panel.classList.contains('open');
}

/** Whether the split shows its panel beside the chat: the Canvas always does, the rest unless it covers them. */
function splitOpen(state: SplitState, drawer: boolean, overlayCanvas: boolean): boolean {
  const open = desktopOpen(state);
  if (isConversation(state)) return open && !drawer && !overlayCanvas;
  return state.widthVar === '--artifact-canvas-width' ? open : open && !overlayCanvas;
}

/** Lay the splits out for the Shell's panels. The Shell says what is open; a sync the splits start for
 *  themselves, a resize or a divider dragged shut, goes on with the last thing it said. */
export function syncPanelSplitState(next?: ShellSplitState): void {
  if (next) shell = next;
  const {canvasOverlay: overlayCanvas, canvasWide: wideCanvas} = shell;
  const drawer = window.matchMedia(COMPACT_SHELL_MEDIA).matches;
  // A panel that opens — or reopens as another grain — asks for its share of the split it is joining,
  // unless the reader has stated a width of their own. An opening is an opening whatever Shell mode
  // the page is in; crossing the compact breakpoint (or a Canvas taking the screen) is a resize, not
  // a fresh opening, so it only clamps what is drawn. Everything else only clamps.
  // A compact drawer asks for no share ("Compact drawers ... keep their existing behavior", ADR 0068):
  // it fills a phone's column and opens at DRAWER_WIDTH beyond it. The share it was owed is asked for
  // when it docks.
  for (const state of states) {
    const open = splitOpen(state, drawer, overlayCanvas);
    const kind = panelKind(state);
    const opened = open && !state.wasOpen;
    const grainChanged = state.wasOpen && open && kind !== state.kind;
    state.wasOpen = open;
    state.kind = kind;
    if (!open) {
      state.awaitsDock = false;
      continue;
    }
    if (state.explicit || !(opened || grainChanged || state.awaitsDock)) continue;
    if (drawer) {
      state.awaitsDock = true;
    } else if (splitAxis(state) > 0) {
      state.awaitsDock = false;
      state.preferred = proportionalDefault(state);
    }
  }
  for (const state of states) {
    const open = splitOpen(state, drawer, overlayCanvas);
    const canvasState = state.widthVar === '--artifact-canvas-width';
    state.split.disabled = drawer || !open || (canvasState && (wideCanvas || overlayCanvas));
    state.split.toggleAttribute('data-open', open);
    state.split.size = open ? state.preferred : 0;
  }
  // Bounds depend on the other split's rendered size, so clamp only after both
  // open/closed positions have been projected in the first pass.
  updateMaximums();
  for (const state of states) {
    if (!splitOpen(state, drawer, overlayCanvas)) continue;
    const canvasState = state.widthVar === '--artifact-canvas-width';
    if (canvasState && (wideCanvas || overlayCanvas)) state.split.size = state.split.max;
    else if (drawer && !state.explicit) state.split.size = drawerWidth(state);
    else state.split.size = Math.min(state.preferred, state.split.max);
  }
  scheduleRenderedWidthSync();
}

/**
 * Ask a panel to put itself away, in whatever way that panel means: a rail, or a closed drawer.
 *
 * The sash leaves with the panel. Whoever holds focus on it, however they came to (Enter on it, or a
 * drag begun after tabbing to it), has the panel hand focus on as its own close control does; where
 * focus is anywhere else, it stays there.
 */
function collapsePanel(state: SplitState): void {
  const panel = state.panel as {
    collapse?: (restoreFocus?: boolean) => void;
    close?: (restoreFocus?: boolean) => void;
  };
  // The sash lives in the split's shadow tree, so the document sees the split itself as focused.
  const holdsFocus = document.activeElement === state.split;
  if (typeof panel.collapse === 'function') panel.collapse(holdsFocus);
  else if (typeof panel.close === 'function') panel.close(holdsFocus);
  syncPanelSplitState();
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
  const unmeasuredWidth = Math.min(maxWidth, fallbackWidth(widthVar));
  const state: SplitState = {
    split,
    panel,
    widthVar,
    storageKey,
    minWidth,
    maxWidth,
    fallbackWidth: unmeasuredWidth,
    preferred: unmeasuredWidth,
    explicit: false,
    kind: panelKind({widthVar}),
    wasOpen: false,
    awaitsDock: false,
  };
  const stored = readStored(storageKey);
  const value = stored === null ? Number.NaN : Number.parseInt(stored, 10);
  if (Number.isFinite(value) && value >= minWidth) {
    state.preferred = Math.min(maxWidth, value);
    state.explicit = true;
  } else {
    state.preferred = proportionalDefault(state);
  }
  return state;
}

function bindState(state: SplitState): void {
  state.split.min = state.minWidth;
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
    if (state.split.size <= 0) {
      // Dragged shut. The panel puts itself away, and the width it had stays `preferred`: that is
      // the width it comes back at, which is what the window splitter pattern asks Enter to restore.
      setResizing(false);
      collapsePanel(state);
      return;
    }
    updateMaximums();
    state.preferred = Math.round(state.split.size);
    state.explicit = true;
    state.split.size = state.preferred;
    savePreferred(state);
    setResizing(false);
    scheduleRenderedWidthSync();
  });
  // Double-clicking the sash asks for the panel's default width — its share of the split it lives
  // in now, for the grain it is showing — never for a collapse: the two gestures are separate,
  // which is how VS Code settled it. The default is stored as a drag stores its width, so it is
  // also what the panel comes back at; a window too narrow for it only clamps what is drawn.
  state.split.addEventListener('dl-split-reset', (event) => {
    if (event.target !== state.split || !desktopOpen(state)) return;
    updateMaximums();
    state.preferred = proportionalDefault(state);
    state.explicit = true;
    state.split.size = state.preferred;
    savePreferred(state);
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
  window.addEventListener('resize', () => { syncPanelSplitState(); });
  syncPanelSplitState();
}
