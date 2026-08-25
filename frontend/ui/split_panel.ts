// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import WaSplitPanel from '@awesome.me/webawesome/dist/components/split-panel/split-panel.js';
import {COMPACT_SHELL_MEDIA} from '../lib/breakpoints.ts';

const MIN_WIDTH = 320;
const RESIZE_KEYS = new Set(['ArrowLeft', 'ArrowRight', 'Home', 'End']);

type WidthVar = '--panel-width' | '--artifact-canvas-width';

interface SplitState {
    split: WaSplitPanel;
    panel: HTMLElement;
    widthVar: WidthVar;
    storageKey: string;
    preferred: number;
    dragging: boolean;
}

let states: SplitState[] = [];
let widthFrame: number | null = null;

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
    } catch (_error) {
        // Storage is an enhancement; the CSS default remains authoritative.
    }
    return cssDefault(widthVar);
}

function savePreferred(state: SplitState): void {
    try {
        localStorage.setItem(state.storageKey, String(state.preferred));
    } catch (_error) {
        // Resizing remains available when storage is blocked.
    }
}

function syncRenderedWidthsNow(): void {
    for (const state of states) {
        if (!state.panel.classList.contains('open')) continue;
        const width = state.panel.getBoundingClientRect().width;
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

export function syncPanelSplitState(): void {
    const drawer = window.matchMedia(COMPACT_SHELL_MEDIA).matches;
    for (const state of states) {
        const open = state.panel.classList.contains('open');
        state.split.disabled = drawer || !open;
        state.split.toggleAttribute('data-open', open);
        const position = open ? state.preferred : 0;
        if (Math.abs(state.split.positionInPixels - position) > 0.5) {
            state.split.positionInPixels = position;
        }
    }
    scheduleRenderedWidthSync();
}

function setResizing(active: boolean): void {
    document.body.toggleAttribute('data-resizing', active);
    document.body.style.userSelect = active ? 'none' : '';
    document.body.style.cursor = active ? 'col-resize' : '';
}

function commitPosition(state: SplitState): void {
    if (!state.panel.classList.contains('open')) return;
    state.preferred = Math.round(state.panel.getBoundingClientRect().width);
    savePreferred(state);
    state.split.positionInPixels = state.preferred;
    scheduleRenderedWidthSync();
}

function cancelDrags(): void {
    if (!states.some((state) => state.dragging)) return;
    document.dispatchEvent(new Event('pointerup'));
    finishDrags();
}

function finishDrags(): void {
    const finished = states.filter((state) => state.dragging);
    if (finished.length === 0) return;
    for (const state of finished) state.dragging = false;
    void Promise.all(finished.map((state) => state.split.updateComplete)).then(() => {
        requestAnimationFrame(() => {
            for (const state of finished) {
                if (!state.dragging) commitPosition(state);
            }
            if (!states.some((state) => state.dragging)) setResizing(false);
        });
    });
}

function bindDivider(state: SplitState): void {
    state.split.divider.setAttribute(
        'aria-label',
        state.widthVar === '--panel-width' ? 'Resize Files or Sources' : 'Resize Artifact Canvas',
    );
    const begin = (): void => {
        if (state.split.disabled) return;
        state.dragging = true;
        setResizing(true);
    };
    state.split.divider.addEventListener('mousedown', begin);
    state.split.divider.addEventListener('touchstart', begin, {passive: true});
    state.split.divider.addEventListener(
        'keydown',
        (event) => {
            if (event.key !== 'Enter') return;
            event.preventDefault();
            event.stopImmediatePropagation();
        },
        {capture: true},
    );
    state.split.divider.addEventListener('keydown', (event) => {
        if (state.split.disabled || !RESIZE_KEYS.has(event.key)) return;
        void state.split.updateComplete.then(() => {
            scheduleRenderedWidthSync();
            requestAnimationFrame(() => {
                commitPosition(state);
            });
        });
    });
}

function createState(
    splitId: string,
    panelId: string,
    widthVar: WidthVar,
    storageKey: string,
): SplitState | null {
    const split = document.getElementById(splitId);
    const panel = document.getElementById(panelId);
    if (!(split instanceof WaSplitPanel) || !panel) return null;
    return {
        split,
        panel,
        widthVar,
        storageKey,
        preferred: loadPreferred(storageKey, widthVar),
        dragging: false,
    };
}

/** Bind Web Awesome's divider to DlightRAG panel state and persisted widths. */
export function setupPanelSplits(): void {
    states = [
        createState('panel-split', 'panel', '--panel-width', 'dlightrag-panel-width'),
        createState(
            'artifact-canvas-split',
            'artifact-canvas',
            '--artifact-canvas-width',
            'dlightrag-artifact-canvas-width',
        ),
    ].filter((state): state is SplitState => state !== null);

    for (const state of states) {
        state.split.addEventListener('wa-reposition', scheduleRenderedWidthSync);
        void state.split.updateComplete.then(() => {
            bindDivider(state);
        });
    }
    for (const event of ['pointerup', 'mouseup', 'touchend']) {
        window.addEventListener(event, finishDrags);
    }
    for (const event of ['pointercancel', 'touchcancel', 'blur']) {
        window.addEventListener(event, cancelDrags);
    }
    syncPanelSplitState();
    window.addEventListener('resize', syncPanelSplitState);
}
