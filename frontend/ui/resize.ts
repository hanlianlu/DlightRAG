// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const MAIN_STORAGE_KEY = 'dlightrag-panel-width';
const REPORT_STORAGE_KEY = 'dlightrag-report-panel-width';
const MIN_WIDTH = 320;
const COMPACT_MAX_WIDTH = 420;
const DRAWER_MEDIA = '(max-width: 1199px)';

type WidthVar = '--panel-width' | '--report-panel-width';

let mainPreferred = 0;
let reportPreferred = 0;

function isDrawer(): boolean {
    return window.matchMedia(DRAWER_MEDIA).matches;
}

function cssDefault(varName: WidthVar): number {
    const raw = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    const px = parseInt(raw, 10);
    return !isNaN(px) && px >= MIN_WIDTH ? px : 420;
}

function chatReserve(): number {
    const styles = getComputedStyle(document.documentElement);
    const chatMinWidth = parseFloat(styles.getPropertyValue('--layout-chat-min-width')) || 520;
    const sidebar = document.body.classList.contains('conversation-sidebar-open')
        ? document.getElementById('chat-sidebar')?.getBoundingClientRect().width || 0
        : 0;
    return chatMinWidth + sidebar;
}

function otherOpenWidth(exclude: WidthVar): number {
    if (exclude === '--panel-width' && document.body.classList.contains('report-panel-open')) {
        return document.getElementById('report-panel')?.getBoundingClientRect().width || 0;
    }
    if (exclude === '--report-panel-width' && (
        document.body.classList.contains('sources-panel-open')
        || document.body.classList.contains('files-panel-open')
    )) {
        return document.getElementById('panel')?.getBoundingClientRect().width || 0;
    }
    return 0;
}

function getMaxWidth(exclude: WidthVar): number {
    if (window.innerWidth <= 640) return window.innerWidth;
    if (window.innerWidth < 1200) return COMPACT_MAX_WIDTH;
    return Math.max(MIN_WIDTH, Math.floor(window.innerWidth - chatReserve() - otherOpenWidth(exclude)));
}

function clampWidth(w: number, exclude: WidthVar): number {
    const maxWidth = getMaxWidth(exclude);
    const minWidth = Math.min(MIN_WIDTH, maxWidth);
    return Math.max(minWidth, Math.min(w, maxWidth));
}

function loadPreferred(key: string, varName: WidthVar): number {
    try {
        const saved = localStorage.getItem(key);
        if (saved !== null) {
            const n = parseInt(saved, 10);
            if (!isNaN(n) && n >= MIN_WIDTH) return n;
        }
    } catch (_) { /* localStorage unavailable */ }
    return cssDefault(varName);
}

function saveWidth(key: string, w: number): void {
    try {
        localStorage.setItem(key, String(w));
    } catch (_) { /* localStorage unavailable */ }
}

function px(varName: WidthVar, preferred: number): number {
    return clampWidth(preferred, varName);
}

export function syncPanelEffectiveWidth(): number {
    const main = px('--panel-width', mainPreferred);
    const report = px('--report-panel-width', reportPreferred);
    document.documentElement.style.setProperty('--panel-width', main + 'px');
    document.documentElement.style.setProperty('--report-panel-width', report + 'px');
    let side = 0;
    if (!isDrawer()) {
        if (document.getElementById('panel')?.classList.contains('open')) side += main;
        if (document.getElementById('report-panel')?.classList.contains('open')) side += report;
    }
    document.documentElement.style.setProperty('--layout-side-width', side + 'px');
    return main;
}

function bindHandle(
    panel: HTMLElement,
    widthVar: WidthVar,
    storageKey: string,
    preferred: {get: () => number; set: (n: number) => void},
): void {
    let handle = panel.querySelector('.panel-resize-handle');
    if (!handle) {
        handle = document.createElement('div');
        handle.className = 'panel-resize-handle';
        panel.insertBefore(handle, panel.firstChild);
    }
    const resizeHandle = handle;
    let dragging = false;
    let startX = 0;
    let startWidth = 0;
    let rafId: number | null = null;
    let activePointerId: number | null = null;

    function isOtherPointer(e: PointerEvent | Event): boolean {
        return 'pointerId' in e && activePointerId !== null && e.pointerId !== activePointerId;
    }

    function onPointerDown(e: Event): void {
        const event = e as PointerEvent;
        if (dragging) return;
        dragging = true;
        activePointerId = event.pointerId;
        startX = event.clientX;
        startWidth = panel.getBoundingClientRect().width;
        event.preventDefault();
        if (resizeHandle.setPointerCapture) {
            resizeHandle.setPointerCapture(event.pointerId);
        }
        resizeHandle.classList.add('active');
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';
        document.body.setAttribute('data-resizing', '');
        document.body.classList.add('resizing');
        panel.style.willChange = 'width';
        panel.style.backdropFilter = 'none';
        panel.style.boxShadow = 'none';
    }

    function onPointerMove(e: Event): void {
        const event = e as PointerEvent;
        if (!dragging) return;
        if (isOtherPointer(event)) return;
        if (rafId !== null) cancelAnimationFrame(rafId);
        const clientX = event.clientX;
        rafId = requestAnimationFrame(function () {
            const deltaX = startX - clientX;
            const newWidth = clampWidth(startWidth + deltaX, widthVar);
            document.documentElement.style.setProperty(widthVar, newWidth + 'px');
            syncPanelEffectiveWidth();
        });
    }

    function finishDrag(e: PointerEvent | Event): void {
        if (!dragging) return;
        if (isOtherPointer(e)) return;
        dragging = false;
        if (rafId !== null) {
            cancelAnimationFrame(rafId);
            rafId = null;
        }
        if (
            activePointerId !== null
            && resizeHandle.hasPointerCapture
            && resizeHandle.hasPointerCapture(activePointerId)
        ) {
            resizeHandle.releasePointerCapture(activePointerId);
        }
        activePointerId = null;
        resizeHandle.classList.remove('active');
        document.body.style.userSelect = '';
        document.body.style.cursor = '';
        document.body.classList.remove('resizing');
        panel.style.willChange = '';
        panel.style.backdropFilter = '';
        panel.style.boxShadow = '';
        const finalWidth = Math.round(panel.getBoundingClientRect().width);
        preferred.set(finalWidth);
        saveWidth(storageKey, finalWidth);
        syncPanelEffectiveWidth();
        setTimeout(function () {
            document.body.removeAttribute('data-resizing');
        }, 0);
    }

    resizeHandle.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', finishDrag);
    document.addEventListener('pointercancel', finishDrag);
    window.addEventListener('blur', finishDrag);
}

export function setupPanelResize(): void {
    const main = document.getElementById('panel');
    const report = document.getElementById('report-panel');
    mainPreferred = loadPreferred(MAIN_STORAGE_KEY, '--panel-width');
    reportPreferred = loadPreferred(REPORT_STORAGE_KEY, '--report-panel-width');
    syncPanelEffectiveWidth();
    if (main) {
        bindHandle(main, '--panel-width', MAIN_STORAGE_KEY, {
            get: () => mainPreferred,
            set: (n) => { mainPreferred = n; },
        });
    }
    if (report) {
        bindHandle(report, '--report-panel-width', REPORT_STORAGE_KEY, {
            get: () => reportPreferred,
            set: (n) => { reportPreferred = n; },
        });
    }
    window.addEventListener('resize', syncPanelEffectiveWidth);
}
