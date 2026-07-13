// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const STORAGE_KEY = 'dlightrag-panel-width';
const MIN_WIDTH = 320;
const COMPACT_MAX_WIDTH = 420;

let preferredWidth = 0;

function cssPanelDefault(): number {
    const raw = getComputedStyle(document.documentElement).getPropertyValue('--panel-width').trim();
    const px = parseInt(raw, 10);
    return !isNaN(px) && px >= MIN_WIDTH ? px : 420;
}

function getMaxWidth(): number {
    if (window.innerWidth <= 640) return window.innerWidth;
    if (window.innerWidth < 1200) return COMPACT_MAX_WIDTH;
    const styles = getComputedStyle(document.documentElement);
    const chatMinWidth = parseFloat(styles.getPropertyValue('--layout-chat-min-width')) || 520;
    const sidebar = document.body.classList.contains('conversation-sidebar-open')
        ? document.getElementById('chat-sidebar')?.getBoundingClientRect().width || 0
        : 0;
    return Math.max(MIN_WIDTH, Math.floor(window.innerWidth - sidebar - chatMinWidth));
}

function clampWidth(w: number): number {
    const maxWidth = getMaxWidth();
    const minWidth = Math.min(MIN_WIDTH, maxWidth);
    return Math.max(minWidth, Math.min(w, maxWidth));
}

function loadPreferredWidth(): number {
    try {
        const saved = localStorage.getItem(STORAGE_KEY);
        if (saved !== null) {
            const n = parseInt(saved, 10);
            if (!isNaN(n) && n >= MIN_WIDTH) return n;
        }
    } catch (_) { /* localStorage unavailable */ }
    return cssPanelDefault();
}

export function syncPanelEffectiveWidth(): number {
    const effectiveWidth = clampWidth(preferredWidth);
    document.documentElement.style.setProperty('--panel-width', effectiveWidth + 'px');
    return effectiveWidth;
}

function saveWidth(w: number): void {
    try {
        localStorage.setItem(STORAGE_KEY, String(w));
    } catch (_) { /* localStorage unavailable */ }
}

export function setupPanelResize(): void {
    const panel = document.getElementById('panel');
    if (!panel) return;

    preferredWidth = loadPreferredWidth();
    syncPanelEffectiveWidth();

    let handle = panel.querySelector('.panel-resize-handle');
    if (!handle) {
        handle = document.createElement('div');
        handle.className = 'panel-resize-handle';
        panel.insertBefore(handle, panel.firstChild);
    }
    const resizePanel = panel;
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
        startWidth = resizePanel.getBoundingClientRect().width;
        event.preventDefault();
        if (resizeHandle.setPointerCapture) {
            resizeHandle.setPointerCapture(event.pointerId);
        }
        resizeHandle.classList.add('active');
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';
        document.body.setAttribute('data-resizing', '');
        document.body.classList.add('resizing');
        resizePanel.style.willChange = 'width';
        resizePanel.style.backdropFilter = 'none';
        resizePanel.style.boxShadow = 'none';
    }

    function onPointerMove(e: Event): void {
        const event = e as PointerEvent;
        if (!dragging) return;
        if (isOtherPointer(event)) return;
        if (rafId !== null) cancelAnimationFrame(rafId);
        const clientX = event.clientX;
        rafId = requestAnimationFrame(function () {
            const deltaX = startX - clientX;
            const newWidth = clampWidth(startWidth + deltaX);
            document.documentElement.style.setProperty('--panel-width', newWidth + 'px');
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
        resizePanel.style.willChange = '';
        resizePanel.style.backdropFilter = '';
        resizePanel.style.boxShadow = '';
        const finalWidth = resizePanel.getBoundingClientRect().width;
        preferredWidth = Math.round(finalWidth);
        saveWidth(preferredWidth);
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

    window.addEventListener('resize', syncPanelEffectiveWidth);
}
