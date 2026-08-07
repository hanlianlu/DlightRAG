// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {closestElement, syncShellInert, wrapTabFocus} from '../lib/dom.ts';


// Source links open their own panel content, so they must not dismiss it first.
const PANEL_KEEP_OPEN_SELECTOR = [
    '[data-action="filter-source"]',
    '[data-action="open-ref-source"]',
].join(', ');
const DRAWER_MEDIA = '(max-width: 1199px)';

let panelReturnFocus: HTMLElement | null = null;

function isDrawer(): boolean {
    return window.matchMedia(DRAWER_MEDIA).matches;
}

function applyPanelModality(): void {
    const panel = document.getElementById('panel');
    const backdrop = document.getElementById('panel-backdrop');
    const open = Boolean(panel?.classList.contains('open'));
    const modal = open && isDrawer();
    document.body.classList.toggle('panel-drawer-open', modal);
    if (backdrop) backdrop.hidden = !modal;
    syncShellInert();
    if (!panel) return;
    panel.inert = !open;
    if (open) panel.removeAttribute('aria-hidden');
    else panel.setAttribute('aria-hidden', 'true');
    if (modal) {
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
    } else {
        panel.removeAttribute('role');
        panel.removeAttribute('aria-modal');
    }
}

function focusablePanelElements(panel: HTMLElement): HTMLElement[] {
    return Array.from(panel.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => !element.hidden && element.getClientRects().length > 0);
}

function trapPanelFocus(event: KeyboardEvent): void {
    if (event.key !== 'Tab' || !isDrawer()) return;
    const panel = document.getElementById('panel');
    if (!panel?.classList.contains('open')) return;
    wrapTabFocus(focusablePanelElements(panel), event);
}

function shouldDismissPanelOnOutsideClick(target: EventTarget | null): boolean {
    const panel = document.getElementById('panel');
    if (!panel || !panel.classList.contains('open')) return false;
    if (document.body.hasAttribute('data-resizing')) return false;
    // Only the conversation area dismisses; persistent chrome never does.
    return Boolean(closestElement(target, '#chat-area'))
        && !closestElement(target, PANEL_KEEP_OPEN_SELECTOR);
}

export function openPanel(title?: string): void {
    const panel = document.getElementById('panel');
    if (!panel) return;
    document.body.dispatchEvent(new CustomEvent('panelOpening', {detail: {title}}));
    if (!panel.classList.contains('open')) {
        panelReturnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    }
    panel.classList.add('open');
    document.body.classList.add('panel-open');
    if (title) {
        const kind = title.toLowerCase();
        const titleEl = document.getElementById('panel-title');
        panel.dataset.panelKind = kind;
        document.body.classList.toggle('files-panel-open', title === 'FILES');
        panel.setAttribute('aria-label', title === 'FILES' ? 'Files' : 'Sources');
        if (titleEl) titleEl.textContent = title === 'FILES' ? '' : 'Sources';
        const ingest = document.querySelector('ingest-target');
        if (ingest) ingest.active = title === 'FILES';
    }
    applyPanelModality();
    if (isDrawer()) window.requestAnimationFrame(function() {
        document.getElementById('panel-close-btn')?.focus();
    });
}

export function closePanel(restoreFocus = true): void {
    const panel = document.getElementById('panel');
    if (panel) panel.classList.remove('open');
    document.body.classList.remove('panel-open');
    document.body.classList.remove('files-panel-open');
    applyPanelModality();
    document.body.dispatchEvent(new CustomEvent('panelClosed'));
    if (restoreFocus) panelReturnFocus?.focus();
    panelReturnFocus = null;
}

export function setupPanel(): void {
    const closeBtn = document.getElementById('panel-close-btn');
    if (closeBtn) closeBtn.addEventListener('click', function() { closePanel(); });
    document.getElementById('panel-backdrop')?.addEventListener('click', function() {
        closePanel();
    });
    document.getElementById('panel')?.addEventListener('keydown', trapPanelFocus);

    document.addEventListener('click', function(e) {
        if (!isDrawer() && shouldDismissPanelOnOutsideClick(e.target)) closePanel();
    });

    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Escape') return;
        if (document.querySelector('dialog[open]')) return;
        if (!document.getElementById('panel')?.classList.contains('open')) return;
        closePanel();
    });
    window.addEventListener('resize', applyPanelModality);
    applyPanelModality();
}
