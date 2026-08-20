// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {closestElement, syncShellInert, wrapTabFocus} from '../lib/dom.ts';
import {syncPanelEffectiveWidth} from './resize.ts';

// Source links and the report control open panel content, so they must not
// dismiss the stack first.
const PANEL_KEEP_OPEN_SELECTOR = [
    '[data-action="filter-source"]',
    '[data-action="open-ref-source"]',
    '[data-action="open-primary-report"]',
].join(', ');
const DRAWER_MEDIA = '(max-width: 1199px)';
const LABELS: Record<string, string> = {
    FILES: 'Files',
    SOURCES: 'Sources',
    REPORT: 'Report',
};

let panelReturnFocus: HTMLElement | null = null;

function isDrawer(): boolean {
    return window.matchMedia(DRAWER_MEDIA).matches;
}

function mainPanel(): HTMLElement | null {
    return document.getElementById('panel');
}

function reportPanel(): HTMLElement | null {
    return document.getElementById('report-panel');
}

function isOpen(el: HTMLElement | null): boolean {
    return Boolean(el?.classList.contains('open'));
}

export function isSourcesOpen(): boolean {
    return mainPanel()?.dataset.panelKind === 'sources' && isOpen(mainPanel());
}

export function isReportOpen(): boolean {
    return isOpen(reportPanel());
}

function anyPaneOpen(): boolean {
    return isOpen(mainPanel()) || isOpen(reportPanel());
}

function applyPanelModality(): void {
    const main = mainPanel();
    const report = reportPanel();
    const backdrop = document.getElementById('panel-backdrop');
    const drawer = isDrawer();
    const top = isOpen(main) ? main : isOpen(report) ? report : null;
    const modal = drawer && top !== null;
    document.body.classList.toggle('panel-drawer-open', modal);
    if (backdrop) backdrop.hidden = !modal;
    syncShellInert();
    for (const pane of [main, report]) {
        if (!pane) continue;
        const open = isOpen(pane);
        pane.inert = !open || (drawer && top !== null && pane !== top);
        if (open) pane.removeAttribute('aria-hidden');
        else pane.setAttribute('aria-hidden', 'true');
    }
    if (top && modal) {
        top.setAttribute('role', 'dialog');
        top.setAttribute('aria-modal', 'true');
    } else {
        main?.removeAttribute('role');
        main?.removeAttribute('aria-modal');
        report?.removeAttribute('role');
        report?.removeAttribute('aria-modal');
    }
}

function focusablePanelElements(panel: HTMLElement): HTMLElement[] {
    return Array.from(panel.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), [tabindex]:not([tabindex="-1"])',
    )).filter((element) => !element.hidden && element.getClientRects().length > 0);
}

function trapPanelFocus(event: KeyboardEvent): void {
    if (event.key !== 'Tab' || !isDrawer()) return;
    const main = mainPanel();
    const report = reportPanel();
    const top = isOpen(main) ? main : isOpen(report) ? report : null;
    if (!top) return;
    wrapTabFocus(focusablePanelElements(top), event);
}

function shouldDismissPanelOnOutsideClick(target: EventTarget | null): boolean {
    if (!anyPaneOpen()) return false;
    if (document.body.hasAttribute('data-resizing')) return false;
    return Boolean(closestElement(target, '#chat-area'))
        && !closestElement(target, PANEL_KEEP_OPEN_SELECTOR);
}

function rememberFocus(): void {
    if (!anyPaneOpen() && document.activeElement instanceof HTMLElement) {
        panelReturnFocus = document.activeElement;
    }
}

function syncBodyFlags(): void {
    const main = mainPanel();
    const files = isOpen(main) && main?.dataset.panelKind === 'files';
    document.body.classList.toggle('panel-open', anyPaneOpen());
    document.body.classList.toggle('files-panel-open', Boolean(files));
    document.body.classList.toggle('sources-panel-open', isSourcesOpen());
    document.body.classList.toggle('report-panel-open', isReportOpen());
    const ingest = document.querySelector('ingest-target');
    if (ingest) ingest.active = Boolean(files);
    syncPanelEffectiveWidth();
}

function openMainPane(title: 'FILES' | 'SOURCES'): void {
    const panel = mainPanel();
    if (!panel) return;
    rememberFocus();
    document.body.dispatchEvent(new CustomEvent('panelOpening', {detail: {title}}));
    panel.classList.add('open');
    panel.dataset.panelKind = title.toLowerCase();
    panel.setAttribute('aria-label', LABELS[title]);
    const titleEl = document.getElementById('panel-title');
    if (titleEl) titleEl.textContent = title === 'FILES' ? '' : LABELS[title];
    syncBodyFlags();
    applyPanelModality();
    if (isDrawer()) window.requestAnimationFrame(function() {
        document.getElementById('panel-close-btn')?.focus();
    });
}

function openReportPane(): void {
    const panel = reportPanel();
    if (!panel) return;
    rememberFocus();
    document.body.dispatchEvent(new CustomEvent('panelOpening', {detail: {title: 'REPORT'}}));
    panel.classList.add('open');
    panel.dataset.panelKind = 'report';
    syncBodyFlags();
    applyPanelModality();
    if (isDrawer() && !isOpen(mainPanel())) window.requestAnimationFrame(function() {
        document.getElementById('report-panel-close-btn')?.focus();
    });
}

export function openPanel(title?: string): void {
    if (title === 'REPORT') {
        if (mainPanel()?.dataset.panelKind === 'files') closeMainPane(false);
        openReportPane();
        return;
    }
    if (title === 'FILES') {
        closeReportPane(false);
        openMainPane('FILES');
        return;
    }
    if (title === 'SOURCES') {
        openMainPane('SOURCES');
        return;
    }
    if (title) openMainPane(title === 'FILES' ? 'FILES' : 'SOURCES');
}

function closeMainPane(restoreFocus: boolean): void {
    const panel = mainPanel();
    if (!isOpen(panel)) return;
    panel?.classList.remove('open');
    delete panel?.dataset.panelKind;
    syncBodyFlags();
    applyPanelModality();
    document.body.dispatchEvent(new CustomEvent('panelClosed'));
    if (restoreFocus && !anyPaneOpen()) {
        panelReturnFocus?.focus();
        panelReturnFocus = null;
    }
}

function closeReportPane(restoreFocus: boolean): void {
    const panel = reportPanel();
    if (!isOpen(panel)) return;
    panel?.classList.remove('open');
    syncBodyFlags();
    applyPanelModality();
    document.body.dispatchEvent(new CustomEvent('reportPanelClosed'));
    if (restoreFocus && !anyPaneOpen()) {
        panelReturnFocus?.focus();
        panelReturnFocus = null;
    }
}

export function closePanel(restoreFocus = true): void {
    closeMainPane(false);
    closeReportPane(false);
    syncBodyFlags();
    applyPanelModality();
    if (restoreFocus) {
        panelReturnFocus?.focus();
        panelReturnFocus = null;
    }
}

/** Close conversation-scoped Sources/Report while preserving workspace Files. */
export function closeConversationPanels(): void {
    if (isSourcesOpen()) closeMainPane(false);
    closeReportPane(false);
    syncBodyFlags();
    applyPanelModality();
}

export function setupPanel(): void {
    document.getElementById('panel-close-btn')?.addEventListener('click', function() {
        closeMainPane(true);
    });
    document.getElementById('report-panel-close-btn')?.addEventListener('click', function() {
        closeReportPane(true);
    });
    document.getElementById('panel-backdrop')?.addEventListener('click', function() {
        if (isOpen(mainPanel())) closeMainPane(true);
        else closeReportPane(true);
    });
    mainPanel()?.addEventListener('keydown', trapPanelFocus);
    reportPanel()?.addEventListener('keydown', trapPanelFocus);

    document.addEventListener('click', function(e) {
        if (!isDrawer() && shouldDismissPanelOnOutsideClick(e.target)) closePanel();
    });

    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Escape') return;
        if (document.querySelector('dialog[open]')) return;
        if (isOpen(mainPanel())) {
            closeMainPane(true);
            return;
        }
        if (isOpen(reportPanel())) closeReportPane(true);
    });
    window.addEventListener('resize', applyPanelModality);
    applyPanelModality();
}
