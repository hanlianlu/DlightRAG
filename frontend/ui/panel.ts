// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
import {closestElement, syncShellInert, wrapTabFocus} from '../lib/dom.ts';
import type {DlArtifactCanvas} from './artifact_canvas.ts';
import {syncPanelSplitState} from './split_panel.ts';

const PANEL_KEEP_OPEN_SELECTOR = [
    '[data-action="filter-source"]',
    '[data-action="open-ref-source"]',
    '[data-action="open-artifact"]',
].join(', ');
const DRAWER_MEDIA = '(max-width: 1199px)';
const LABELS: Record<string, string> = {FILES: 'Files', SOURCES: 'Sources'};

let panelReturnFocus: HTMLElement | null = null;

function isDrawer(): boolean {
    return window.matchMedia(DRAWER_MEDIA).matches;
}

function mainPanel(): HTMLElement | null {
    return document.getElementById('panel');
}

function artifactCanvas(): DlArtifactCanvas | null {
    return document.querySelector<DlArtifactCanvas>('#artifact-canvas');
}

function isOpen(el: HTMLElement | null): boolean {
    return Boolean(el?.classList.contains('open'));
}

export function isSourcesOpen(): boolean {
    return mainPanel()?.dataset.panelKind === 'sources' && isOpen(mainPanel());
}

function anyPaneOpen(): boolean {
    return isOpen(mainPanel()) || isOpen(artifactCanvas());
}

function applyPanelModality(): void {
    const main = mainPanel();
    const backdrop = document.getElementById('panel-backdrop');
    const modal = isDrawer() && isOpen(main);
    document.body.classList.toggle('panel-drawer-open', modal);
    if (backdrop) backdrop.hidden = !modal;
    syncShellInert();
    if (main) {
        main.inert = !isOpen(main);
        if (isOpen(main)) main.removeAttribute('aria-hidden');
        else main.setAttribute('aria-hidden', 'true');
        if (modal) {
            main.setAttribute('role', 'dialog');
            main.setAttribute('aria-modal', 'true');
        } else {
            main.removeAttribute('role');
            main.removeAttribute('aria-modal');
        }
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
    if (!isOpen(main) || !main) return;
    wrapTabFocus(focusablePanelElements(main), event);
}

function shouldDismissPanelOnOutsideClick(target: EventTarget | null): boolean {
    if (!anyPaneOpen() || document.body.hasAttribute('data-resizing')) return false;
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
    const ingest = document.querySelector('ingest-target');
    if (ingest) ingest.active = Boolean(files);
    syncPanelSplitState();
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
    if (isDrawer()) window.requestAnimationFrame(() => {
        document.getElementById('panel-close-btn')?.focus();
    });
}

export function openPanel(title?: string): void {
    if (title === 'FILES') {
        artifactCanvas()?.close(false);
        openMainPane('FILES');
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

export function closePanel(restoreFocus = true): void {
    closeMainPane(false);
    artifactCanvas()?.close(false);
    syncBodyFlags();
    applyPanelModality();
    if (restoreFocus) {
        panelReturnFocus?.focus();
        panelReturnFocus = null;
    }
}

/** Close conversation-scoped Sources and Artifact Canvas while preserving Files. */
export function closeConversationPanels(): void {
    if (isSourcesOpen()) closeMainPane(false);
    artifactCanvas()?.close(false);
    syncBodyFlags();
    applyPanelModality();
}

export function setupPanel(): void {
    document.getElementById('panel-close-btn')?.addEventListener('click', () => {
        closeMainPane(true);
    });
    document.getElementById('panel-backdrop')?.addEventListener('click', () => {
        closeMainPane(true);
    });
    mainPanel()?.addEventListener('keydown', trapPanelFocus);
    document.addEventListener('click', (event) => {
        if (!isDrawer() && shouldDismissPanelOnOutsideClick(event.target)) closePanel();
    });
    document.addEventListener('keydown', (event) => {
        if (event.key !== 'Escape' || document.querySelector('dialog[open]')) return;
        if (isOpen(mainPanel())) closeMainPane(true);
    });
    window.addEventListener('resize', applyPanelModality);
    document.body.addEventListener('artifact-canvas-state-changed', syncBodyFlags);
    applyPanelModality();
}
