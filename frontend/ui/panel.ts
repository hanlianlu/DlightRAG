// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import chatStyles from '../styles/chat.module.css';
import {renderMath} from './mathjax.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {showToast} from './toast.ts';
import {createWorkspace} from './workspaces.ts';
import {setSanitizedHtml} from '../lib/safe_html.ts';

const AI_MESSAGE_SELECTOR = '.' + chatStyles.aiMessage;

type WorkspaceEventDetail = {
    workspace?: string;
    display_name?: string;
    displayName?: string;
    next_workspace?: string;
    nextWorkspace?: string;
    value?: WorkspaceEventDetail;
};

type IngestWorkspaceInput = string | {
    workspace?: string;
    id?: string;
    display_name?: string;
    displayName?: string;
};

interface IngestWorkspaceRecord {
    workspace: string;
    display_name: string;
}

let ingestPopoverEl: HTMLElement | null = null;

export {showToast};

const PANEL_DISMISS_EXEMPT_SELECTOR = [
    '.panel',
    '#files-btn',
    '#composer',
    '[data-action="filter-source"]',
    '[data-action="open-ref-source"]',
].join(', ');

function closestElement<T extends Element = Element>(target: EventTarget | null, selector: string): T | null {
    if (!(target instanceof Element)) return null;
    return target.closest(selector) as T | null;
}

function shouldDismissPanelOnOutsideClick(target: EventTarget | null): boolean {
    const panel = document.getElementById('panel');
    if (!panel || !panel.classList.contains('open')) return false;
    if (document.body.hasAttribute('data-resizing')) return false;
    return !closestElement(target, PANEL_DISMISS_EXEMPT_SELECTOR);
}

function workspaceEventDetail(event: Event): WorkspaceEventDetail {
    const detail = (event as CustomEvent<WorkspaceEventDetail>).detail;
    return detail?.value || detail || {};
}

function htmxEvent(event: Event): HTMXEvent {
    return event as HTMXEvent;
}

function createCaretIcon(): SVGSVGElement {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '8');
    svg.setAttribute('height', '8');
    svg.setAttribute('viewBox', '0 0 10 10');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.5');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M2.5 4 L5 6.5 L7.5 4');
    svg.appendChild(path);
    return svg;
}

export function openPanel(title?: string): void {
    const panel = document.getElementById('panel');
    if (!panel) return;
    panel.classList.add('open');
    document.body.classList.add('panel-open');
    if (!title) return;

    const titleEl = document.getElementById('panel-title');
    const ingestTarget = document.getElementById('ingest-target');
    if (!titleEl) return;
    if (title === 'FILES') {
        ingestStore.resetToPrimary();
        titleEl.textContent = 'FILES';
        renderIngestPill();
    } else {
        titleEl.textContent = title;
        if (ingestTarget) ingestTarget.replaceChildren();
    }
}

function showEmptyPanelPlaceholder(): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent || panelContent.children.length > 0) return;
    const placeholder = document.createElement('div');
    placeholder.className = 'empty-state';
    placeholder.textContent = 'No files yet. Drop files or folders here to ingest.';
    panelContent.appendChild(placeholder);
}

export function applyPanelHtml(html: string): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    setSanitizedHtml(panelContent, html);
    if (window.htmx && typeof window.htmx.process === 'function') {
        window.htmx.process(panelContent);
    }
    applyProgressBars(panelContent);
    showEmptyPanelPlaceholder();
}

function applyProgressBars(root: Element | null | undefined): void {
    if (!root) return;
    const bars: Element[] = [];
    if (root.id === 'ingest-progress') bars.push(root);
    root.querySelectorAll('#ingest-progress').forEach(function(bar) {
        bars.push(bar);
    });
    bars.forEach(function(bar) {
        const fill = bar.querySelector<HTMLElement>('.progress-bar-fill');
        if (!fill) return;
        const pct = fill.getAttribute('data-pct');
        if (pct !== null) {
            fill.style.width = Math.max(0, Math.min(100, parseInt(pct, 10) || 0)) + '%';
        }
    });
}

function renderIngestPill(): void {
    const container = document.getElementById('ingest-target');
    if (!container) return;
    container.replaceChildren();

    const pill = document.createElement('span');
    pill.className = 'ingest-target-pill';
    pill.setAttribute('role', 'button');
    pill.setAttribute('tabindex', '0');
    pill.setAttribute('aria-label', 'Select ingest workspace');

    const dot = document.createElement('span');
    dot.className = 'ingest-target-dot';
    pill.appendChild(dot);

    const name = document.createElement('span');
    name.className = 'ingest-target-name';
    name.textContent = ingestStore.workspace;
    pill.appendChild(name);

    const caret = document.createElement('span');
    caret.className = 'ingest-target-caret';
    caret.appendChild(createCaretIcon());
    pill.appendChild(caret);

    pill.addEventListener('click', (e) => {
        e.stopPropagation();
        toggleIngestPopover(container);
    });
    pill.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            toggleIngestPopover(container);
        }
    });

    container.appendChild(pill);
}

function getIngestWorkspaceRecords(): IngestWorkspaceRecord[] {
    const selector = document.getElementById('workspace-selector');
    if (!selector) return [];
    try {
        const raw = JSON.parse(selector.getAttribute('data-all') || '[]') as IngestWorkspaceInput[];
        return raw.map(function (record) {
            if (typeof record === 'string') {
                return {workspace: record, display_name: record};
            }
            const workspace = record.workspace || record.id;
            if (!workspace) return null;
            return {
                workspace: workspace,
                display_name: record.display_name || record.displayName || workspace,
            };
        }).filter((record): record is IngestWorkspaceRecord => record !== null);
    } catch (_) {
        return [];
    }
}

function toggleIngestPopover(container: HTMLElement): void {
    if (ingestPopoverEl) {
        closeIngestPopover();
        return;
    }
    container.classList.add('open');
    const pill = container.querySelector('.ingest-target-pill');
    if (pill) pill.setAttribute('aria-expanded', 'true');

    const popover = document.createElement('div');
    popover.className = 'ui-popover ui-popover--ingest';
    popover.setAttribute('role', 'listbox');
    popover.setAttribute('aria-label', 'Select ingest workspace');

    const records = getIngestWorkspaceRecords();
    const current = ingestStore.workspace;

    records.slice().sort((a, b) => a.display_name.localeCompare(b.display_name)).forEach((record) => {
        const item = document.createElement('div');
        item.className = 'ui-popover-item';
        item.setAttribute('role', 'option');
        item.setAttribute('tabindex', '0');
        item.setAttribute('aria-selected', record.workspace === current ? 'true' : 'false');

        const radio = document.createElement('div');
        radio.className = 'ingest-target-popover-radio' + (record.workspace === current ? ' on' : '');
        item.appendChild(radio);

        const nameSpan = document.createElement('span');
        nameSpan.textContent = record.display_name;
        item.appendChild(nameSpan);

        item.addEventListener('click', (e) => {
            e.stopPropagation();
            selectIngestWorkspace(record.workspace);
        });
        item.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                selectIngestWorkspace(record.workspace);
            }
        });
        popover.appendChild(item);
    });

    // Create row
    const createRow = document.createElement('div');
    createRow.className = 'ui-popover-create';

    const input = document.createElement('input');
    input.className = 'ui-popover-input';
    input.type = 'text';
    input.placeholder = 'New workspace...';
    input.addEventListener('click', (e) => e.stopPropagation());
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            createWorkspace(input);
        }
    });
    createRow.appendChild(input);

    const btn = document.createElement('button');
    btn.className = 'ui-popover-create-btn';
    btn.type = 'button';
    btn.textContent = '+';
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        createWorkspace(input);
    });
    createRow.appendChild(btn);
    popover.appendChild(createRow);

    container.appendChild(popover);
    ingestPopoverEl = popover;

    setTimeout(() => {
        document.addEventListener('click', onIngestPopoverOutside);
        document.addEventListener('keydown', onIngestPopoverEscape);
    }, 0);
}

function selectIngestWorkspace(workspace: string): void {
    ingestStore.set(workspace);
    updateIngestPillLabel();
    closeIngestPopover();
    // Refresh file list for the new workspace.
    // Pass workspace explicitly — syncWorkspaceParameter won't catch manual htmx.ajax calls.
    htmx.ajax('GET', '/web/files', {
        target: '#panel-content',
        swap: 'innerHTML',
        values: {workspace: workspace},
    });
}

function updateIngestPillLabel(): void {
    const nameEl = document.querySelector('.ingest-target-name');
    if (nameEl) nameEl.textContent = ingestStore.workspace;
}

function closeIngestPopover(): void {
    if (ingestPopoverEl) {
        ingestPopoverEl.remove();
        ingestPopoverEl = null;
    }
    const container = document.getElementById('ingest-target');
    if (container) {
        container.classList.remove('open');
        const pill = container.querySelector('.ingest-target-pill');
        if (pill) pill.setAttribute('aria-expanded', 'false');
    }
    document.removeEventListener('click', onIngestPopoverOutside);
    document.removeEventListener('keydown', onIngestPopoverEscape);
}

function onIngestPopoverOutside(e: MouseEvent): void {
    const container = document.getElementById('ingest-target');
    if (container && e.target instanceof Node && !container.contains(e.target)) closeIngestPopover();
}

function onIngestPopoverEscape(e: KeyboardEvent): void {
    if (e.key === 'Escape') closeIngestPopover();
}

export function closePanel(): void {
    closeIngestPopover();
    const panel = document.getElementById('panel');
    if (panel) panel.classList.remove('open');
    document.body.classList.remove('panel-open');
}

function openRefSource(refItem: HTMLElement): void {
    const ref = refItem.dataset.ref;
    const answerEl = refItem.closest(AI_MESSAGE_SELECTOR);
    if (!answerEl) return;
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    const sourceData = answerEl.querySelector('.source-data');
    if (!sourceData) return;

    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    if (!(clone instanceof HTMLElement)) return;
    clone.classList.remove('hidden');
    while (clone.firstChild) panelContent.appendChild(clone.firstChild);

    panelContent.querySelectorAll<HTMLElement>('.source-doc').forEach(function(doc) {
        const chunksContainer = doc.querySelector<HTMLElement>('.source-doc-chunks');
        if (doc.dataset.ref === ref) {
            doc.classList.add('expanded');
            if (chunksContainer) chunksContainer.hidden = false;
        } else {
            doc.classList.remove('expanded');
            if (chunksContainer) chunksContainer.hidden = true;
        }
    });

    const showAllBtn = panelContent.querySelector<HTMLElement>('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = false;
    renderExpandedSourceMath(panelContent);
    openPanel('SOURCES');
}

export function filterSource(badge: HTMLElement): void {
    const ref = badge.dataset.ref;
    const chunk = badge.dataset.chunk;
    const panelContent = document.getElementById('panel-content');
    const answerEl = badge.closest(AI_MESSAGE_SELECTOR);
    if (!panelContent || !answerEl) return;
    const sourceData = answerEl.querySelector('.source-data');
    if (!sourceData) return;

    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    if (!(clone instanceof HTMLElement)) return;
    clone.classList.remove('hidden');
    while (clone.firstChild) panelContent.appendChild(clone.firstChild);

    panelContent.querySelectorAll<HTMLElement>('.source-doc').forEach(function(doc) {
        const chunksContainer = doc.querySelector<HTMLElement>('.source-doc-chunks');
        if (doc.dataset.ref === ref) {
            doc.classList.add('expanded');
            if (chunksContainer) chunksContainer.hidden = false;
            if (chunk) {
                doc.querySelectorAll<HTMLElement>('.source-chunk').forEach(function(c) {
                    const active = c.dataset.chunk === chunk;
                    c.hidden = !active;
                    c.classList.toggle('active', active);
                });
            }
        } else {
            doc.classList.remove('expanded');
            if (chunksContainer) chunksContainer.hidden = true;
        }
    });

    const showAllBtn = panelContent.querySelector<HTMLElement>('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = false;
    renderExpandedSourceMath(panelContent);
    openPanel('SOURCES');
}

function renderExpandedSourceMath(root: Element): void {
    if (!root) return;
    root.querySelectorAll('.source-doc.expanded .source-doc-chunks').forEach(function(container) {
        renderMath(container);
    });
}

function toggleDoc(header: Element): void {
    const doc = header.closest<HTMLElement>('.source-doc');
    if (!doc) return;
    const panelContent = doc.closest('#panel-content') || doc.parentElement;
    if (!panelContent) return;
    panelContent.querySelectorAll('.source-doc').forEach(function(d) {
        if (d !== doc) {
            d.classList.remove('expanded');
            const chunks = d.querySelector<HTMLElement>('.source-doc-chunks');
            if (chunks) chunks.hidden = true;
        }
    });
    const wasOpen = doc.classList.contains('expanded');
    doc.classList.toggle('expanded', !wasOpen);
    const chunksContainer = doc.querySelector<HTMLElement>('.source-doc-chunks');
    if (chunksContainer) {
        chunksContainer.hidden = wasOpen;
        if (!wasOpen) {
            chunksContainer.querySelectorAll<HTMLElement>('.source-chunk').forEach(function(c) {
                c.hidden = false;
                c.classList.remove('active');
            });
            renderExpandedSourceMath(doc);
        }
    }
}

function showAllSources(): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    panelContent.querySelectorAll<HTMLElement>('.source-doc').forEach(function(doc) {
        doc.classList.add('expanded');
        const chunks = doc.querySelector<HTMLElement>('.source-doc-chunks');
        if (chunks) {
            chunks.hidden = false;
            chunks.querySelectorAll<HTMLElement>('.source-chunk').forEach(function(c) {
                c.hidden = false;
                c.classList.remove('active');
            });
        }
    });
    const showAllBtn = panelContent.querySelector<HTMLElement>('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = true;
    renderExpandedSourceMath(panelContent);
}

export function setupPanel(): void {
    const closeBtn = document.getElementById('panel-close-btn');
    if (closeBtn) closeBtn.addEventListener('click', closePanel);

    document.addEventListener('click', function(e) {
        const badge = closestElement<HTMLElement>(e.target, '[data-action="filter-source"]');
        if (badge) {
            e.preventDefault();
            filterSource(badge);
            return;
        }
        const refItem = closestElement<HTMLElement>(e.target, '[data-action="open-ref-source"]');
        if (refItem) {
            e.preventDefault();
            openRefSource(refItem);
            return;
        }
        if (shouldDismissPanelOnOutsideClick(e.target)) closePanel();
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closePanel();
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const badge = closestElement<HTMLElement>(e.target, '[data-action="filter-source"]');
        if (badge) {
            e.preventDefault();
            filterSource(badge);
        }
    });

    // Apply progress-bar width from data-pct and render math in swapped content.
    document.body.addEventListener('htmx:afterSettle', function(ev) {
        const target = htmxEvent(ev).detail.target;
        applyProgressBars(target);
        // Auto-render math in any HTMX-swapped content.
        if (target) renderMath(target);
        showEmptyPanelPlaceholder();
    });

    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    panelContent.addEventListener('click', function(e) {
        const toggleButton = closestElement<HTMLElement>(e.target, '[data-action="toggle-doc"]');
        if (toggleButton) {
            toggleDoc(toggleButton);
            return;
        }
        if (closestElement(e.target, '[data-action="show-all-sources"]')) {
            showAllSources();
            return;
        }
        const uploadZone = closestElement(e.target, '#upload-zone');
        if (uploadZone) {
            const uploadFileBtn = closestElement(e.target, '[data-action="upload-file"]');
            if (uploadFileBtn) {
                const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
                if (fileInput) fileInput.click();
                return;
            }
            // Clicking the zone itself (not a button) opens file picker
            const fi = document.getElementById('file-input') as HTMLInputElement | null;
            if (fi && !closestElement(e.target, '[data-action]')) fi.click();
            return;
        }
    });
    panelContent.addEventListener('change', function(e) {
        const target = e.target instanceof Element ? e.target : null;
        if (target?.id === 'file-input') {
            htmx.trigger(document.getElementById('upload-form'), 'submit');
        }
    });

    // Sync ingest pill when workspace is created from either popover
    document.body.addEventListener('workspaceCreated', (event) => {
        const detail = workspaceEventDetail(event);
        const workspace = detail.workspace;
        if (!workspace) return;
        ingestStore.set(workspace);
        updateIngestPillLabel();
        closeIngestPopover();
        htmx.ajax('GET', '/web/files', {
            target: '#panel-content',
            swap: 'innerHTML',
            values: {workspace: workspace},
        });
    });

    document.body.addEventListener('workspaceDeleted', (event) => {
        const detail = workspaceEventDetail(event);
        ingestStore.set(detail.next_workspace || detail.nextWorkspace || workspaceStore.primary);
        updateIngestPillLabel();
    });
}
