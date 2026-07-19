// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import chatStyles from '../styles/chat.module.css';
import {renderMath} from './mathjax.ts';
import {closestElement, openPanel} from './panel.ts';

const AI_MESSAGE_SELECTOR = '.' + chatStyles.aiMessage;

interface ActiveCitation {
    ref: string;
    chunk: string;
}

// Remembered citation highlight. Set when a citation badge is clicked and kept
// through show-all / collapse-all / manual doc toggles so the gold-edge chunk
// stays marked. Cleared when the panel closes, switches away from SOURCES, or
// the user navigates via a different citation or a doc-level reference.
let activeCitation: ActiveCitation | null = null;

function getToggleAllBtn(): HTMLButtonElement | null {
    return document.getElementById('source-toggle-all-btn') as HTMLButtonElement | null;
}

function applyActiveHighlight(scope: ParentNode): void {
    scope.querySelectorAll<HTMLElement>('.source-chunk').forEach(function(c) {
        const active = activeCitation !== null
            && c.dataset.ref === activeCitation.ref
            && c.dataset.chunk === activeCitation.chunk;
        c.classList.toggle('active', active);
    });
}

function isFullyExpanded(panelContent: HTMLElement): boolean {
    const docs = Array.from(panelContent.querySelectorAll<HTMLElement>('.source-doc'));
    if (docs.length === 0) return false;
    return docs.every(function(doc) {
        if (!doc.classList.contains('expanded')) return false;
        return Array.from(doc.querySelectorAll<HTMLElement>('.source-chunk'))
            .every(function(c) { return !c.hidden; });
    });
}

function updateToggleAllButton(): void {
    const btn = getToggleAllBtn();
    const panelContent = document.getElementById('panel-content');
    if (!btn || !panelContent) return;
    const hasSources = panelContent.querySelector('.source-doc') !== null;
    btn.hidden = !hasSources;
    if (!hasSources) return;
    const expanded = isFullyExpanded(panelContent);
    btn.textContent = expanded ? 'Collapse all' : 'Show all';
    btn.setAttribute('aria-pressed', expanded ? 'true' : 'false');
}

function hideToggleAllButton(): void {
    const btn = getToggleAllBtn();
    if (btn) btn.hidden = true;
}

function findSourceData(answerEl: Element): HTMLElement | null {
    for (const child of answerEl.children) {
        if (child instanceof HTMLElement && child.classList.contains('source-data')) return child;
    }
    return null;
}

function copySourcePanelFromAnswer(answerEl: Element): boolean {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return false;
    const sourceData = findSourceData(answerEl);
    if (!sourceData) return false;

    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    if (!(clone instanceof HTMLElement)) return false;
    clone.classList.remove('hidden');
    while (clone.firstChild) panelContent.appendChild(clone.firstChild);
    return true;
}

function setExpandedSource(ref?: string, chunk?: string): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;

    panelContent.querySelectorAll<HTMLElement>('.source-doc').forEach(function(doc) {
        const isActiveDoc = doc.dataset.ref === ref;
        const chunksContainer = doc.querySelector<HTMLElement>('.source-doc-chunks');
        doc.classList.toggle('expanded', isActiveDoc);
        if (chunksContainer) chunksContainer.hidden = !isActiveDoc;

        if (isActiveDoc) {
            doc.querySelectorAll<HTMLElement>('.source-chunk').forEach(function(c) {
                c.hidden = chunk ? c.dataset.chunk !== chunk : false;
            });
        }
    });

    applyActiveHighlight(panelContent);
    updateToggleAllButton();
    renderExpandedSourceMath(panelContent);
}

function openRefSource(refItem: HTMLElement): void {
    const ref = refItem.dataset.ref;
    const answerEl = refItem.closest(AI_MESSAGE_SELECTOR);
    if (!answerEl) return;
    if (!copySourcePanelFromAnswer(answerEl)) return;
    activeCitation = null;
    setExpandedSource(ref);
    openPanel('SOURCES');
}

export function filterSource(badge: HTMLElement): void {
    const ref = badge.dataset.ref;
    const chunk = badge.dataset.chunk;
    const answerEl = badge.closest(AI_MESSAGE_SELECTOR);
    if (!answerEl) return;
    if (!copySourcePanelFromAnswer(answerEl)) return;
    activeCitation = (ref && chunk) ? {ref, chunk} : null;
    setExpandedSource(ref, chunk);
    openPanel('SOURCES');
}

function renderExpandedSourceMath(root: Element): void {
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
            });
            applyActiveHighlight(doc);
            renderExpandedSourceMath(doc);
        }
    }
    updateToggleAllButton();
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
            });
        }
    });
    applyActiveHighlight(panelContent);
    updateToggleAllButton();
    renderExpandedSourceMath(panelContent);
}

function collapseAllSources(): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    panelContent.querySelectorAll<HTMLElement>('.source-doc').forEach(function(doc) {
        doc.classList.remove('expanded');
        const chunks = doc.querySelector<HTMLElement>('.source-doc-chunks');
        if (chunks) chunks.hidden = true;
    });
    updateToggleAllButton();
}

function toggleAllSources(): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    if (isFullyExpanded(panelContent)) collapseAllSources();
    else showAllSources();
}

export function setupSourcePanel(): void {
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
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' && e.key !== ' ') return;
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
        }
    });

    getToggleAllBtn()?.addEventListener('click', function() {
        toggleAllSources();
    });

    document.body.addEventListener('panelOpening', function(e) {
        const title = (e as CustomEvent<{title?: string}>).detail?.title;
        if (title && title !== 'SOURCES') {
            activeCitation = null;
            hideToggleAllButton();
        }
    });

    document.body.addEventListener('panelClosed', function() {
        activeCitation = null;
        hideToggleAllButton();
    });

    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    panelContent.addEventListener('click', function(e) {
        const toggleButton = closestElement<HTMLElement>(e.target, '[data-action="toggle-doc"]');
        if (toggleButton) {
            toggleDoc(toggleButton);
        }
    });
}
