// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import chatStyles from '../styles/chat.module.css';
import {renderMath} from './mathjax.ts';
import {closestElement, openPanel} from './panel.ts';

const AI_MESSAGE_SELECTOR = '.' + chatStyles.aiMessage;

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

        if (isActiveDoc && chunk) {
            doc.querySelectorAll<HTMLElement>('.source-chunk').forEach(function(c) {
                const active = c.dataset.chunk === chunk;
                c.hidden = !active;
                c.classList.toggle('active', active);
            });
        }
    });

    const showAllBtn = panelContent.querySelector<HTMLElement>('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = false;
    renderExpandedSourceMath(panelContent);
}

function openRefSource(refItem: HTMLElement): void {
    const ref = refItem.dataset.ref;
    const answerEl = refItem.closest(AI_MESSAGE_SELECTOR);
    if (!answerEl) return;
    if (!copySourcePanelFromAnswer(answerEl)) return;
    setExpandedSource(ref);
    openPanel('SOURCES');
}

export function filterSource(badge: HTMLElement): void {
    const ref = badge.dataset.ref;
    const chunk = badge.dataset.chunk;
    const answerEl = badge.closest(AI_MESSAGE_SELECTOR);
    if (!answerEl) return;
    if (!copySourcePanelFromAnswer(answerEl)) return;
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
        }
    });
}
