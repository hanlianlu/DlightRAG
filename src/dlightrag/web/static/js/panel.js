// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {openLightbox} from './images.js';
import {renderMath} from './chat_renderer.js';
import {getIngestWorkspace, setIngestWorkspace, resetIngestWorkspace} from './state.js';
import {showToast} from './toast.js';
import {createWorkspace} from './workspaces.js';

let ingestPopoverEl = null;

export {showToast};

export function openPanel(title) {
    document.getElementById('panel').classList.add('open');
    document.body.classList.add('panel-open');
    if (!title) return;

    const titleEl = document.getElementById('panel-title');
    if (title === 'FILES') {
        resetIngestWorkspace();
        titleEl.style.letterSpacing = '0';
        renderIngestPill(titleEl);
    } else {
        titleEl.style.letterSpacing = '';
        titleEl.innerHTML = '';
        titleEl.textContent = title;
    }
}

function renderIngestPill(titleEl) {
    titleEl.innerHTML = '';

    const label = document.createElement('span');
    label.className = 'ingest-target-label';
    label.textContent = 'FILES';
    titleEl.appendChild(label);

    const sep = document.createElement('span');
    sep.className = 'ingest-target-sep';
    sep.textContent = '›';
    titleEl.appendChild(sep);

    const container = document.createElement('span');
    container.className = 'ingest-target';
    container.id = 'ingest-target';

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
    name.textContent = getIngestWorkspace();
    pill.appendChild(name);

    const caret = document.createElement('span');
    caret.className = 'ingest-target-caret';
    caret.innerHTML = '<svg width="8" height="8" viewBox="0 0 10 10" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M2.5 4 L5 6.5 L7.5 4"/></svg>';
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
    titleEl.appendChild(container);
}

function getIngestWorkspaceRecords() {
    const selector = document.getElementById('workspace-selector');
    if (!selector) return [];
    try {
        const raw = JSON.parse(selector.getAttribute('data-all') || '[]');
        return raw.map(function (record) {
            if (typeof record === 'string') {
                return {workspace: record, display_name: record};
            }
            return {
                workspace: record.workspace || record.id || 'default',
                display_name: record.display_name || record.workspace || record.id || 'default',
            };
        });
    } catch (_) {
        return [];
    }
}

function toggleIngestPopover(container) {
    if (ingestPopoverEl) {
        closeIngestPopover();
        return;
    }
    container.classList.add('open');
    const pill = container.querySelector('.ingest-target-pill');
    if (pill) pill.setAttribute('aria-expanded', 'true');

    const popover = document.createElement('div');
    popover.className = 'ingest-target-popover';
    popover.setAttribute('role', 'listbox');
    popover.setAttribute('aria-label', 'Select ingest workspace');

    const records = getIngestWorkspaceRecords();
    const current = getIngestWorkspace();

    records.slice().sort((a, b) => a.display_name.localeCompare(b.display_name)).forEach((record) => {
        const item = document.createElement('div');
        item.className = 'ingest-target-popover-item';
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
    createRow.className = 'ingest-target-popover-create';

    const input = document.createElement('input');
    input.className = 'ingest-target-popover-input';
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
    btn.className = 'ingest-target-popover-create-btn';
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

function selectIngestWorkspace(workspace) {
    setIngestWorkspace(workspace);
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

function updateIngestPillLabel() {
    const nameEl = document.querySelector('.ingest-target-name');
    if (nameEl) nameEl.textContent = getIngestWorkspace();
}

function closeIngestPopover() {
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

function onIngestPopoverOutside(e) {
    const container = document.getElementById('ingest-target');
    if (container && !container.contains(e.target)) closeIngestPopover();
}

function onIngestPopoverEscape(e) {
    if (e.key === 'Escape') closeIngestPopover();
}

export function closePanel() {
    closeIngestPopover();
    document.getElementById('panel').classList.remove('open');
    document.body.classList.remove('panel-open');
}

function openRefSource(refItem) {
    const ref = refItem.dataset.ref;
    const answerEl = refItem.closest('.ai-message');
    if (!answerEl) return;
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    const sourceData = answerEl.querySelector('.source-data');
    if (!sourceData) return;

    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    clone.classList.remove('visually-hidden');
    while (clone.firstChild) panelContent.appendChild(clone.firstChild);

    panelContent.querySelectorAll('.source-doc').forEach(function(doc) {
        const chunksContainer = doc.querySelector('.source-doc-chunks');
        if (doc.dataset.ref === ref) {
            doc.classList.add('expanded');
            if (chunksContainer) chunksContainer.hidden = false;
        } else {
            doc.classList.remove('expanded');
            if (chunksContainer) chunksContainer.hidden = true;
        }
    });

    const showAllBtn = panelContent.querySelector('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = false;
    renderMath(panelContent);
    openPanel('SOURCES');
}

export function filterSource(badge) {
    const ref = badge.dataset.ref;
    const chunk = badge.dataset.chunk;
    const panelContent = document.getElementById('panel-content');
    const answerEl = badge.closest('.ai-message');
    if (!panelContent || !answerEl) return;
    const sourceData = answerEl.querySelector('.source-data');
    if (!sourceData) return;

    panelContent.replaceChildren();
    const clone = sourceData.cloneNode(true);
    clone.classList.remove('visually-hidden');
    while (clone.firstChild) panelContent.appendChild(clone.firstChild);

    panelContent.querySelectorAll('.source-doc').forEach(function(doc) {
        const chunksContainer = doc.querySelector('.source-doc-chunks');
        if (doc.dataset.ref === ref) {
            doc.classList.add('expanded');
            if (chunksContainer) chunksContainer.hidden = false;
            if (chunk) {
                doc.querySelectorAll('.source-chunk').forEach(function(c) {
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

    const showAllBtn = panelContent.querySelector('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = false;
    renderMath(panelContent);
    openPanel('SOURCES');
}

function toggleDoc(header) {
    const doc = header.closest('.source-doc');
    if (!doc) return;
    const panelContent = doc.closest('#panel-content') || doc.parentElement;
    panelContent.querySelectorAll('.source-doc').forEach(function(d) {
        if (d !== doc) {
            d.classList.remove('expanded');
            const chunks = d.querySelector('.source-doc-chunks');
            if (chunks) chunks.hidden = true;
        }
    });
    const wasOpen = doc.classList.contains('expanded');
    doc.classList.toggle('expanded', !wasOpen);
    const chunksContainer = doc.querySelector('.source-doc-chunks');
    if (chunksContainer) {
        chunksContainer.hidden = wasOpen;
        if (!wasOpen) {
            chunksContainer.querySelectorAll('.source-chunk').forEach(function(c) {
                c.hidden = false;
                c.classList.remove('active');
            });
        }
    }
}

function showAllSources() {
    const panelContent = document.getElementById('panel-content');
    panelContent.querySelectorAll('.source-doc').forEach(function(doc) {
        doc.classList.add('expanded');
        const chunks = doc.querySelector('.source-doc-chunks');
        if (chunks) {
            chunks.hidden = false;
            chunks.querySelectorAll('.source-chunk').forEach(function(c) {
                c.hidden = false;
                c.classList.remove('active');
            });
        }
    });
    const showAllBtn = panelContent.querySelector('.show-all-btn');
    if (showAllBtn) showAllBtn.hidden = true;
}

export function setupPanel() {
    const closeBtn = document.getElementById('panel-close-btn');
    if (closeBtn) closeBtn.addEventListener('click', closePanel);

    document.addEventListener('click', function(e) {
        const panel = document.getElementById('panel');
        if (panel.classList.contains('open') && !e.target.closest('.panel') && !e.target.closest('.citation-badge')) {
            if (document.body.hasAttribute('data-resizing')) return;
            closePanel();
        }
        const badge = e.target.closest('[data-action="filter-source"]');
        if (badge) {
            e.preventDefault();
            filterSource(badge);
            return;
        }
        const refItem = e.target.closest('[data-action="open-ref-source"]');
        if (refItem) {
            e.preventDefault();
            openRefSource(refItem);
            return;
        }
        const image = e.target.closest('[data-action="open-lightbox"]');
        if (image && image.hasAttribute('data-full-src')) {
            e.preventDefault();
            const lightboxSrc = image.getAttribute('data-full-src');
            if (lightboxSrc && /^(?:\/|blob:|data:)/.test(lightboxSrc)) openLightbox(lightboxSrc);
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') closePanel();
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const badge = e.target.closest('[data-action="filter-source"]');
        if (badge) {
            e.preventDefault();
            filterSource(badge);
        }
    });

    // Apply progress-bar width from data-pct and render math in swapped content.
    document.body.addEventListener('htmx:afterSettle', function(ev) {
        const bar = document.getElementById('ingest-progress');
        if (bar) {
            const fill = bar.querySelector('.progress-bar-fill');
            if (fill) {
                const pct = fill.getAttribute('data-pct');
                if (pct !== null) {
                    fill.style.width = Math.max(0, Math.min(100, parseInt(pct, 10) || 0)) + '%';
                }
            }
        }
        // Auto-render math in any HTMX-swapped content.
        if (ev.detail.target) renderMath(ev.detail.target);
    });

    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    panelContent.addEventListener('click', function(e) {
        const toggleButton = e.target.closest('[data-action="toggle-doc"]');
        if (toggleButton) {
            toggleDoc(toggleButton);
            return;
        }
        if (e.target.closest('[data-action="show-all-sources"]')) {
            showAllSources();
            return;
        }
        const uploadZone = e.target.closest('#upload-zone');
        if (uploadZone) {
            const uploadFileBtn = e.target.closest('[data-action="upload-file"]');
            if (uploadFileBtn) {
                const fileInput = document.getElementById('file-input');
                if (fileInput) fileInput.click();
                return;
            }
            // Clicking the zone itself (not a button) opens file picker
            const fi = document.getElementById('file-input');
            if (fi && !e.target.closest('[data-action]')) fi.click();
            return;
        }
    });
    panelContent.addEventListener('change', function(e) {
        if (e.target.id === 'file-input') {
            htmx.trigger(document.getElementById('upload-form'), 'submit');
        }
    });

    // Sync ingest pill when workspace is created from either popover
    document.body.addEventListener('workspaceCreated', (event) => {
        const detail = event.detail && (event.detail.value || event.detail);
        const workspace = detail && detail.workspace;
        if (!workspace) return;
        setIngestWorkspace(workspace);
        updateIngestPillLabel();
        closeIngestPopover();
        htmx.ajax('GET', '/web/files', {
            target: '#panel-content',
            swap: 'innerHTML',
            values: {workspace: workspace},
        });
    });
}
