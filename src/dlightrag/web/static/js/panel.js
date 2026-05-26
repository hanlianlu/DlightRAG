// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {openLightbox} from './images.js';
import {renderMath} from './chat_renderer.js';

let toastTimer = null;

export function openPanel(title) {
    document.getElementById('panel').classList.add('open');
    document.body.classList.add('panel-open');
    if (title) document.getElementById('panel-title').textContent = title;
}

export function closePanel() {
    document.getElementById('panel').classList.remove('open');
    document.body.classList.remove('panel-open');
}

export function showToast(msg, duration) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('visible');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function() { el.classList.remove('visible'); }, duration || 3000);
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
        if (image) {
            e.preventDefault();
            var lightboxSrc = image.getAttribute('data-full-src') || image.getAttribute('src');
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
        var uploadZone = e.target.closest('#upload-zone');
        if (uploadZone) {
            var uploadFileBtn = e.target.closest('[data-action="upload-file"]');
            if (uploadFileBtn) {
                var fileInput = document.getElementById('file-input');
                if (fileInput) fileInput.click();
                return;
            }
            // Clicking the zone itself (not a button) opens file picker
            var fi = document.getElementById('file-input');
            if (fi && !e.target.closest('[data-action]')) fi.click();
            return;
        }
    });
    panelContent.addEventListener('change', function(e) {
        if (e.target.id === 'file-input') {
            htmx.trigger(document.getElementById('upload-form'), 'submit');
        }
    });
}
