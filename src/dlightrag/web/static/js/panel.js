// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {openLightbox} from './images.js';
import {toggleWorkspace, updateWorkspacePanelCheckboxes} from './workspaces.js';

let toastTimer = null;

export function openPanel(title) {
    document.getElementById('panel').classList.add('open');
    if (title) document.getElementById('panel-title').textContent = title;
}

export function closePanel() {
    document.getElementById('panel').classList.remove('open');
}

export function showToast(msg, duration) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('visible');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function() { el.classList.remove('visible'); }, duration || 3000);
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
            closePanel();
        }
        const badge = e.target.closest('[data-action="filter-source"]');
        if (badge) {
            e.preventDefault();
            filterSource(badge);
            return;
        }
        const image = e.target.closest('[data-action="open-lightbox"]');
        if (image) {
            e.preventDefault();
            openLightbox(image.getAttribute('data-full-src') || image.getAttribute('src'));
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

    const filesBtn = document.getElementById('files-btn');
    if (filesBtn) {
        filesBtn.addEventListener('htmx:afterRequest', function() { openPanel('FILES'); });
    }

    const wsAddBtn = document.getElementById('ws-add-btn');
    if (wsAddBtn) {
        wsAddBtn.addEventListener('htmx:afterRequest', function() {
            openPanel('WORKSPACES');
            updateWorkspacePanelCheckboxes();
        });
    }

    document.body.addEventListener('htmx:afterRequest', function(e) {
        const el = e.detail.elt;
        if (!el) return;
        if (el.id === 'upload-form' || (el.closest && el.closest('#upload-form'))) {
            showToast(e.detail.successful ? 'Ingestion complete.' : 'Ingestion failed.', e.detail.successful ? 3000 : 5000);
        }
        if (el.classList && el.classList.contains('file-delete')) {
            showToast(e.detail.successful ? 'File deleted.' : 'Deletion failed.', e.detail.successful ? 3000 : 5000);
        }
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
        const deleteOpen = e.target.closest('[data-action="workspace-delete-open"]');
        if (deleteOpen) {
            const confirmBox = deleteOpen.closest('.workspace-check-item').querySelector('.ws-delete-confirm');
            if (confirmBox) {
                confirmBox.classList.add('open');
                confirmBox.setAttribute('aria-hidden', 'false');
            }
            return;
        }
        const deleteCancel = e.target.closest('[data-action="workspace-delete-cancel"]');
        if (deleteCancel) {
            const openBox = deleteCancel.closest('.ws-delete-confirm');
            if (openBox) {
                openBox.classList.remove('open');
                openBox.setAttribute('aria-hidden', 'true');
            }
            return;
        }
        const deleteConfirm = e.target.closest('.ws-delete-confirm');
        if (deleteConfirm) {
            if (!e.target.closest('.workspace-delete-card')) {
                deleteConfirm.classList.remove('open');
                deleteConfirm.setAttribute('aria-hidden', 'true');
            }
            return;
        }
        const uploadZone = e.target.closest('#upload-zone');
        if (uploadZone) {
            const fileInput = uploadZone.querySelector('#file-input');
            if (fileInput) fileInput.click();
            return;
        }
        const item = e.target.closest('.workspace-check-item');
        if (item) {
            const ws = item.getAttribute('data-ws');
            if (ws) toggleWorkspace(ws);
        }
    });
    panelContent.addEventListener('change', function(e) {
        if (e.target.id === 'file-input') {
            htmx.trigger(document.getElementById('upload-form'), 'submit');
        }
    });
}
