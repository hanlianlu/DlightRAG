// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {getPrimaryWorkspace} from './state.js';
import {openPanel, showToast} from './panel.js';

function isPanelTarget(event) {
    return event.detail && event.detail.target && event.detail.target.id === 'panel-content';
}

function isFileRequest(el) {
    if (!el) return false;
    if (el.id === 'files-btn' || el.id === 'upload-form') return true;
    if (el.classList && el.classList.contains('file-delete')) return true;
    return Boolean(el.closest && el.closest('#upload-form'));
}

function syncWorkspaceParameter(event) {
    const el = event.detail && event.detail.elt;
    if (!isFileRequest(el)) return;
    const workspace = getPrimaryWorkspace();
    if (!event.detail.parameters) event.detail.parameters = {};
    event.detail.parameters.workspace = workspace;

    if (el.id === 'upload-form') {
        const input = el.querySelector('input[name="workspace"]');
        if (input) input.value = workspace;
    }
}

function openPanelAfterRequest(event) {
    const el = event.detail && event.detail.elt;
    if (!el) return;

    if (el.id === 'files-btn') {
        openPanel('FILES');
        return;
    }

    if (el.id === 'upload-form' || (el.closest && el.closest('#upload-form'))) {
        showToast(
            event.detail.successful ? 'Ingestion complete.' : 'Ingestion failed.',
            event.detail.successful ? 3000 : 5000,
        );
    }

    if (el.classList && el.classList.contains('file-delete')) {
        showToast(
            event.detail.successful ? 'File deleted.' : 'Deletion failed.',
            event.detail.successful ? 3000 : 5000,
        );
    }
}

export function setupHtmxInteractions() {
    document.body.addEventListener('htmx:configRequest', syncWorkspaceParameter);

    document.body.addEventListener('htmx:beforeSwap', function(event) {
        if (isPanelTarget(event) && event.detail.xhr.status >= 400) {
            event.detail.shouldSwap = true;
            event.detail.isError = false;
        }
    });

    document.body.addEventListener('htmx:afterRequest', openPanelAfterRequest);
}
