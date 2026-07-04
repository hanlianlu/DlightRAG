// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {ingestStore} from '../stores/ingestStore.ts';
import {openPanel, showToast} from './panel.ts';

function htmxEvent(event: Event): HTMXEvent {
    return event as HTMXEvent;
}

function isPanelTarget(event: Event): boolean {
    const detail = htmxEvent(event).detail;
    return detail.target?.id === 'panel-content';
}

function isFileRequest(el: Element | undefined): boolean {
    if (!el) return false;
    if (el.id === 'files-btn' || el.id === 'upload-form') return true;
    if (el.classList.contains('file-delete')) return true;
    return Boolean(el.closest('#upload-form'));
}

function syncWorkspaceParameter(event: Event): void {
    const detail = htmxEvent(event).detail;
    const el = detail.elt;
    if (!isFileRequest(el)) return;
    const workspace = ingestStore.workspace;
    if (!detail.parameters) detail.parameters = {};
    detail.parameters.workspace = workspace;

    if (el?.id === 'upload-form') {
        const input = el.querySelector<HTMLInputElement>('input[name="workspace"]');
        if (input) input.value = workspace;
    }
}

function openPanelAfterRequest(event: Event): void {
    const detail = htmxEvent(event).detail;
    const el = detail.elt;
    if (!el) return;

    if (el.id === 'files-btn') {
        openPanel('FILES');
        return;
    }

    if (el.id === 'upload-form' || el.closest('#upload-form')) {
        showToast(
            detail.successful ? 'Files received — processing in background' : 'Upload failed.',
            detail.successful ? 3000 : 5000,
        );
    }

    if (el.classList.contains('file-delete')) {
        showToast(
            detail.successful ? 'File deleted.' : 'Deletion failed.',
            detail.successful ? 3000 : 5000,
        );
    }
}

export function setupHtmxInteractions(): void {
    document.body.addEventListener('htmx:configRequest', syncWorkspaceParameter);

    document.body.addEventListener('htmx:beforeSwap', function(event) {
        const detail = htmxEvent(event).detail;
        if (isPanelTarget(event) && detail.xhr.status >= 400) {
            detail.shouldSwap = true;
            detail.isError = false;
        }
    });

    document.body.addEventListener('htmx:afterRequest', openPanelAfterRequest);
}
