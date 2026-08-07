// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {bus, type WorkspaceRecord} from '../events/bus.ts';
import {toWorkspaceRecord, type WorkspaceEventDetail} from '../events/workspace_events.ts';
import {WorkspaceApiError, deleteWorkspaceRequest} from '../api/workspaces.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {showToast} from './toast.ts';
import './workspace_create.ts';
import './workspace_scope.ts';
import {closestElement} from '../lib/dom.ts';


function normalizeRecord(record: string | WorkspaceEventDetail): WorkspaceRecord | null {
    if (typeof record === 'string') {
        return {workspace: record, displayName: record, embeddingModel: ''};
    }
    return toWorkspaceRecord(record);
}

function workspaceName(workspace: string): string {
    const match = workspaceStore.records.find((item) => item.workspace === workspace);
    return match ? match.displayName : workspace;
}

export function initWorkspaces(): void {
    const selector = document.querySelector('workspace-scope');
    if (!selector) return;

    try {
        const records = (JSON.parse(selector.getAttribute('data-all') || '[]') as (string | WorkspaceEventDetail)[])
            .map(normalizeRecord)
            .filter((record): record is WorkspaceRecord => record !== null);
        let active: string[] = [];
        try {
            active = JSON.parse(selector.getAttribute('data-active') || '[]');
        } catch (_) {
            active = [];
        }
        const primary = selector.getAttribute('data-primary') || '';
        workspaceStore.init(records, active, primary);
    } catch (_) {
        // data-all attribute may be absent; start with empty records
    }
    setupWorkspaceEvents();
}

function setDeleteWorkspacePending(pending: boolean): void {
    const confirmBtn = document.getElementById('delete-workspace-confirm-btn') as HTMLButtonElement | null;
    const confirmInput = document.getElementById('delete-workspace-confirm-input') as HTMLInputElement | null;
    if (!confirmBtn || !confirmInput) return;
    confirmBtn.textContent = pending ? 'Deleting\u2026' : 'Delete';
    confirmInput.readOnly = pending;
    confirmBtn.disabled = true;
    // Let the typed-name rule decide again rather than keeping a second copy of it.
    if (!pending) confirmInput.dispatchEvent(new Event('input'));
}

function showDeleteWorkspaceDialog(workspace: string): void {
    const record = workspaceStore.records.find((item) => item.workspace === workspace);
    const displayName = record ? record.displayName : workspace;
    const dialog = document.getElementById('delete-workspace-dialog') as HTMLDialogElement | null;
    const name = document.getElementById('delete-workspace-name');
    const idInput = document.getElementById('delete-workspace-id') as HTMLInputElement | null;
    const confirmInput = document.getElementById('delete-workspace-confirm-input') as HTMLInputElement | null;
    const confirmBtn = document.getElementById('delete-workspace-confirm-btn') as HTMLButtonElement | null;
    if (!dialog || !name || !idInput || !confirmInput || !confirmBtn) return;

    name.textContent = displayName;
    idInput.value = workspace;
    confirmInput.value = '';
    confirmInput.readOnly = false;
    confirmBtn.textContent = 'Delete';
    confirmBtn.disabled = true;
    confirmInput.oninput = function() {
        const value = confirmInput.value.trim();
        confirmBtn.disabled = value !== displayName && value !== workspace;
    };
    dialog.showModal();
}

function setupWorkspaceEvents(): void {
    const selector = document.querySelector('workspace-scope');
    selector?.addEventListener('workspace-delete', function({detail}) {
        showDeleteWorkspaceDialog(detail.workspace);
    });

    document.addEventListener('click', (event) => {
        const closeButton = closestElement(event.target, '[data-action="close-delete-workspace-dialog"]');
        if (!closeButton) return;
        event.preventDefault();
        closeButton.closest('dialog')?.close();
    });

    // Dropping a workspace clears every store it owns, so the request outlives
    // the click and the dialog would otherwise sit unchanged.
    const deleteForm = document.getElementById('delete-workspace-form');
    if (deleteForm && !deleteForm.dataset.bound) {
        deleteForm.dataset.bound = 'true';
        deleteForm.addEventListener('submit', async (event) => {
            event.preventDefault();
            const workspace = (document.getElementById('delete-workspace-id') as HTMLInputElement | null)?.value;
            if (!workspace) return;
            setDeleteWorkspacePending(true);
            try {
                const deleted = await deleteWorkspaceRequest(workspace);
                workspaceStore.remove(deleted.workspace, deleted.next_workspace);
            } catch (error) {
                showToast(
                    error instanceof WorkspaceApiError ? error.message : 'Could not delete workspace.',
                    4000,
                );
            } finally {
                setDeleteWorkspacePending(false);
            }
        });
    }

    // The scope control re-renders itself from the store; only the toast and the
    // dialog are the shell's business.
    bus.on('workspaceCreated', ({workspace}) => {
        selector?.close();
        showToast(`Workspace ${workspaceName(workspace)} created.`);
    });
    bus.on('workspaceDeleted', ({workspace}) => {
        const dialog = document.getElementById('delete-workspace-dialog') as HTMLDialogElement | null;
        if (dialog && dialog.open) dialog.close();
        showToast(`Workspace ${workspace} deleted.`);
    });
}
