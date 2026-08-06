// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {bus, type WorkspaceRecord} from '../events/bus.ts';
import {toWorkspaceRecord, type WorkspaceEventDetail} from '../events/workspace_events.ts';
import {
    WorkspaceApiError,
    createWorkspaceRequest,
    deleteWorkspaceRequest,
} from '../api/workspaces.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {showToast} from './toast.ts';
import workspaceStyles from '../styles/workspaces.module.css';
import {installRovingArrowNavigation} from '../lib/listbox.ts';
import {createAutoDismiss} from '../lib/popover.ts';
import {closestElement} from '../lib/dom.ts';

let popoverEl: HTMLElement | null = null;

const workspaceDismiss = createAutoDismiss({
    getAnchor: () => document.getElementById('workspace-selector'),
    isOpen: () => popoverEl !== null,
    onDismiss: () => closeWorkspacePopover(),
});

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
    const selector = document.getElementById('workspace-selector');
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
    renderWorkspaceSelector();
    setupWorkspaceEvents();
}

export function toggleWorkspace(workspace: string): void {
    workspaceStore.toggle(workspace);
}

function renderWorkspaceSelector(): void {
    const label = document.getElementById('workspace-label');
    const dot = document.getElementById('workspace-dot');
    if (!label || !dot) return;

    const total = workspaceStore.records.length;
    const activeCount = workspaceStore.active.length;
    const allSelected = total > 0 && workspaceStore.records.every(
        (item) => workspaceStore.active.indexOf(item.workspace) >= 0,
    );

    dot.classList.toggle('multi', activeCount > 1 || allSelected);

    const scopeAnchor = workspaceStore.active.includes(workspaceStore.primary)
        ? workspaceStore.primary
        : workspaceStore.active[0] || '';

    if (activeCount === 0 || allSelected) {
        label.textContent = total > 0 ? `All workspaces (${total})` : 'All workspaces';
    } else if (activeCount === 1) {
        label.textContent = workspaceName(scopeAnchor);
    } else {
        label.textContent = `${workspaceName(scopeAnchor)} + ${activeCount - 1}`;
    }
}

function selectAllWorkspaces(): void {
    if (workspaceStore.records.length === 0) return;
    workspaceStore.selectAll();
    closeWorkspacePopover();
    openWorkspacePopover();
}

function allWorkspacesSelected(): boolean {
    const allIds = workspaceStore.records.map((item) => item.workspace);
    return allIds.length > 0 && allIds.every(
        (workspace) => workspaceStore.active.indexOf(workspace) >= 0,
    );
}

function updateAllWorkspaceItemState(container: Element): void {
    const item = container.querySelector<HTMLElement>('[data-workspace-all="true"]');
    if (!item) return;
    const selected = allWorkspacesSelected();
    item.setAttribute('aria-selected', selected ? 'true' : 'false');
    const check = item.firstElementChild;
    if (check instanceof HTMLElement) {
        check.classList.toggle(workspaceStyles.on, selected);
    }
}

export function openWorkspacePopover(): void {
    if (popoverEl) {
        closeWorkspacePopover();
        return;
    }
    const selector = document.getElementById('workspace-selector');
    if (!selector) return;
    selector.classList.add('open');

    const popover = document.createElement('div');
    popover.className = 'ui-popover ui-popover--workspace';
    popover.setAttribute('role', 'listbox');
    popover.setAttribute('aria-label', 'Workspaces');

    const allItem = document.createElement('div');
    allItem.className = `ui-popover-item ${workspaceStyles.workspacePopoverAll}`;
    allItem.dataset.workspaceAll = 'true';
    allItem.setAttribute('tabindex', '0');
    allItem.setAttribute('role', 'option');
    const allCheck = document.createElement('div');
    const isAllSelected = allWorkspacesSelected();
    allItem.setAttribute('aria-selected', isAllSelected ? 'true' : 'false');
    allCheck.className = `${workspaceStyles.workspacePopoverCheck}${isAllSelected ? ' ' + workspaceStyles.on : ''}`;
    allItem.appendChild(allCheck);
    allItem.appendChild(document.createTextNode('All workspaces'));
    allItem.addEventListener('click', (event) => {
        event.stopPropagation();
        selectAllWorkspaces();
    });
    allItem.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            selectAllWorkspaces();
        }
    });
    popover.appendChild(allItem);

    workspaceStore.records.slice().sort((a, b) => a.displayName.localeCompare(b.displayName)).forEach((record) => {
        const item = document.createElement('div');
        item.className = `ui-popover-item ${workspaceStyles.workspacePopoverItem}`;
        item.setAttribute('tabindex', '0');
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', workspaceStore.active.indexOf(record.workspace) >= 0 ? 'true' : 'false');
        const check = document.createElement('div');
        check.className = `${workspaceStyles.workspacePopoverCheck}${workspaceStore.active.indexOf(record.workspace) >= 0 ? ' ' + workspaceStyles.on : ''}`;
        item.appendChild(check);

        const name = document.createElement('span');
        name.className = workspaceStyles.workspacePopoverName;
        name.textContent = record.displayName;
        item.appendChild(name);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = workspaceStyles.workspacePopoverDelete;
        deleteBtn.type = 'button';
        deleteBtn.textContent = '✕';
        deleteBtn.title = 'Delete workspace';
        deleteBtn.addEventListener('click', (event) => {
            event.stopPropagation();
            closeWorkspacePopover();
            showDeleteWorkspaceDialog(record.workspace);
        });
        item.appendChild(deleteBtn);

        item.addEventListener('click', (event) => {
            if (event.target === deleteBtn) return;
            event.stopPropagation();
            toggleWorkspace(record.workspace);
            check.classList.toggle(workspaceStyles.on, workspaceStore.active.indexOf(record.workspace) >= 0);
            item.setAttribute('aria-selected', workspaceStore.active.indexOf(record.workspace) >= 0 ? 'true' : 'false');
            updateAllWorkspaceItemState(popover);
        });
        item.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                toggleWorkspace(record.workspace);
                check.classList.toggle(workspaceStyles.on, workspaceStore.active.indexOf(record.workspace) >= 0);
                item.setAttribute('aria-selected', workspaceStore.active.indexOf(record.workspace) >= 0 ? 'true' : 'false');
                updateAllWorkspaceItemState(popover);
            }
        });
        popover.appendChild(item);
    });

    popover.appendChild(createRow());
    selector.appendChild(popover);
    popoverEl = popover;
    installRovingArrowNavigation(popover, '[role="option"]');
    workspaceDismiss.activate();
}

function createRow(): HTMLDivElement {
    const row = document.createElement('div');
    row.className = 'ui-popover-create';

    const input = document.createElement('input');
    input.className = 'ui-popover-input';
    input.type = 'text';
    input.placeholder = 'New workspace...';
    input.addEventListener('click', (event) => event.stopPropagation());
    input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            createWorkspace(input);
        }
    });
    row.appendChild(input);

    const button = document.createElement('button');
    button.className = 'ui-popover-create-btn';
    button.type = 'button';
    button.textContent = '+';
    button.addEventListener('click', (event) => {
        event.stopPropagation();
        createWorkspace(input);
    });
    row.appendChild(button);
    return row;
}

export async function createWorkspace(input: HTMLInputElement): Promise<void> {
    const name = input.value.trim();
    if (!name) return;
    input.disabled = true;
    try {
        const created = await createWorkspaceRequest(name);
        workspaceStore.add({
            workspace: created.workspace,
            displayName: created.display_name,
            embeddingModel: '',
        });
    } catch (error) {
        showToast(error instanceof WorkspaceApiError ? error.message : 'Failed to create workspace', 5000);
    } finally {
        input.disabled = false;
    }
}

function closeWorkspacePopover(): void {
    if (popoverEl) {
        popoverEl.remove();
        popoverEl = null;
    }
    const selector = document.getElementById('workspace-selector');
    if (selector) selector.classList.remove('open');
    workspaceDismiss.deactivate();
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
    const selector = document.getElementById('workspace-selector');
    if (selector && !selector.dataset.bound) {
        selector.dataset.bound = 'true';
        selector.addEventListener('click', (event) => {
            if (closestElement(event.target, '.ui-popover')) return;
            openWorkspacePopover();
        });
        selector.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            openWorkspacePopover();
        });
    }

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

    // Store change -> workspace-selector UI (typed bus, camelCase).
    bus.on('workspaceCreated', ({workspace}) => {
        renderWorkspaceSelector();
        closeWorkspacePopover();
        showToast(`Workspace ${workspaceName(workspace)} created.`);
    });
    bus.on('workspaceDeleted', ({workspace}) => {
        renderWorkspaceSelector();
        const dialog = document.getElementById('delete-workspace-dialog') as HTMLDialogElement | null;
        if (dialog && dialog.open) dialog.close();
        showToast(`Workspace ${workspace} deleted.`);
    });
    bus.on('workspaceToggled', () => renderWorkspaceSelector());
}
