// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {workspaceStore} from '../stores/workspaceStore.ts';
import {showToast} from './toast.ts';
import workspaceStyles from '../styles/workspaces.module.css';
import type {WorkspaceRecord} from '../events/bus.ts';

type WorkspacePayload = {
    workspace?: string;
    id?: string;
    display_name?: string;
    displayName?: string;
    embedding_model?: string;
    embeddingModel?: string;
    next_workspace?: string;
    nextWorkspace?: string;
    value?: WorkspacePayload;
};

type WorkspaceRecordInput = string | WorkspacePayload;

let popoverEl: HTMLElement | null = null;

export function getWorkspaceRecords(): readonly WorkspaceRecord[] {
    return workspaceStore.records;
}

function payload(event: Event): WorkspacePayload {
    const detail = (event as CustomEvent<WorkspacePayload>).detail;
    return detail?.value || detail || {};
}

function normalizeRecord(record: WorkspaceRecordInput): WorkspaceRecord | null {
    if (typeof record === 'string') {
        return {workspace: record, displayName: record, embeddingModel: ''};
    }
    const workspace = record.workspace || record.id;
    if (!workspace) return null;
    return {
        workspace,
        displayName: record.display_name || record.displayName || workspace,
        embeddingModel: record.embedding_model || record.embeddingModel || '',
    };
}

function workspaceName(workspace: string): string {
    const match = workspaceStore.records.find((item) => item.workspace === workspace);
    return match ? match.displayName : workspace;
}

function eventElement(event: Event): Element | null {
    return event.target instanceof Element ? event.target : null;
}

export function initWorkspaces(): void {
    const selector = document.getElementById('workspace-selector');
    if (!selector) return;

    try {
        const records = (JSON.parse(selector.getAttribute('data-all') || '[]') as WorkspaceRecordInput[])
            .map(normalizeRecord)
            .filter((record): record is WorkspaceRecord => record !== null);
        let active: string[] = [];
        try {
            active = JSON.parse(selector.getAttribute('data-active') || '[]');
        } catch (_) {
            active = [];
        }
        workspaceStore.init(records, active);
    } catch (_) {
        // data-all attribute may be absent; start with empty records
    }
    renderWorkspaceSelector();
    setupWorkspaceEvents();
}

export function toggleWorkspace(workspace: string): void {
    workspaceStore.toggle(workspace);
    renderWorkspaceSelector();
}

export function selectWorkspace(workspace: string): void {
    if (!workspace) return;
    workspaceStore.select(workspace);
    renderWorkspaceSelector();
}

export function removeWorkspace(workspace: string, nextWorkspace?: string): void {
    workspaceStore.remove(workspace, nextWorkspace || '');
    renderWorkspaceSelector();
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

    if (activeCount === 0 || allSelected) {
        label.textContent = total > 0 ? `All Workspaces (${total})` : 'All Workspaces';
    } else if (activeCount === 1) {
        label.textContent = workspaceName(workspaceStore.active[0]);
    } else {
        label.textContent = `Workspaces (${activeCount})`;
    }
}

function selectAllWorkspaces(): void {
    if (workspaceStore.records.length === 0) return;
    workspaceStore.selectAll();
    renderWorkspaceSelector();
    closeWorkspacePopover();
    openWorkspacePopover();
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
    allItem.setAttribute('tabindex', '0');
    allItem.setAttribute('role', 'option');
    const allCheck = document.createElement('div');
    const allIds = workspaceStore.records.map((item) => item.workspace);
    const isAllSelected = allIds.length > 0 && allIds.every(
        (workspace) => workspaceStore.active.indexOf(workspace) >= 0,
    );
    allCheck.className = `${workspaceStyles.workspacePopoverCheck}${isAllSelected ? ' ' + workspaceStyles.on : ''}`;
    allItem.appendChild(allCheck);
    allItem.appendChild(document.createTextNode('All Workspaces'));
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
        });
        item.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                toggleWorkspace(record.workspace);
                check.classList.toggle(workspaceStyles.on, workspaceStore.active.indexOf(record.workspace) >= 0);
                item.setAttribute('aria-selected', workspaceStore.active.indexOf(record.workspace) >= 0 ? 'true' : 'false');
            }
        });
        popover.appendChild(item);
    });

    popover.appendChild(createRow());
    selector.appendChild(popover);
    popoverEl = popover;

    setTimeout(() => {
        document.addEventListener('click', onOutsideClick);
        document.addEventListener('keydown', onEscapeKey);
    }, 0);
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

export function createWorkspace(input: HTMLInputElement): void {
    const name = input.value.trim();
    if (!name) return;
    input.disabled = true;
    htmx.ajax('POST', '/web/workspaces/create', {
        values: {workspace_name: name},
        swap: 'none',
    }).catch(() => {
        input.disabled = false;
        showToast('Failed to create workspace', 5000);
    });
}

function closeWorkspacePopover(): void {
    if (popoverEl) {
        popoverEl.remove();
        popoverEl = null;
    }
    const selector = document.getElementById('workspace-selector');
    if (selector) selector.classList.remove('open');
    document.removeEventListener('click', onOutsideClick);
    document.removeEventListener('keydown', onEscapeKey);
}

function onOutsideClick(event: MouseEvent): void {
    const selector = document.getElementById('workspace-selector');
    if (selector && event.target instanceof Node && !selector.contains(event.target)) closeWorkspacePopover();
}

function onEscapeKey(event: KeyboardEvent): void {
    if (event.key === 'Escape') closeWorkspacePopover();
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
    confirmBtn.disabled = true;
    confirmInput.oninput = function() {
        const value = confirmInput.value.trim();
        confirmBtn.disabled = value !== displayName && value !== workspace;
    };
    dialog.showModal();
}

function syncDataAllAttribute(): void {
    const selector = document.getElementById('workspace-selector');
    if (selector) {
        const data = workspaceStore.records.map(r => ({
            workspace: r.workspace,
            display_name: r.displayName,
            embedding_model: r.embeddingModel,
        }));
        selector.setAttribute('data-all', JSON.stringify(data));
    }
}

function setupWorkspaceEvents(): void {
    const selector = document.getElementById('workspace-selector');
    if (selector && !selector.dataset.bound) {
        selector.dataset.bound = 'true';
        selector.addEventListener('click', (event) => {
            if (eventElement(event)?.closest('.ui-popover')) return;
            openWorkspacePopover();
        });
        selector.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            openWorkspacePopover();
        });
    }

    document.addEventListener('click', (event) => {
        const closeButton = eventElement(event)?.closest('[data-action="close-delete-workspace-dialog"]');
        if (!closeButton) return;
        event.preventDefault();
        closeButton.closest('dialog')?.close();
    });

    document.body.addEventListener('workspaceCreated', (event) => {
        const detail = payload(event);
        const workspace = detail.workspace;
        if (!workspace) return;
        workspaceStore.add({
            workspace,
            displayName: detail.display_name || detail.displayName || workspace,
            embeddingModel: detail.embedding_model || detail.embeddingModel || '',
        });
        renderWorkspaceSelector();
        closeWorkspacePopover();
        syncDataAllAttribute();
        showToast(`Workspace ${workspaceName(workspace)} created.`);
    });

    document.body.addEventListener('workspaceDeleted', (event) => {
        const detail = payload(event);
        const workspace = detail.workspace;
        if (!workspace) return;
        removeWorkspace(workspace, detail.next_workspace || detail.nextWorkspace);
        syncDataAllAttribute();
        const dialog = document.getElementById('delete-workspace-dialog') as HTMLDialogElement | null;
        if (dialog && dialog.open) dialog.close();
        showToast(`Workspace ${workspace} deleted.`);
    });
}
