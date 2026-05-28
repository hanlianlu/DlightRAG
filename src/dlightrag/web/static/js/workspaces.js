// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {activeWorkspaces, setActiveWorkspaces} from './state.js';
import {showToast} from './toast.js';

const PRIMARY_COOKIE = 'dlightrag_workspace';
const ACTIVE_COOKIE = 'dlightrag_workspace_ids';

let workspaceRecords = [];
let popoverEl = null;

export function getWorkspaceRecords() {
    return workspaceRecords;
}

function payload(event) {
    if (!event.detail) return {};
    return event.detail.value || event.detail;
}

function normalizeRecord(record) {
    if (typeof record === 'string') {
        return {workspace: record, display_name: record, embedding_model: ''};
    }
    const workspace = record.workspace || record.id || 'default';
    return {
        workspace,
        display_name: record.display_name || workspace,
        embedding_model: record.embedding_model || '',
    };
}

function workspaceName(workspace) {
    const match = workspaceRecords.find((item) => item.workspace === workspace);
    return match ? match.display_name : workspace;
}

function saveWorkspaceCookies() {
    const primary = activeWorkspaces.length > 0 ? activeWorkspaces[0] : 'default';
    document.cookie = `${PRIMARY_COOKIE}=${encodeURIComponent(primary)};path=/;SameSite=Lax`;
    document.cookie = `${ACTIVE_COOKIE}=${encodeURIComponent(activeWorkspaces.join(','))};path=/;SameSite=Lax`;
}

export function initWorkspaces() {
    const selector = document.getElementById('workspace-selector');
    if (!selector) return;

    try {
        workspaceRecords = JSON.parse(selector.getAttribute('data-all') || '[]').map(normalizeRecord);
    } catch (_) {
        workspaceRecords = [];
    }

    let active = [];
    try {
        active = JSON.parse(selector.getAttribute('data-active') || '[]');
    } catch (_) {
        active = [];
    }

    if (workspaceRecords.length === 0) {
        workspaceRecords = [{workspace: 'default', display_name: 'default', embedding_model: ''}];
    }
    if (active.length === 0) active = [workspaceRecords[0].workspace];

    setActiveWorkspaces(active);
    renderWorkspaceSelector();
    saveWorkspaceCookies();
    setupWorkspaceEvents();
}

export function toggleWorkspace(workspace) {
    const idx = activeWorkspaces.indexOf(workspace);
    if (idx >= 0) {
        if (activeWorkspaces.length <= 1) return;
        activeWorkspaces.splice(idx, 1);
    } else {
        activeWorkspaces.push(workspace);
    }
    renderWorkspaceSelector();
    saveWorkspaceCookies();
}

export function selectWorkspace(workspace) {
    setActiveWorkspaces([workspace || 'default']);
    renderWorkspaceSelector();
    saveWorkspaceCookies();
}

export function removeWorkspace(workspace, fallback) {
    workspaceRecords = workspaceRecords.filter((item) => item.workspace !== workspace);
    const remaining = activeWorkspaces.filter((active) => active !== workspace);
    if (remaining.length > 0) {
        setActiveWorkspaces(remaining);
    } else if (fallback) {
        setActiveWorkspaces([fallback]);
    } else if (workspaceRecords.length > 0) {
        setActiveWorkspaces([workspaceRecords[0].workspace]);
    } else {
        setActiveWorkspaces(['default']);
    }
    renderWorkspaceSelector();
    saveWorkspaceCookies();
}

function renderWorkspaceSelector() {
    const label = document.getElementById('workspace-label');
    const dot = document.getElementById('workspace-dot');
    if (!label || !dot) return;

    const total = workspaceRecords.length;
    const activeCount = activeWorkspaces.length;
    const allSelected = total > 0 && workspaceRecords.every(
        (item) => activeWorkspaces.indexOf(item.workspace) >= 0,
    );

    dot.classList.toggle('multi', activeCount > 1 || allSelected);

    if (activeCount === 0 || allSelected) {
        label.textContent = total > 0 ? `All Workspaces (${total})` : 'All Workspaces';
    } else if (activeCount === 1) {
        label.textContent = workspaceName(activeWorkspaces[0]);
    } else {
        label.textContent = `Workspaces (${activeCount})`;
    }
}

function selectAllWorkspaces() {
    if (workspaceRecords.length === 0) return;
    const allIds = workspaceRecords.map((item) => item.workspace);
    const allSelected = allIds.every((workspace) => activeWorkspaces.indexOf(workspace) >= 0);
    setActiveWorkspaces(allSelected ? [allIds[0]] : allIds);
    renderWorkspaceSelector();
    saveWorkspaceCookies();
    closeWorkspacePopover();
    openWorkspacePopover();
}

export function openWorkspacePopover() {
    if (popoverEl) {
        closeWorkspacePopover();
        return;
    }
    const selector = document.getElementById('workspace-selector');
    if (!selector) return;
    selector.classList.add('open');

    const popover = document.createElement('div');
    popover.className = 'workspace-popover';
    popover.setAttribute('role', 'listbox');
    popover.setAttribute('aria-label', 'Workspaces');

    const allItem = document.createElement('div');
    allItem.className = 'workspace-popover-item workspace-popover-all';
    allItem.setAttribute('tabindex', '0');
    allItem.setAttribute('role', 'option');
    const allCheck = document.createElement('div');
    const allIds = workspaceRecords.map((item) => item.workspace);
    const isAllSelected = allIds.length > 0 && allIds.every(
        (workspace) => activeWorkspaces.indexOf(workspace) >= 0,
    );
    allCheck.className = `workspace-popover-check${isAllSelected ? ' on' : ''}`;
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

    workspaceRecords.slice().sort((a, b) => a.display_name.localeCompare(b.display_name)).forEach((record) => {
        const item = document.createElement('div');
        item.className = 'workspace-popover-item';
        item.setAttribute('tabindex', '0');
        item.setAttribute('role', 'option');
        item.setAttribute('aria-selected', activeWorkspaces.indexOf(record.workspace) >= 0 ? 'true' : 'false');
        const check = document.createElement('div');
        check.className = `workspace-popover-check${activeWorkspaces.indexOf(record.workspace) >= 0 ? ' on' : ''}`;
        item.appendChild(check);

        const name = document.createElement('span');
        name.className = 'workspace-popover-name';
        name.textContent = record.display_name;
        item.appendChild(name);

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'workspace-popover-delete';
        deleteBtn.type = 'button';
        deleteBtn.textContent = '\u2715';
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
            check.classList.toggle('on', activeWorkspaces.indexOf(record.workspace) >= 0);
            item.setAttribute('aria-selected', activeWorkspaces.indexOf(record.workspace) >= 0 ? 'true' : 'false');
        });
        item.addEventListener('keydown', (event) => {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                toggleWorkspace(record.workspace);
                check.classList.toggle('on', activeWorkspaces.indexOf(record.workspace) >= 0);
                item.setAttribute('aria-selected', activeWorkspaces.indexOf(record.workspace) >= 0 ? 'true' : 'false');
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

function createRow() {
    const row = document.createElement('div');
    row.className = 'workspace-popover-create';

    const input = document.createElement('input');
    input.className = 'workspace-popover-input';
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
    button.className = 'workspace-popover-create-btn';
    button.type = 'button';
    button.textContent = '+';
    button.addEventListener('click', (event) => {
        event.stopPropagation();
        createWorkspace(input);
    });
    row.appendChild(button);
    return row;
}

export function createWorkspace(input) {
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

function closeWorkspacePopover() {
    if (popoverEl) {
        popoverEl.remove();
        popoverEl = null;
    }
    const selector = document.getElementById('workspace-selector');
    if (selector) selector.classList.remove('open');
    document.removeEventListener('click', onOutsideClick);
    document.removeEventListener('keydown', onEscapeKey);
}

function onOutsideClick(event) {
    const selector = document.getElementById('workspace-selector');
    if (selector && !selector.contains(event.target)) closeWorkspacePopover();
}

function onEscapeKey(event) {
    if (event.key === 'Escape') closeWorkspacePopover();
}

function showDeleteWorkspaceDialog(workspace) {
    const record = workspaceRecords.find((item) => item.workspace === workspace);
    const displayName = record ? record.display_name : workspace;
    const dialog = document.getElementById('delete-workspace-dialog');
    const name = document.getElementById('delete-workspace-name');
    const idInput = document.getElementById('delete-workspace-id');
    const confirmInput = document.getElementById('delete-workspace-confirm-input');
    const confirmBtn = document.getElementById('delete-workspace-confirm-btn');
    if (!dialog || !name || !idInput || !confirmInput || !confirmBtn) return;

    name.textContent = displayName;
    idInput.value = workspace;
    confirmInput.value = '';
    confirmBtn.disabled = true;
    confirmInput.oninput = function() {
        const value = this.value.trim();
        confirmBtn.disabled = value !== displayName && value !== workspace;
    };
    dialog.showModal();
}

function syncDataAllAttribute() {
    const selector = document.getElementById('workspace-selector');
    if (selector) {
        selector.setAttribute('data-all', JSON.stringify(workspaceRecords));
    }
}

function setupWorkspaceEvents() {
    const selector = document.getElementById('workspace-selector');
    if (selector && !selector.dataset.bound) {
        selector.dataset.bound = 'true';
        selector.addEventListener('click', (event) => {
            if (event.target.closest('.workspace-popover')) return;
            openWorkspacePopover();
        });
        selector.addEventListener('keydown', (event) => {
            if (event.key !== 'Enter' && event.key !== ' ') return;
            event.preventDefault();
            openWorkspacePopover();
        });
    }

    document.addEventListener('click', (event) => {
        const target = event.target instanceof Element ? event.target : null;
        const closeButton = target?.closest('[data-action="close-delete-workspace-dialog"]');
        if (!closeButton) return;
        event.preventDefault();
        closeButton.closest('dialog')?.close();
    });

    document.body.addEventListener('workspaceCreated', (event) => {
        const detail = payload(event);
        const workspace = detail.workspace;
        if (!workspace) return;
        if (!workspaceRecords.some((item) => item.workspace === workspace)) {
            workspaceRecords.push(normalizeRecord({
                workspace,
                display_name: detail.display_name || workspace,
                embedding_model: detail.embedding_model || '',
            }));
        }
        selectWorkspace(workspace);
        closeWorkspacePopover();
        syncDataAllAttribute();
        showToast(`Workspace ${workspaceName(workspace)} created.`);
    });

    document.body.addEventListener('workspaceDeleted', (event) => {
        const detail = payload(event);
        const workspace = detail.workspace;
        if (!workspace) return;
        removeWorkspace(workspace, detail.fallback);
        syncDataAllAttribute();
        const dialog = document.getElementById('delete-workspace-dialog');
        if (dialog && dialog.open) dialog.close();
        showToast(`Workspace ${workspace} deleted.`);
    });
}
