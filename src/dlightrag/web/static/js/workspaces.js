// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {activeWorkspaces, setActiveWorkspaces} from './state.js';

export function initWorkspaces() {
    const chips = document.querySelectorAll('#workspace-chips .ws-chip');
    const workspaces = [];
    chips.forEach(function(chip) {
        const ws = chip.getAttribute('data-ws');
        if (ws) workspaces.push(ws);
    });
    setActiveWorkspaces(workspaces.length ? workspaces : ['default']);
}

export function toggleWorkspace(ws) {
    const idx = activeWorkspaces.indexOf(ws);
    if (idx >= 0) {
        if (activeWorkspaces.length <= 1) return;
        activeWorkspaces.splice(idx, 1);
    } else {
        activeWorkspaces.push(ws);
    }
    renderWorkspaceChips();
    updateWorkspacePanelCheckboxes();
}

export function renderWorkspaceChips() {
    const container = document.getElementById('workspace-chips');
    if (!container) return;
    const addBtn = document.getElementById('ws-add-btn');
    container.replaceChildren();
    activeWorkspaces.forEach(function(ws) {
        const chip = document.createElement('span');
        chip.className = 'ws-chip';
        chip.setAttribute('data-ws', ws);
        chip.textContent = ws;
        if (activeWorkspaces.length > 1) {
            const removeBtn = document.createElement('button');
            removeBtn.className = 'ws-chip-remove';
            removeBtn.textContent = 'x';
            removeBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                toggleWorkspace(ws);
            });
            chip.appendChild(removeBtn);
        }
        container.appendChild(chip);
    });
    if (addBtn) container.appendChild(addBtn);
}

export function updateWorkspacePanelCheckboxes() {
    document.querySelectorAll('#panel-content .workspace-check-item').forEach(function(item) {
        const ws = item.getAttribute('data-ws');
        const selected = activeWorkspaces.indexOf(ws) >= 0;
        item.classList.toggle('selected', selected);
        const checkbox = item.querySelector('.ws-checkbox');
        if (checkbox) checkbox.classList.toggle('checked', selected);
    });
}
