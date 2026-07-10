// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {bus} from '../events/bus.ts';
import {setSanitizedHtml} from '../lib/safe_html.ts';
import {ingestStore} from '../stores/ingestStore.ts';
import {workspaceStore} from '../stores/workspaceStore.ts';
import {closestElement, openPanel} from './panel.ts';
import {type RelativeFile, withRelativePath} from './folder-upload.ts';
import {showToast} from './toast.ts';
import {createWorkspace} from './workspaces.ts';

type IngestWorkspaceRecord = {
    workspace: string;
    displayName: string;
};

type PanelRequest = {
    id: number;
    controller: AbortController;
    workspace: string;
};

let ingestPopoverEl: HTMLElement | null = null;
let activePanelRequest: PanelRequest | null = null;
let nextPanelRequestId = 0;
let ingestPollController: AbortController | null = null;
let ingestPollTimer: number | null = null;
let ingestPollWorkspace: string | null = null;

function isAbortError(error: unknown): boolean {
    return error instanceof DOMException && error.name === 'AbortError';
}

function isFilesPanelActive(): boolean {
    const panel = document.getElementById('panel');
    const title = document.getElementById('panel-title');
    return Boolean(panel?.classList.contains('open') && title?.textContent === 'FILES');
}

function filePanelUrl(path: string, workspace: string): string {
    const url = new URL(path, window.location.origin);
    if (workspace) url.searchParams.set('workspace', workspace);
    return url.pathname + url.search;
}

function startPanelRequest(workspace: string): PanelRequest {
    activePanelRequest?.controller.abort();
    stopIngestPolling();
    const request = {
        id: ++nextPanelRequestId,
        controller: new AbortController(),
        workspace,
    };
    activePanelRequest = request;
    return request;
}

function finishPanelRequest(request: PanelRequest): void {
    if (activePanelRequest?.id === request.id) {
        activePanelRequest = null;
    }
}

function isCurrentPanelRequest(request: PanelRequest): boolean {
    return activePanelRequest?.id === request.id && ingestStore.workspace === request.workspace;
}

function abortPanelRequest(): void {
    activePanelRequest?.controller.abort();
    activePanelRequest = null;
    nextPanelRequestId++;
}

function applyProgressBars(root: Element | null | undefined): void {
    if (!root) return;
    const bars: Element[] = [];
    if (root.id === 'ingest-progress') bars.push(root);
    root.querySelectorAll('#ingest-progress').forEach(function(bar) {
        bars.push(bar);
    });
    bars.forEach(function(bar) {
        const fill = bar.querySelector<HTMLElement>('.progress-bar-fill');
        if (!fill) return;
        const pct = fill.getAttribute('data-pct');
        if (pct !== null) {
            fill.style.width = Math.max(0, Math.min(100, parseInt(pct, 10) || 0)) + '%';
        }
    });
}

function showEmptyPanelPlaceholder(): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent || panelContent.children.length > 0) return;
    const placeholder = document.createElement('div');
    placeholder.className = 'empty-state';
    placeholder.textContent = 'No files yet. Drop files or folders here to ingest.';
    panelContent.appendChild(placeholder);
}

function showFilePanelLoading(): void {
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    panelContent.replaceChildren();

    const status = document.createElement('div');
    status.className = 'file-status file-status--loading';

    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    status.appendChild(spinner);

    const text = document.createElement('span');
    text.textContent = 'Loading files...';
    status.appendChild(text);

    panelContent.appendChild(status);
}

function processSwappedFilePanelContent(panelContent: HTMLElement): void {
    if (window.htmx && typeof window.htmx.process === 'function') {
        window.htmx.process(panelContent);
    }
    applyProgressBars(panelContent);
    showEmptyPanelPlaceholder();
}

function applyFilePanelHtml(html: string, workspace: string, request?: PanelRequest): void {
    if (request && !isCurrentPanelRequest(request)) return;
    const panelContent = document.getElementById('panel-content');
    if (!panelContent) return;
    setSanitizedHtml(panelContent, html);
    processSwappedFilePanelContent(panelContent);

    if (workspace === ingestStore.workspace && panelContent.querySelector('#ingest-progress')) {
        startIngestPolling(workspace);
    } else {
        stopIngestPolling();
    }
}

async function requestFilePanelHtml(
    request: PanelRequest,
    path: string,
    init?: RequestInit,
): Promise<Response> {
    return fetch(filePanelUrl(path, request.workspace), {
        ...init,
        signal: request.controller.signal,
    });
}

export async function refreshFilePanel(workspace = ingestStore.workspace): Promise<void> {
    const request = startPanelRequest(workspace);
    showFilePanelLoading();
    try {
        const response = await requestFilePanelHtml(request, '/web/files');
        const html = await response.text();
        applyFilePanelHtml(html, workspace, request);
    } catch (error: unknown) {
        if (isAbortError(error)) return;
        if (isCurrentPanelRequest(request)) {
            showToast('Failed to load files.', 5000);
        }
    } finally {
        finishPanelRequest(request);
    }
}

function createCaretIcon(): SVGSVGElement {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '8');
    svg.setAttribute('height', '8');
    svg.setAttribute('viewBox', '0 0 10 10');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '1.5');
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M2.5 4 L5 6.5 L7.5 4');
    svg.appendChild(path);
    return svg;
}

function renderIngestPill(): void {
    const container = document.getElementById('ingest-target');
    if (!container) return;
    container.replaceChildren();

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
    name.textContent = ingestStore.workspace;
    pill.appendChild(name);

    const caret = document.createElement('span');
    caret.className = 'ingest-target-caret';
    caret.appendChild(createCaretIcon());
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
}

function getIngestWorkspaceRecords(): IngestWorkspaceRecord[] {
    return workspaceStore.records.map((record) => ({
        workspace: record.workspace,
        displayName: record.displayName,
    }));
}

function toggleIngestPopover(container: HTMLElement): void {
    if (ingestPopoverEl) {
        closeIngestPopover();
        return;
    }
    container.classList.add('open');
    const pill = container.querySelector('.ingest-target-pill');
    if (pill) pill.setAttribute('aria-expanded', 'true');

    const popover = document.createElement('div');
    popover.className = 'ui-popover ui-popover--ingest';
    popover.setAttribute('role', 'listbox');
    popover.setAttribute('aria-label', 'Select ingest workspace');

    const records = getIngestWorkspaceRecords();
    const current = ingestStore.workspace;

    records.slice().sort((a, b) => a.displayName.localeCompare(b.displayName)).forEach((record) => {
        const item = document.createElement('div');
        item.className = 'ui-popover-item';
        item.setAttribute('role', 'option');
        item.setAttribute('tabindex', '0');
        item.setAttribute('aria-selected', record.workspace === current ? 'true' : 'false');

        const radio = document.createElement('div');
        radio.className = 'ingest-target-popover-radio' + (record.workspace === current ? ' on' : '');
        item.appendChild(radio);

        const nameSpan = document.createElement('span');
        nameSpan.textContent = record.displayName;
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

    const createRow = document.createElement('div');
    createRow.className = 'ui-popover-create';

    const input = document.createElement('input');
    input.className = 'ui-popover-input';
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
    btn.className = 'ui-popover-create-btn';
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

function selectIngestWorkspace(workspace: string): void {
    ingestStore.set(workspace);
    updateIngestPillLabel();
    closeIngestPopover();
    void refreshFilePanel(workspace);
}

function updateIngestPillLabel(): void {
    const nameEl = document.querySelector('.ingest-target-name');
    if (nameEl) nameEl.textContent = ingestStore.workspace;
}

function closeIngestPopover(): void {
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

function onIngestPopoverOutside(e: MouseEvent): void {
    const container = document.getElementById('ingest-target');
    if (container && e.target instanceof Node && !container.contains(e.target)) closeIngestPopover();
}

function onIngestPopoverEscape(e: KeyboardEvent): void {
    if (e.key === 'Escape') closeIngestPopover();
}

function setUploadBusy(isBusy: boolean): void {
    document.getElementById('upload-zone')?.classList.toggle('is-uploading', isBusy);
}

function uploadLabel(files: readonly File[], label?: string | null): string {
    if (label) return label;
    return files.length === 1 ? files[0].name : files.length + ' files';
}

function uploadPath(file: RelativeFile): string {
    return file._relativePath || file.webkitRelativePath || file.name;
}

export async function uploadFilesToWorkspace(
    files: readonly File[],
    label?: string | null,
): Promise<void> {
    if (files.length === 0) return;
    const workspace = ingestStore.workspace;
    const request = startPanelRequest(workspace);
    const formData = new FormData();
    formData.append('workspace', workspace);
    files.forEach(function(file) {
        const relativeFile = file as RelativeFile;
        formData.append('files', file, uploadPath(relativeFile));
    });

    const name = uploadLabel(files, label);
    setUploadBusy(true);
    showToast('Uploading ' + name + '...');

    try {
        const response = await fetch('/web/files/upload', {
            method: 'POST',
            body: formData,
            signal: request.controller.signal,
        });
        const html = await response.text();
        applyFilePanelHtml(html, workspace, request);
        showToast(
            response.ok ? 'Files received — processing in background' : 'Upload failed.',
            response.ok ? 3000 : 5000,
        );
    } catch (error: unknown) {
        if (isAbortError(error)) return;
        showToast('Upload failed: ' + (error instanceof Error ? error.message : String(error)), 5000);
    } finally {
        if (isCurrentPanelRequest(request)) setUploadBusy(false);
        finishPanelRequest(request);
    }
}

async function deleteFile(filePath: string): Promise<void> {
    if (!filePath) return;
    const filename = filePath.split('/').pop() || filePath;
    if (!window.confirm('Delete ' + filename + '?')) return;

    const workspace = ingestStore.workspace;
    const request = startPanelRequest(workspace);
    const url = new URL('/web/files', window.location.origin);
    if (workspace) url.searchParams.set('workspace', workspace);
    url.searchParams.set('file_path', filePath);

    try {
        const response = await fetch(url.pathname + url.search, {
            method: 'DELETE',
            signal: request.controller.signal,
        });
        const html = await response.text();
        applyFilePanelHtml(html, workspace, request);
        showToast(response.ok ? 'File deleted.' : 'Deletion failed.', response.ok ? 3000 : 5000);
    } catch (error: unknown) {
        if (isAbortError(error)) return;
        showToast('Deletion failed.', 5000);
    } finally {
        finishPanelRequest(request);
    }
}

function replaceIngestProgressHtml(html: string, workspace: string): void {
    if (workspace !== ingestStore.workspace) return;
    const template = document.createElement('template');
    setSanitizedHtml(template, html);
    const replacement = template.content.querySelector('#ingest-progress');
    const current = document.getElementById('ingest-progress');
    if (!replacement || !current) {
        applyFilePanelHtml(html, workspace);
        return;
    }
    current.replaceWith(replacement);
    applyProgressBars(replacement);
    startIngestPolling(workspace);
}

async function pollIngestStatus(workspace: string): Promise<void> {
    const controller = new AbortController();
    ingestPollController = controller;
    try {
        const response = await fetch(filePanelUrl('/web/ingest-status', workspace), {
            signal: controller.signal,
        });
        const html = await response.text();
        if (workspace !== ingestStore.workspace) return;
        if (response.headers.get('HX-Retarget') === '#panel-content') {
            applyFilePanelHtml(html, workspace);
            return;
        }
        replaceIngestProgressHtml(html, workspace);
    } catch (error: unknown) {
        if (isAbortError(error)) return;
        if (workspace === ingestStore.workspace) startIngestPolling(workspace);
    } finally {
        if (ingestPollController === controller) ingestPollController = null;
    }
}

function startIngestPolling(workspace: string): void {
    stopIngestPolling();
    ingestPollWorkspace = workspace;
    ingestPollTimer = window.setTimeout(function() {
        ingestPollTimer = null;
        if (ingestPollWorkspace === workspace) void pollIngestStatus(workspace);
    }, 2000);
}

function stopIngestPolling(): void {
    if (ingestPollTimer !== null) {
        window.clearTimeout(ingestPollTimer);
        ingestPollTimer = null;
    }
    ingestPollController?.abort();
    ingestPollController = null;
    ingestPollWorkspace = null;
}

function handleFileInputChange(input: HTMLInputElement): void {
    const files = Array.from(input.files || []);
    if (files.length > 0) void uploadFilesToWorkspace(files);
    input.value = '';
}

function handleFolderInputChange(input: HTMLInputElement): void {
    const fileList = input.files;
    if (!fileList || fileList.length === 0) return;
    const rawFiles = Array.from(fileList);
    let folderName: string | null = null;
    const files = rawFiles.map(function(file) {
        const path = file.webkitRelativePath || file.name;
        const relativeFile = withRelativePath(file, path);
        if (!folderName && file.webkitRelativePath) {
            folderName = file.webkitRelativePath.split('/')[0];
        }
        return relativeFile;
    });
    void uploadFilesToWorkspace(files, folderName);
    input.value = '';
}

export function openFilesPanel(): void {
    ingestStore.resetToPrimary();
    openPanel('FILES');
    renderIngestPill();
    void refreshFilePanel(ingestStore.workspace);
}

export function setupFilesPanel(): void {
    const filesButton = document.getElementById('files-btn');
    if (filesButton) {
        filesButton.addEventListener('click', function(e) {
            e.preventDefault();
            openFilesPanel();
        });
    }

    const panelContent = document.getElementById('panel-content');
    if (panelContent) {
        panelContent.addEventListener('click', function(e) {
            const deleteButton = closestElement<HTMLElement>(e.target, '[data-action="delete-file"]');
            if (deleteButton) {
                e.preventDefault();
                void deleteFile(deleteButton.dataset.filePath || '');
                return;
            }

            const uploadZone = closestElement(e.target, '#upload-zone');
            if (!uploadZone) return;
            if (closestElement(e.target, '[data-action="upload-file"]')) {
                document.getElementById('file-input')?.click();
                return;
            }
            if (closestElement(e.target, '[data-action="upload-folder"]')) {
                document.getElementById('folder-input')?.click();
                return;
            }
            if (!closestElement(e.target, '[data-action]')) {
                document.getElementById('file-input')?.click();
            }
        });
        panelContent.addEventListener('change', function(e) {
            const target = e.target instanceof HTMLInputElement ? e.target : null;
            if (target?.id === 'file-input') handleFileInputChange(target);
        });
    }

    const folderInput = document.getElementById('folder-input') as HTMLInputElement | null;
    if (folderInput) {
        folderInput.addEventListener('change', function() {
            handleFolderInputChange(folderInput);
        });
    }

    bus.on('workspaceCreated', ({workspace}) => {
        ingestStore.set(workspace);
        updateIngestPillLabel();
        closeIngestPopover();
        if (isFilesPanelActive()) void refreshFilePanel(workspace);
    });

    bus.on('workspaceDeleted', ({nextWorkspace}) => {
        ingestStore.set(nextWorkspace || workspaceStore.primary);
        updateIngestPillLabel();
        if (isFilesPanelActive()) void refreshFilePanel(ingestStore.workspace);
    });

    document.body.addEventListener('panelClosed', function() {
        closeIngestPopover();
        stopIngestPolling();
        abortPanelRequest();
    });
}
