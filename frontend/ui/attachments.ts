// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {canAddImage, getImageAdmissionPolicy} from './image_policy.ts';
import {detectDropItems} from './folder-upload.ts';

export type AttachmentKind = 'image' | 'document' | 'unsupported';

export interface DocumentAdmissionPolicy {
    currentUploadLimit: number;
    maxUploadBytes: number;
}

interface PendingDocument {
    file: File;
    objectUrl: string;
}

// Browser-only capabilities are loaded lazily so this module keeps a CSS-free
// static import graph (the `images.ts`/`files-panel.ts` modules pull CSS module
// stylesheets that the Node test runner cannot parse). `setupAttachmentInputs()`
// resolves the bridge before any composer interaction can occur.
interface AttachmentBrowserBridge {
    addImage: (file: File) => void;
    clearImages: () => void;
    getPendingImageData: () => string[];
    renderPendingImageThumbnails: (strip: Element) => void;
    uploadFilesToWorkspace: (files: readonly File[], label?: string | null) => Promise<void>;
    styles: {readonly [key: string]: string};
}

// Mirrors the server-side `_DOCUMENT_SUFFIXES` admission set (without the dot).
const DOCUMENT_EXTENSIONS = new Set([
    'pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'md', 'markdown', 'textpack',
    'txt', 'csv', 'json', 'html', 'htm', 'xml', 'yaml', 'yml', 'rtf', 'odt', 'epub',
    'tex', 'log', 'py', 'js', 'ts', 'css', 'scss', 'sql', 'sh', 'conf', 'ini', 'properties',
]);

const pendingDocuments: PendingDocument[] = [];
let bridge: AttachmentBrowserBridge | null = null;

function parseInteger(value: string | undefined, minimum: number): number | null {
    if (!value || !/^\d+$/.test(value)) return null;
    const parsed = Number(value);
    if (!Number.isSafeInteger(parsed) || parsed < minimum) return null;
    return parsed;
}

export function classifyAttachmentFile(file: File): AttachmentKind {
    if (file.type.startsWith('image/')) return 'image';
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    return DOCUMENT_EXTENSIONS.has(extension) ? 'document' : 'unsupported';
}

export function getDocumentAdmissionPolicy(
    root: Pick<HTMLElement, 'dataset'> | null = document.getElementById('app'),
): DocumentAdmissionPolicy | null {
    if (!root) return null;
    const currentUploadLimit = parseInteger(root.dataset.documentCurrentUploadLimit, 0);
    const maxUploadBytes = parseInteger(root.dataset.documentMaxUploadBytes, 1);
    if (currentUploadLimit === null || maxUploadBytes === null) return null;
    return {currentUploadLimit, maxUploadBytes};
}

export function acceptsDocumentUpload(
    file: File,
    currentCount: number,
    policy: DocumentAdmissionPolicy,
): boolean {
    return (
        classifyAttachmentFile(file) === 'document' &&
        currentCount < policy.currentUploadLimit &&
        file.size <= policy.maxUploadBytes
    );
}

export function addAttachmentFile(file: File): void {
    const kind = classifyAttachmentFile(file);
    if (kind === 'image') {
        bridge?.addImage(file);
        return;
    }
    if (kind !== 'document') return;
    const policy = getDocumentAdmissionPolicy();
    if (!policy || !acceptsDocumentUpload(file, pendingDocuments.length, policy)) return;
    pendingDocuments.push({file, objectUrl: URL.createObjectURL(file)});
    renderAttachmentStrip();
}

export function getPendingDocumentFiles(): File[] {
    return pendingDocuments.map(function(item) { return item.file; });
}

export function getPendingImagesForSubmit(): string[] {
    return bridge?.getPendingImageData() ?? [];
}

export function clearAttachments(): void {
    pendingDocuments.forEach(function(item) { URL.revokeObjectURL(item.objectUrl); });
    pendingDocuments.length = 0;
    if (bridge) {
        bridge.clearImages();
        return;
    }
    renderAttachmentStrip();
}

function removeDocument(target: PendingDocument): void {
    const index = pendingDocuments.indexOf(target);
    if (index < 0) return;
    URL.revokeObjectURL(target.objectUrl);
    pendingDocuments.splice(index, 1);
    renderAttachmentStrip();
}

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    const units = ['KB', 'MB', 'GB'];
    let value = bytes / 1024;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    const rounded = value >= 10 || Number.isInteger(value) ? Math.round(value) : Math.round(value * 10) / 10;
    return `${rounded} ${units[unitIndex]}`;
}

export function renderAttachmentStrip(): void {
    const strip = document.getElementById('thumbnail-strip');
    if (!strip || !bridge) return;
    const styles = bridge.styles;
    strip.replaceChildren();
    bridge.renderPendingImageThumbnails(strip);
    pendingDocuments.forEach(function(doc) {
        const chip = document.createElement('div');
        chip.className = styles.documentChip;
        chip.dataset.documentAttachment = 'true';

        const info = document.createElement('div');
        info.className = styles.documentChipInfo;

        const name = document.createElement('span');
        name.className = styles.documentChipName;
        name.textContent = doc.file.name;
        name.title = doc.file.name;

        const meta = document.createElement('span');
        meta.className = styles.documentChipMeta;
        meta.textContent = formatBytes(doc.file.size);

        info.append(name, meta);

        const remove = document.createElement('button');
        remove.type = 'button';
        remove.className = styles.documentChipRemove;
        remove.setAttribute('aria-label', `Remove ${doc.file.name}`);
        remove.textContent = '\u00d7';
        remove.addEventListener('click', function() { removeDocument(doc); });

        chip.append(info, remove);
        strip.appendChild(chip);
    });
}

function applyAttachmentCapabilityGate(plusBtn: HTMLElement): void {
    const imagePolicy = getImageAdmissionPolicy();
    const documentPolicy = getDocumentAdmissionPolicy();
    const imagesAllowed =
        imagePolicy !== null &&
        canAddImage({
            status: imagePolicy.capabilityStatus,
            limit: imagePolicy.effectiveCurrentUploadLimit,
            current: 0,
        });
    const documentsAllowed = documentPolicy !== null && documentPolicy.currentUploadLimit > 0;
    if (imagesAllowed || documentsAllowed) return;
    if (plusBtn instanceof HTMLButtonElement) plusBtn.disabled = true;
    plusBtn.setAttribute('aria-disabled', 'true');
    plusBtn.title = 'Attachments are currently unavailable.';
}

export async function setupAttachmentInputs(): Promise<void> {
    const [imagesModule, filesModule, chatStylesModule] = await Promise.all([
        import('./images.ts'),
        import('./files-panel.ts'),
        import('../styles/chat.module.css'),
    ]);
    bridge = {
        addImage: imagesModule.addImage,
        clearImages: imagesModule.clearImages,
        getPendingImageData: imagesModule.getPendingImageData,
        renderPendingImageThumbnails: imagesModule.renderPendingImageThumbnails,
        uploadFilesToWorkspace: filesModule.uploadFilesToWorkspace,
        styles: chatStylesModule.default,
    };
    imagesModule.setPendingImagesChangedHandler(renderAttachmentStrip);

    const plusBtn = document.getElementById('composer-plus');
    const attachmentInput = document.getElementById('attachment-input') as HTMLInputElement | null;
    if (plusBtn && attachmentInput) {
        applyAttachmentCapabilityGate(plusBtn);
        plusBtn.addEventListener('click', function() { attachmentInput.click(); });
        attachmentInput.addEventListener('change', function() {
            Array.from(attachmentInput.files || []).forEach(function(f) { addAttachmentFile(f); });
            attachmentInput.value = '';
        });
    }

    const dropOverlay = document.getElementById('drop-overlay');
    let dragCounter = 0;
    document.addEventListener('dragenter', function(e) {
        e.preventDefault();
        if (e.dataTransfer && e.dataTransfer.types.indexOf('Files') >= 0) {
            dragCounter++;
            if (dropOverlay) dropOverlay.classList.add('active');
        }
    });
    document.addEventListener('dragleave', function(e) {
        e.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            if (dropOverlay) dropOverlay.classList.remove('active');
        }
    });
    document.addEventListener('dragover', function(e) { e.preventDefault(); });
    document.addEventListener('drop', async function(e) {
        e.preventDefault();
        dragCounter = 0;
        if (dropOverlay) dropOverlay.classList.remove('active');

        const items = e.dataTransfer?.items;
        if (!items || items.length === 0) return;

        const result = await detectDropItems(
            items,
            function(imageFile) { addAttachmentFile(imageFile); },
        );

        if (result.files.length > 0) {
            await bridge?.uploadFilesToWorkspace(result.files, result.folderName);
        }
    });
    document.addEventListener('paste', function(e) {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        for (let i = 0; i < items.length; i++) {
            if (items[i].type.startsWith('image/')) {
                const file = items[i].getAsFile();
                if (file) addAttachmentFile(file);
            }
        }
    });
}
