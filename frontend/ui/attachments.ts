// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {canAddImage, getImageAdmissionPolicy} from './image_policy.ts';
import {
    acceptsDocumentUpload,
    classifyAttachmentFile,
    getDocumentAdmissionPolicy,
} from './attachment_policy.ts';
import {
    addImage,
    clearImages,
    renderPendingImageThumbnails,
    setPendingImagesChangedHandler,
} from './images.ts';
import {uploadFilesToWorkspace} from './files-panel.ts';
import {detectDropItems} from './folder-upload.ts';
import {createDocumentChip} from '../lib/document_chip.ts';

interface PendingDocument {
    file: File;
    objectUrl: string;
}

// Query attachments are current query images plus Composer documents.
const pendingDocuments: PendingDocument[] = [];

export function addAttachmentFile(file: File): void {
    const kind = classifyAttachmentFile(file);
    if (kind === 'image') {
        addImage(file);
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

export function clearAttachments(): void {
    pendingDocuments.forEach(function(item) { URL.revokeObjectURL(item.objectUrl); });
    pendingDocuments.length = 0;
    // clearImages() re-renders the shared strip through the registered handler
    // (renderAttachmentStrip), so cleared document chips disappear too.
    clearImages();
}

function removeDocument(target: PendingDocument): void {
    const index = pendingDocuments.indexOf(target);
    if (index < 0) return;
    URL.revokeObjectURL(target.objectUrl);
    pendingDocuments.splice(index, 1);
    renderAttachmentStrip();
}

export function renderAttachmentStrip(): void {
    const strip = document.getElementById('thumbnail-strip');
    if (!strip) return;
    strip.replaceChildren();
    renderPendingImageThumbnails(strip);
    pendingDocuments.forEach(function(doc) {
        strip.appendChild(
            createDocumentChip({
                filename: doc.file.name,
                byteSize: doc.file.size,
                onRemove: function() { removeDocument(doc); },
            }),
        );
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

export function setupAttachmentInputs(): void {
    setPendingImagesChangedHandler(renderAttachmentStrip);

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
            await uploadFilesToWorkspace(result.files, result.folderName);
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
