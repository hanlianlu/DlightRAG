// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {
    acceptsAttachmentUpload,
    attachmentsEnabled,
    classifyAttachmentFile,
    getAttachmentPolicy,
} from './attachment_policy.ts';
import {attachmentStore, type PendingAttachment} from '../stores/attachmentStore.ts';
import {uploadFilesToWorkspace} from './files-panel.ts';
import {detectDropItems} from './folder-upload.ts';
import {createDocumentChip} from '../lib/document_chip.ts';
import chatStyles from '../styles/chat.module.css';

// One ordered pending attachment collection (images + documents) lives in
// attachmentStore. This module admits files against the unified server policy
// and renders the composer strip: image thumbnails and compact document chips.

export function addAttachmentFile(file: File): void {
    const policy = getAttachmentPolicy();
    if (!policy) return;
    const kind = classifyAttachmentFile(file, policy.extensions);
    if (kind === 'unsupported') return;
    const counts = {total: attachmentStore.size, images: attachmentStore.imageCount};
    if (!acceptsAttachmentUpload(file, counts, policy)) return;
    attachmentStore.add(file, kind);
}

export function getPendingAttachments(): readonly PendingAttachment[] {
    return attachmentStore.list();
}

export function clearAttachments(): void {
    attachmentStore.clear();
}

function renderImageThumbnail(item: PendingAttachment): HTMLElement {
    const thumbnail = document.createElement('div');
    thumbnail.className = chatStyles.thumbnailItem;

    const imgEl = document.createElement('img');
    imgEl.className = chatStyles.thumbnailImg;
    imgEl.src = item.objectUrl.startsWith('blob:') ? item.objectUrl : '';
    imgEl.alt = item.file.name;
    thumbnail.appendChild(imgEl);

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = chatStyles.thumbnailRemove;
    remove.textContent = '\u00d7';
    remove.setAttribute('aria-label', `Remove ${item.file.name}`);
    remove.addEventListener('click', function() { attachmentStore.remove(item.id); });
    thumbnail.appendChild(remove);

    return thumbnail;
}

export function renderAttachmentStrip(): void {
    const strip = document.getElementById('thumbnail-strip');
    if (!strip) return;
    strip.replaceChildren();
    attachmentStore.list().forEach(function(item) {
        if (item.kind === 'image') {
            strip.appendChild(renderImageThumbnail(item));
            return;
        }
        strip.appendChild(
            createDocumentChip({
                filename: item.file.name,
                byteSize: item.file.size,
                onRemove: function() { attachmentStore.remove(item.id); },
            }),
        );
    });
}

function applyAttachmentCapabilityGate(plusBtn: HTMLElement): void {
    const policy = getAttachmentPolicy();
    if (policy && attachmentsEnabled(policy)) return;
    if (plusBtn instanceof HTMLButtonElement) plusBtn.disabled = true;
    plusBtn.setAttribute('aria-disabled', 'true');
    plusBtn.title = 'Attachments are currently unavailable.';
}

export function setupAttachmentInputs(): void {
    attachmentStore.subscribe(renderAttachmentStrip);

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
