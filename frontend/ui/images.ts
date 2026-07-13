// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {detectDropItems} from './folder-upload.ts';
import {uploadFilesToWorkspace} from './files-panel.ts';
import chatStyles from '../styles/chat.module.css';
import lightboxStyles from '../styles/lightbox.module.css';
import type {ConversationImageReference} from '../api/conversations.ts';
import {getImageAdmissionPolicy, ImageReadAdmissionController} from './image_policy.ts';

const SAFE_DATA_IMAGE_SRC_RE = /^data:image\/(?:avif|bmp|gif|jpeg|jpg|png|webp);base64,[a-z0-9+/=]+$/i;
const pendingImages: PendingImage[] = [];

interface PendingImage {
    file: File;
    dataUrl: string;
    objectUrl: string;
}

const imageAdmission = new ImageReadAdmissionController({
    getPolicy: getImageAdmissionPolicy,
    getCommittedCount: function() { return pendingImages.length; },
    onReady: function(file, dataUrl) {
        const objectUrl = URL.createObjectURL(file);
        pendingImages.push({file, dataUrl, objectUrl});
        renderThumbnails();
    },
});

type LightboxElement = HTMLDivElement & {
    __lightboxPrev?: HTMLButtonElement;
    __lightboxNext?: HTMLButtonElement;
    __lightboxImg?: HTMLImageElement;
};

function eventElement(event: Event): Element | null {
    return event.target instanceof Element ? event.target : null;
}

export function addImage(file: File): void {
    imageAdmission.admit(file);
}

export function getPendingImageData(): string[] {
    return pendingImages.map(function(img) { return img.dataUrl; });
}

export function renderMessageImages(
    container: Element,
    storedImages?: readonly ConversationImageReference[],
): void {
    if (storedImages) {
        renderStoredMessageImages(container, storedImages);
        return;
    }
    if (pendingImages.length === 0) return;
    const msgImages = document.createElement('div');
    msgImages.className = chatStyles.messageImages;
    pendingImages.forEach(function(img) {
        const imgEl = document.createElement('img');
        imgEl.className = chatStyles.messageImg;
        imgEl.src = img.dataUrl;
        imgEl.alt = 'Attached image';
        msgImages.appendChild(imgEl);
    });
    container.appendChild(msgImages);
}

function renderStoredMessageImages(
    container: Element,
    images: readonly ConversationImageReference[],
): void {
    if (images.length === 0) return;
    const msgImages = document.createElement('div');
    msgImages.className = chatStyles.messageImages;
    images.forEach(function(reference) {
        const card = document.createElement('div');
        card.className = chatStyles.historyImageCard;

        const imageButton = document.createElement('button');
        imageButton.type = 'button';
        imageButton.className = chatStyles.historyImageButton;
        imageButton.setAttribute('aria-label', `Open ${reference.label}`);

        const status = document.createElement('span');
        status.className = chatStyles.historyImageStatus;
        status.setAttribute('role', 'status');
        status.textContent = `Loading ${reference.label}`;

        const retry = document.createElement('button');
        retry.type = 'button';
        retry.className = chatStyles.historyImageRetry;
        retry.textContent = 'Retry image';
        retry.setAttribute('aria-label', `Retry image: ${reference.label}`);
        retry.hidden = true;

        const imgEl = document.createElement('img');
        imgEl.className = chatStyles.messageImg;
        imgEl.alt = reference.label;
        imgEl.loading = 'lazy';
        imgEl.decoding = 'async';
        const thumbnailSrc = _safeImageSrc(reference.thumbnail_url);
        const fullSrc = _safeImageSrc(reference.url);

        const showError = (): void => {
            imgEl.hidden = true;
            imageButton.disabled = true;
            imageButton.removeAttribute('data-action');
            status.textContent = `History image failed to load: ${reference.label}`;
            retry.hidden = false;
        };
        imgEl.addEventListener('load', function() {
            status.hidden = true;
            retry.hidden = true;
        });
        imgEl.addEventListener('error', showError);
        retry.addEventListener('click', function() {
            if (!thumbnailSrc || !fullSrc) return;
            retry.hidden = true;
            status.hidden = false;
            status.textContent = `Loading ${reference.label}`;
            imageButton.disabled = false;
            imageButton.setAttribute('data-action', 'open-lightbox');
            imgEl.hidden = false;
            imgEl.removeAttribute('src');
            window.requestAnimationFrame(function() { imgEl.src = thumbnailSrc; });
        });

        if (thumbnailSrc && fullSrc) {
            imageButton.setAttribute('data-action', 'open-lightbox');
            imageButton.setAttribute('data-full-src', fullSrc);
            imgEl.src = thumbnailSrc;
        } else {
            showError();
        }

        imageButton.append(imgEl, status);
        card.append(imageButton, retry);
        msgImages.appendChild(card);
    });
    container.appendChild(msgImages);
}

export function clearImages(): void {
    imageAdmission.clear();
    pendingImages.forEach(function(img) { URL.revokeObjectURL(img.objectUrl); });
    pendingImages.length = 0;
    renderThumbnails();
}

function removeImage(index: number): void {
    const removed = pendingImages.splice(index, 1);
    if (removed.length > 0) URL.revokeObjectURL(removed[0].objectUrl);
    renderThumbnails();
}

function renderThumbnails(): void {
    const strip = document.getElementById('thumbnail-strip');
    if (!strip) return;
    strip.replaceChildren();
    pendingImages.forEach(function(img, idx) {
        const item = document.createElement('div');
        item.className = chatStyles.thumbnailItem;

        const imgEl = document.createElement('img');
        imgEl.className = chatStyles.thumbnailImg;
        imgEl.src = img.objectUrl && img.objectUrl.startsWith('blob:') ? img.objectUrl : '';
        imgEl.alt = 'Attached image';
        item.appendChild(imgEl);

        const btn = document.createElement('button');
        btn.className = chatStyles.thumbnailRemove;
        btn.textContent = 'x';
        btn.setAttribute('data-idx', String(idx));
        btn.addEventListener('click', function() { removeImage(idx); });
        item.appendChild(btn);

        strip.appendChild(item);
    });
}

function _safeImageUrl(src: unknown): URL | null {
    if (typeof src !== 'string') return null;
    const value = src.trim();
    if (!value) return null;

    let url: URL;
    try {
        url = new URL(value, window.location.origin);
    } catch {
        return null;
    }

    if (
        (url.protocol === 'http:' || url.protocol === 'https:') &&
        url.origin === window.location.origin
    ) {
        return url;
    }
    if (url.protocol === 'blob:' && url.origin === window.location.origin) return url;
    if (url.protocol === 'data:' && SAFE_DATA_IMAGE_SRC_RE.test(value)) return url;
    return null;
}

function _safeImageSrc(src: unknown): string {
    return _safeImageUrl(src)?.href || '';
}

function _getLightboxImageSrc(el: Element): string {
    const s = el.getAttribute('data-full-src') || el.getAttribute('data-src') || '';
    return _safeImageSrc(s);
}

function _collectGalleryImages(): string[] {
    const items = document.querySelectorAll('[data-action="open-lightbox"]');
    const srcs: string[] = [];
    items.forEach(function(el) {
        const s = _getLightboxImageSrc(el);
        if (s) srcs.push(s);
    });
    return srcs;
}

function _currentGalleryIndex(currentSrc: string): number {
    const images = _collectGalleryImages();
    for (let i = 0; i < images.length; i++) {
        if (images[i] === currentSrc) return i;
    }
    return -1;
}

function _updateNavButtons(box: LightboxElement): void {
    if (!box || !box.classList.contains(lightboxStyles.open)) return;
    const currentSrc = box.getAttribute('data-current-src') || '';
    const images = _collectGalleryImages();
    const idx = _currentGalleryIndex(currentSrc);
    const prev = box.__lightboxPrev;
    const next = box.__lightboxNext;
    if (images.length <= 1) {
        if (prev) prev.style.display = 'none';
        if (next) next.style.display = 'none';
    } else {
        if (prev) {
            prev.style.display = idx <= 0 ? 'none' : '';
        }
        if (next) {
            next.style.display = idx >= images.length - 1 ? 'none' : '';
        }
    }
}

function _showLightboxImage(box: LightboxElement, src: string): void {
    const safeUrl = _safeImageUrl(src);
    if (!safeUrl) return;
    const img = box.__lightboxImg;
    if (!img) return;
    img.src = safeUrl.href;
    box.setAttribute('data-current-src', safeUrl.href);
    _updateNavButtons(box);
}

function _navigateLightbox(direction: number): void {
    const box = document.getElementById('image-lightbox') as LightboxElement | null;
    if (!box || !box.classList.contains(lightboxStyles.open)) return;
    const currentSrc = box.getAttribute('data-current-src') || '';
    const images = _collectGalleryImages();
    if (images.length <= 1) return;
    const idx = _currentGalleryIndex(currentSrc);
    if (idx < 0) return;
    let newIdx = idx + direction;
    if (newIdx < 0) newIdx = images.length - 1;
    if (newIdx >= images.length) newIdx = 0;
    _showLightboxImage(box, images[newIdx]);
}

let _lightboxReturnFocus: HTMLElement | null = null;

function _lightboxFocusables(box: HTMLElement): HTMLElement[] {
    return Array.from(box.querySelectorAll<HTMLElement>('button')).filter(
        (el) => el.offsetParent !== null,
    );
}

function ensureLightbox(): LightboxElement {
    let box = document.getElementById('image-lightbox') as LightboxElement | null;
    if (box) return box;

    box = document.createElement('div') as LightboxElement;
    box.id = 'image-lightbox';
    box.className = lightboxStyles.imageLightbox;
    box.setAttribute('aria-hidden', 'true');
    box.setAttribute('role', 'dialog');
    box.setAttribute('aria-modal', 'true');
    box.setAttribute('aria-label', 'Image viewer');
    box.tabIndex = -1;

    const prev = document.createElement('button');
    prev.className = lightboxStyles.imageLightboxPrev;
    prev.type = 'button';
    prev.setAttribute('aria-label', 'Previous');
    prev.textContent = '‹';
    box.appendChild(prev);

    const next = document.createElement('button');
    next.className = lightboxStyles.imageLightboxNext;
    next.type = 'button';
    next.setAttribute('aria-label', 'Next');
    next.textContent = '›';
    box.appendChild(next);

    const img = document.createElement('img');
    img.className = lightboxStyles.imageLightboxImg;
    img.alt = 'Source image';
    box.appendChild(img);

    box.__lightboxPrev = prev;
    box.__lightboxNext = next;
    box.__lightboxImg = img;

    document.body.appendChild(box);
    box.addEventListener('click', function(e) {
        if (e.target === box) {
            closeLightbox();
            return;
        }
        const target = eventElement(e);
        if (target?.closest('.' + lightboxStyles.imageLightboxPrev)) {
            _navigateLightbox(-1);
            return;
        }
        if (target?.closest('.' + lightboxStyles.imageLightboxNext)) {
            _navigateLightbox(1);
            return;
        }
    });
    return box;
}

export function openLightbox(src: unknown): void {
    const safeUrl = _safeImageUrl(src);
    if (!safeUrl) return;
    _lightboxReturnFocus =
        document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const box = ensureLightbox();
    box.setAttribute('data-current-src', safeUrl.href);
    const img = box.__lightboxImg;
    if (!img) return;
    img.src = safeUrl.href;
    box.classList.add(lightboxStyles.open);
    box.setAttribute('aria-hidden', 'false');
    _updateNavButtons(box);
    box.focus();
}

export function closeLightbox(): void {
    const box = document.getElementById('image-lightbox') as LightboxElement | null;
    if (!box) return;
    box.classList.remove(lightboxStyles.open);
    box.setAttribute('aria-hidden', 'true');
    box.removeAttribute('data-current-src');
    const img = box.__lightboxImg;
    if (img) img.removeAttribute('src');
    if (_lightboxReturnFocus) {
        _lightboxReturnFocus.focus();
        _lightboxReturnFocus = null;
    }
}

export function setupImageInputs(): void {
    const plusBtn = document.getElementById('composer-plus');
    const imageInput = document.getElementById('image-input') as HTMLInputElement | null;
    if (plusBtn && imageInput) {
        plusBtn.addEventListener('click', function() { imageInput.click(); });
        imageInput.addEventListener('change', function() {
            Array.from(imageInput.files || []).forEach(function(f) { addImage(f); });
            imageInput.value = '';
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
            function(imageFile) { addImage(imageFile); }
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
                if (file) addImage(file);
            }
        }
    });
    document.addEventListener('click', function(e) {
        const item = eventElement(e)?.closest('[data-action="open-lightbox"]');
        if (!item) return;
        const src = _getLightboxImageSrc(item);
        if (!src) return;
        e.preventDefault();
        openLightbox(src);
    });
    document.addEventListener('keydown', function(e) {
        const box = document.getElementById('image-lightbox');
        if (!box || !box.classList.contains(lightboxStyles.open)) return;
        if (e.key === 'Escape') {
            if (document.querySelector('dialog[open]')) return;
            e.preventDefault();
            e.stopImmediatePropagation();
            closeLightbox();
            return;
        }
        if (e.key === 'ArrowLeft') { e.preventDefault(); _navigateLightbox(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); _navigateLightbox(1); }
        if (e.key === 'Tab') {
            const focusables = _lightboxFocusables(box);
            if (focusables.length === 0) {
                e.preventDefault();
                box.focus();
                return;
            }
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            const active = document.activeElement;
            if (e.shiftKey && active === first) {
                e.preventDefault();
                last.focus();
            } else if (!e.shiftKey && active === last) {
                e.preventDefault();
                first.focus();
            } else if (!box.contains(active)) {
                e.preventDefault();
                first.focus();
            }
        }
    }, true);
}
