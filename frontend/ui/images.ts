// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import chatStyles from '../styles/chat.module.css';
import lightboxStyles from '../styles/lightbox.module.css';
import type {ConversationAttachmentReference} from '../api/conversations.ts';
import {closestElement, wrapTabFocus} from '../lib/dom.ts';

const SAFE_DATA_IMAGE_SRC_RE = /^data:image\/(?:avif|bmp|gif|jpeg|jpg|png|webp);base64,[a-z0-9+/=]+$/i;

type LightboxElement = HTMLDivElement & {
    __lightboxPrev?: HTMLButtonElement;
    __lightboxNext?: HTMLButtonElement;
    __lightboxImg?: HTMLImageElement;
};

// Render image attachments (composer previews and stored history alike) as lazy
// async thumbnails that open their full-resolution original in the lightbox on
// demand. `thumbnail_url` falls back to `url` when a derived thumbnail is absent
// (e.g. live blob previews that carry a single object URL).
export function renderMessageAttachmentImages(
    container: Element,
    images: readonly ConversationAttachmentReference[],
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
        const thumbnailSrc = _safeImageSrc(reference.thumbnail_url || reference.url);
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
            if (fullSrc.startsWith('blob:')) imageButton.dataset.liveObjectUrl = fullSrc;
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

export function releaseMessageAttachmentObjectUrls(root: ParentNode): void {
    const urls = new Set<string>();
    root.querySelectorAll<HTMLElement>('[data-live-object-url]').forEach((element) => {
        const url = element.dataset.liveObjectUrl;
        if (url) urls.add(url);
        delete element.dataset.liveObjectUrl;
    });
    for (const url of urls) URL.revokeObjectURL(url);
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

function _updateNavButtons(box: LightboxElement): void {
    if (!box || !box.classList.contains(lightboxStyles.open)) return;
    const currentSrc = box.getAttribute('data-current-src') || '';
    const images = _collectGalleryImages();
    const idx = images.indexOf(currentSrc);
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
    const idx = images.indexOf(currentSrc);
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
        const target = e.target instanceof Element ? e.target : null;
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

export function setupImageLightbox(): void {
    document.addEventListener('click', function(e) {
        const item = closestElement(e.target, '[data-action="open-lightbox"]');
        if (!item) return;
        const src = _getLightboxImageSrc(item);
        if (!src) return;
        e.preventDefault();
        openLightbox(src);
    });
    document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        const item = closestElement(e.target, '[data-action="open-lightbox"]');
        if (!item) return;
        e.preventDefault();
        const src = _getLightboxImageSrc(item);
        if (src) openLightbox(src);
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
            if (!wrapTabFocus(focusables, e) && !box.contains(document.activeElement)) {
                e.preventDefault();
                focusables[0].focus();
            }
        }
    }, true);
}
