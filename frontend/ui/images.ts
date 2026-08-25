// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import lightboxStyles from '../styles/lightbox.module.css';
import {closestElement, wrapTabFocus} from '../lib/dom.ts';
import {safeImageSrc} from '../lib/urls.ts';

type LightboxElement = HTMLDivElement & {
    __lightboxPrev?: HTMLButtonElement;
    __lightboxNext?: HTMLButtonElement;
    __lightboxImg?: HTMLImageElement;
};

function _getLightboxImageSrc(el: Element): string {
    const s = el.getAttribute('data-full-src') || el.getAttribute('data-src') || '';
    return safeImageSrc(s);
}

function _collectGalleryImages(): string[] {
    const items = document.querySelectorAll(
        '[data-action="open-lightbox"], [data-answer-image]',
    );
    const srcs = new Set<string>();
    items.forEach(function(el) {
        const source = _getLightboxImageSrc(el);
        if (source) srcs.add(source);
    });
    return [...srcs];
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
    const safeUrl = safeImageSrc(src);
    if (!safeUrl) return;
    const img = box.__lightboxImg;
    if (!img) return;
    img.src = safeUrl;
    box.setAttribute('data-current-src', safeUrl);
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
    const safeUrl = safeImageSrc(src);
    if (!safeUrl) return;
    _lightboxReturnFocus =
        document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const box = ensureLightbox();
    box.setAttribute('data-current-src', safeUrl);
    const img = box.__lightboxImg;
    if (!img) return;
    img.src = safeUrl;
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
