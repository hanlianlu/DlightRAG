// @ts-nocheck — full types deferred per spec
// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {detectDropItems, uploadFolderToWorkspace} from './folder-upload.ts';
import chatStyles from '../styles/chat.module.css';
import lightboxStyles from '../styles/lightbox.module.css';

const MAX_IMAGES = 3;
const MAX_IMAGE_SIZE = 10 * 1024 * 1024;
const pendingImages = [];

export function addImage(file) {
    if (pendingImages.length >= MAX_IMAGES) return;
    if (!file.type.startsWith('image/')) return;
    if (file.size > MAX_IMAGE_SIZE) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const dataUrl = e.target.result;
        const objectUrl = URL.createObjectURL(file);
        pendingImages.push({file, dataUrl, objectUrl});
        renderThumbnails();
    };
    reader.readAsDataURL(file);
}

export function getPendingImageData() {
    return pendingImages.map(function(img) { return img.dataUrl; });
}

export function renderMessageImages(container) {
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

export function clearImages() {
    pendingImages.forEach(function(img) { URL.revokeObjectURL(img.objectUrl); });
    pendingImages.length = 0;
    renderThumbnails();
}

function removeImage(index) {
    const removed = pendingImages.splice(index, 1);
    if (removed.length > 0) URL.revokeObjectURL(removed[0].objectUrl);
    renderThumbnails();
}

function renderThumbnails() {
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
        btn.setAttribute('data-idx', idx);
        btn.addEventListener('click', function() { removeImage(idx); });
        item.appendChild(btn);

        strip.appendChild(item);
    });
}

function _isSafeImageSrc(src) {
    return typeof src === 'string' && /^(?!\/\/)(?:\/|blob:|data:)/.test(src);
}

function _getLightboxImageSrc(el) {
    const s =el.getAttribute('data-full-src') || el.getAttribute('data-src') || '';
    return _isSafeImageSrc(s) ? s : '';
}

function _collectGalleryImages() {
    const items = document.querySelectorAll('[data-action="open-lightbox"]');
    const srcs = [];
    items.forEach(function(el) {
        const s =_getLightboxImageSrc(el);
        if (s) srcs.push(s);
    });
    return srcs;
}

function _currentGalleryIndex(currentSrc) {
    const images = _collectGalleryImages();
    for (let i = 0; i < images.length; i++) {
        if (images[i] === currentSrc) return i;
    }
    return -1;
}

function _updateNavButtons(box) {
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

function _showLightboxImage(box, src) {
    if (typeof src !== 'string') return;
    if (!_isSafeImageSrc(src)) return;
    const img = box.__lightboxImg;
    if (!img) return;
    img.src = src;
    box.setAttribute('data-current-src', src);
    _updateNavButtons(box);
}

function _navigateLightbox(direction) {
    const box = document.getElementById('image-lightbox');
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

function ensureLightbox() {
    let box = document.getElementById('image-lightbox');
    if (box) return box;

    box = document.createElement('div');
    box.id = 'image-lightbox';
    box.className = lightboxStyles.imageLightbox;
    box.setAttribute('aria-hidden', 'true');

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

    // Store element references for later use
    box.__lightboxPrev = prev;
    box.__lightboxNext = next;
    box.__lightboxImg = img;

    document.body.appendChild(box);
    box.addEventListener('click', function(e) {
        if (e.target === box) {
            closeLightbox();
            return;
        }
        if (e.target.closest('.' + lightboxStyles.imageLightboxPrev)) {
            _navigateLightbox(-1);
            return;
        }
        if (e.target.closest('.' + lightboxStyles.imageLightboxNext)) {
            _navigateLightbox(1);
            return;
        }
    });
    return box;
}

export function openLightbox(src) {
    if (typeof src !== 'string') return;
    if (!_isSafeImageSrc(src)) return;
    const box = ensureLightbox();
    box.setAttribute('data-current-src', src);
    const img = box.__lightboxImg;
    img.src = src;
    box.classList.add(lightboxStyles.open);
    box.setAttribute('aria-hidden', 'false');
    _updateNavButtons(box);
}

export function closeLightbox() {
    const box = document.getElementById('image-lightbox');
    if (!box) return;
    box.classList.remove(lightboxStyles.open);
    box.setAttribute('aria-hidden', 'true');
    box.removeAttribute('data-current-src');
    const img = box.__lightboxImg;
    if (img) img.removeAttribute('src');
}

export function setupImageInputs() {
    const plusBtn = document.getElementById('composer-plus');
    const imageInput = document.getElementById('image-input');
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

        const items = e.dataTransfer.items;
        if (!items || items.length === 0) return;

        const result = await detectDropItems(
            items,
            function(imageFile) { addImage(imageFile); }
        );

        if (result.files.length > 0) {
            await uploadFolderToWorkspace(result.files, result.folderName);
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
        const item = e.target.closest('[data-action="open-lightbox"]');
        if (!item) return;
        const src = _getLightboxImageSrc(item);
        if (!src) return;
        e.preventDefault();
        openLightbox(src);
    });
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') { closeLightbox(); return; }
        const box = document.getElementById('image-lightbox');
        if (!box || !box.classList.contains(lightboxStyles.open)) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); _navigateLightbox(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); _navigateLightbox(1); }
    });
}
