// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {detectDropItems, uploadFolderToWorkspace} from './folder-upload.js';

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
        const base64 = dataUrl.split(',')[1];
        const objectUrl = URL.createObjectURL(file);
        pendingImages.push({file, base64, dataUrl, objectUrl});
        renderThumbnails();
    };
    reader.readAsDataURL(file);
}

export function getPendingImageData() {
    return pendingImages.map(function(img) { return img.base64; });
}

export function renderMessageImages(container) {
    if (pendingImages.length === 0) return;
    const msgImages = document.createElement('div');
    msgImages.className = 'message-images';
    pendingImages.forEach(function(img) {
        const imgEl = document.createElement('img');
        imgEl.className = 'message-img';
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
        item.className = 'thumbnail-item';

        const imgEl = document.createElement('img');
        imgEl.className = 'thumbnail-img';
        imgEl.src = img.objectUrl && img.objectUrl.startsWith('blob:') ? img.objectUrl : '';
        imgEl.alt = 'Attached image';
        item.appendChild(imgEl);

        const btn = document.createElement('button');
        btn.className = 'thumbnail-remove';
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
    if (!box || !box.classList.contains('open')) return;
    const currentSrc = box.getAttribute('data-current-src') || '';
    const images = _collectGalleryImages();
    const idx = _currentGalleryIndex(currentSrc);
    const prev = box.querySelector('.image-lightbox-prev');
    const next = box.querySelector('.image-lightbox-next');
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
    const img = box.querySelector('.image-lightbox-img');
    if (!img) return;
    img.src = src;
    box.setAttribute('data-current-src', src);
    _updateNavButtons(box);
}

function _navigateLightbox(direction) {
    const box = document.getElementById('image-lightbox');
    if (!box || !box.classList.contains('open')) return;
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
    box.className = 'image-lightbox';
    box.setAttribute('aria-hidden', 'true');
    box.innerHTML = '<button class="image-lightbox-close" type="button" aria-label="Close">×</button>' +
        '<button class="image-lightbox-prev" type="button" aria-label="Previous">‹</button>' +
        '<button class="image-lightbox-next" type="button" aria-label="Next">›</button>' +
        '<img class="image-lightbox-img" alt="Source image">';
    document.body.appendChild(box);
    box.addEventListener('click', function(e) {
        if (e.target === box || e.target.closest('.image-lightbox-close')) {
            closeLightbox();
            return;
        }
        if (e.target.closest('.image-lightbox-prev')) {
            _navigateLightbox(-1);
            return;
        }
        if (e.target.closest('.image-lightbox-next')) {
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
    const img = box.querySelector('.image-lightbox-img');
    img.src = src;
    box.classList.add('open');
    box.setAttribute('aria-hidden', 'false');
    _updateNavButtons(box);
}

export function closeLightbox() {
    const box = document.getElementById('image-lightbox');
    if (!box) return;
    box.classList.remove('open');
    box.setAttribute('aria-hidden', 'true');
    box.removeAttribute('data-current-src');
    const img = box.querySelector('.image-lightbox-img');
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
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') { closeLightbox(); return; }
        const box = document.getElementById('image-lightbox');
        if (!box || !box.classList.contains('open')) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); _navigateLightbox(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); _navigateLightbox(1); }
    });
}
