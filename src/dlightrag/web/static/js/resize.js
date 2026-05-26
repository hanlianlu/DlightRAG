// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

var STORAGE_KEY = 'dlightrag-panel-width';
var MIN_WIDTH = 320;

function getMaxWidth() {
    return Math.floor(window.innerWidth * 0.5);
}

function clampWidth(w) {
    return Math.max(MIN_WIDTH, Math.min(w, getMaxWidth()));
}

function loadWidth() {
    try {
        var saved = localStorage.getItem(STORAGE_KEY);
        if (saved !== null) {
            var n = parseInt(saved, 10);
            if (!isNaN(n) && n >= MIN_WIDTH) return clampWidth(n);
        }
    } catch (_) { /* localStorage unavailable */ }
    return 420;
}

function saveWidth(w) {
    try {
        localStorage.setItem(STORAGE_KEY, String(w));
    } catch (_) { /* localStorage unavailable */ }
}

export function setupPanelResize() {
    var panel = document.getElementById('panel');
    if (!panel) return;

    // Set initial width from storage
    var initial = loadWidth();
    document.documentElement.style.setProperty('--panel-width', initial + 'px');

    // Create handle if not in DOM
    var handle = panel.querySelector('.panel-resize-handle');
    if (!handle) {
        handle = document.createElement('div');
        handle.className = 'panel-resize-handle';
        panel.insertBefore(handle, panel.firstChild);
    }

    var dragging = false;
    var startX = 0;
    var startWidth = 0;

    function onPointerDown(e) {
        dragging = true;
        startX = e.clientX;
        startWidth = panel.getBoundingClientRect().width;
        handle.classList.add('active');
        document.body.style.userSelect = 'none';
        document.body.style.cursor = 'col-resize';
        document.body.setAttribute('data-resizing', '');
        document.body.classList.add('resizing');
    }

    function onPointerMove(e) {
        if (!dragging) return;
        var deltaX = startX - e.clientX;
        var newWidth = clampWidth(startWidth + deltaX);
        document.documentElement.style.setProperty('--panel-width', newWidth + 'px');
    }

    function onPointerUp() {
        if (!dragging) return;
        dragging = false;
        handle.classList.remove('active');
        document.body.style.userSelect = '';
        document.body.style.cursor = '';
        document.body.classList.remove('resizing');
        var finalWidth = panel.getBoundingClientRect().width;
        saveWidth(Math.round(finalWidth));
        // Defer removing data-resizing so the post-drag click event
        // doesn't trigger panel-close in the click-outside handler.
        setTimeout(function () {
            document.body.removeAttribute('data-resizing');
        }, 0);
    }

    handle.addEventListener('pointerdown', onPointerDown);
    document.addEventListener('pointermove', onPointerMove);
    document.addEventListener('pointerup', onPointerUp);

    // Re-clamp on window resize
    window.addEventListener('resize', function () {
        var current = panel.getBoundingClientRect().width;
        var clamped = clampWidth(current);
        document.documentElement.style.setProperty('--panel-width', clamped + 'px');
    });
}
