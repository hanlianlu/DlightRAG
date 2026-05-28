// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

let toastTimer = null;

export function showToast(msg, duration) {
    const el = document.getElementById('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('visible');
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
        el.classList.remove('visible');
    }, duration || 3000);
}
