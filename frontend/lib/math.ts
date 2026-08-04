// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Typesetting entry point, kept separate from the MathJax loader/config so that
// editing the loader does not change the bytes of every module that only
// typesets.
export function renderMath(container: Element): void {
    if (!window.MathJax || !window.MathJax.typesetPromise) return;
    window.MathJax.typesetPromise([container]).catch(function () {
        // MathJax may fail on genuinely malformed input; ignore.
    });
}
