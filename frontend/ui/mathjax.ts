// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const MATHJAX_SRC = 'https://cdn.jsdelivr.net/npm/mathjax@4.1.2/tex-mml-chtml.js';

let loading = false;

export function renderMath(container: Element): void {
    if (!window.MathJax || !window.MathJax.typesetPromise) return;
    window.MathJax.typesetPromise([container]).catch(function () {
        // MathJax may fail on genuinely malformed input; ignore.
    });
}

function configureMathJax(): void {
    if (window.MathJax) return;
    window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']],
            processEscapes: false,
            tags: 'all',
        },
        options: {
            skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        },
    };
}

function appendMathJaxScript(): void {
    if (loading || (window.MathJax && window.MathJax.typesetPromise)) return;
    loading = true;
    configureMathJax();

    const script = document.createElement('script');
    script.src = MATHJAX_SRC;
    script.async = true;
    script.onload = function() {
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise().catch(function() {
                // Malformed math should not affect the chat shell.
            });
        }
    };
    script.onerror = function() {
        loading = false;
    };
    document.head.appendChild(script);
}

export function setupMathRendering(): void {
    const idle = window.requestIdleCallback ?? ((callback: () => void) => window.setTimeout(callback, 0));
    idle(appendMathJaxScript, {timeout: 2000});
}
