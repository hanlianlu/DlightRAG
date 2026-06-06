// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

const MATHJAX_SRC = 'https://cdn.jsdelivr.net/npm/mathjax@4.1.2/tex-mml-chtml.js';

let loading = false;

function configureMathJax() {
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

function appendMathJaxScript() {
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

export function setupMathRendering() {
    if ('requestIdleCallback' in window) {
        window.requestIdleCallback(appendMathJaxScript, {timeout: 2000});
    } else {
        window.setTimeout(appendMathJaxScript, 0);
    }
}
