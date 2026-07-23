// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

// Self-hosted MathJax v4 (SVG output), served same-origin so math rendering has
// no CDN dependency (enterprise/offline reliability + no metadata leak). Files
// are vendored under /static/vendor/mathjax from the npm `mathjax` and
// `@mathjax/mathjax-newcm-font` packages (v4.1.3). The loader/svg paths below
// keep dynamic font ranges and the SRE a11y worker on our origin — verified to
// make zero third-party requests. Re-vendor on upgrade: `npm i -D mathjax@<ver>`,
// copy `tex-mml-svg.js`, `@mathjax/mathjax-newcm-font/svg/dynamic`, and `sre/`
// into that directory, then `npm rm mathjax`.
const MATHJAX_ROOT = '/static/vendor/mathjax';
const MATHJAX_SRC = MATHJAX_ROOT + '/tex-mml-svg.js';

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
        loader: {paths: {fonts: MATHJAX_ROOT + '/@mathjax'}},
        svg: {dynamicPrefix: MATHJAX_ROOT + '/@mathjax/mathjax-newcm-font/svg/dynamic'},
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
