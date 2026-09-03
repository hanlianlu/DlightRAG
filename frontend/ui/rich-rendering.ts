// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

/** The one rich-content rendering pipeline for server-sanitized HTML.

 *  Every surface that shows model or document content runs the same stages in
 *  the same order: sanitize (defense in depth on top of the server pass),
 *  harden links, typeset math, upgrade Mermaid fences. The stages split across
 *  two narrow entries because typesetting needs laid-out content: mounting is
 *  safe while a host is hidden, typesetting must wait until it is visible.
 */

import {setSanitizedLlmHtml} from '../lib/safe-html.ts';
import {renderMath} from '../lib/math.ts';
import {renderDiagrams} from './mermaid.ts';

/** Harden the anchors the sanitizer keeps: external links open in a new tab
 *  and never inherit this page's origin. Download anchors keep their contract
 *  so save-to-file behavior is untouched. */
export function secureExternalLinks(container: ParentNode): void {
  container.querySelectorAll<HTMLAnchorElement>('a[href]').forEach((link) => {
    if (link.hasAttribute('download')) return;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  });
}

/** Replace a host's content with sanitized rich HTML and harden its links.

 *  Layout-independent: safe to run while the host is hidden. Run once per
 *  content change; it resets the DOM, so it is not a typesetting refresh. */
export function mountRichHtml(host: HTMLElement, html: string): void {
  setSanitizedLlmHtml(host, html);
  secureExternalLinks(host);
}

/** Typeset math and upgrade Mermaid fences inside laid-out content.

 *  Idempotent: MathJax skips already-processed nodes and each Mermaid fence is
 *  replaced by a diagram, so visibility toggles can re-run it freely. Requires
 *  a visible container — hidden content has no layout for typesetting. */
export function typesetRichContent(container: Element): void {
  renderMath(container);
  renderDiagrams(container);
}
