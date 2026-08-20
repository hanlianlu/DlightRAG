// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {AnswerPresentation} from '../api/conversations.ts';
import {LightElement} from '../lib/lit_host.ts';
import {setSanitizedLlmHtml} from '../lib/safe_html.ts';
import {renderMath} from '../lib/math.ts';
import {renderDiagrams} from './mermaid.ts';
import {safeImageSrc} from '../lib/urls.ts';

function secureExternalLinks(container: ParentNode): void {
  container.querySelectorAll<HTMLAnchorElement>('a[href]').forEach((link) => {
    if (link.hasAttribute('download')) return;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  });
}

/** Safe answer body plus Lit-owned images and reference controls. */
export class AnswerPresentationElement extends LightElement {
  static properties = {
    presentation: {attribute: false},
  };

  declare presentation: AnswerPresentation | null;

  constructor() {
    super();
    this.presentation = null;
  }

  protected override updated(): void {
    const host = this.querySelector('.answer-rich-content');
    if (!host || !this.presentation) return;
    setSanitizedLlmHtml(host, this.presentation.answer_html);
    renderMath(host);
    renderDiagrams(host);
    secureExternalLinks(host);
  }

  protected override render(): TemplateResult | typeof nothing {
    const presentation = this.presentation;
    if (!presentation) return nothing;
    return html`
      <div class="answer-rich-content"></div>
      ${presentation.answer_images.length > 0 ? html`
        <div class="answer-image-strip">
          ${repeat(
            presentation.answer_images,
            (image) => image.id || image.url,
            (image) => {
              const source = safeImageSrc(image.url);
              const thumbnail = safeImageSrc(image.thumbnail_url || image.url);
              if (!source || !thumbnail) return nothing;
              return html`
                <button class="answer-image-item" type="button" data-action="open-lightbox"
                        data-src=${source} aria-label=${`Open image: ${image.label}`}>
                  <img src=${thumbnail} alt=${image.label} loading="lazy">
                  <span class="answer-image-label">${image.label}</span>
                </button>
              `;
            },
          )}
        </div>
      ` : nothing}
      ${presentation.sources.length > 0 ? html`
        <div class="answer-references">
          <div class="answer-references-title">References</div>
          ${repeat(
            presentation.sources,
            (source) => source.id,
            (source) => html`
              <button class="answer-ref-item" type="button" data-ref=${source.id}>
                <span class="answer-ref-id">${source.id}</span>
                <span class="answer-ref-title">${source.title}</span>
              </button>
            `,
          )}
        </div>
      ` : nothing}
    `;
  }
}

customElements.define('answer-presentation', AnswerPresentationElement);

declare global {
  interface HTMLElementTagNameMap {
    'answer-presentation': AnswerPresentationElement;
  }
}
