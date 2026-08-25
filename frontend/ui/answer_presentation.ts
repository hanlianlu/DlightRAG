// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {html, nothing, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {
  AnswerArtifact,
  AnswerPresentation,
  PresentationImage,
  PresentationPart,
} from '../api/conversations.ts';
import {LightElement} from '../lib/lit_host.ts';
import {setSanitizedLlmHtml} from '../lib/safe_html.ts';
import {renderMath} from '../lib/math.ts';
import {safeImageSrc} from '../lib/urls.ts';
import {renderDiagrams} from './mermaid.ts';

export interface ArtifactOpenDetail {
  artifact: AnswerArtifact;
  returnFocus: HTMLElement;
}

export interface AnswerSourceOpenDetail {
  referenceId: string;
  chunkId?: string;
  returnFocus: HTMLElement;
}

export interface AnswerImageOpenDetail {
  src: string;
  returnFocus: HTMLElement;
}

function secureExternalLinks(container: ParentNode): void {
  container.querySelectorAll<HTMLAnchorElement>('a[href]').forEach((link) => {
    if (link.hasAttribute('download')) return;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  });
}

/** Canonical Answer body, placed Artifacts, Evidence Images, and References. */
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
    const presentation = this.presentation;
    if (!presentation) return;
    this.querySelectorAll<HTMLElement>('[data-answer-part]').forEach((host) => {
      const index = Number(host.dataset.answerPart);
      const part = presentation.parts[index];
      if (!part || part.type !== 'markdown') return;
      setSanitizedLlmHtml(host, part.html);
      renderMath(host);
      renderDiagrams(host);
      secureExternalLinks(host);
    });
  }

  protected override render(): TemplateResult | typeof nothing {
    const presentation = this.presentation;
    if (!presentation) return nothing;
    return html`
      ${presentation.artifact_outcome.status === 'complete' ? nothing : html`
        <div class="artifact-publication-warning" role="alert">
          Some requested Artifacts could not be published.
        </div>
      `}
      <div class="answer-parts" @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
        ${presentation.parts.map((part, index) => this.#part(part, index))}
      </div>
      ${presentation.evidence_images.length > 0 ? html`
        <section class="answer-evidence" aria-label="Visual Evidence"
                 @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
          <h3>Visual Evidence</h3>
          <div class="answer-image-strip">
            ${repeat(
              presentation.evidence_images,
              (image) => image.id || image.url,
              (image) => this.#evidenceImage(image),
            )}
          </div>
        </section>
      ` : nothing}
      ${presentation.sources.length > 0 ? html`
        <section class="answer-references" aria-label="References"
                 @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
          <h3 class="answer-references-title">References</h3>
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
        </section>
      ` : nothing}
    `;
  }

  #part(part: PresentationPart, index: number): TemplateResult | typeof nothing {
    if (part.type === 'markdown') {
      return html`<div class="answer-rich-content" data-answer-part=${String(index)}></div>`;
    }
    if (part.type === 'evidence_image' && part.evidence_image) {
      return html`<div class="answer-inline-evidence">${this.#evidenceImage(part.evidence_image)}</div>`;
    }
    if (part.type === 'artifact' && part.artifact) return this.#artifact(part.artifact, part.inline);
    return nothing;
  }

  #artifact(artifact: AnswerArtifact, inline: boolean): TemplateResult {
    if (artifact.status === 'unavailable') {
      return html`
        <article class="answer-artifact-card answer-artifact-unavailable" role="group"
                 aria-label=${`${artifact.label}, unavailable`}>
          <strong>${artifact.label}</strong>
          <span>${artifact.filename}</span>
          <p>${artifact.issue?.description || 'This Artifact is unavailable.'}</p>
        </article>
      `;
    }
    if (inline && artifact.presentation === 'image') {
      const source = safeImageSrc(artifact.data_url || '');
      if (source) {
        return html`
          <figure class="answer-artifact-image">
            <button type="button" data-answer-image data-src=${source}
                    aria-label=${`Open image: ${artifact.label}`}>
              <img src=${source} alt=${artifact.label} loading="lazy">
            </button>
            <figcaption>${artifact.label}</figcaption>
          </figure>
        `;
      }
    }
    const primary = artifact.role === 'primary_report';
    return html`
      <article class="answer-artifact-card" role="group" aria-label=${artifact.label}>
        <div>
          <strong>${artifact.label}</strong>
          <span>${artifact.filename}</span>
        </div>
        <button class="ui-btn" type="button" data-action="open-artifact" @click=${(event: Event) => {
          this.#openArtifact(artifact, event.currentTarget as HTMLElement);
        }}>${primary ? 'View report' : 'Open Artifact'}</button>
      </article>
    `;
  }

  #evidenceImage(image: PresentationImage): TemplateResult | typeof nothing {
    const source = safeImageSrc(image.url);
    const thumbnail = safeImageSrc(image.thumbnail_url || image.url);
    if (!source || !thumbnail) return nothing;
    return html`
      <div class="answer-evidence-image">
        <button class="answer-image-item" type="button" data-answer-image
                data-src=${source} aria-label=${`Open image: ${image.label}`}>
          <img src=${thumbnail} alt=${image.label} loading="lazy">
          <span class="answer-image-label">${image.label}</span>
        </button>
        ${image.source_ref ? html`
          <button class="answer-image-source" type="button" data-ref=${image.source_ref}
                  aria-label=${`Open source ${image.source_ref}`}>
            Source ${image.source_ref}
          </button>
        ` : nothing}
      </div>
    `;
  }

  #handleIntent = (event: Event): void => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    const source = target.closest<HTMLElement>(
      '.citation-badge[data-ref], .answer-ref-item[data-ref], .answer-image-source[data-ref]',
    );
    if (source && this.contains(source)) {
      event.preventDefault();
      event.stopPropagation();
      this.dispatchEvent(new CustomEvent<AnswerSourceOpenDetail>('answer-source-open', {
        bubbles: true,
        composed: true,
        detail: {
          referenceId: source.dataset.ref || '',
          ...(source.dataset.chunk ? {chunkId: source.dataset.chunk} : {}),
          returnFocus: source,
        },
      }));
      return;
    }
    const image = target.closest<HTMLElement>('[data-answer-image][data-src]');
    if (!image || !this.contains(image)) return;
    event.preventDefault();
    event.stopPropagation();
    this.dispatchEvent(new CustomEvent<AnswerImageOpenDetail>('answer-image-open', {
      bubbles: true,
      composed: true,
      detail: {src: image.dataset.src || '', returnFocus: image},
    }));
  };

  #handleKeyIntent = (event: KeyboardEvent): void => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (!target || target instanceof HTMLButtonElement) return;
    this.#handleIntent(event);
  };

  #openArtifact(artifact: AnswerArtifact, returnFocus: HTMLElement): void {
    this.dispatchEvent(new CustomEvent<ArtifactOpenDetail>('artifact-open', {
      bubbles: true,
      composed: true,
      detail: {artifact, returnFocus},
    }));
  }
}

customElements.define('dl-answer-presentation', AnswerPresentationElement);

declare global {
  interface HTMLElementTagNameMap {
    'dl-answer-presentation': AnswerPresentationElement;
  }
}
