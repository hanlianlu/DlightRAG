// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {
  AnswerArtifact,
  AnswerPresentation,
  PresentationImage,
  PresentationPart,
} from '../api/conversations.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeImageSrc} from '../lib/urls.ts';
import type {ImageOpenDetail} from './image-lightbox.ts';
import {mountRichHtml, typesetRichContent} from './rich-rendering.ts';
import chatStyles from '../styles/chat.module.css';
import answerStyles from '../styles/answer-presentation.module.css';

export interface ArtifactOpenDetail {
  artifact: AnswerArtifact;
  returnFocus: HTMLElement;
}

export interface AnswerSourceOpenDetail {
  presentation: AnswerPresentation;
  referenceId: string;
  chunkId?: string;
  returnFocus: HTMLElement;
}

/** Canonical Answer body, placed Artifacts, Evidence Images, and References. */
export class AnswerPresentationElement extends LightElement {
  static properties = {
    presentation: {attribute: false},
    referencesExpanded: {state: true},
  };

  declare presentation: AnswerPresentation | null;
  declare referencesExpanded: boolean;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.presentation = null;
    this.referencesExpanded = false;
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('presentation')) this.referencesExpanded = false;
  }

  protected override updated(): void {
    const presentation = this.presentation;
    if (!presentation) return;
    this.querySelectorAll<HTMLElement>('[data-answer-part]').forEach((host) => {
      const index = Number(host.dataset.answerPart);
      const part = presentation.parts[index];
      if (part?.type !== 'markdown') return;
      mountRichHtml(host, part.html);
      typesetRichContent(host);
    });
  }

  protected override render(): TemplateResult | typeof nothing {
    const presentation = this.presentation;
    if (!presentation) return nothing;
    return html`
      ${presentation.artifactOutcome.status === 'complete' ? nothing : html`
        <div class="artifact-publication-warning" role="alert">
          ${msg('Some requested Artifacts could not be published.', {id: 'answerPresentation.publicationWarning'})}
        </div>
      `}
      <div class="answer-parts" @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
        ${presentation.parts.map((part, index) => this.#part(part, index))}
      </div>
      ${presentation.evidenceImages.length > 0 ? html`
        <section class="answer-evidence" aria-label=${msg('Visual Evidence', {id: 'answerPresentation.visualEvidenceAria'})}
                 @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
          <h3>${msg('Visual Evidence', {id: 'answerPresentation.visualEvidence'})}</h3>
          <div class=${answerStyles['answer-image-strip']}>
            ${repeat(
              presentation.evidenceImages,
              (image) => image.id || image.url,
              (image) => this.#evidenceImage(image),
            )}
          </div>
        </section>
      ` : nothing}
      ${presentation.sources.length > 0 ? html`
        <section class=${answerStyles['answer-references']} aria-label=${msg('References', {id: 'answerPresentation.referencesAria'})}
                 @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
          <h3 class=${answerStyles['answer-references-title']}>${msg('References', {id: 'answerPresentation.references'})}</h3>
          <div class=${`${answerStyles['answer-reference-list']}${this.referencesExpanded ? ` ${answerStyles.expanded}` : ''}`}>
          ${repeat(
            presentation.sources,
            (source) => source.id,
            (source) => html`
              <button class=${answerStyles['answer-ref-item']} type="button" data-ref=${source.id}>
                <span class=${answerStyles['answer-ref-id']}>${source.id}</span>
                <span class=${answerStyles['answer-ref-title']}>${source.title}</span>
              </button>
            `,
          )}
          </div>
          ${presentation.sources.length > 3 ? html`
            <button class=${answerStyles['answer-references-show-all']} type="button"
                    aria-expanded=${String(this.referencesExpanded)}
                    @click=${this.#toggleReferences}>
              ${this.referencesExpanded
                ? msg('Show fewer', {id: 'answerPresentation.showFewerReferences'})
                : msg(str`Show all ${presentation.sources.length}`, {id: 'answerPresentation.showAllReferences'})}
            </button>
          ` : nothing}
        </section>
      ` : nothing}
    `;
  }

  #part(part: PresentationPart, index: number): TemplateResult | typeof nothing {
    if (part.type === 'markdown') {
      return html`<div class="answer-rich-content ${chatStyles.aiMessageContent}" data-answer-part=${String(index)}></div>`;
    }
    if (part.type === 'evidence_image' && part.evidenceImage) {
      return html`<div class="answer-inline-evidence">${this.#evidenceImage(part.evidenceImage)}</div>`;
    }
    if (part.type === 'artifact' && part.artifact) return this.#artifact(part.artifact, part.inline);
    return nothing;
  }

  #artifact(artifact: AnswerArtifact, inline: boolean): TemplateResult {
    if (artifact.status === 'unavailable') {
      return html`
        <article class="answer-artifact-card answer-artifact-unavailable" role="group"
                 aria-label=${msg(str`${artifact.label}, unavailable`, {id: 'answerPresentation.artifactUnavailableAria'})}>
          <strong>${artifact.label}</strong>
          <span>${artifact.filename}</span>
          <p>${artifact.issue?.description || msg('This Artifact is unavailable.', {id: 'answerPresentation.artifactUnavailable'})}</p>
        </article>
      `;
    }
    if (inline && artifact.presentation === 'image') {
      const source = safeImageSrc(artifact.dataUrl || '');
      if (source) {
        return html`
          <figure class="answer-artifact-image">
            <button type="button" data-answer-image data-src=${source}
                    aria-label=${msg(str`Open image: ${artifact.label}`, {id: 'answerPresentation.openImage'})}>
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
        <button class="dl-btn" type="button" @click=${(event: Event) => {
          this.#openArtifact(artifact, event.currentTarget as HTMLElement);
        }}>${primary ? msg('View report', {id: 'answerPresentation.viewReport'}) : msg('Open Artifact', {id: 'answerPresentation.openArtifact'})}</button>
      </article>
    `;
  }

  #evidenceImage(image: PresentationImage): TemplateResult | typeof nothing {
    const source = safeImageSrc(image.url);
    const thumbnail = safeImageSrc(image.thumbnailUrl || image.url);
    if (!source || !thumbnail) return nothing;
    return html`
      <div class="answer-evidence-image">
        <button class=${answerStyles['answer-image-item']} type="button" data-answer-image
                data-src=${source} aria-label=${msg(str`Open image: ${image.label}`, {id: 'answerPresentation.openImage'})}>
          <img src=${thumbnail} alt=${image.label} loading="lazy">
          <span class=${answerStyles['answer-image-label']}>${image.label}</span>
        </button>
        ${image.sourceRef ? html`
          <button class="answer-image-source" type="button" data-ref=${image.sourceRef}
                  aria-label=${msg(str`Open source ${image.sourceRef}`, {id: 'answerPresentation.openSourceAria'})}>
            ${msg(str`Source ${image.sourceRef}`, {id: 'answerPresentation.openSource'})}
          </button>
        ` : nothing}
      </div>
    `;
  }

  #toggleReferences = (): void => {
    this.referencesExpanded = !this.referencesExpanded;
  };

  #handleIntent = (event: Event): void => {
    const target = event.target instanceof Element ? event.target : null;
    if (!target) return;
    const source = target.closest<HTMLElement>(
      '.citation-badge[data-ref], button[data-ref]',
    );
    const presentation = this.presentation;
    if (source && this.contains(source) && presentation) {
      event.preventDefault();
      event.stopPropagation();
      this.dispatchEvent(new CustomEvent<AnswerSourceOpenDetail>('dl-answer-source-open', {
        bubbles: true,
        composed: true,
        detail: {
          presentation,
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
    this.dispatchEvent(new CustomEvent<ImageOpenDetail>('dl-image-open', {
      bubbles: true,
      composed: true,
      detail: {
        src: image.dataset.src || '',
        gallery: this.#galleryImages(),
        returnFocus: image,
      },
    }));
  };

  #galleryImages(): string[] {
    const presentation = this.presentation;
    if (!presentation) return [];
    const candidates = [
      ...presentation.parts.flatMap((part) => {
        if (part.type === 'evidence_image' && part.evidenceImage) {
          return [part.evidenceImage.url];
        }
        if (part.type === 'artifact' && part.artifact?.presentation === 'image') {
          return [part.artifact.dataUrl || ''];
        }
        return [];
      }),
      ...presentation.evidenceImages.map((image) => image.url),
    ];
    return [...new Set(candidates.map(safeImageSrc).filter(Boolean))];
  }

  #handleKeyIntent = (event: KeyboardEvent): void => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (!target || target instanceof HTMLButtonElement) return;
    this.#handleIntent(event);
  };

  #openArtifact(artifact: AnswerArtifact, returnFocus: HTMLElement): void {
    this.dispatchEvent(new CustomEvent<ArtifactOpenDetail>('dl-artifact-open', {
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

  interface HTMLElementEventMap {
    'dl-answer-source-open': CustomEvent<AnswerSourceOpenDetail>;
    'dl-artifact-open': CustomEvent<ArtifactOpenDetail>;
  }
}
