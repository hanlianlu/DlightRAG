// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, render, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {
  AnswerArtifact,
  AnswerPresentation,
  PresentationImage,
  PresentationPart,
} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeExternalHttpHref, safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import answerStyles from '../styles/answer-presentation.module.css';
import chatStyles from '../styles/chat.module.css';
import type {ImageOpenDetail} from './image-lightbox.ts';
import {artifactDownloadLink} from './artifact-download.ts';
import {mountRichHtml, typesetRichContent} from './rich-rendering.ts';

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

interface MountedRichPart {
  html: string;
  links: {anchor: HTMLAnchorElement; slot: HTMLSpanElement | null}[];
}

/** Canonical Answer body, placed Artifacts, Evidence Images, and References. */
export class AnswerPresentationElement extends LightElement {
  static properties = {
    presentation: {attribute: false},
    referencesExpanded: {state: true},
  };

  declare presentation: AnswerPresentation | null;
  declare referencesExpanded: boolean;

  #mountedRichParts = new WeakMap<HTMLElement, MountedRichPart>();

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
      let mounted = this.#mountedRichParts.get(host);
      if (!mounted || mounted.html !== part.html) {
        mountRichHtml(host, part.html);
        mounted = {
          html: part.html,
          links: [...host.querySelectorAll<HTMLAnchorElement>('a[href]')]
            .map((anchor) => ({anchor, slot: null})),
        };
        this.#mountedRichParts.set(host, mounted);
      }
      this.#upgradeLinkCards(mounted, presentation);
      for (const [partIndex, placed] of presentation.parts.entries()) {
        if (placed.slot == null) continue;
        const slot = host.querySelector<HTMLElement>(`.answer-resource-slot-${placed.slot}`);
        if (slot) {
          slot.dataset.answerTyped = '';
          render(this.#part(placed, partIndex), slot);
        }
      }
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
        ${presentation.parts.map((part, index) => part.slot == null ? this.#part(part, index) : nothing)}
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
              <button class=${answerStyles['answer-ref-item']} type="button" data-ref=${source.id} data-answer-ref>
                <span class=${answerStyles['answer-ref-id']}>${source.id}</span>
                <span class=${answerStyles['answer-ref-title']}>${source.title}</span>
              </button>
            `,
          )}
          </div>
          ${presentation.sources.length > 3 ? html`
            <button class=${answerStyles['answer-references-toggle']} type="button"
                    aria-expanded=${String(this.referencesExpanded)}
                    @click=${this.#toggleReferences}>
              <span class=${answerStyles['answer-references-toggle-icon']}>${icon('disclosure', {size: 'xs'})}</span>
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
    if (part.type === 'link_card' && part.card) return this.#linkCard(part.card);
    return nothing;
  }

  #upgradeLinkCards(mounted: MountedRichPart, presentation: AnswerPresentation): void {
    // Only actual anchors survived Markdown parsing and both sanitizers. Match
    // metadata to each such occurrence, never to source text or a URL elsewhere
    // in the answer. Code, titles, math and image alt text cannot authorize one.
    const sources = new Set(presentation.sources.map((source) => safeExternalHttpHref(source.sourceUrl || '')));
    const cards = new Map((presentation.linkCards ?? []).map((card) => [safeExternalHttpHref(card.url), card]));
    for (const occurrence of mounted.links) {
      const {anchor} = occurrence;
      const href = safeExternalHttpHref(anchor.getAttribute('href') || '');
      const card = href && !sources.has(href) && !anchor.closest('code, pre, .citation-badge')
        ? cards.get(href) : undefined;
      if (!card) {
        if (occurrence.slot) {
          render(nothing, occurrence.slot);
          occurrence.slot.replaceWith(anchor);
          occurrence.slot = null;
        }
        continue;
      }
      // Keep the original anchor and the Lit root for this occurrence. Metadata
      // or References updates must not remount the surrounding typed resources.
      if (!occurrence.slot) {
        occurrence.slot = document.createElement('span');
        occurrence.slot.dataset.answerTyped = '';
        anchor.replaceWith(occurrence.slot);
      }
      render(this.#linkCard(card), occurrence.slot);
    }
  }

  #linkCard(card: import('../api/conversations.ts').LinkCard): TemplateResult {
    // The card is a link out. The page's own cover image is the only subresource
    // it adds, and the platform is never embedded (ADR 0026).
    const cover = safeExternalHttpHref(card.image || '');
    return html`
      <a class=${answerStyles['answer-link-card']} data-answer-link-card
         href=${safeExternalHttpHref(card.url) || '#'}
         target="_blank" rel="noopener noreferrer"
         aria-label=${msg(str`Open ${card.title}`, {id: 'answerPresentation.openLinkCard'})}>
        ${cover ? html`<img class=${answerStyles['answer-link-card-cover']} src=${cover} alt="" loading="lazy">` : nothing}
        <span class=${answerStyles['answer-link-card-body']}>
          <strong class=${answerStyles['answer-link-card-title']}>${card.title}</strong>
          ${card.description ? html`<span class=${answerStyles['answer-link-card-description']}>${card.description}</span>` : nothing}
          <span class=${answerStyles['answer-link-card-site']}>${card.site}</span>
        </span>
      </a>
    `;
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
    if (inline && artifact.presentation === 'video') {
      // The browser owns decoding, so the Artifact URL is the player's source:
      // a range-capable same-origin request streams and seeks without inlining
      // the bytes. The caption keeps an escape hatch for a container this
      // browser cannot decode, which leaves the element itself empty.
      const source = safeSameOriginHref(artifact.dataUrl || '');
      if (source) {
        return html`
          <figure class=${answerStyles['answer-artifact-video']}>
            <video data-answer-video controls preload="metadata" playsinline src=${source}></video>
            <figcaption>
              <span>${artifact.label}</span>
              <a href=${source} target="_blank" rel="noopener noreferrer">${msg('Open in a new tab', {id: 'answerPresentation.openVideo'})}</a>
              ${artifactDownloadLink(
                artifact.downloadUrl,
                msg('Download', {id: 'answerPresentation.downloadVideo'}),
              )}
            </figcaption>
          </figure>
        `;
      }
    }
    return html`
      <article class="answer-artifact-card" role="group" aria-label=${artifact.label}>
        <div>
          <strong>${artifact.label}</strong>
          <span>${artifact.filename}</span>
        </div>
        <button class="dl-btn" type="button" @click=${(event: Event) => {
          this.#openArtifact(artifact, event.currentTarget as HTMLElement);
        }}>${msg('Open Artifact', {id: 'answerPresentation.openArtifact'})}</button>
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
