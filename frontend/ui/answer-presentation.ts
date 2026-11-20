// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type PropertyValues, render, type TemplateResult} from 'lit';
import {repeat} from 'lit/directives/repeat.js';
import type {
  AnswerArtifact,
  AnswerPresentation,
  ArtifactAttestation,
  PresentationImage,
  PresentationPart,
  PresentationSource,
} from '../api/conversations.ts';
import {raise} from '../lib/dom.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeExternalHttpHref, safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import answerStyles from '../styles/answer-presentation.module.css';
import chatStyles from '../styles/chat.module.css';
import {artifactDeliveryRecord} from './artifact-attestation.ts';
import {artifactDownloadLink} from './artifact-download.ts';
import {holdsDiagramSource} from './mermaid.ts';
import {mountRichHtml, typesetRichContent} from './rich-rendering.ts';
import './video-playback.ts';

/** How long a reference the trace points at stays lit.
 *
 * Long enough for an eye to land on the row it was sent to, short enough that the page is not
 * left carrying a highlight the reader did not ask for. */
const REFERENCE_HIGHLIGHT_MS = 2000;

interface ArtifactOpenDetail {
  artifact: AnswerArtifact;
  returnFocus: HTMLElement;
}

export interface AnswerSourceOpenDetail {
  presentation: AnswerPresentation;
  referenceId: string;
  chunkId?: string;
  returnFocus: HTMLElement;
}

/** One source a trace asked an Answer to show, named by the id the presentation publishes.
 *
 *  ``runId`` names the Answer that published it: two Answers on one page can number their
 *  sources the same way, and only the one the trace was opened from should answer. */
export interface ShowReferenceDetail {
  sourceId: string;
  runId?: string | null;
}

interface MountedRichPart {
  html: string;
  links: {anchor: HTMLAnchorElement; slot: HTMLSpanElement | null}[];
}

/** The name of the bytes behind one Answer image, for the places a reader gets one name.
 *
 * An Evidence Image's `label` is the source's own name -- a document title, or "Page 3" -- and the
 * card already shows it beside the thumbnail. Saying it a second time as the `alt` told a screen
 * reader one thing twice and dumped a title into the layout whenever the thumbnail failed; the
 * file the picture actually is answers both, and it is the one name a broken image can carry
 * honestly. */
function imageName(url: string): string {
  const path = url.split(/[?#]/, 1)[0];
  const name = path.slice(path.lastIndexOf('/') + 1);
  return name || path || url;
}

/** Canonical Answer body, placed Artifacts, Evidence Images, and References. */
export class AnswerPresentationElement extends LightElement {
  static properties = {
    presentation: {attribute: false},
    runId: {attribute: false},
    referencesExpanded: {state: true},
    recordExpanded: {state: true},
    highlightedReferenceId: {state: true},
  };

  declare presentation: AnswerPresentation | null;
  /** The Run this Answer answers, when the page knows it: a trace request names one Run's source. */
  declare runId: string | null;
  declare referencesExpanded: boolean;
  declare recordExpanded: boolean;
  declare highlightedReferenceId: string | null;

  #mountedRichParts = new WeakMap<HTMLElement, MountedRichPart>();
  #highlightTimer: ReturnType<typeof setTimeout> | undefined;
  /** The one id the delivery record's disclosure names; every Answer on a page needs its own. */
  readonly #recordEvidenceId = `answer-artifact-record-${crypto.randomUUID()}`;

  constructor() {
    super();
    this.presentation = null;
    this.runId = null;
    this.referencesExpanded = false;
    this.recordExpanded = false;
    this.highlightedReferenceId = null;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    // Every Answer on the page hears one trace request and answers only for the sources it holds.
    document.addEventListener('dl-show-reference', this.#handleShowReference, {
      signal: this.lifetime,
    });
  }

  override disconnectedCallback(): void {
    this.#clearHighlight();
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (!changed.has('presentation')) return;
    this.referencesExpanded = false;
    // A new Answer starts with its record folded, whatever the one before it had opened.
    this.recordExpanded = false;
    // A lit row belongs to the Answer the trace was pointing at, not to the one that replaces it.
    this.#clearHighlight();
  }

  protected override updated(changed: PropertyValues<this>): void {
    const presentation = this.presentation;
    if (!presentation) return;
    // What the words say is typeset once for each presentation. A redraw for anything else (the language,
    // the References opening) draws the typed resources again but leaves the typeset content alone, unless
    // a diagram in it is still its source: the load that would draw it failed, and the redraw is another try.
    const typeset = changed.has('presentation');
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
      if (typeset || holdsDiagramSource(host)) typesetRichContent(host);
    });
  }

  protected override render(): TemplateResult | typeof nothing {
    const presentation = this.presentation;
    if (!presentation) return nothing;
    return html`
      <p class="dl-sr-only" role="status" data-reference-announcer>${this.highlightedReferenceId === null
        ? ''
        : msg(str`Reference ${this.highlightedReferenceId} shown`, {id: 'answerPresentation.referenceShown'})}</p>
      ${presentation.artifactOutcome.status === 'complete' ? nothing : html`
        <div class="artifact-publication-warning" role="alert">
          ${msg('Some requested files could not be published.', {id: 'answerPresentation.publicationWarning'})}
        </div>
      `}
      <div class="answer-parts" @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
        ${presentation.parts.map((part, index) => part.slot == null ? this.#part(part, index) : nothing)}
      </div>
      ${this.#deliveryRecord()}
      ${presentation.evidenceImages.length > 0 ? html`
        <section class="answer-evidence" aria-label=${msg('Visual Evidence', {id: 'answerPresentation.visualEvidenceAria'})}
                 @click=${this.#handleIntent} @keydown=${this.#handleKeyIntent}>
          <h3 class=${answerStyles['answer-section-title']}>${msg('Visual Evidence', {id: 'answerPresentation.visualEvidence'})}</h3>
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
          ${presentation.sources.length > 2 ? html`
            <h3 class=${answerStyles['answer-section-title']}>
              <button class=${answerStyles['answer-references-toggle']} type="button"
                      aria-expanded=${String(this.referencesExpanded)}
                      @click=${this.#toggleReferences}>
                ${msg('References', {id: 'answerPresentation.references'})}
                <span class=${answerStyles['answer-references-count']}>
                  ${this.referencesExpanded
                    ? msg('Show fewer', {id: 'answerPresentation.showFewerReferences'})
                    : msg(str`Show all ${presentation.sources.length}`, {id: 'answerPresentation.showAllReferences'})}
                </span>
              </button>
            </h3>
          ` : html`
            <h3 class=${answerStyles['answer-section-title']}>${msg('References', {id: 'answerPresentation.references'})}</h3>
          `}
          <div class=${`${answerStyles['answer-reference-list']}${this.referencesExpanded ? ` ${answerStyles.expanded}` : ''}`}>
          ${repeat(
            presentation.sources,
            (source) => source.id,
            (source) => html`
              <button class=${this.#referenceItemClass(source)} type="button" data-ref=${source.id}
                      data-reference-id=${source.id} data-answer-ref>
                <span class=${answerStyles['answer-ref-id']}>${source.id}</span>
                <span class=${answerStyles['answer-ref-title']}>${source.title}</span>
              </button>
            `,
          )}
          </div>
        </section>
      ` : nothing}
    `;
  }

  #part(part: PresentationPart, index: number): TemplateResult | typeof nothing {
    if (part.type === 'markdown') {
      return html`<div class="answer-rich-content ${chatStyles.aiMessageContent} ${chatStyles.markdownContent}" data-answer-part=${String(index)}></div>`;
    }
    if (part.type === 'evidence_image' && part.evidenceImage) {
      return html`<div class="answer-inline-evidence">${this.#evidenceImage(part.evidenceImage)}</div>`;
    }
    if (part.type === 'artifact' && part.artifact) return this.#artifact(part.artifact, part.inline);
    return nothing;
  }

  #upgradeLinkCards(mounted: MountedRichPart, presentation: AnswerPresentation): void {
    // Only actual anchors survived Markdown parsing and both sanitizers. Match
    // metadata to each such occurrence, never to source text or a URL elsewhere
    // in the answer. Code, titles, math and image alt text cannot authorize one.
    const cards = new Map((presentation.linkCards ?? []).map((card) => [safeExternalHttpHref(card.url), card]));
    for (const occurrence of mounted.links) {
      const {anchor} = occurrence;
      const href = safeExternalHttpHref(anchor.getAttribute('href') || '');
      // Citation controls keep their role. A recommendation anchor in the body
      // is still a recommendation when the same URL also appears in References.
      const eligible = href && !anchor.closest('code, pre, .citation-badge, .answer-citation-link');
      const card = eligible ? cards.get(href) : undefined;
      const video = eligible && presentation.videoLinks?.some((link) => safeExternalHttpHref(link.url) === href);
      if (!card && !video) {
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
      const preview = card
        ? (video ? this.#videoCardLabel(card) : this.#linkCard(card))
        : anchor;
      render(this.#videoLink(href, preview, video ? card?.image || '' : ''), occurrence.slot);
    }
  }

  #videoLink(url: string, preview: TemplateResult | HTMLAnchorElement, cover = ''): TemplateResult {
    const link = this.presentation?.videoLinks?.find((item) => safeExternalHttpHref(item.url) === safeExternalHttpHref(url));
    return link
      ? html`<dl-video-playback .link=${link} .preview=${preview} .cover=${cover}></dl-video-playback>`
      : html`${preview}`;
  }

  #videoCardLabel(card: import('../api/conversations.ts').LinkCard): TemplateResult {
    return html`
      <strong>${card.title}</strong>
      ${card.description ? html`<span>${card.description}</span>` : nothing}
    `;
  }

  #linkCard(card: import('../api/conversations.ts').LinkCard): TemplateResult {
    // Non-video cards remain a link out. Playback candidates own a separate
    // native Play control; OG metadata alone never authorizes an iframe (ADR 0028).
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
          <p>${artifact.issue?.description || msg('This file is unavailable.', {id: 'answerPresentation.artifactUnavailable'})}</p>
          ${/* The issue already names the reason a withheld Artifact was withheld; a badge here
                would repeat it in other words. */ ''}
        </article>
      `;
    }
    if (inline && artifact.presentation === 'image') {
      const source = safeImageSrc(artifact.dataUrl || '');
      if (source) {
        // The Model writes this Artifact's link text as the picture's own sentence -- "2026-10-15
        // Photo of the Day: Rotation Station (Garret Suhrie)" -- and drawing it three times told a
        // reader nothing twice. The caption is where a sentence belongs, and the alt is where a
        // broken image and a screen reader need one: the file's own name, which is what the bytes
        // under the picture currently are. The control's accessible name keeps the sentence for the
        // reader who navigates by control rather than by picture, and nothing repeats it visually.
        return html`
          <figure class="answer-artifact-image">
            <button type="button" data-answer-image data-src=${source}
                    aria-label=${msg(str`Open image: ${artifact.label}`, {id: 'answerPresentation.openImage'})}>
              <img src=${source} alt=${artifact.filename} loading="lazy">
            </button>
            <figcaption>
              <span class="answer-artifact-image-caption">${artifact.label}</span>
              ${artifact.width && artifact.height
                ? html`<span class="answer-artifact-image-size">${artifact.width}×${artifact.height}</span>`
                : nothing}
            </figcaption>
          </figure>
        `;
      }
    }
    if (inline && artifact.presentation === 'video') {
      // The browser owns decoding, so the Artifact URL is the player's source:
      // a range-capable same-origin request streams and seeks without inlining
      // the bytes. The caption keeps an escape hatch for a container this
      // browser cannot decode, which leaves the element itself empty. Both
      // caption links wear the shared button face, so neither relies on colour
      // to read as a link and both keep text contrast on every surface.
      const source = safeSameOriginHref(artifact.dataUrl || '');
      if (source) {
        return html`
          <figure class=${answerStyles['answer-artifact-video']}>
            <video data-answer-video controls preload="metadata" playsinline src=${source}></video>
            <figcaption>
              <span>${artifact.label}</span>
              <a class="dl-btn" href=${source} target="_blank" rel="noopener noreferrer">${msg('Open in a new tab', {id: 'answerPresentation.openVideo'})}</a>
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
        }}>${msg('Open file', {id: 'answerPresentation.openArtifact'})}</button>
      </article>
    `;
  }

  /** The one delivery record this Answer's files carry, or nothing when it has none to draw.
   *
   * The server stamps one record on every Artifact of a Run (`stamp_projected_artifacts`), so the
   * record under the files is any available placement's. A withheld Artifact is not counted: its
   * own issue already names why it was withheld, and a record here would say it again. */
  #deliveryRecord(): TemplateResult | typeof nothing {
    const presentation = this.presentation;
    if (!presentation) return nothing;
    let record: ArtifactAttestation | null = null;
    for (const part of presentation.parts) {
      const artifact = part.type === 'artifact' ? part.artifact : null;
      if (artifact?.status === 'available' && artifact.attestation !== null) {
        record = artifact.attestation;
        break;
      }
    }
    return artifactDeliveryRecord(record, {
      expanded: this.recordExpanded,
      onToggle: this.#toggleRecord,
      evidenceId: this.#recordEvidenceId,
    });
  }

  #evidenceImage(image: PresentationImage): TemplateResult | typeof nothing {
    const source = safeImageSrc(image.url);
    const thumbnail = safeImageSrc(image.thumbnailUrl || image.url);
    if (!source || !thumbnail) return nothing;
    return html`
      <div class="answer-evidence-image">
        <button class=${answerStyles['answer-image-item']} type="button" data-answer-image
                data-src=${source} aria-label=${msg(str`Open image: ${image.label}`, {id: 'answerPresentation.openImage'})}>
          <img src=${thumbnail} alt=${imageName(image.url)} loading="lazy">
          <span class=${answerStyles['answer-image-label']}>${image.label}</span>
        </button>
        ${image.sourceRef ? html`
          <button class="dl-text-action answer-image-source" type="button" data-ref=${image.sourceRef}
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

  #toggleRecord = (): void => {
    this.recordExpanded = !this.recordExpanded;
  };

  /** The row a trace sent the reader to, with the light that says which row it is. */
  #referenceItemClass(source: PresentationSource): string {
    const classes = [answerStyles['answer-ref-item']];
    if (this.highlightedReferenceId === source.id) {
      classes.push(answerStyles['answer-ref-item-highlighted']);
    }
    return classes.join(' ');
  }

  /** A trace asks the page for one source by the id its Answer publishes.
   *
   * The request is dispatched on the document, so every Answer on the page hears it; an Answer
   * that does not hold that source has nothing to show and says nothing. */
  #handleShowReference = (event: CustomEvent<ShowReferenceDetail>): void => {
    const sourceId = event.detail?.sourceId;
    if (typeof sourceId !== 'string' || sourceId === '') return;
    const askedFor = event.detail?.runId;
    if (typeof askedFor === 'string' && this.runId !== null && askedFor !== this.runId) return;
    const presentation = this.presentation;
    if (!presentation?.sources.some((source) => source.id === sourceId)) return;
    this.#revealReference(sourceId);
  };

  #revealReference(sourceId: string): void {
    const presentation = this.presentation;
    if (!presentation) return;
    // The list shows two rows folded; the one named may be behind the caption, so revealing it
    // means opening the list first. The row it names is the Answer's, not its summary's.
    if (presentation.sources.length > 2) this.referencesExpanded = true;
    this.highlightedReferenceId = sourceId;
    clearTimeout(this.#highlightTimer);
    this.#highlightTimer = setTimeout(() => {
      this.#highlightTimer = undefined;
      this.highlightedReferenceId = null;
    }, REFERENCE_HIGHLIGHT_MS);
    // The lit row has to be laid out before it can be brought into view, and the reader lands on
    // it: a request from the trace moved them here, so the focus follows the thing they asked for.
    void this.updateComplete.then(() => {
      const row = this.querySelector<HTMLElement>(`[data-reference-id="${CSS.escape(sourceId)}"]`);
      row?.focus({preventScroll: true});
      row?.scrollIntoView({block: 'nearest'});
    });
  }

  #clearHighlight(): void {
    clearTimeout(this.#highlightTimer);
    this.#highlightTimer = undefined;
    this.highlightedReferenceId = null;
  }

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
      raise(this, 'dl-answer-source-open', {
        presentation,
        referenceId: source.dataset.ref || '',
        ...(source.dataset.chunk ? {chunkId: source.dataset.chunk} : {}),
        returnFocus: source,
      });
      return;
    }
    const image = target.closest<HTMLElement>('[data-answer-image][data-src]');
    if (!image || !this.contains(image)) return;
    event.preventDefault();
    event.stopPropagation();
    raise(this, 'dl-image-open', {
      src: image.dataset.src || '',
      gallery: this.#galleryImages(),
      returnFocus: image,
    });
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
    raise(this, 'dl-artifact-open', {artifact, returnFocus});
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

  interface DocumentEventMap {
    'dl-show-reference': CustomEvent<ShowReferenceDetail>;
  }
}
