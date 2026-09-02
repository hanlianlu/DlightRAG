// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import type {AnswerArtifact, AnswerPresentation} from '../api/conversations.ts';
import {icon} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA, MOBILE_MEDIA} from '../lib/breakpoints.ts';
import {wrapTabFocus} from '../lib/dom.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import type {DlActiveArtifactFrame} from './active-artifact-frame.ts';
import type {ImageOpenDetail} from './image-lightbox.ts';
import './active-artifact-frame.ts';
import './answer-presentation.ts';

type CanvasLayout = 'side' | 'wide' | 'fullscreen';
type CanvasState = 'idle' | 'loading' | 'ready' | 'error';

export interface ArtifactCanvasStateDetail {
  open: boolean;
  modal: boolean;
  overlay: boolean;
}

const TEXT_PREVIEW_BYTES = 1024 * 1024;

/** General presentation surface for every Answer Artifact, including Primary Reports. */
export class DlArtifactCanvas extends LightElement {
  static properties = {
    activePreviewEnabled: {type: Boolean, attribute: 'active-preview-enabled'},
    canvasState: {state: true},
    layout: {state: true},
    artifact: {state: true},
    textPreview: {state: true},
    presentation: {state: true},
    interactive: {state: true},
  };

  declare activePreviewEnabled: boolean;
  declare canvasState: CanvasState;
  declare layout: CanvasLayout;
  declare artifact: AnswerArtifact | null;
  declare textPreview: string;
  declare presentation: AnswerPresentation | null;
  declare interactive: boolean;

  #controller: AbortController | null = null;
  #returnFocus: HTMLElement | null = null;
  #compactMedia: MediaQueryList | null = null;
  #focusGeneration = 0;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.activePreviewEnabled = true;
    this.canvasState = 'idle';
    this.layout = 'side';
    this.artifact = null;
    this.textPreview = '';
    this.presentation = null;
    this.interactive = false;
    this.addEventListener('keydown', (event) => this.#onKeyDown(event));
  }

  override connectedCallback(): void {
    super.connectedCallback();
    if (!this.classList.contains('open')) {
      this.inert = true;
      this.setAttribute('aria-hidden', 'true');
    }
    this.#compactMedia = window.matchMedia(COMPACT_SHELL_MEDIA);
    this.#compactMedia.addEventListener('change', this.#compactLayoutChanged);
  }

  override disconnectedCallback(): void {
    this.#compactMedia?.removeEventListener('change', this.#compactLayoutChanged);
    this.#compactMedia = null;
    this.#focusGeneration += 1;
    this.#destroyPreview();
    this.#controller?.abort();
    super.disconnectedCallback();
  }

  async open(artifact: AnswerArtifact, returnFocus?: HTMLElement | null): Promise<void> {
    this.#focusGeneration += 1;
    const entering = !this.classList.contains('open');
    this.#controller?.abort();
    this.#destroyPreview();
    if (entering) {
      this.#returnFocus = returnFocus ?? (
        document.activeElement instanceof HTMLElement ? document.activeElement : null
      );
    }
    this.artifact = artifact;
    this.canvasState = 'loading';
    this.#setLayout(this.#suggestedLayout(artifact));
    this.textPreview = '';
    this.presentation = null;
    this.interactive = false;
    this.classList.add('open');
    this.inert = false;
    this.removeAttribute('aria-hidden');
    this.setAttribute('role', 'dialog');
    this.setAttribute('aria-labelledby', 'artifact-canvas-title');
    this.#syncModalState();
    await this.updateComplete;
    this.querySelector<HTMLButtonElement>('[data-action="close"]')?.focus();
    await this.#load(artifact);
  }

  /** Make the Inspector reachable and return focus owned by a closed compact Canvas. */
  prepareForInspector(): HTMLElement | null {
    if (!this.classList.contains('open')) return null;
    if (this.#compactMedia?.matches ?? window.matchMedia(COMPACT_SHELL_MEDIA).matches) {
      const returnFocus = this.#returnFocus;
      this.close(false);
      return returnFocus;
    }
    this.#setLayout('side');
    return null;
  }

  close(restoreFocus = true): void {
    if (!this.classList.contains('open')) return;
    const focusGeneration = ++this.#focusGeneration;
    this.#controller?.abort();
    this.#controller = null;
    this.#destroyPreview();
    const focusedInside = this.contains(document.activeElement);
    this.classList.remove('open', 'layout-wide', 'layout-fullscreen');
    this.inert = true;
    this.setAttribute('aria-hidden', 'true');
    this.removeAttribute('role');
    this.removeAttribute('aria-labelledby');
    this.removeAttribute('aria-modal');
    this.artifact = null;
    this.canvasState = 'idle';
    this.presentation = null;
    this.textPreview = '';
    this.#stateChanged();
    const returnFocus = this.#returnFocus;
    this.#returnFocus = null;
    if (restoreFocus) {
      window.requestAnimationFrame(() => {
        if (focusGeneration !== this.#focusGeneration || this.classList.contains('open')) return;
        if (returnFocus?.isConnected && !returnFocus.inert) returnFocus.focus();
      });
    } else if (focusedInside && document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }
  }

  reload(): void {
    if (this.artifact) void this.#load(this.artifact);
  }

  protected override render(): TemplateResult {
    const artifact = this.artifact;
    return html`
      <div class="artifact-canvas-header">
        <div class="artifact-canvas-heading">
          <h2 class="artifact-canvas-title" id="artifact-canvas-title">${artifact?.label || msg('Artifact', {id: 'artifactCanvas.fallbackTitle'})}</h2>
          ${artifact ? html`<span class="artifact-canvas-filename">${artifact.filename}</span>` : nothing}
        </div>
        <div class="artifact-canvas-actions">
          <div class="artifact-canvas-layout-actions" role="group"
               aria-labelledby="artifact-canvas-title">
            <button class="dl-btn" type="button" @click=${() => this.#setLayout('side')}
                    aria-pressed=${this.layout === 'side'}>${msg('Side', {id: 'artifactCanvas.layoutSide'})}</button>
            <button class="dl-btn" type="button" @click=${() => this.#setLayout('wide')}
                    aria-pressed=${this.layout === 'wide'}>${msg('Wide', {id: 'artifactCanvas.layoutWide'})}</button>
            <button class="dl-btn" type="button" @click=${() => this.#setLayout('fullscreen')}
                    aria-pressed=${this.layout === 'fullscreen'}>${msg('Fullscreen', {id: 'artifactCanvas.layoutFullscreen'})}</button>
          </div>
          ${artifact?.downloadUrl ? html`
            <a class="dl-btn" href=${safeSameOriginHref(artifact.downloadUrl) || '#'} download>
              ${msg('Download', {id: 'artifactCanvas.download'})}
            </a>` : nothing}
          <button class="panel-close" data-action="close" type="button"
                  aria-label=${msg('Close Artifact', {id: 'artifactCanvas.close'})} @click=${() => this.close()}>${icon('close', {size: 'sm'})}</button>
        </div>
      </div>
      <div class="artifact-canvas-content">
        ${this.#content()}
      </div>
    `;
  }

  #content(): TemplateResult {
    const artifact = this.artifact;
    if (!artifact) return html``;
    if (artifact.status === 'unavailable') {
      return html`<div class="artifact-unavailable" role="alert">
        <strong>${msg('Artifact unavailable', {id: 'artifactCanvas.unavailableTitle'})}</strong>
        <p>${artifact.issue?.description || msg('This Artifact could not be published.', {id: 'artifactCanvas.unavailableDescription'})}</p>
      </div>`;
    }
    if (this.canvasState === 'loading') {
      return html`<div class="artifact-loading" role="status">${msg('Loading Artifact…', {id: 'artifactCanvas.loading'})}</div>`;
    }
    if (this.canvasState === 'error') {
      return html`<div class="artifact-error" role="alert">
        <p>${msg('Could not load this Artifact safely.', {id: 'artifactCanvas.error'})}</p>
        <button class="dl-btn" type="button" @click=${() => this.reload()}>${msg('Retry', {id: 'artifactCanvas.retry'})}</button>
        ${this.textPreview ? html`<pre>${this.textPreview}</pre>` : nothing}
      </div>`;
    }
    switch (artifact.presentation) {
      case 'markdown':
        return this.presentation
          ? html`<dl-answer-presentation .presentation=${this.presentation}></dl-answer-presentation>`
          : html``;
      case 'image': {
        const source = safeImageSrc(artifact.dataUrl || '');
        return source
          ? html`<button class="artifact-image" type="button"
                  aria-label=${msg(str`Open image: ${artifact.label}`, {id: 'artifactCanvas.openImage'})}
                  @click=${(event: Event) => this.#openImage(
                    source,
                    event.currentTarget as HTMLElement,
                  )}>
              <img src=${source} alt=${artifact.label}>
            </button>`
          : this.#downloadOnly();
      }
      case 'html':
        return this.#htmlPreview();
      case 'pdf': {
        const source = safeSameOriginHref(artifact.dataUrl || '');
        return source
          ? html`<iframe class="artifact-pdf" title=${artifact.label} src=${source}
                  sandbox="" referrerpolicy="no-referrer"></iframe>`
          : this.#downloadOnly();
      }
      case 'text':
        return html`<pre class="artifact-source">${this.textPreview}</pre>`;
      default:
        return this.#downloadOnly();
    }
  }

  #htmlPreview(): TemplateResult {
    if (!this.activePreviewEnabled) {
      return html`
        <dl-active-artifact-frame
          .source=${this.textPreview}
          .active=${false}
          .label=${this.artifact?.label || msg('HTML Artifact', {id: 'artifactCanvas.htmlFallbackLabel'})}
        ></dl-active-artifact-frame>
        ${this.#htmlSource(msg('Source', {id: 'artifactCanvas.source'}))}
      `;
    }
    if (!this.interactive) {
      return html`
        <div class="artifact-active-consent">
          <strong>${msg('Untrusted interactive report', {id: 'artifactCanvas.untrustedTitle'})}</strong>
          <p>${msg('Active code is isolated from DlightRAG. Normal external loads are blocked by browser policy.', {id: 'artifactCanvas.untrustedDescription'})}</p>
          <button class="dl-btn" type="button" @click=${() => { this.interactive = true; }}>
            ${msg('Open interactive report', {id: 'artifactCanvas.openInteractive'})}
          </button>
        </div>
        ${this.#htmlSource(msg('Static source', {id: 'artifactCanvas.staticSource'}))}
      `;
    }
    return html`
      <dl-active-artifact-frame
        .source=${this.textPreview}
        .active=${true}
        .label=${this.artifact?.label || msg('HTML Artifact', {id: 'artifactCanvas.htmlFallbackLabel'})}
        @dl-artifact-frame-escape=${() => this.close()}
      ></dl-active-artifact-frame>
      ${this.#htmlSource(msg('Source', {id: 'artifactCanvas.source'}))}
    `;
  }

  #htmlSource(summary: string): TemplateResult {
    return html`<details>
      <summary>${summary}</summary>
      <pre class="artifact-source">${this.textPreview}</pre>
    </details>`;
  }

  #downloadOnly(): TemplateResult {
    return html`<div class="artifact-download-only">
      <p>${msg('No browser-safe inline preview is available for this file.', {id: 'artifactCanvas.downloadOnly'})}</p>
      ${this.artifact?.downloadUrl ? html`
        <a class="dl-btn" href=${safeSameOriginHref(this.artifact.downloadUrl) || '#'} download>
          ${msg(str`Download ${this.artifact.filename}`, {id: 'artifactCanvas.downloadFile'})}
        </a>` : nothing}
    </div>`;
  }

  async #load(artifact: AnswerArtifact): Promise<void> {
    const controller = new AbortController();
    this.#controller = controller;
    try {
      if (artifact.status === 'unavailable') {
        this.canvasState = 'ready';
        return;
      }
      if (artifact.presentation === 'markdown') {
        if (!artifact.presentationUrl) throw new Error('missing presentation URL');
        const response = await fetch(artifact.presentationUrl, {signal: controller.signal});
        if (!response.ok) throw new Error('presentation failed');
        const presentation = await response.json() as AnswerPresentation;
        if (this.#controller !== controller) return;
        this.presentation = presentation;
      } else if (artifact.presentation === 'html' || artifact.presentation === 'text') {
        if (!artifact.dataUrl) throw new Error('missing Artifact URL');
        const response = await fetch(artifact.dataUrl, {
          headers: artifact.presentation === 'text'
            ? {Range: `bytes=0-${TEXT_PREVIEW_BYTES - 1}`}
            : undefined,
          signal: controller.signal,
        });
        if (!response.ok) throw new Error('Artifact data failed');
        const textPreview = await response.text();
        if (this.#controller !== controller) return;
        this.textPreview = textPreview;
      }
      if (this.#controller === controller) this.canvasState = 'ready';
    } catch {
      if (!controller.signal.aborted && this.#controller === controller) this.canvasState = 'error';
    }
  }

  #destroyPreview(): void {
    this.querySelector<DlActiveArtifactFrame>('dl-active-artifact-frame')?.destroy();
  }

  #openImage(src: string, returnFocus: HTMLElement): void {
    this.dispatchEvent(new CustomEvent<ImageOpenDetail>('dl-image-open', {
      bubbles: true,
      composed: true,
      detail: {src, gallery: [src], returnFocus},
    }));
  }

  #stateChanged(): void {
    const open = this.classList.contains('open');
    this.dispatchEvent(new CustomEvent<ArtifactCanvasStateDetail>(
      'dl-artifact-canvas-state-change',
      {
        bubbles: true,
        composed: true,
        detail: {open, modal: this.#isModal(), overlay: open && this.layout !== 'side'},
      },
    ));
  }

  #setLayout(layout: CanvasLayout): void {
    this.layout = layout;
    this.classList.toggle('layout-wide', layout === 'wide');
    this.classList.toggle('layout-fullscreen', layout === 'fullscreen');
    this.#syncModalState();
  }

  #compactLayoutChanged = (): void => {
    this.#syncModalState();
    if (this.#isModal() && !this.contains(document.activeElement)) {
      this.querySelector<HTMLButtonElement>('[data-action="close"]')?.focus();
    }
  };

  #isModal(): boolean {
    const compact = this.#compactMedia?.matches
      ?? window.matchMedia(COMPACT_SHELL_MEDIA).matches;
    return this.classList.contains('open') && (this.layout !== 'side' || compact);
  }

  #syncModalState(): void {
    const modal = this.#isModal();
    if (modal) this.setAttribute('aria-modal', 'true');
    else this.removeAttribute('aria-modal');
    this.#stateChanged();
  }

  #suggestedLayout(artifact: AnswerArtifact): CanvasLayout {
    if (window.matchMedia(MOBILE_MEDIA).matches) return 'fullscreen';
    return artifact.presentation === 'markdown' ? 'side' : 'wide';
  }

  #onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      event.preventDefault();
      this.close();
      return;
    }
    if (event.key === 'Tab' && this.#isModal()) {
      const focusable = Array.from(this.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], iframe, [tabindex]:not([tabindex="-1"])',
      )).filter((element) => element.getClientRects().length > 0);
      wrapTabFocus(focusable, event);
    }
  }
}

customElements.define('dl-artifact-canvas', DlArtifactCanvas);

declare global {
  interface HTMLElementTagNameMap {
    'dl-artifact-canvas': DlArtifactCanvas;
  }

  interface HTMLElementEventMap {
    'dl-artifact-canvas-state-change': CustomEvent<ArtifactCanvasStateDetail>;
  }
}
