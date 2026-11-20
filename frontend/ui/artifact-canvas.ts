// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {
  type AnswerArtifact,
  type AnswerPresentation,
  getArtifactPresentationAt,
  getArtifactTextAt,
} from '../api/conversations.ts';
import {ApiError} from '../api/wire.ts';
import {isImeKey} from '../design-system/index.ts';
import {COMPACT_SHELL_MEDIA, MOBILE_MEDIA} from '../lib/breakpoints.ts';
import {focusedElement, raise, tabbables, wrapTabFocus} from '../lib/dom.ts';
import {FocusReturn} from '../lib/focus-return.ts';
import {LatestRequest} from '../lib/latest-request.ts';
import {LightElement, MediaController} from '../lib/lit-host.ts';
import {safeImageSrc, safeSameOriginHref} from '../lib/urls.ts';
import canvasStyles from '../styles/artifact-canvas.module.css';
import type {DlActiveArtifactFrame} from './active-artifact-frame.ts';
import {artifactAttestationBadge} from './artifact-attestation.ts';
import {artifactDownloadLink} from './artifact-download.ts';
import './active-artifact-frame.ts';
import './answer-presentation.ts';

type CanvasLayout = 'side' | 'wide' | 'fullscreen';
type CanvasState = 'idle' | 'loading' | 'ready' | 'error';

/** Why a load left the Canvas empty. A refusal or an unreachable server is not
 *  a judgement about the bytes, and only the third says the Canvas could not
 *  establish what it was handed. */
type LoadFailure =
  | {kind: 'answered'; status: number}
  | {kind: 'unreachable'}
  | {kind: 'unsafe'};

/** One Artifact's loaded bytes, held so reopening it is not a second read. */
interface HeldArtifact {
  key: string;
  presentation: AnswerPresentation | null;
  text: string;
}

export interface ArtifactCanvasStateDetail {
  open: boolean;
  modal: boolean;
  overlay: boolean;
  wide: boolean;
}

const TEXT_PREVIEW_BYTES = 1024 * 1024;

/** What a failed load tells the reader: the server's own answer, an unreachable
 *  server, or content the validated edge could not vouch for. */
function loadFailureOf(error: unknown): LoadFailure {
  if (error instanceof ApiError && error.status >= 400) {
    return {kind: 'answered', status: error.status};
  }
  // A fetch that never reached a response rejects with a TypeError; anything
  // else — a URL the Canvas will not follow, bytes that fail their schema —
  // is the Canvas refusing what it was handed.
  if (error instanceof TypeError) return {kind: 'unreachable'};
  return {kind: 'unsafe'};
}

/** General presentation surface for every presentable Answer Artifact. */
export class DlArtifactCanvas extends LightElement {
  static properties = {
    activePreviewEnabled: {type: Boolean, attribute: 'active-preview-enabled'},
    canvasState: {state: true},
    layout: {state: true},
    artifact: {state: true},
    textPreview: {state: true},
    presentation: {state: true},
  };

  declare activePreviewEnabled: boolean;
  declare canvasState: CanvasState;
  declare layout: CanvasLayout;
  declare artifact: AnswerArtifact | null;
  declare textPreview: string;
  declare presentation: AnswerPresentation | null;

  /** The read that fills the Canvas; opening another Artifact, closing, or leaving ends the one under way. */
  readonly #reads = new LatestRequest();
  readonly #focus = new FocusReturn();
  readonly #compact = new MediaController(this, COMPACT_SHELL_MEDIA, () => { this.#compactLayoutChanged(); });
  #loadFailure: LoadFailure | null = null;
  #held: HeldArtifact | null = null;

  constructor() {
    super();
    this.activePreviewEnabled = true;
    this.canvasState = 'idle';
    this.layout = 'side';
    this.artifact = null;
    this.textPreview = '';
    this.presentation = null;
    this.addEventListener('keydown', (event) => this.#onKeyDown(event));
  }

  override connectedCallback(): void {
    super.connectedCallback();
    if (!this.classList.contains('open')) {
      this.inert = true;
      this.setAttribute('aria-hidden', 'true');
    }
  }

  override disconnectedCallback(): void {
    this.#focus.invalidate();
    this.#destroyPreview();
    this.#reads.cancel();
    super.disconnectedCallback();
  }

  async open(artifact: AnswerArtifact, returnFocus?: HTMLElement | null): Promise<void> {
    this.#focus.invalidate();
    const entering = !this.classList.contains('open');
    this.#reads.cancel();
    this.#destroyPreview();
    if (entering) this.#focus.remember(returnFocus);
    this.artifact = artifact;
    this.canvasState = 'loading';
    this.#setLayout(this.#suggestedLayout(artifact));
    this.textPreview = '';
    this.presentation = null;
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
    if (this.#compact.matches) {
      const returnFocus = this.#focus.target;
      this.close(false);
      return returnFocus;
    }
    if (this.layout === 'fullscreen') this.#setLayout('wide');
    return null;
  }

  close(restoreFocus = true): void {
    if (!this.classList.contains('open')) return;
    this.#reads.cancel();
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
    this.#loadFailure = null;
    this.#stateChanged();
    if (restoreFocus) {
      this.#focus.restore(() => this.classList.contains('open'));
    } else {
      this.#focus.discard();
      if (focusedInside) focusedElement()?.blur();
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
          <h2 class="artifact-canvas-title" id="artifact-canvas-title">${artifact?.label || msg('File', {id: 'artifactCanvas.fallbackTitle'})}</h2>
          ${artifact ? html`<span class="artifact-canvas-filename">${artifact.filename}</span>` : nothing}
          ${artifactAttestationBadge(artifact?.attestation ?? null)}
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
          ${artifactDownloadLink(
            artifact?.downloadUrl,
            msg('Download', {id: 'artifactCanvas.download'}),
          )}
          <dl-icon-button class="panel-close" data-action="close" name="close" size="sm"
                  aria-label=${msg('Close file', {id: 'artifactCanvas.close'})} @click=${() => this.close()}></dl-icon-button>
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
        <strong>${msg('File unavailable', {id: 'artifactCanvas.unavailableTitle'})}</strong>
        <p>${artifact.issue?.description || msg('This file could not be published.', {id: 'artifactCanvas.unavailableDescription'})}</p>
      </div>`;
    }
    if (this.canvasState === 'loading') {
      return html`<div class="artifact-loading" role="status">${msg('Loading file…', {id: 'artifactCanvas.loading'})}</div>`;
    }
    if (this.canvasState === 'error') {
      return html`<div class="artifact-error" role="alert">
        <p>${this.#failureMessage()}</p>
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
      case 'video': {
        const source = safeSameOriginHref(artifact.dataUrl || '');
        return source
          ? html`<video class=${canvasStyles['artifact-video']} data-artifact-video controls preload="metadata" playsinline src=${source}></video>`
          : this.#downloadOnly();
      }
      case 'pdf': {
        // Deliberately not sandboxed, unlike the HTML preview: Chrome refuses to instantiate
        // its PDF viewer in a sandboxed frame — `sandbox=""` and every flag alike, measured on
        // Chrome 155 — so a sandbox here renders a broken-page icon instead of a document. The
        // frame can still only hold a PDF, because the route serves it same-origin, `inline`,
        // and `nosniff`; Chrome then renders it in its own extension origin and process.
        const source = safeSameOriginHref(artifact.dataUrl || '');
        return source
          ? html`<iframe class="artifact-pdf" title=${artifact.label} src=${source}
                  referrerpolicy="no-referrer"></iframe>`
          : this.#downloadOnly();
      }
      case 'text':
        return html`<pre class="artifact-source">${this.textPreview}</pre>`;
      default:
        return this.#downloadOnly();
    }
  }

  #htmlPreview(): TemplateResult {
    return html`
      <dl-active-artifact-frame
        .source=${this.textPreview}
        .active=${this.activePreviewEnabled}
        .label=${this.artifact?.label || msg('HTML file', {id: 'artifactCanvas.htmlFallbackLabel'})}
        @dl-artifact-frame-escape=${() => this.close()}
      ></dl-active-artifact-frame>
    `;
  }

  /** Why the load failed, in the reader's words: a safety refusal is only one of
   *  the three, and saying so of a server's 502 sends the reader after the wrong
   *  problem. */
  #failureMessage(): string {
    const failure = this.#loadFailure;
    if (failure?.kind === 'answered') {
      return msg(
        str`Could not load this file: the server answered HTTP ${failure.status}.`,
        {id: 'artifactCanvas.answered'},
      );
    }
    if (failure?.kind === 'unreachable') {
      return msg(
        'Could not load this file: the server could not be reached.',
        {id: 'artifactCanvas.unreachable'},
      );
    }
    return msg('Could not load this file safely.', {id: 'artifactCanvas.error'});
  }

  #downloadOnly(): TemplateResult {
    return html`<div class="artifact-download-only">
      <p>${msg('No browser-safe inline preview is available for this file.', {id: 'artifactCanvas.downloadOnly'})}</p>
      ${artifactDownloadLink(
        this.artifact?.downloadUrl,
        msg(str`Download ${this.artifact?.filename}`, {id: 'artifactCanvas.downloadFile'}),
      )}
    </div>`;
  }

  async #load(artifact: AnswerArtifact): Promise<void> {
    this.#loadFailure = null;
    if (artifact.status === 'unavailable') {
      this.canvasState = 'ready';
      return;
    }
    const key = this.#loadKey(artifact);
    // The digest is the content: an Artifact this Canvas already drew is drawn
    // from what it holds, so reopening one is not a second read of a route that
    // may be behind a slow or failing edge. A new publication has its own
    // digest (and its own route), so it is read.
    if (this.#held?.key === key) {
      this.presentation = this.#held.presentation;
      this.textPreview = this.#held.text;
      this.canvasState = 'ready';
      return;
    }
    const signal = this.#reads.start();
    try {
      if (artifact.presentation === 'markdown') {
        // Checked here as well as at the wire edge, so a route this Canvas may
        // not follow is its own refusal: `new URL` inside that helper throws a
        // TypeError, which the failure reading would otherwise take for a fetch
        // that never reached a server.
        const source = safeSameOriginHref(artifact.presentationUrl);
        if (!source) throw new Error('presentation URL is not one this Canvas may follow');
        const presentation = await getArtifactPresentationAt(source, signal);
        if (!this.#reads.isCurrent(signal)) return;
        this.presentation = presentation;
      } else if (artifact.presentation === 'html' || artifact.presentation === 'text') {
        const source = safeSameOriginHref(artifact.dataUrl);
        if (!source) throw new Error('Artifact URL is not one this Canvas may follow');
        const textPreview = await getArtifactTextAt(
          source,
          artifact.presentation === 'text' ? {firstBytes: TEXT_PREVIEW_BYTES} : {},
          signal,
        );
        if (!this.#reads.isCurrent(signal)) return;
        this.textPreview = textPreview;
      }
      if (this.#reads.isCurrent(signal)) {
        this.#held = {key, presentation: this.presentation, text: this.textPreview};
        this.canvasState = 'ready';
      }
    } catch (error) {
      if (this.#reads.isCurrent(signal)) {
        this.#loadFailure = loadFailureOf(error);
        this.canvasState = 'error';
      }
    }
  }

  /** What identifies the bytes a load reads: the same digest is the same content,
   *  and the route is part of the identity because one publication is served
   *  under the Run that published it. */
  #loadKey(artifact: AnswerArtifact): string {
    return JSON.stringify([
      artifact.resourceId,
      artifact.digest,
      artifact.dataUrl ?? '',
      artifact.presentationUrl ?? '',
    ]);
  }

  #destroyPreview(): void {
    this.querySelector<DlActiveArtifactFrame>('dl-active-artifact-frame')?.destroy();
  }

  #openImage(src: string, returnFocus: HTMLElement): void {
    raise(this, 'dl-image-open', {src, gallery: [src], returnFocus});
  }

  #stateChanged(): void {
    const open = this.classList.contains('open');
    raise(this, 'dl-artifact-canvas-state-change', {
      open,
      modal: this.#isModal(),
      overlay: open && this.layout === 'fullscreen',
      wide: open && this.layout === 'wide' && !this.#compact.matches,
    });
  }

  #setLayout(layout: CanvasLayout): void {
    this.layout = layout;
    this.classList.toggle('layout-wide', layout === 'wide');
    this.classList.toggle('layout-fullscreen', layout === 'fullscreen');
    this.#syncModalState();
  }

  #compactLayoutChanged(): void {
    this.#syncModalState();
    if (this.#isModal() && !this.contains(document.activeElement)) {
      this.querySelector<HTMLButtonElement>('[data-action="close"]')?.focus();
    }
  }

  #isModal(): boolean {
    return this.classList.contains('open')
      && (this.layout === 'fullscreen' || this.#compact.matches);
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
    // An Escape that cancels an IME composition belongs to the composition, not to the Canvas.
    if (isImeKey(event)) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      this.close();
      return;
    }
    if (event.key === 'Tab' && this.#isModal()) {
      wrapTabFocus(tabbables(this), event);
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
