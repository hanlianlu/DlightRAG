// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {styleMap} from 'lit/directives/style-map.js';
import {resolveVideoPlayback, type VideoPlaybackLink} from '../api/video-playback.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeExternalHttpHref} from '../lib/urls.ts';
import styles from '../styles/video-playback.module.css';

const ACTIVATE = 'dl-video-playback-activate';

/** Reader-owned external playback; never an execution mode of Artifact HTML. */
export class DlVideoPlayback extends LightElement {
  static properties = {
    link: {attribute: false}, preview: {attribute: false},
    state: {state: true}, source: {state: true}, aspectRatio: {state: true},
  };

  declare link: VideoPlaybackLink | null;
  declare preview: TemplateResult | Node | null;
  declare state: 'idle' | 'loading' | 'active' | 'error';
  declare source: string;
  declare aspectRatio: number;
  #controller: AbortController | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.link = null;
    this.preview = null;
    this.state = 'idle';
    this.source = '';
    this.aspectRatio = 16 / 9;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.ownerDocument.addEventListener(ACTIVATE, this.#anotherPlayer);
  }

  override disconnectedCallback(): void {
    this.ownerDocument.removeEventListener(ACTIVATE, this.#anotherPlayer);
    this.#stop();
    super.disconnectedCallback();
  }

  protected override willUpdate(changed: PropertyValues<this>): void {
    if (changed.has('link') && changed.get('link')?.url !== this.link?.url) this.#stop();
  }

  protected override render(): TemplateResult | typeof nothing {
    const link = this.link;
    const href = safeExternalHttpHref(link?.url ?? '');
    if (!link || !href) return nothing;
    return html`
      <span class=${styles.playback}>
        ${this.source ? html`
          <iframe class=${styles.player} data-external-video style=${styleMap({'aspect-ratio': String(this.aspectRatio)})}
            title=${msg(str`${link.provider} video player`, {id: 'videoPlayback.frameTitle'})}
            src=${this.source} sandbox="allow-scripts allow-same-origin allow-presentation"
            allow="autoplay; encrypted-media; fullscreen; picture-in-picture"
            referrerpolicy="strict-origin-when-cross-origin"
            @error=${this.#failed}
          ></iframe>
        ` : this.preview}
        <span class=${styles.actions}>
          ${this.state === 'loading' || this.source ? html`
            <button class="dl-btn" type="button" data-video-stop @click=${this.#stop}>
              ${msg('Stop playback', {id: 'videoPlayback.stop'})}
            </button>
          ` : html`
            <button class="dl-btn" type="button" data-video-play @click=${this.#play}
              aria-label=${msg(str`Try playback here — connects to ${link.provider}`, {id: 'videoPlayback.playAria'})}>
              ${msg('Try playback here', {id: 'videoPlayback.play'})}
            </button>
          `}
          ${this.source ? html`
            <a href=${href} target="_blank" rel="noopener noreferrer">
              ${msg('Open at source', {id: 'videoPlayback.open'})}
            </a>
          ` : nothing}
          ${this.state === 'loading' ? html`<span role="status">${msg('Loading player…', {id: 'videoPlayback.loading'})}</span>` : nothing}
          ${this.state === 'error' ? html`<span role="alert">${msg('Player unavailable. Try again or open the original link.', {id: 'videoPlayback.error'})}</span>` : nothing}
        </span>
      </span>
    `;
  }

  #anotherPlayer = (event: Event): void => {
    if ((event as CustomEvent).detail !== this) this.#stop();
  };

  #stop = (): void => {
    this.#controller?.abort();
    this.#controller = null;
    // Tear down synchronously: B must not overlap A while Lit queues a render.
    this.querySelector('iframe')?.remove();
    this.source = '';
    this.state = 'idle';
  };

  #failed = (): void => {
    this.#stop();
    this.state = 'error';
  };

  #play = async (): Promise<void> => {
    if (!this.link) return;
    this.ownerDocument.dispatchEvent(new CustomEvent(ACTIVATE, {detail: this}));
    this.#stop();
    const controller = new AbortController();
    this.#controller = controller;
    this.state = 'loading';
    try {
      const player = await resolveVideoPlayback(this.link, controller.signal);
      if (this.#controller !== controller || !this.isConnected) return;
      this.source = player.embedUrl;
      this.aspectRatio = player.aspectRatio;
      this.state = 'active';
    } catch {
      if (this.#controller === controller && this.isConnected) this.#failed();
    }
  };
}

customElements.define('dl-video-playback', DlVideoPlayback);

declare global {
  interface HTMLElementTagNameMap {
    'dl-video-playback': DlVideoPlayback;
  }
}
