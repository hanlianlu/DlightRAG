// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0

import {msg, str, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {styleMap} from 'lit/directives/style-map.js';
import {resolveVideoPlayback, type VideoPlaybackLink} from '../api/video-playback.ts';
import {icon} from '../design-system/index.ts';
import {LightElement} from '../lib/lit-host.ts';
import {safeExternalHttpHref} from '../lib/urls.ts';
import styles from '../styles/video-playback.module.css';

const ACTIVATE = 'dl-video-playback-activate';

/** Reader-owned external playback; never an execution mode of Artifact HTML. */
export class DlVideoPlayback extends LightElement {
  static properties = {
    link: {attribute: false}, preview: {attribute: false}, cover: {attribute: false},
    state: {state: true}, source: {state: true}, aspectRatio: {state: true},
  };

  declare link: VideoPlaybackLink | null;
  declare preview: TemplateResult | Node | null;
  declare cover: string;
  declare state: 'idle' | 'loading' | 'active' | 'error';
  declare source: string;
  declare aspectRatio: number;
  #controller: AbortController | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.link = null;
    this.preview = null;
    this.cover = '';
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
    const playing = Boolean(this.source);
    const awaiting = this.state === 'loading';
    return html`
      <span class=${styles.card} data-video-card>
        <span class=${styles.media} data-video-media
              style=${styleMap({'aspect-ratio': String(this.aspectRatio)})}>
          ${playing ? html`
            <iframe class=${styles.player} data-external-video
              title=${msg(str`${link.provider} video player`, {id: 'videoPlayback.frameTitle'})}
              src=${this.source} sandbox="allow-scripts allow-same-origin allow-presentation"
              allow="autoplay; encrypted-media; fullscreen; picture-in-picture"
              referrerpolicy="strict-origin-when-cross-origin"
              @error=${this.#failed}
            ></iframe>
          ` : awaiting ? this.#previewFace() : html`
            <button class=${styles.play} type="button" data-video-play
              aria-label=${msg(str`Play ${link.provider} video`, {id: 'videoPlayback.playAria'})}
              @click=${this.#play}>
              ${this.#previewFace()}
              <span class=${styles.icon}>${icon('play', {size: 'lg'})}</span>
            </button>
          `}
        </span>
        <span class=${styles.body}>
          <span class=${styles.label}>${this.preview}</span>
          <span class=${styles.footer}>
            <span class=${styles.provider}>${link.provider}</span>
            ${awaiting || playing ? html`
              <button class="dl-btn" type="button" data-video-stop @click=${this.#stop}>
                ${msg('Stop playback', {id: 'videoPlayback.stop'})}
              </button>
            ` : nothing}
            <a class=${styles.source} data-video-open
               href=${href} target="_blank" rel="noopener noreferrer"
               aria-label=${msg('Open at source', {id: 'videoPlayback.open'})}>
              ${icon('open-external', {size: 'sm'})}
            </a>
          </span>
          ${awaiting ? html`<span role="status">${msg('Loading player…', {id: 'videoPlayback.loading'})}</span>` : nothing}
          ${this.state === 'error' ? html`<span role="alert">${msg('Player unavailable. Try again or open the original link.', {id: 'videoPlayback.error'})}</span>` : nothing}
        </span>
      </span>
    `;
  }

  #previewFace(): TemplateResult {
    const cover = safeExternalHttpHref(this.cover);
    return cover
      ? html`<img class=${styles.cover} src=${cover} alt="" loading="lazy">`
      : html`<span class=${styles.neutral} data-video-preview></span>`;
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
