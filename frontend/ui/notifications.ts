// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Notification Offer Feature and browser Notification lifecycle. */

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, type PropertyValues, type TemplateResult} from 'lit';
import {LightElement} from '../lib/lit-host.ts';

const ASKED_STORAGE_KEY = 'dlightrag-notify-asked';

function away(): boolean {
  // App switching can preserve visibility; focus completes the browser signal.
  return document.hidden || !document.hasFocus();
}

function supported(): boolean {
  return typeof window !== 'undefined' && 'Notification' in window;
}

function alreadyAsked(): boolean {
  try {
    return window.localStorage.getItem(ASKED_STORAGE_KEY) === '1';
  } catch {
    return false;
  }
}

function rememberAsked(): void {
  try {
    window.localStorage.setItem(ASKED_STORAGE_KEY, '1');
  } catch {
    // Browser storage is an optional enhancement.
  }
}

/** Owns missed-answer state, permission intent, and page-presence listeners. */
export class DlNotificationOffer extends LightElement {
  static properties = {
    running: {attribute: false},
    visible: {state: true},
  };

  declare running: boolean;
  declare visible: boolean;

  #missedAnswer = false;
  #events: AbortController | null = null;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.running = false;
    this.visible = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    const events = new AbortController();
    this.#events = events;
    window.addEventListener('blur', this.#leftPage, {signal: events.signal});
    window.addEventListener('focus', this.#cameBack, {signal: events.signal});
    document.addEventListener('visibilitychange', this.#visibilityChanged, {
      signal: events.signal,
    });
    this.#syncHost();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    super.disconnectedCallback();
  }

  protected override updated(changed: PropertyValues<this>): void {
    if (changed.has('running')) {
      const previous = Boolean(changed.get('running'));
      if (this.running && !previous) {
        this.#missedAnswer = away();
      } else if (!this.running && previous && away()) {
        this.#missedAnswer = true;
        if (supported() && Notification.permission === 'granted') this.#notifyAnswerReady();
      }
    }
    this.#syncHost();
  }

  protected override render(): TemplateResult {
    return html`
      <span class="notify-offer-text">${msg('Notify you when an answer finishes?', {id: 'notifications.offerText'})}</span>
      <button class="dl-btn" type="button" @click=${this.#accept}>${msg('Enable', {id: 'notifications.enable'})}</button>
      <button class="dl-btn" type="button" @click=${this.#decline}>${msg('Not now', {id: 'notifications.notNow'})}</button>
    `;
  }

  #syncHost(): void {
    this.hidden = !supported() || !this.visible;
  }

  #notifyAnswerReady(): void {
    try {
      const notification = new Notification(
        msg('Answer ready', {id: 'notifications.answerReadyTitle'}),
        {body: msg('DlightRAG finished generating your answer.', {id: 'notifications.answerReadyBody'})},
      );
      notification.onclick = () => {
        window.focus();
        notification.close();
      };
    } catch {
      // Some browsers reject construction outside a service worker.
    }
  }

  #leftPage = (): void => {
    if (this.running) this.#missedAnswer = true;
  };

  #cameBack = (): void => {
    if (away()) return;
    if (!this.running && this.#missedAnswer && supported()
        && Notification.permission === 'default' && !alreadyAsked()) {
      this.visible = true;
    }
    this.#missedAnswer = false;
  };

  #visibilityChanged = (): void => {
    if (document.hidden) this.#leftPage();
    else this.#cameBack();
  };

  #accept = async (): Promise<void> => {
    this.visible = false;
    try {
      if (await Notification.requestPermission() !== 'default') rememberAsked();
    } catch {
      rememberAsked();
    }
  };

  #decline = (): void => {
    rememberAsked();
    this.visible = false;
  };
}

customElements.define('dl-notification-offer', DlNotificationOffer);

declare global {
  interface HTMLElementTagNameMap {
    'dl-notification-offer': DlNotificationOffer;
  }
}
