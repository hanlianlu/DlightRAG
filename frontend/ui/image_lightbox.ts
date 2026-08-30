// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Image Lightbox Feature: gallery navigation, focus, keyboard, and safe URLs. */

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type TemplateResult} from 'lit';
import {icon} from '../design-system/index.ts';
import {wrapTabFocus} from '../lib/dom.ts';
import {LightElement} from '../lib/lit_host.ts';
import {safeImageSrc} from '../lib/urls.ts';
import lightboxStyles from '../styles/lightbox.module.css';

export interface ImageOpenDetail {
  src: string;
  gallery: readonly string[];
  returnFocus: HTMLElement;
}

/** Owns the modal image viewer and its document-level keyboard lifecycle. */
export class DlImageLightbox extends LightElement {
  static properties = {
    openState: {state: true},
    current: {state: true},
    gallery: {state: true},
  };

  declare openState: boolean;
  declare current: string;
  declare gallery: readonly string[];

  #returnFocus: HTMLElement | null = null;
  #events: AbortController | null = null;
  #focusGeneration = 0;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.openState = false;
    this.current = '';
    this.gallery = [];
    this.addEventListener('click', this.#backdropClick);
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.classList.add(lightboxStyles.imageLightbox);
    this.setAttribute('role', 'dialog');
    this.setAttribute('aria-modal', 'true');
    this.setAttribute('aria-label', msg('Image viewer', {id: 'imageLightbox.ariaLabel'}));
    this.tabIndex = -1;
    const events = new AbortController();
    this.#events = events;
    document.addEventListener('keydown', this.#keydown, {capture: true, signal: events.signal});
    this.#syncHost();
  }

  override disconnectedCallback(): void {
    this.#events?.abort();
    this.#events = null;
    this.#focusGeneration += 1;
    super.disconnectedCallback();
  }

  /** Open one safe image and an optional ordered gallery. */
  async open(
    src: unknown,
    returnFocus?: HTMLElement | null,
    gallery: readonly string[] = [],
  ): Promise<void> {
    const current = safeImageSrc(src);
    if (!current) return;
    this.#focusGeneration += 1;
    const safeGallery = [...new Set(gallery.map(safeImageSrc).filter(Boolean))];
    this.gallery = safeGallery.includes(current) ? safeGallery : [current, ...safeGallery];
    this.current = current;
    this.#returnFocus = returnFocus ?? (
      document.activeElement instanceof HTMLElement ? document.activeElement : null
    );
    this.openState = true;
    this.#syncHost();
    this.#publishState();
    await this.updateComplete;
    this.focus();
  }

  close(): void {
    if (!this.openState) return;
    const focusGeneration = ++this.#focusGeneration;
    this.openState = false;
    this.current = '';
    this.gallery = [];
    this.#syncHost();
    this.#publishState();
    const returnFocus = this.#returnFocus;
    this.#returnFocus = null;
    window.requestAnimationFrame(() => {
      if (focusGeneration !== this.#focusGeneration || this.openState) return;
      if (returnFocus?.isConnected && !returnFocus.inert) returnFocus.focus();
    });
  }

  protected override updated(): void {
    this.#syncHost();
  }

  protected override render(): TemplateResult | typeof nothing {
    if (!this.openState) return nothing;
    const index = this.gallery.indexOf(this.current);
    const multiple = this.gallery.length > 1;
    return html`
      <button class=${lightboxStyles.imageLightboxPrev} type="button" aria-label=${msg('Previous', {id: 'imageLightbox.previous'})}
              ?hidden=${!multiple || index <= 0} @click=${() => this.#navigate(-1)}>${icon('previous', {size: 'lg'})}</button>
      <button class=${lightboxStyles.imageLightboxNext} type="button" aria-label=${msg('Next', {id: 'imageLightbox.next'})}
              ?hidden=${!multiple || index >= this.gallery.length - 1}
              @click=${() => this.#navigate(1)}>${icon('next', {size: 'lg'})}</button>
      <img class=${lightboxStyles.imageLightboxImg} src=${this.current} alt=${msg('Source image', {id: 'imageLightbox.sourceImageAlt'})}>
    `;
  }

  #syncHost(): void {
    this.classList.toggle(lightboxStyles.open, this.openState);
    this.setAttribute('aria-hidden', this.openState ? 'false' : 'true');
    this.inert = !this.openState;
  }

  #navigate(direction: number): void {
    const index = this.gallery.indexOf(this.current);
    if (index < 0 || this.gallery.length <= 1) return;
    const next = (index + direction + this.gallery.length) % this.gallery.length;
    this.current = this.gallery[next];
  }

  #publishState(): void {
    this.dispatchEvent(new CustomEvent<{open: boolean}>('dl-image-lightbox-state-change', {
      bubbles: true,
      composed: true,
      detail: {open: this.openState},
    }));
  }

  #backdropClick = (event: Event): void => {
    if (event.target === this) this.close();
  };

  #focusables(): HTMLElement[] {
    return Array.from(this.querySelectorAll<HTMLElement>('button:not([hidden])'))
      .filter((element) => element.getClientRects().length > 0);
  }

  #keydown = (event: KeyboardEvent): void => {
    if (!this.openState) return;
    if (event.key === 'Escape') {
      // Native dialogs opened above the viewer own Escape while they are active.
      if (document.querySelector('dialog[open]')) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      this.close();
      return;
    }
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      this.#navigate(-1);
      return;
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      this.#navigate(1);
      return;
    }
    if (event.key !== 'Tab') return;
    const focusables = this.#focusables();
    if (focusables.length === 0) {
      event.preventDefault();
      this.focus();
      return;
    }
    const active = document.activeElement;
    if (active === this || !this.contains(active)) {
      event.preventDefault();
      (event.shiftKey ? focusables.at(-1) : focusables[0])?.focus();
      return;
    }
    wrapTabFocus(focusables, event);
  };
}

customElements.define('dl-image-lightbox', DlImageLightbox);

declare global {
  interface HTMLElementTagNameMap {
    'dl-image-lightbox': DlImageLightbox;
  }

  interface HTMLElementEventMap {
    'dl-image-open': CustomEvent<ImageOpenDetail>;
    'dl-image-lightbox-state-change': CustomEvent<{open: boolean}>;
  }
}
