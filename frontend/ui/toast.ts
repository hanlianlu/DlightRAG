// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One replace-in-place toast Feature with optional asynchronous action. */

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {LightElement} from '../lib/lit-host.ts';

/** How long a receipt stays, not counting the time it is paused. */
const TOAST_DURATION = 3000;

export interface ActionToastOptions {
  actionLabel: string;
  onAction: () => Promise<string | undefined>;
  /** Move focus to the action once it shows: for a command whose own control has just gone away. */
  focus?: boolean;
}

export interface ToastRequestDetail {
  message: string;
  action?: ActionToastOptions;
}

/** Accessible toast state, timer lifecycle, and asynchronous action ownership. */
export class DlToastRegion extends LightElement {
  static properties = {
    shellInert: {attribute: false},
    request: {state: true},
    visible: {state: true},
    pending: {state: true},
  };

  declare shellInert: boolean;
  /** The receipt on screen, or the one fading out: it goes with its fade, so the pill never fades out empty. */
  declare request: ToastRequestDetail | null;
  declare visible: boolean;
  declare pending: boolean;

  #timer: ReturnType<typeof setTimeout> | null = null;
  #remaining = 0;
  #startedAt = 0;
  #hovered = false;
  #focused = false;

  constructor() {
    super();
    this.shellInert = false;
    this.request = null;
    this.visible = false;
    this.pending = false;
    this.addEventListener('mouseenter', this.#pointerEntered);
    this.addEventListener('mouseleave', this.#pointerLeft);
    this.addEventListener('focusin', this.#focusEntered);
    this.addEventListener('focusout', this.#focusLeft);
    this.addEventListener('transitionend', this.#faded);
  }

  override disconnectedCallback(): void {
    this.#stopTimer();
    this.request = null;
    this.visible = false;
    this.pending = false;
    this.#hovered = false;
    this.#focused = false;
    super.disconnectedCallback();
  }

  /** Replace the current receipt with a `dl-toast-request`: a plain message, or one with an asynchronous action. */
  show(request: ToastRequestDetail): void {
    this.request = request;
    this.visible = true;
    this.pending = false;
    this.#startTimer();
    if (request.action?.focus) void this.#focusAction();
  }

  protected override updated(changed: PropertyValues<this>): void {
    this.classList.toggle('visible', this.visible);
    this.inert = !this.visible || this.shellInert;
    if (changed.has('shellInert') && this.visible && this.request?.action) {
      if (this.shellInert) this.#pause();
      else this.#resume();
    }
  }

  protected override render(): TemplateResult | typeof nothing {
    const request = this.request;
    if (!request) return nothing;
    return html`
      <span class="toast-message">${request.message}</span>
      ${request.action ? html`
        <button class="dl-btn toast-action" type="button" ?disabled=${this.pending}
                @click=${this.#runAction}>${request.action.actionLabel}</button>
      ` : nothing}
    `;
  }

  /** The action takes focus once it is on screen, unless another receipt has already replaced it. */
  async #focusAction(): Promise<void> {
    const request = this.request;
    await this.updateComplete;
    if (this.request === request) this.querySelector<HTMLElement>('.toast-action')?.focus();
  }

  #hide(): void {
    this.#stopTimer();
    this.visible = false;
    this.pending = false;
    this.#remaining = 0;
  }

  #stopTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #startTimer(): void {
    this.#stopTimer();
    this.#remaining = TOAST_DURATION;
    this.#resume();
  }

  #hasPauseReason(): boolean {
    return this.#hovered || this.#focused
      || Boolean(this.request?.action && this.shellInert);
  }

  #pause(): void {
    if (!this.#timer) return;
    this.#remaining = Math.max(0, this.#remaining - (performance.now() - this.#startedAt));
    this.#stopTimer();
  }

  #resume = (): void => {
    if (!this.visible || this.#timer || this.pending || this.#hasPauseReason()) return;
    this.#startedAt = performance.now();
    this.#timer = setTimeout(() => {
      this.#timer = null;
      this.#hide();
    }, this.#remaining);
  };

  #faded = (event: TransitionEvent): void => {
    if (event.target === this && event.propertyName === 'opacity' && !this.visible) this.request = null;
  };

  #pointerEntered = (): void => {
    this.#hovered = true;
    this.#pause();
  };

  #pointerLeft = (): void => {
    this.#hovered = false;
    this.#resume();
  };

  #focusEntered = (): void => {
    this.#focused = true;
    this.#pause();
  };

  #focusLeft = (event: FocusEvent): void => {
    if (event.relatedTarget instanceof Node && this.contains(event.relatedTarget)) return;
    this.#focused = false;
    this.#resume();
  };

  #runAction = async (): Promise<void> => {
    const request = this.request;
    if (!request?.action || this.pending) return;
    this.#stopTimer();
    this.pending = true;
    let message: string;
    try {
      message = await request.action.onAction()
        || msg('Change undone.', {id: 'toast.changeUndone'});
    } catch {
      message = msg('Could not undo the change.', {id: 'toast.undoFailed'});
    }
    if (this.request !== request) return;
    const settled: ToastRequestDetail = {message};
    this.request = settled;
    this.pending = false;
    await this.updateComplete;
    if (this.request !== settled) return;
    this.#focused = this.contains(document.activeElement);
    this.#startTimer();
  };
}

customElements.define('dl-toast-region', DlToastRegion);

declare global {
  interface HTMLElementTagNameMap {
    'dl-toast-region': DlToastRegion;
  }

  interface HTMLElementEventMap {
    'dl-toast-request': CustomEvent<ToastRequestDetail>;
  }
}
