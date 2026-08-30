// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** One replace-in-place toast Feature with optional asynchronous action. */

import {msg, updateWhenLocaleChanges} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {LightElement} from '../lib/lit_host.ts';

const MAX_TOAST_DURATION = 3000;

export interface ActionToastOptions {
  actionLabel: string;
  onAction: () => Promise<string | void>;
  duration?: number;
}

export type ToastRequestDetail =
  | {message: string; duration?: number; action?: never}
  | {message: string; action: ActionToastOptions; duration?: never};

interface ToastRequest {
  message: string;
  duration: number;
  action: ActionToastOptions | null;
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
  declare request: ToastRequest | null;
  declare visible: boolean;
  declare pending: boolean;

  #timer: ReturnType<typeof setTimeout> | null = null;
  #remaining = 0;
  #startedAt = 0;
  #hovered = false;
  #focused = false;

  constructor() {
    super();
    updateWhenLocaleChanges(this);
    this.shellInert = false;
    this.request = null;
    this.visible = false;
    this.pending = false;
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.addEventListener('mouseenter', this.#pointerEntered);
    this.addEventListener('mouseleave', this.#pointerLeft);
    this.addEventListener('focusin', this.#focusEntered);
    this.addEventListener('focusout', this.#focusLeft);
  }

  override disconnectedCallback(): void {
    this.removeEventListener('mouseenter', this.#pointerEntered);
    this.removeEventListener('mouseleave', this.#pointerLeft);
    this.removeEventListener('focusin', this.#focusEntered);
    this.removeEventListener('focusout', this.#focusLeft);
    this.#stopTimer();
    this.request = null;
    this.visible = false;
    this.pending = false;
    this.#hovered = false;
    this.#focused = false;
    super.disconnectedCallback();
  }

  /** Replace the current receipt with a plain status message. */
  show(message: string, duration = 3000): void {
    this.#show({
      message,
      duration,
      action: null,
    });
  }

  /** Replace the current receipt with one asynchronous action. */
  showAction(message: string, options: ActionToastOptions): void {
    this.#show({
      message,
      duration: options.duration ?? MAX_TOAST_DURATION,
      action: options,
    });
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

  #show(request: ToastRequest): void {
    const bounded = {
      ...request,
      duration: Math.min(request.duration, MAX_TOAST_DURATION),
    };
    this.request = bounded;
    this.visible = true;
    this.pending = false;
    this.#startTimer(bounded.duration);
  }

  #hide(): void {
    this.#stopTimer();
    this.request = null;
    this.visible = false;
    this.pending = false;
    this.#remaining = 0;
  }

  #stopTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #startTimer(duration: number): void {
    this.#stopTimer();
    this.#remaining = duration;
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
    let duration: number;
    try {
      message = await request.action.onAction()
        || msg('Change undone.', {id: 'toast.changeUndone'});
      duration = 3000;
    } catch {
      message = msg('Could not undo the change.', {id: 'toast.undoFailed'});
      duration = 3000;
    }
    if (this.request !== request) return;
    const settled: ToastRequest = {message, duration, action: null};
    this.request = settled;
    this.pending = false;
    await this.updateComplete;
    if (this.request !== settled) return;
    this.#focused = this.contains(document.activeElement);
    this.#startTimer(duration);
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
