// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The toast Feature: a replace-in-place region with an optional asynchronous action.
 *
 *  A host shows two, one for each tone. A receipt passes in a moment and is announced politely. An error
 *  is announced at once and stays until the reader dismisses it or eight seconds are up, not counting the
 *  time a modal keeps the reader from it; a modal pauses its clock but does not make it inert. A new
 *  toast replaces the one of its own tone and leaves the other alone, so an error is not lost to the
 *  receipt that follows it.
 */

import {msg} from '@lit/localize';
import {html, nothing, type PropertyValues, type TemplateResult} from 'lit';
import {raise} from '../lib/dom.ts';
import {LightElement} from '../lib/lit-host.ts';

export type ToastTone = 'receipt' | 'error';

/** How long a toast stays, not counting the time it is paused: a glance for a receipt, a reading for an error. */
const TOAST_DURATION: Record<ToastTone, number> = {receipt: 1500, error: 8000};

interface ActionToastOptions {
  actionLabel: string;
  onAction: () => Promise<string | undefined>;
  /** Move focus to the action once it shows: for a command whose own control has just gone away. */
  focus?: boolean;
}

export interface ToastRequestDetail {
  message: string;
  /** What the message reports: something done, or something that failed or was refused.
   *  A receipt unless it says otherwise. */
  tone?: ToastTone;
  action?: ActionToastOptions;
}

/** The region under `root` that shows toasts of this tone. */
export function toastRegionOf(root: ParentNode, tone: ToastTone = 'receipt'): DlToastRegion | undefined {
  return Array.from(root.querySelectorAll('dl-toast-region')).find((region) => region.tone === tone);
}

/** Accessible toast state, timer lifecycle, and asynchronous action ownership. */
export class DlToastRegion extends LightElement {
  static properties = {
    tone: {reflect: true},
    shellInert: {attribute: false},
    request: {state: true},
    visible: {state: true},
    pending: {state: true},
  };

  /** Which toasts this region shows. It is a live region of the matching role: a status for receipts,
   *  an alert for errors. */
  declare tone: ToastTone;
  declare shellInert: boolean;
  /** The toast on screen, or the one fading out: it goes with its fade, so the pill never fades out empty. */
  declare request: ToastRequestDetail | null;
  declare visible: boolean;
  declare pending: boolean;

  #timer: ReturnType<typeof setTimeout> | null = null;
  #remaining = 0;
  #startedAt = 0;
  #hovered = false;
  #focused = false;
  /** Where focus was before it came into the toast, for the reader who dismisses it from the keyboard. */
  #cameFrom: HTMLElement | null = null;

  constructor() {
    super();
    this.tone = 'receipt';
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

  override connectedCallback(): void {
    super.connectedCallback();
    this.setAttribute('role', this.tone === 'error' ? 'alert' : 'status');
  }

  override disconnectedCallback(): void {
    this.#stopTimer();
    this.request = null;
    this.visible = false;
    this.pending = false;
    this.#hovered = false;
    this.#focused = false;
    this.#cameFrom = null;
    super.disconnectedCallback();
  }

  /** Replace the current toast with a `dl-toast-request`: a plain message, or one with an asynchronous action. */
  show(request: ToastRequestDetail): void {
    this.request = request;
    this.visible = true;
    this.pending = false;
    this.#startTimer();
    if (request.action?.focus) void this.#focusAction();
  }

  protected override updated(changed: PropertyValues<this>): void {
    this.classList.toggle('visible', this.visible);
    // A receipt is out of reach while a modal holds the shell; an error is not, because it must be read.
    this.inert = !this.visible || (this.shellInert && this.tone === 'receipt');
    if (changed.has('shellInert') && this.visible && this.#awaitsReader()) {
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
      ${this.tone === 'error' ? html`
        <dl-icon-button class="toast-close" name="close" size="sm" aria-label=${msg('Dismiss', {id: 'toast.dismiss'})}
                        @click=${this.#dismiss}></dl-icon-button>
      ` : nothing}
    `;
  }

  /** The action takes focus once it is on screen, unless another receipt has already replaced it. */
  async #focusAction(): Promise<void> {
    const request = this.request;
    await this.updateComplete;
    if (this.request === request) this.querySelector<HTMLElement>('.toast-action')?.focus();
  }

  /** Take the toast away, whether its time is up or the reader pressed its close button. The pointer and the
   *  focus that were on it go with it. */
  #hide = (): void => {
    this.#stopTimer();
    this.visible = false;
    this.pending = false;
    this.#remaining = 0;
    this.#hovered = false;
    this.#focused = false;
    this.#cameFrom = null;
  };

  /** The reader pressed Dismiss. From the keyboard the toast holds focus and is about to go inert, which
   *  would drop focus to the top of the page: it goes back to where it was before it came here, and where
   *  that is gone the Shell is asked to place it. A pointer has no place to keep. */
  #dismiss = (event: MouseEvent): void => {
    if (event.detail === 0 && this.contains(document.activeElement)) {
      this.#cameFrom?.focus();
      if (this.contains(document.activeElement)) raise(this, 'dl-toast-focus-lost');
    }
    this.#hide();
  };

  #stopTimer(): void {
    if (this.#timer) clearTimeout(this.#timer);
    this.#timer = null;
  }

  #startTimer(): void {
    this.#stopTimer();
    this.#remaining = TOAST_DURATION[this.tone];
    this.#resume();
  }

  /** An error is read before it goes and an Undo is pressed before it goes, so neither runs out while a
   *  modal keeps the reader from it. A receipt with nothing to press is a glance, and passes. */
  #awaitsReader(): boolean {
    return this.tone === 'error' || Boolean(this.request?.action);
  }

  #hasPauseReason(): boolean {
    return this.#hovered || this.#focused || (this.shellInert && this.#awaitsReader());
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

  #focusEntered = (event: FocusEvent): void => {
    const from = event.relatedTarget;
    if (!(from instanceof Node && this.contains(from))) {
      this.#cameFrom = from instanceof HTMLElement ? from : null;
    }
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
      // The Undo is gone either way; what failed is told as an error, where errors are told.
      if (this.request === request) this.#hide();
      raise(this, 'dl-toast-request', {
        message: msg('Could not undo the change.', {id: 'toast.undoFailed'}),
        tone: 'error',
      });
      return;
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
    /** A toast the reader dismissed from the keyboard held focus, and the place it came from cannot take it
     *  back: the Shell puts focus where the reader works. */
    'dl-toast-focus-lost': CustomEvent<void>;
  }
}
