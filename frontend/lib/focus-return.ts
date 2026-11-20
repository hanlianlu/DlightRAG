// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The focus a dialog or panel gives back to what opened it.
 *
 * Closing hands focus back a frame later, once the surface is out of the way. A surface that opens again
 * before that frame, or leaves the page, has made the pending hand-back stale, so the frame checks that
 * nothing happened in between before it moves focus.
 */

import {focusedElement} from './dom.ts';

export class FocusReturn {
  #target: HTMLElement | null = null;
  #generation = 0;

  /** The element focus goes back to, while one is remembered. */
  get target(): HTMLElement | null {
    return this.#target;
  }

  /** Remember where focus goes back to: the opener when the caller names one, else what holds focus now. */
  remember(opener?: HTMLElement | null): void {
    this.#target = opener ?? focusedElement();
  }

  /** Make a hand-back still waiting for its frame stale: the surface opened again or left the page. */
  invalidate(): void {
    this.#generation += 1;
  }

  /** The surface closed and focus stays where it is: forget the opener and any hand-back waiting. */
  discard(): void {
    this.invalidate();
    this.#target = null;
  }

  /** The surface closed: forget the opener and, a frame later, focus it unless `isOpen` says the surface
   *  has opened again or `invalidate` made this hand-back stale. */
  restore(isOpen: () => boolean): void {
    this.invalidate();
    const generation = this.#generation;
    const target = this.#target;
    this.#target = null;
    window.requestAnimationFrame(() => {
      if (generation !== this.#generation || isOpen()) return;
      target?.focus();
    });
  }
}
