// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Native split-layout behavior. Product state and persistence belong to adapters. */

import {isImeKey} from './ime.ts';

type SplitOrientation = 'horizontal' | 'vertical';
type SplitPrimary = 'start' | 'end';

interface SplitPositionDetail {
  readonly position: number;
}

const template = document.createElement('template');
template.innerHTML = `
  <style>
    :host {
      --dl-split-size: 0px;
      display: block;
      min-block-size: 0;
      min-inline-size: 0;
    }
    #layout {
      block-size: 100%;
      display: flex;
      inline-size: 100%;
      isolation: isolate;
      min-block-size: 0;
      min-inline-size: 0;
      position: relative;
    }
    :host(:not([orientation='vertical'])) #layout { flex-direction: row; }
    :host([orientation='vertical']) #layout { flex-direction: column; }
    slot {
      display: block;
      min-block-size: 0;
      min-inline-size: 0;
      overflow: hidden;
      position: relative;
    }
    #start, #end { flex: 1 1 auto; }
    #start { z-index: var(--split-start-layer, 0); }
    #end { z-index: var(--split-end-layer, 0); }
    :host([primary='start']) #start,
    :host(:not([primary='end'])) #start { flex: 0 0 var(--dl-split-size); }
    :host([primary='end']) #end { flex: 0 0 var(--dl-split-size); }
    ::slotted(*) {
      block-size: 100%;
      box-sizing: border-box;
      inline-size: 100%;
      min-block-size: 0;
      min-inline-size: 0;
    }
    #divider {
      background: var(--color-border-subtle, rgb(120 113 108 / 16%));
      flex: 0 0 var(--split-divider-size, 1px);
      opacity: 1;
      outline: none;
      position: relative;
      touch-action: none;
      transition: background var(--duration-control, 150ms), opacity var(--duration-control, 150ms);
      z-index: 1;
    }
    #divider::before {
      content: '';
      inset: 0;
      pointer-events: auto;
      position: absolute;
    }
    :host(:not([orientation='vertical'])) #divider { cursor: col-resize; }
    :host(:not([orientation='vertical'])) #divider::before {
      inset-inline: calc((1px - var(--split-hit-target, 12px)) / 2);
    }
    :host([orientation='vertical']) #divider { cursor: row-resize; }
    :host([orientation='vertical']) #divider::before {
      inset-block: calc((1px - var(--split-hit-target, 12px)) / 2);
    }
    :host([disabled]) #divider,
    :host([data-collapsed]) #divider {
      flex-basis: 0;
      opacity: 0;
      pointer-events: none;
    }
    :host(:not([disabled]):not([data-collapsed])) #divider:hover,
    :host(:not([disabled]):not([data-collapsed])) #divider:focus-visible {
      background: var(--focus-ring-color, rgb(210 182 97 / 64%));
    }
    :host(:not([disabled]):not([data-collapsed])) #divider:focus-visible {
      box-shadow: 0 0 0 2px var(--focus-ring-color, rgb(210 182 97 / 64%));
    }
    @media (forced-colors: active) {
      #divider { background: CanvasText; }
      :host(:not([disabled]):not([data-collapsed])) #divider:focus-visible {
        background: Highlight;
      }
    }
  </style>
  <div id="layout">
    <slot id="start" name="start" part="start"></slot>
    <div id="divider" part="divider" role="separator" tabindex="0"></div>
    <slot id="end" name="end" part="end"></slot>
  </div>
`;

export class DlSplitLayout extends HTMLElement {
  static readonly observedAttributes = ['size', 'min', 'max', 'primary', 'orientation', 'disabled', 'label'];

  readonly #divider: HTMLDivElement;
  #dragStartCoordinate = 0;
  #dragStartSize = 0;
  #activePointerId: number | null = null;
  /** What the panel was before it was collapsed, which Enter puts back (APG window splitter). */
  #restoreSize: number | null = null;

  constructor() {
    super();
    const shadow = this.attachShadow({mode: 'open'});
    shadow.append(template.content.cloneNode(true));
    const divider = shadow.querySelector<HTMLDivElement>('#divider');
    // The template above is this file's own, so a miss is a mistake here, not a state to run in.
    if (divider === null) throw new Error('dl-split-layout template has no #divider');
    this.#divider = divider;
    this.#divider.addEventListener('pointerdown', this.#pointerDown);
    this.#divider.addEventListener('keydown', this.#keyDown);
    this.#divider.addEventListener('lostpointercapture', this.#lostPointerCapture);
    // Double-clicking a sash asks for the panel's own default width, which the caller owns: VS Code
    // learned to keep that apart from collapsing, and from the window's idea of a good width.
    this.#divider.addEventListener('dblclick', () => {
      this.dispatchEvent(new CustomEvent('dl-split-reset', {bubbles: true, composed: true}));
    });
  }

  connectedCallback(): void {
    if (!this.hasAttribute('orientation')) this.orientation = 'horizontal';
    if (!this.hasAttribute('primary')) this.primary = 'start';
    this.#sync();
  }

  disconnectedCallback(): void {
    this.#cancelDrag(false);
  }

  attributeChangedCallback(name: string): void {
    if (name === 'label') {
      this.#divider.setAttribute('aria-label', this.getAttribute('label') ?? '');
      return;
    }
    if (name === 'disabled' && this.disabled) this.#cancelDrag(false);
    this.#sync();
  }

  get size(): number { return this.#numberAttribute('size', 0); }
  set size(value: number) { this.setAttribute('size', String(Math.max(0, value))); }

  get min(): number { return this.#numberAttribute('min', 0); }
  set min(value: number) { this.setAttribute('min', String(Math.max(0, value))); }

  get max(): number { return this.#numberAttribute('max', Number.POSITIVE_INFINITY); }
  set max(value: number) {
    if (Number.isFinite(value)) this.setAttribute('max', String(Math.max(0, value)));
    else this.removeAttribute('max');
  }

  get primary(): SplitPrimary { return this.getAttribute('primary') === 'end' ? 'end' : 'start'; }
  set primary(value: SplitPrimary) { this.setAttribute('primary', value); }

  get orientation(): SplitOrientation {
    return this.getAttribute('orientation') === 'vertical' ? 'vertical' : 'horizontal';
  }
  set orientation(value: SplitOrientation) { this.setAttribute('orientation', value); }

  get disabled(): boolean { return this.hasAttribute('disabled'); }
  set disabled(value: boolean) { this.toggleAttribute('disabled', value); }

  get divider(): HTMLDivElement { return this.#divider; }

  #numberAttribute(name: string, fallback: number): number {
    if (!this.hasAttribute(name)) return fallback;
    const value = Number(this.getAttribute(name));
    return Number.isFinite(value) ? value : fallback;
  }

  #effectiveMaximum(): number {
    const axisSize = this.orientation === 'horizontal' ? this.clientWidth : this.clientHeight;
    const configuredDivider = Number.parseFloat(
      getComputedStyle(this).getPropertyValue('--split-divider-size'),
    );
    const dividerSize = this.disabled
      ? 0
      : (Number.isFinite(configuredDivider) ? configuredDivider : 1);
    const containerMaximum = axisSize > 0
      ? Math.max(this.min, axisSize - dividerSize)
      : Number.POSITIVE_INFINITY;
    if (!Number.isFinite(this.max)) return containerMaximum;
    return Math.max(this.min, Math.min(this.max, containerMaximum));
  }

  #normalize(value: number, allowCollapsed = false): number {
    if (allowCollapsed && value <= 0) return 0;
    return Math.min(Math.max(value, this.min), this.#effectiveMaximum());
  }

  #setInteractiveSize(
    value: number,
    eventName: 'dl-split-input' | 'dl-split-change',
    allowCollapsed = false,
  ): void {
    const position = Math.round(this.#normalize(value, allowCollapsed));
    this.size = position;
    this.dispatchEvent(new CustomEvent<SplitPositionDetail>(eventName, {
      bubbles: true,
      composed: true,
      detail: {position},
    }));
  }

  /**
   * Where a divider being dragged settles, in the manner `react-resizable-panels` states it: a
   * collapsible panel collapses when dragged past `min` by the threshold, and expands when dragged
   * past the collapsed size by the same threshold. The two thresholds differ whenever the collapsed
   * size is not zero, and the state a drag is in decides which one applies — so a panel held near
   * the boundary does not flicker between the two, which is the hysteresis every snap needs.
   */
  #snapped(value: number): number {
    const threshold = this.#collapseThreshold();
    return this.size > 0
      ? (value < this.min - threshold ? 0 : value)
      : (value > threshold ? this.min : 0);
  }

  /** How far past the edge a drag has to go, half the gap by default, as the reference does. */
  #collapseThreshold(): number {
    const configured = Number.parseFloat(this.getAttribute('collapse-threshold') ?? '');
    if (Number.isFinite(configured) && configured >= 0) return configured;
    return this.min / 2;
  }

  /**
   * Shut a panel the reader asked to shut, or restore it to the width it had before — what the
   * APG window splitter pattern says Enter does, which is not the same as its smallest width.
   */
  #toggleCollapsed(): void {
    if (this.size > 0) {
      this.#restoreSize = this.size;
      this.#setInteractiveSize(0, 'dl-split-change', true);
      return;
    }
    this.#setInteractiveSize(this.#restoreSize ?? this.min, 'dl-split-change', true);
  }

  #sync(): void {
    const requested = this.size;
    const position = Math.round(this.#normalize(requested, true));
    if (this.isConnected && requested !== position) {
      this.setAttribute('size', String(position));
      return;
    }
    this.style.setProperty('--dl-split-size', `${position}px`);
    const collapsed = position <= 0;
    this.toggleAttribute('data-collapsed', collapsed);
    if (collapsed) {
      this.#divider.removeAttribute('role');
      this.#divider.setAttribute('aria-hidden', 'true');
      for (const name of [
        'aria-orientation', 'aria-valuemin', 'aria-valuemax', 'aria-valuenow', 'aria-disabled',
      ]) this.#divider.removeAttribute(name);
      this.#divider.tabIndex = -1;
      return;
    }
    this.#divider.setAttribute('role', 'separator');
    this.#divider.removeAttribute('aria-hidden');
    const verticalDivider = this.orientation === 'horizontal';
    this.#divider.setAttribute('aria-orientation', verticalDivider ? 'vertical' : 'horizontal');
    this.#divider.setAttribute('aria-valuemin', String(this.min));
    const maximum = this.#effectiveMaximum();
    if (Number.isFinite(maximum)) this.#divider.setAttribute('aria-valuemax', String(maximum));
    else this.#divider.removeAttribute('aria-valuemax');
    this.#divider.setAttribute('aria-valuenow', String(position));
    this.#divider.setAttribute('aria-disabled', String(this.disabled));
    this.#divider.tabIndex = this.disabled ? -1 : 0;
  }

  #coordinate(event: PointerEvent): number {
    return this.orientation === 'horizontal' ? event.clientX : event.clientY;
  }

  #writingDirection(): number {
    return this.orientation === 'horizontal' && getComputedStyle(this).direction === 'rtl' ? -1 : 1;
  }

  #cancelDrag(commit: boolean): void {
    const pointerId = this.#activePointerId;
    if (pointerId === null) return;
    this.#activePointerId = null;
    this.#divider.removeEventListener('pointermove', this.#pointerMove);
    this.#divider.removeEventListener('pointerup', this.#pointerUp);
    this.#divider.removeEventListener('pointercancel', this.#pointerUp);
    this.ownerDocument.removeEventListener('keydown', this.#escapeDrag, {capture: true});
    if (this.#divider.hasPointerCapture(pointerId)) {
      this.#divider.releasePointerCapture(pointerId);
    }
    if (commit && this.isConnected && !this.disabled) {
      // Commit the state the drag ended in: a panel it took past the threshold stays shut, rather
      // than being clamped back to its smallest width by the act of letting go.
      this.#setInteractiveSize(this.size, 'dl-split-change', this.size <= 0);
    }
  }

  readonly #pointerDown = (event: PointerEvent): void => {
    if (this.disabled || event.button !== 0 || this.#activePointerId !== null) return;
    event.preventDefault();
    this.#activePointerId = event.pointerId;
    this.#dragStartCoordinate = this.#coordinate(event);
    this.#dragStartSize = this.size;
    try {
      this.#divider.setPointerCapture(event.pointerId);
    } catch {
      // Synthetic and already-cancelled pointers can lack an active capture target.
    }
    this.#divider.addEventListener('pointermove', this.#pointerMove);
    this.#divider.addEventListener('pointerup', this.#pointerUp);
    this.#divider.addEventListener('pointercancel', this.#pointerUp);
    // A pointer does not focus the divider, so a key pressed during the drag goes to the page: listen
    // for Escape there until the drag is over, in the capture phase, which puts it ahead of the panels'
    // own handlers. A listener the page added to that phase earlier still goes first: an open popover
    // takes the first Escape to close itself, and the drag the next.
    this.ownerDocument.addEventListener('keydown', this.#escapeDrag, {capture: true});
  };

  /**
   * Escape during a pointer drag: the width it started at, not the one it reached. The panels close
   * themselves on Escape, from the page or from inside, so the key is spent here, before they see it.
   * Cancelling is not committing: only the width drawn goes back, as `dl-split-input`, and no
   * `dl-split-change` says the reader chose it. The width drawn can be a clamped one, and a clamp is
   * not a preference.
   */
  readonly #escapeDrag = (event: KeyboardEvent): void => {
    // An Escape that cancels an IME composition belongs to the composition.
    if (event.key !== 'Escape' || event.defaultPrevented || isImeKey(event)) return;
    event.preventDefault();
    event.stopPropagation();
    this.#setInteractiveSize(this.#dragStartSize, 'dl-split-input', this.#dragStartSize <= 0);
    this.#cancelDrag(false);
  };

  readonly #pointerMove = (event: PointerEvent): void => {
    if (event.pointerId !== this.#activePointerId) return;
    if (this.disabled) {
      this.#cancelDrag(false);
      return;
    }
    const primaryDirection = this.primary === 'start' ? 1 : -1;
    const direction = primaryDirection * this.#writingDirection();
    const delta = (this.#coordinate(event) - this.#dragStartCoordinate) * direction;
    this.#setInteractiveSize(this.#snapped(this.#dragStartSize + delta), 'dl-split-input', true);
  };

  readonly #pointerUp = (event: PointerEvent): void => {
    if (event.pointerId !== this.#activePointerId) return;
    this.#cancelDrag(true);
  };

  readonly #lostPointerCapture = (event: PointerEvent): void => {
    if (event.pointerId === this.#activePointerId) this.#cancelDrag(true);
  };

  readonly #keyDown = (event: KeyboardEvent): void => {
    if (this.disabled) return;
    const horizontal = this.orientation === 'horizontal';
    const decreasingKey = horizontal ? 'ArrowLeft' : 'ArrowUp';
    const increasingKey = horizontal ? 'ArrowRight' : 'ArrowDown';
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      this.#toggleCollapsed();
      return;
    }
    let next: number | null = null;
    if (event.key === 'Home') next = this.min;
    else if (event.key === 'End') {
      const maximum = this.#effectiveMaximum();
      if (Number.isFinite(maximum)) next = maximum;
    }
    else if (event.key === decreasingKey || event.key === increasingKey) {
      const axisDirection = event.key === increasingKey ? 1 : -1;
      const primaryDirection = this.primary === 'start' ? 1 : -1;
      next = this.size
        + (event.shiftKey ? 50 : 10)
          * axisDirection * primaryDirection * this.#writingDirection();
    } else {
      return;
    }
    if (next === null) return;
    event.preventDefault();
    this.#setInteractiveSize(next, 'dl-split-input');
    this.#setInteractiveSize(this.size, 'dl-split-change');
  };
}

declare global {
  interface HTMLElementTagNameMap {
    'dl-split-layout': DlSplitLayout;
  }
  interface HTMLElementEventMap {
    'dl-split-input': CustomEvent<SplitPositionDetail>;
    'dl-split-change': CustomEvent<SplitPositionDetail>;
  }
}
