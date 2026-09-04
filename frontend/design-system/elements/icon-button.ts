// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Icon-only control. Chrome is Shadow; the host owns the accessible name. */

import {render} from 'lit';
import {type IconName, type IconSize, icon } from '../icons/icon.ts';

const SIZES = new Set<IconSize>(['xs', 'sm', 'md', 'lg']);

const template = document.createElement('template');
template.innerHTML = `
  <style>
    :host { display: inline-flex; color: var(--color-text-dim, currentColor); }
    button {
      align-items: center;
      background: none;
      border: none;
      border-radius: var(--radius-control, 6px);
      color: inherit;
      cursor: pointer;
      display: inline-flex;
      justify-content: center;
      margin: 0;
      min-block-size: var(--control-hit-target, 44px);
      min-inline-size: var(--control-hit-target, 44px);
      padding: var(--space-inline, 0.25rem) var(--space-tight, 0.5rem);
    }
    button:hover { background: var(--color-bg-elevated, transparent); }
    button:focus-visible {
      outline: 2px solid var(--color-border-focus, currentColor);
      outline-offset: 2px;
    }
    button:disabled { cursor: default; opacity: 0.5; }
    #icon, #icon svg { display: block; }
    #icon svg {
      color: inherit;
      height: var(--dl-icon-size-sm, 16px);
      overflow: visible;
      stroke: currentColor;
      stroke-width: var(--dl-icon-stroke, 1.75);
      width: var(--dl-icon-size-sm, 16px);
    }
    :host([size='xs']) #icon svg {
      height: var(--dl-icon-size-xs, 12px);
      width: var(--dl-icon-size-xs, 12px);
    }
    :host([size='md']) #icon svg {
      height: var(--dl-icon-size-md, 20px);
      width: var(--dl-icon-size-md, 20px);
    }
    :host([size='lg']) #icon svg {
      height: var(--dl-icon-size-lg, 24px);
      width: var(--dl-icon-size-lg, 24px);
    }
  </style>
  <button type="button" part="button"><span id="icon"></span><slot></slot></button>
`;

export class DlIconButton extends HTMLElement {
  static readonly observedAttributes = ['name', 'size', 'disabled'];

  readonly #button: HTMLButtonElement;
  readonly #icon: HTMLSpanElement;

  constructor() {
    super();
    const shadow = this.attachShadow({mode: 'open'});
    shadow.append(template.content.cloneNode(true));
    this.#button = shadow.querySelector('button')!;
    this.#icon = shadow.querySelector('#icon')!;
  }

  connectedCallback(): void {
    this.#sync();
  }

  attributeChangedCallback(): void {
    this.#sync();
  }

  get name(): string {
    return this.getAttribute('name') ?? '';
  }
  set name(value: string) {
    this.setAttribute('name', value);
  }

  get size(): IconSize {
    const value = this.getAttribute('size');
    return value && SIZES.has(value as IconSize) ? value as IconSize : 'sm';
  }
  set size(value: IconSize) {
    this.setAttribute('size', value);
  }

  get disabled(): boolean {
    return this.hasAttribute('disabled');
  }
  set disabled(value: boolean) {
    this.toggleAttribute('disabled', value);
  }

  override focus(options?: FocusOptions): void {
    this.#button.focus(options);
  }

  #sync(): void {
    this.#button.disabled = this.disabled;
    const label = this.getAttribute('aria-label');
    if (label) this.#button.setAttribute('aria-label', label);
    else this.#button.removeAttribute('aria-label');
    const name = this.name as IconName;
    if (!name) {
      render(null, this.#icon);
      return;
    }
    try {
      render(icon(name, {size: this.size}), this.#icon);
    } catch {
      render(null, this.#icon);
    }
  }
}
