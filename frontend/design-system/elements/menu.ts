// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Slotted menuitems stay Light DOM; the host is the menu chrome and keyboard. */

const template = document.createElement('template');
template.innerHTML = `
  <style>
    :host { display: block; }
    slot { display: contents; }
  </style>
  <slot></slot>
`;

export class DlMenu extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({mode: 'open'});
    shadow.append(template.content.cloneNode(true));
    this.addEventListener('keydown', this.#onKeydown);
  }

  connectedCallback(): void {
    if (!this.hasAttribute('role')) this.setAttribute('role', 'menu');
  }

  #items(): HTMLElement[] {
    return [...this.querySelectorAll<HTMLElement>('[role="menuitem"]')]
      .filter((item) => !item.hasAttribute('disabled'));
  }

  #onKeydown = (event: KeyboardEvent): void => {
    const items = this.#items();
    if (items.length === 0) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      this.dispatchEvent(new CustomEvent('dl-menu-dismiss', {bubbles: true, composed: true}));
      return;
    }
    if (
      event.key !== 'ArrowDown'
      && event.key !== 'ArrowUp'
      && event.key !== 'Home'
      && event.key !== 'End'
    ) return;
    event.preventDefault();
    const current = items.indexOf(document.activeElement as HTMLElement);
    let next: number;
    if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = items.length - 1;
    else {
      const delta = event.key === 'ArrowDown' ? 1 : -1;
      next = current < 0
        ? (delta > 0 ? 0 : items.length - 1)
        : (current + delta + items.length) % items.length;
    }
    items[next]?.focus();
  };
}
