// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The one keyboard contract for menus, and the element that applies it.

 *  A menu button opens its menu with ArrowDown, Enter, or Space on the first
 *  item and with ArrowUp on the last. Inside, ArrowDown and ArrowUp move with
 *  wrapping, Home and End jump to the ends, a printable character moves to
 *  the next item whose label starts with it, and Escape dismisses so the
 *  owner can return focus to the button. Enter and Space stay with each
 *  item's own activation. Pickers that are not menus reuse the same roving
 *  step through rovingFocusKeydown().
 *
 *  Slotted items stay Light DOM; the host is the menu chrome and keyboard.
 */

const template = document.createElement('template');
template.innerHTML = `
  <style>
    :host { display: block; }
    slot { display: contents; }
  </style>
  <slot></slot>
`;

const MENU_ITEMS = '[role="menuitem"], [role="menuitemradio"], [role="menuitemcheckbox"]';

export type MenuFocus = 'first' | 'last';

/** Which item a key on a menu button opens its menu on; null when the key is not the button's. */
export function menuButtonFocus(event: KeyboardEvent): MenuFocus | null {
  if (event.key === 'ArrowUp') return 'last';
  if (event.key === 'ArrowDown' || event.key === 'Enter' || event.key === ' ') return 'first';
  return null;
}

function itemLabel(item: HTMLElement): string {
  return (item.getAttribute('aria-label') ?? item.textContent ?? '').trim().toLocaleLowerCase();
}

/** The next item whose label starts with a typed character, searching on from the current one. */
function typeaheadIndex(event: KeyboardEvent, items: readonly HTMLElement[], current: number): number | null {
  const typed = event.key;
  if (typed.length !== 1 || typed === ' ' || event.altKey || event.ctrlKey || event.metaKey) return null;
  // A field inside a picker keeps its own typing.
  if (!items.includes(event.target as HTMLElement)) return null;
  const letter = typed.toLocaleLowerCase();
  for (let step = 1; step <= items.length; step += 1) {
    const index = (current + step + items.length) % items.length;
    if (itemLabel(items[index]!).startsWith(letter)) return index;
  }
  return null;
}

/** Apply the roving step for one key; true when the key moved focus. */
export function rovingFocusKeydown(event: KeyboardEvent, items: readonly HTMLElement[]): boolean {
  if (items.length === 0) return false;
  const current = items.indexOf(document.activeElement as HTMLElement);
  const inField = event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement;
  // A field keeps its own caret keys; only the arrows leave it for the items.
  if (inField && (event.key === 'Home' || event.key === 'End')) return false;
  let next: number | null;
  if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = items.length - 1;
  else if (event.key === 'ArrowDown') next = current < 0 ? 0 : (current + 1) % items.length;
  else if (event.key === 'ArrowUp') {
    next = current < 0 ? items.length - 1 : (current - 1 + items.length) % items.length;
  } else next = typeaheadIndex(event, items, current);
  if (next === null) return false;
  event.preventDefault();
  items[next]?.focus();
  return true;
}

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

  /** Focus the first or last enabled item, as the menu button's key asked. */
  focusItem(which: MenuFocus): void {
    const items = this.#items();
    (which === 'first' ? items[0] : items.at(-1))?.focus();
  }

  #items(): HTMLElement[] {
    return [...this.querySelectorAll<HTMLElement>(MENU_ITEMS)]
      .filter((item) => !item.hasAttribute('disabled') && item.getAttribute('aria-disabled') !== 'true');
  }

  #onKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      this.dispatchEvent(new CustomEvent('dl-menu-dismiss', {bubbles: true, composed: true}));
      return;
    }
    rovingFocusKeydown(event, this.#items());
  };
}
