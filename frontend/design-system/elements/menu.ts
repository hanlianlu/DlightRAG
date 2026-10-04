// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The one keyboard contract for menus, and the element that applies it.

 *  A menu button opens its menu with ArrowDown, Enter, or Space on the first
 *  item and with ArrowUp on the last. Inside, ArrowDown and ArrowUp move with
 *  wrapping, Home and End jump to the ends, a printable character moves to
 *  the next item whose label starts with it, and Enter and Space stay with
 *  each item's own activation. An aria-disabled item still takes focus but
 *  never activates. The menu asks its owner to close it with
 *  dl-menu-dismiss: Escape asks for focus back on the menu button, while Tab,
 *  or focus moving to an element other than the menu or the button that
 *  controls it (aria-controls), leaves focus where it went. Focus that lands
 *  on no element stays the owner's to handle with its outside-click
 *  dismissal: WebKit does not focus a pressed button, so a press on the
 *  menu's own button reports none. Pickers that are not menus reuse the same
 *  roving step through rovingFocusKeydown().
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

/** Why a menu asks to close: only Escape returns focus to its button. */
export interface MenuDismissDetail {
  readonly restoreFocus: boolean;
}

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
  const item = items[next];
  // Engines scroll focus differently (Firefox leaves a partly visible item where it is), so
  // the step owns the scroll: the item lands wholly in view, with any scroll-margin it sets.
  item?.focus({preventScroll: true});
  item?.scrollIntoView({block: 'nearest'});
  return true;
}

export class DlMenu extends HTMLElement {
  constructor() {
    super();
    const shadow = this.attachShadow({mode: 'open'});
    shadow.append(template.content.cloneNode(true));
    this.addEventListener('keydown', this.#onKeydown);
    this.addEventListener('focusout', this.#onFocusout);
    // Capture runs before an item's own click handler, so a disabled item never activates.
    this.addEventListener('click', this.#onClick, {capture: true});
  }

  connectedCallback(): void {
    if (!this.hasAttribute('role')) this.setAttribute('role', 'menu');
    // A press on the menu's own chrome keeps focus inside it instead of leaving it.
    if (!this.hasAttribute('tabindex')) this.tabIndex = -1;
  }

  /** Focus the first or last item, as the menu button's key asked. */
  focusItem(which: MenuFocus): void {
    const items = this.#items();
    (which === 'first' ? items[0] : items.at(-1))?.focus();
  }

  /** The items focus moves among: aria-disabled ones included, natively disabled ones cannot take it. */
  #items(): HTMLElement[] {
    return [...this.querySelectorAll<HTMLElement>(MENU_ITEMS)]
      .filter((item) => !item.hasAttribute('disabled'));
  }

  #dismiss(restoreFocus: boolean): void {
    this.dispatchEvent(new CustomEvent<MenuDismissDetail>('dl-menu-dismiss', {
      bubbles: true,
      composed: true,
      detail: {restoreFocus},
    }));
  }

  #onKeydown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      this.#dismiss(true);
      return;
    }
    // Tab leaves the menu: the browser moves focus on and the menu closes behind it.
    if (event.key === 'Tab') {
      this.#dismiss(false);
      return;
    }
    rovingFocusKeydown(event, this.#items());
  };

  #onFocusout = (event: FocusEvent): void => {
    if (this.hidden || !this.isConnected) return;
    const next = event.relatedTarget;
    // No element: a press WebKit does not focus (the menu's own button among
    // them) or the window losing focus. Closing here would let that button's
    // click reopen the menu; the owner's outside-click dismissal decides.
    if (!(next instanceof Node) || this.contains(next)) return;
    // The menu's own button closes it on its click; closing here would reopen it there.
    if (next instanceof Element && this.id
        && (next.getAttribute('aria-controls') ?? '').split(/\s+/).includes(this.id)) return;
    this.#dismiss(false);
  };

  #onClick = (event: MouseEvent): void => {
    const item = event.target instanceof Element ? event.target.closest(MENU_ITEMS) : null;
    if (!item || !this.contains(item) || item.getAttribute('aria-disabled') !== 'true') return;
    event.preventDefault();
    event.stopImmediatePropagation();
  };
}

declare global {
  interface HTMLElementEventMap {
    'dl-menu-dismiss': CustomEvent<MenuDismissDetail>;
  }
}
