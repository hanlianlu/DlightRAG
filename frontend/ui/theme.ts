// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Theme Control Feature and document color-mode capability. */

import {msg} from '@lit/localize';
import {html, type TemplateResult} from 'lit';
import {
  type DlMenu,
  icon,
  type IconName,
  menuButtonFocus,
  type MenuDismissDetail,
} from '../design-system/index.ts';
import {
  parseThemePreference,
  resolveColorMode,
  THEME_STORAGE_KEY,
  type ThemePreference,
} from '../lib/theme.ts';
import {LightElement} from '../lib/lit-host.ts';
import {TriggerPopover} from '../lib/popover.ts';
import {isLocalStorageEvent, readStored, writeStored} from '../lib/storage.ts';

function readPreference(): ThemePreference {
  const stored = parseThemePreference(readStored(THEME_STORAGE_KEY));
  return stored === 'system'
    ? parseThemePreference(document.documentElement.getAttribute('data-theme'))
    : stored;
}

function writePreference(preference: ThemePreference): void {
  writeStored(THEME_STORAGE_KEY, preference === 'system' ? null : preference);
}

/** Owns theme preference, menu accessibility, persistence, and system changes. */
export class DlThemeControl extends LightElement {
  static properties = {preference: {state: true}};

  declare preference: ThemePreference;

  #media: MediaQueryList | null = null;
  readonly #menu = new TriggerPopover(this, {
    trigger: () => this.querySelector<HTMLButtonElement>('#theme-trigger'),
    enter: (which) => { this.querySelector<DlMenu>('#theme-menu')?.focusItem(which); },
  });

  constructor() {
    super();
    this.preference = 'system';
  }

  override connectedCallback(): void {
    super.connectedCallback();
    this.preference = readPreference();
    this.#media = window.matchMedia('(prefers-color-scheme: dark)');
    this.#media.addEventListener('change', this.#mediaChanged, {signal: this.lifetime});
    window.addEventListener('storage', this.#storageChanged, {signal: this.lifetime});
    this.#apply();
  }

  override disconnectedCallback(): void {
    this.#media = null;
    super.disconnectedCallback();
  }

  protected override updated(): void {
    this.#apply();
  }

  protected override render(): TemplateResult {
    const appearance = msg('Appearance', {id: 'theme.appearance'});
    return html`
      <button id="theme-trigger" type="button" aria-label=${appearance} title=${appearance}
              aria-haspopup="menu" aria-controls="theme-menu"
              aria-expanded=${this.#menu.open ? 'true' : 'false'}
              @click=${this.#menu.toggle} @keydown=${this.#triggerKeydown}>
        ${icon('moon', {size: 'sm', className: 'theme-icon theme-icon-moon'})}
        ${icon('sun', {size: 'sm', className: 'theme-icon theme-icon-sun'})}
      </button>
      <dl-menu id="theme-menu" class="dl-anchored dl-anchored--end" role="menu" aria-label=${appearance}
           ?hidden=${!this.#menu.open} @dl-menu-dismiss=${this.#menuDismissed}>
        ${this.#option('system', msg('System', {id: 'theme.system'}), 'system')}
        ${this.#option('light', msg('Light', {id: 'theme.light'}), 'sun')}
        ${this.#option('dark', msg('Dark', {id: 'theme.dark'}), 'moon')}
      </dl-menu>
    `;
  }

  #option(value: ThemePreference, label: string, iconName: IconName): TemplateResult {
    const checked = this.preference === value;
    return html`
      <button type="button" role="menuitemradio" data-theme-value=${value} aria-label=${label}
              aria-checked=${checked ? 'true' : 'false'} tabindex="-1"
              @click=${() => this.#select(value)}>
        <span class="theme-menu-icon" aria-hidden="true">${icon(iconName, {size: 'sm'})}</span>
        <span class="theme-menu-label">${label}</span>
        <span class="theme-menu-check" aria-hidden="true">${icon('check', {size: 'xs'})}</span>
      </button>
    `;
  }

  #apply(): void {
    // Theme is an approved top-level browser capability; the root is its interface.
    const root = document.documentElement;
    const colorMode = resolveColorMode(this.preference, this.#media?.matches ?? false);
    root.setAttribute('data-theme', this.preference);
    root.setAttribute('data-color-mode', colorMode);
    root.style.colorScheme = colorMode;
  }

  #select(preference: ThemePreference): void {
    this.preference = preference;
    writePreference(preference);
    this.#menu.close(true);
  }

  #triggerKeydown = (event: KeyboardEvent): void => {
    const focus = menuButtonFocus(event);
    if (!focus) return;
    event.preventDefault();
    this.#menu.show(focus);
  };

  #menuDismissed = (event: CustomEvent<MenuDismissDetail>): void => {
    this.#menu.close(event.detail.restoreFocus);
  };

  #mediaChanged = (): void => {
    if (this.preference === 'system') this.#apply();
  };

  #storageChanged = (event: StorageEvent): void => {
    if (!isLocalStorageEvent(event)) return;
    if (event.key !== null && event.key !== THEME_STORAGE_KEY) return;
    this.preference = parseThemePreference(event.newValue);
  };
}

customElements.define('dl-theme-control', DlThemeControl);

declare global {
  interface HTMLElementTagNameMap {
    'dl-theme-control': DlThemeControl;
  }
}
